"""Thin multi-scenario orchestrator over :class:`CppSimulator`.

Joint-shared-θ inference often runs the same theta pool across multiple
scenarios (e.g. baseline + treatment arms). The per-scenario fan-out
already lives in :class:`CppSimulator` (3-tier cache, top-up,
combine + download). This module adds the joint-θ layer on top:

- one upload of the shared (sample_index, theta) CSV before the loop
- per-scenario ``run_hpc(n, samples_csv_remote=..., skip_setup=...)`` so
  scenarios 2..N skip the redundant venv refresh / git pull / cmake
- a :func:`joint_nan_mask` helper for cross-scenario alignment
- :meth:`MultiScenarioRunner.simulate_with_parameters_all` — the local,
  posterior-predictive counterpart: one fused C++ batch evaluates a
  user-supplied theta matrix under every scenario, evolving each theta
  once (#90 Phase 3)

Inference-agnostic: returns per-scenario ``(theta, x, sample_index)``
arrays the way :meth:`CppSimulator.run_hpc` already produces them.
Tensor packing, copula transforms, and NPE training stay in the caller.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from qsp_hpc.cpp.batch_runner import CppBatchRunner, FusedScenarioSpec, write_pool_manifest
from qsp_hpc.simulation.cpp_simulator import CppSimulator, PpcContext

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """One scenario's downloaded test_statistics + provenance."""

    theta: np.ndarray
    x: np.ndarray
    sample_index: np.ndarray
    pool_id: str
    pool_path: str


class MultiScenarioRunner:
    """Joint-θ orchestrator over a :class:`CppSimulator` per scenario.

    Args:
        simulators: ``{name: CppSimulator}``. Insertion order is the
            submit order; the first scenario does setup (venv refresh /
            git pull / cmake), the rest pass ``skip_setup=True``. All
            simulators MUST share the same ``priors_csv``, ``seed``,
            ``submodel_priors_yaml``, and ``binary_path`` — otherwise
            ``theta_for_indices(i)`` produces a different theta in
            scenario A vs B at the same ``sample_index``, breaking
            joint-inference alignment.
        job_manager: Optional override; defaults to the job_manager on
            the first simulator. Required only by :meth:`run_all` (the HPC
            training path); :meth:`simulate_with_parameters_all` runs the
            fused C++ batch locally and needs no transport, so the runner
            can be constructed without one for local-only PPC.

    Top-up: each :meth:`CppSimulator.run_hpc` call already handles
    deficit submission. Top-up cases silently drop ``samples_csv_remote``
    (the shared CSV's row layout doesn't match a deficit's expected
    indices) and fall back to per-pool deficit upload — caller pays a
    second upload only on partial pools.
    """

    def __init__(
        self,
        simulators: Mapping[str, CppSimulator],
        *,
        job_manager=None,
        discard_trajectories: bool = False,
    ):
        if not simulators:
            raise ValueError("simulators cannot be empty")
        self.simulators: Dict[str, CppSimulator] = dict(simulators)
        first = next(iter(self.simulators.values()))
        # job_manager is required only by the HPC training path
        # (:meth:`run_all`). The local posterior-predictive path
        # (:meth:`simulate_with_parameters_all`) runs the fused C++ batch
        # on this host and needs no transport — so the "missing
        # job_manager" failure is deferred to :meth:`run_all`, letting a
        # local-only PPC caller construct the runner without one.
        self.job_manager = job_manager or first.job_manager
        self._validate_alignment()

        self._shared_samples_remote: Optional[str] = None
        self._shared_samples_local: Optional[Path] = None
        self._aux_samples_local: Optional[Path] = None
        self._aux_samples_remote: Optional[str] = None
        self._auxiliary_units: Optional[dict] = None
        # When True, array tasks unlink their raw trajectory parquet after
        # inline-derive — only the test stats are kept. The trajectories are
        # the bulk of pool disk and dead weight for an SBI run.
        self.discard_trajectories = discard_trajectories

    def _validate_alignment(self) -> None:
        """Fail fast if simulators disagree on joint-inference invariants.

        A different priors_csv, seed, submodel_priors_yaml, or binary
        across simulators means ``theta_for_indices(i)`` returns
        different theta per scenario. Joint NPE training would then mix
        rows under different θ at the same ``sample_index`` — silent and
        catastrophic. Catch it at runner construction.
        """
        first_name, first = next(iter(self.simulators.items()))
        for name, sim in self.simulators.items():
            for attr in ("priors_csv", "seed", "submodel_priors_yaml", "binary_path"):
                a, b = getattr(first, attr), getattr(sim, attr)
                if a != b:
                    raise ValueError(
                        f"MultiScenarioRunner: simulators[{name!r}].{attr}={b!r} "
                        f"differs from simulators[{first_name!r}].{attr}={a!r}; "
                        f"joint-θ alignment requires every simulator to draw from "
                        f"the same theta pool."
                    )

    # ---- shared samples upload ------------------------------------------

    def upload_shared_samples_csv(self, n: int) -> str:
        """Render the shared (sample_index, theta) CSV for ``n`` rows and
        upload once. Idempotent within a runner instance.

        Pulls theta from the first simulator's deterministic theta pool
        (``theta_for_indices(0, n)``). Validation in
        :meth:`_validate_alignment` guarantees every simulator's pool
        produces the same theta at the same sample_index, so the upload
        is shareable.
        """
        if self._shared_samples_remote is not None:
            return self._shared_samples_remote

        first = next(iter(self.simulators.values()))
        # Reuse CppSimulator's params-CSV writer so the schema (header,
        # sample_index column, theta order) matches what the SLURM
        # worker expects.
        local_path = first._write_params_csv(n, start_index=0)
        try:
            content_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()[:12]
            remote_filename = f"samples_shared_{content_hash}.csv"
            self._shared_samples_remote = self.job_manager.upload_shared_samples_csv(
                str(local_path), remote_filename
            )
        finally:
            # Keep the local file alive until run_all finishes (the
            # legacy fallback in CppSimulator.run_hpc still passes
            # samples_csv= when samples_csv_remote points at a stale
            # path — although v1 rebuilds locally per call so this
            # mostly belt-and-braces).
            self._shared_samples_local = local_path
        return self._shared_samples_remote

    def attach_auxiliary_samples(
        self,
        aux_samples_csv: str | Path,
        auxiliary_units: dict,
    ) -> None:
        """Register a per-sim auxiliary-samples CSV for the session.

        Aux draws are sampled once by the inference orchestrator, keyed
        on ``sample_index``, and shared across every scenario in a
        sweep (since aux is a measurement-bridge concept, not a
        scenario-specific perturbation). When called before
        :meth:`run_all`, the runner uploads the CSV via the
        skip-if-exists shared-upload rail and threads its remote path
        + ``auxiliary_units`` map into each scenario's
        :meth:`CppSimulator.run_hpc` call.

        Args:
            aux_samples_csv: Local path to a CSV with one row per
                ``sample_index``. Required columns: ``sample_index`` plus
                one float column per auxiliary parameter.
            auxiliary_units: ``{aux_name: pint-parseable units string}``.
                The derive worker uses this to attach Pint units when
                merging aux values into ``species_dict``.
        """
        self._aux_samples_local = Path(aux_samples_csv)
        self._auxiliary_units = dict(auxiliary_units)

    def upload_shared_aux_samples_csv(self) -> Optional[str]:
        """Upload the registered aux samples CSV once and return remote path.

        Returns None when no aux samples were attached. Idempotent within
        a runner instance.
        """
        if self._aux_samples_local is None:
            return None
        if self._aux_samples_remote is not None:
            return self._aux_samples_remote
        content_hash = hashlib.sha256(self._aux_samples_local.read_bytes()).hexdigest()[:12]
        remote_filename = f"aux_samples_shared_{content_hash}.csv"
        self._aux_samples_remote = self.job_manager.upload_shared_aux_samples_csv(
            str(self._aux_samples_local), remote_filename
        )
        return self._aux_samples_remote

    # ---- core run_all ---------------------------------------------------

    def prepare_session(self) -> None:
        """Run the per-session HPC setup once: venv refresh + qsp-hpc-tools
        upgrade + git fetch + cmake build. Decoupled from any scenario's
        cache-tier resolution so scenarios that hit a local cache (and
        never call submit_cpp_jobs) still leave the session in a state
        where downstream skip_setup=True is safe.

        Idempotent within a runner instance — re-running is a near-no-op
        (incremental cmake on unchanged source).
        """
        if getattr(self, "_session_prepared", False):
            return
        logger.info("MSR: prepare_session — ensure_hpc_venv + ensure_cpp_binary")
        self.job_manager.ensure_hpc_venv()
        self.job_manager.ensure_cpp_binary()
        self._session_prepared = True

    def _preupload_per_scenario_files(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Pre-upload each scenario's per-scenario shared files once.

        scenario.yaml, drug_metadata.yaml, and test_statistics.csv differ
        across scenarios but are session-stable per scenario (the same
        scenario re-run has byte-identical files). Hash-keyed shared
        upload + skip-if-exists turns repeat sessions on the same
        scenario set into one ``test -f`` per file. Within a single
        session this is a no-op vs the per-pool upload — the win is
        across iterative re-runs.

        Returns ``{name: {scenario_yaml, drug_metadata_yaml,
        test_stats_csv}}`` with each value being the absolute remote
        path or None when the simulator has no such file.
        """
        out: Dict[str, Dict[str, Optional[str]]] = {}
        for name, sim in self.simulators.items():
            entry: Dict[str, Optional[str]] = {
                "scenario_yaml": None,
                "drug_metadata_yaml": None,
                "test_stats_csv": None,
            }
            if getattr(sim, "scenario_yaml", None):
                entry["scenario_yaml"] = self.job_manager.upload_shared_scenario_yaml(
                    str(sim.scenario_yaml)
                )
            if getattr(sim, "drug_metadata_yaml", None):
                entry["drug_metadata_yaml"] = self.job_manager.upload_shared_drug_metadata_yaml(
                    str(sim.drug_metadata_yaml)
                )
            if getattr(sim, "test_stats_csv", None):
                entry["test_stats_csv"] = self.job_manager.upload_shared_test_stats_csv(
                    str(sim.test_stats_csv)
                )
            out[name] = entry
        return out

    def upload_shared_model_structure(self) -> Optional[str]:
        """Upload model_structure.json once and return the remote path.

        Returns None when no simulator carries a model_structure_file.
        Same fall-back semantics as :meth:`upload_shared_healthy_state`:
        only hoist when every simulator agrees on the same local path;
        otherwise let the per-pool upload happen.
        """
        if getattr(self, "_shared_model_structure_remote", None) is not None:
            return self._shared_model_structure_remote

        local_paths = {
            getattr(sim, "model_structure_file", None) for sim in self.simulators.values()
        }
        local_paths.discard(None)
        if len(local_paths) != 1:
            logger.info(
                "MSR.upload_shared_model_structure: simulators don't agree on a single "
                "model_structure.json (%s) — falling back to per-pool upload",
                local_paths,
            )
            self._shared_model_structure_remote = None
            return None

        local = next(iter(local_paths))
        self._shared_model_structure_remote = self.job_manager.upload_shared_model_structure(
            str(local)
        )
        return self._shared_model_structure_remote

    def upload_shared_healthy_state(self) -> Optional[str]:
        """Upload the healthy_state YAML once and return the remote path.

        Returns None when no simulator carries a healthy_state_yaml.
        Validation in :meth:`_validate_alignment` does NOT enforce that
        every simulator's healthy_state matches — typical shared usage
        does match, but legacy mixed-IC sweeps would silently break if
        we forced a single upload. So: only hoist when *every*
        simulator agrees on the same local path. Mismatches fall back
        to the per-pool upload (no shared remote).
        """
        if getattr(self, "_shared_healthy_remote", None) is not None:
            return self._shared_healthy_remote

        local_paths = {getattr(sim, "healthy_state_yaml", None) for sim in self.simulators.values()}
        local_paths.discard(None)
        if len(local_paths) != 1:
            logger.info(
                "MSR.upload_shared_healthy_state: simulators don't agree on a single "
                "healthy_state YAML (%s) — falling back to per-pool upload",
                local_paths,
            )
            self._shared_healthy_remote = None
            return None

        local = next(iter(local_paths))
        self._shared_healthy_remote = self.job_manager.upload_shared_healthy_state(str(local))
        return self._shared_healthy_remote

    def _all_local(self, n: int) -> bool:
        """True when *every* simulator's local test-stats cache covers ``n``.

        Pre-check that lets :meth:`run_all` skip the entire HPC preamble
        (``prepare_session`` + shared-file uploads) when no scenario
        needs the cluster. All-or-nothing: a single scenario that needs
        HPC forces the session prep, since the others' ``run_hpc`` calls
        run with ``skip_setup=True`` and depend on it.

        The aux-samples CSV (when attached) is folded into each
        simulator's test-stats hash, so the same local aux path that the
        per-scenario :meth:`CppSimulator.run_hpc` call would use is
        threaded into the probe — otherwise the probe resolves a
        different cache key than the call.
        """
        aux = str(self._aux_samples_local) if self._aux_samples_local else None
        return all(
            sim.local_cache_satisfies(n, aux_samples_csv=aux) for sim in self.simulators.values()
        )

    def run_all(self, n: int) -> Dict[str, ScenarioResult]:
        """Run every scenario for ``n`` simulations against the shared
        theta pool, wait, return per-scenario ``(theta, x, sample_index)``.

        Strategy (#90 Phase 2 — scenario-fused submission):

        - If every scenario is already satisfied from its local
          test-stats cache (:meth:`_all_local`), skip the HPC preamble
          entirely and go straight to the per-scenario download loop,
          where each :meth:`CppSimulator.run_hpc` Tier 1 returns from
          the local Parquet without touching the cluster.
        - Otherwise hoist :meth:`prepare_session` (venv + binary setup)
          and the shared uploads, then submit **one fused array** for
          every scenario still needing sims (:meth:`_plan_fused` /
          :meth:`_submit_fused_array`). Each fused task evolves each
          theta once and runs all scenarios from that state — the fixed
          per-task overhead is paid once instead of N times.
        - Then the per-scenario :meth:`CppSimulator.run_hpc` loop runs
          purely as a **download**: the fused array has already
          populated every scenario's HPC test stats, so each
          ``run_hpc`` resolves at Tier 1/2 without submitting. Scenarios
          excluded from the fused set (already fully cached) are picked
          up by the same loop.

        Returns ``{name: ScenarioResult}`` in scenario-insertion order.
        """
        if self.job_manager is None:
            raise ValueError(
                "MultiScenarioRunner.run_all needs a job_manager (set it on the "
                "simulators or pass it explicitly); for a local posterior-"
                "predictive run use simulate_with_parameters_all instead."
            )
        aux = str(self._aux_samples_local) if self._aux_samples_local else None
        if self._all_local(n):
            logger.info(
                "MSR: all %d scenario(s) satisfied from local cache — "
                "skipping HPC session prep + shared uploads",
                len(self.simulators),
            )
            none_remotes = {
                name: {
                    "scenario_yaml": None,
                    "drug_metadata_yaml": None,
                    "test_stats_csv": None,
                }
                for name in self.simulators
            }
            return self._assemble_results(
                n,
                shared_remote=None,
                shared_healthy_remote=None,
                shared_model_structure_remote=None,
                shared_aux_remote=None,
                per_scen_remotes=none_remotes,
            )

        self.prepare_session()
        # Batched shared-input upload: register every shared file (samples,
        # healthy state, model structure, aux, per-scenario YAMLs/CSVs)
        # then rsync them all in ONE round-trip, instead of one ~2s
        # ``test -f`` probe per file.
        self.job_manager.begin_deferred_shared_uploads()
        shared_remote = self.upload_shared_samples_csv(n)
        shared_healthy_remote = self.upload_shared_healthy_state()
        shared_model_structure_remote = self.upload_shared_model_structure()
        shared_aux_remote = self.upload_shared_aux_samples_csv()
        per_scen_remotes = self._preupload_per_scenario_files()
        self.job_manager.flush_shared_uploads()

        # One fused array for every scenario still needing sims. The
        # persistent evolve cache (#90 Phase 1) composes underneath:
        # fusion shares the evolve across scenarios within this run; the
        # cache carries it across runs and to scenarios that drop out of
        # the fused set on a later top-up.
        plan = self._plan_fused(n, aux)
        if plan:
            self._submit_fused_array(
                n,
                plan,
                aux,
                shared_remote=shared_remote,
                shared_healthy_remote=shared_healthy_remote,
                shared_model_structure_remote=shared_model_structure_remote,
                shared_aux_remote=shared_aux_remote,
                per_scen_remotes=per_scen_remotes,
            )
        else:
            logger.info(
                "MSR: every scenario already has %d test stats on HPC/local — "
                "no fused array needed, downloading directly",
                n,
            )

        return self._assemble_results(
            n,
            shared_remote=shared_remote,
            shared_healthy_remote=shared_healthy_remote,
            shared_model_structure_remote=shared_model_structure_remote,
            shared_aux_remote=shared_aux_remote,
            per_scen_remotes=per_scen_remotes,
        )

    # ---- Phase 2 fused submission ---------------------------------------

    def _plan_fused(self, n: int, aux: Optional[str]) -> list:
        """Decide which scenarios go into the fused array, at what depth.

        Probes each scenario's existing test-stats depth
        (:meth:`CppSimulator.hpc_existing_depth`). A scenario already at
        ``>= n`` is **not** fused — its :meth:`CppSimulator.run_hpc` in
        the download loop will just resolve it from cache. A scenario
        short of ``n`` joins the fused set with ``start_index`` = its
        current depth, so the fused array sims only its deficit tail
        ``[depth, n)`` (#90 partial-miss design).

        Returns a list of ``(name, sim, start_index)`` for the fused
        scenarios, in scenario-insertion order.
        """
        plan = []
        for name, sim in self.simulators.items():
            depth = sim.hpc_existing_depth(n, aux_samples_csv=aux)
            if depth >= n:
                logger.info(
                    "MSR: scenario %s already has %d/%d test stats — not fused",
                    name,
                    depth,
                    n,
                )
            else:
                logger.info(
                    "MSR: scenario %s needs sims [%d, %d) — joining fused array",
                    name,
                    depth,
                    n,
                )
                plan.append((name, sim, depth))
        return plan

    def _submit_fused_array(
        self,
        n: int,
        plan: list,
        aux: Optional[str],
        *,
        shared_remote: Optional[str],
        shared_healthy_remote: Optional[str],
        shared_model_structure_remote: Optional[str],
        shared_aux_remote: Optional[str],
        per_scen_remotes: Dict[str, Dict[str, Optional[str]]],
    ) -> None:
        """Submit the one fused array for the scenarios in ``plan`` and block.

        The array spans ``[min_offset, n)`` where ``min_offset`` is the
        smallest per-scenario deficit start — the union of every fused
        scenario's deficit. Each fused task evolves a theta once and
        runs from that state every scenario whose ``start_index`` admits
        it.
        """
        first = next(iter(self.simulators.values()))
        min_offset = min(start for (_n, _s, start) in plan)

        scenarios = []
        for name, sim, start in plan:
            scenarios.append(
                {
                    "name": name,
                    "simulation_pool_id": sim.simulation_pool_id,
                    "scenario_yaml_remote": per_scen_remotes[name]["scenario_yaml"],
                    "drug_metadata_yaml_remote": per_scen_remotes[name]["drug_metadata_yaml"],
                    "test_stats_csv_remote": per_scen_remotes[name]["test_stats_csv"],
                    "test_stats_hash": sim._compute_test_stats_hash(aux_samples_csv=aux),
                    "samples_start_offset": start,
                }
            )

        # HPC-side binary/template: per-sim override → credentials.
        remote_binary = first.remote_binary_path or self.job_manager.config.cpp_binary_path
        remote_template = first.remote_template_xml or self.job_manager.config.cpp_template_path

        logger.info(
            "MSR: submitting ONE fused array — %d scenario(s), deficit [%d, %d)",
            len(plan),
            min_offset,
            n,
        )
        job_info = self.job_manager.submit_cpp_fused_jobs(
            scenarios=scenarios,
            samples_csv_remote=shared_remote,
            num_simulations=n - min_offset,
            samples_start_offset=min_offset,
            t_end_days=first.t_end_days,
            min_cadence_hours=first.min_cadence_hours,
            seed=first.seed,
            binary_path=remote_binary,
            template_path=remote_template,
            subtree=first.subtree,
            healthy_state_yaml_remote=shared_healthy_remote,
            model_structure_remote=shared_model_structure_remote,
            aux_samples_csv_remote=shared_aux_remote,
            auxiliary_units=self._auxiliary_units,
            evolve_cache=first.evolve_cache_root is not None,
            per_sim_timeout_s=first.per_sim_timeout_s or 300.0,
            discard_trajectories=self.discard_trajectories,
            skip_setup=True,  # prepare_session already ran
        )
        logger.info("MSR: fused array submitted (%s) — waiting", job_info.job_ids)
        first._wait_for_jobs(job_info.job_ids)

    def _assemble_results(
        self,
        n: int,
        *,
        shared_remote: Optional[str],
        shared_healthy_remote: Optional[str],
        shared_model_structure_remote: Optional[str],
        shared_aux_remote: Optional[str],
        per_scen_remotes: Dict[str, Dict[str, Optional[str]]],
    ) -> Dict[str, ScenarioResult]:
        """Assemble ``{name: ScenarioResult}`` with a fused teardown.

        Scenarios split two ways:

        - **Locally cached** — read straight from the local test-stats
          cache via :meth:`CppSimulator.run_hpc` (a Tier-1 read, no HPC).
        - **From the fused array** — every remaining scenario is combined
          and downloaded in ONE shot via
          :meth:`HPCJobManager.download_test_stats_fused`: one remote
          combine, one tarball. ``combined_params.csv`` is never
          downloaded — theta is regenerated locally from the
          deterministic theta pool (:meth:`CppSimulator._generate_parameters`),
          keyed by the ``sample_index`` sidecar.

        Replaces the old per-scenario ``run_hpc`` download loop (N
        combines + 2N file transfers). Note: per-scenario auto-top-up is
        gone — a fused array runs every scenario over the same theta set,
        so a genuine shortfall just leaves NaN rows for the joint filter.
        """
        aux = str(self._aux_samples_local) if self._aux_samples_local else None
        pool_root = self.job_manager.config.simulation_pool_path
        results: Dict[str, ScenarioResult] = {}

        # Partition: locally-cached scenarios vs. those that need the
        # fused HPC combine+download.
        fused: list = []  # [(name, sim)]
        for name, sim in self.simulators.items():
            if sim.local_cache_satisfies(n, aux_samples_csv=aux):
                logger.info("MSR: %s — local test-stats cache, reading directly", name)
                theta_scen, x_scen = sim.run_hpc(
                    n,
                    samples_csv_remote=shared_remote,
                    healthy_state_yaml_remote=shared_healthy_remote,
                    model_structure_remote=shared_model_structure_remote,
                    scenario_yaml_remote=per_scen_remotes[name]["scenario_yaml"],
                    drug_metadata_yaml_remote=per_scen_remotes[name]["drug_metadata_yaml"],
                    test_stats_csv_remote=per_scen_remotes[name]["test_stats_csv"],
                    aux_samples_csv=aux,
                    aux_samples_csv_remote=shared_aux_remote,
                    auxiliary_units=self._auxiliary_units,
                    skip_setup=True,
                    discard_trajectories=self.discard_trajectories,
                )
                sample_index_scen = (
                    np.asarray(sim.last_sample_index, dtype=np.int64)
                    if getattr(sim, "last_sample_index", None) is not None
                    else np.arange(len(theta_scen), dtype=np.int64)
                )
                results[name] = ScenarioResult(
                    theta=theta_scen,
                    x=x_scen,
                    sample_index=sample_index_scen,
                    pool_id=sim.simulation_pool_id,
                    pool_path=f"{pool_root}/{sim.simulation_pool_id}",
                )
            else:
                fused.append((name, sim))

        # One fused combine + download for every non-cached scenario.
        if fused:
            scen_specs = [
                {
                    "name": name,
                    "pool_path": f"{pool_root}/{sim.simulation_pool_id}",
                    "test_stats_hash": sim._compute_test_stats_hash(aux_samples_csv=aux),
                }
                for name, sim in fused
            ]
            logger.info(
                "MSR: fused combine+download — %d scenario(s) in one round-trip",
                len(fused),
            )
            with tempfile.TemporaryDirectory(prefix="msr_fused_dl_") as tmp:
                fused_dl = self.job_manager.download_test_stats_fused(scen_specs, Path(tmp))
            for name, sim in fused:
                sample_index_scen, x_scen = fused_dl[name]
                # Theta regenerated locally from the deterministic theta
                # pool — never downloaded. sample_index is the join key.
                theta_scen = sim._generate_parameters(sample_index_scen)
                sim.last_sample_index = sample_index_scen
                # Persist the local Tier-1 parquet the single-scenario
                # path writes via ``_download_and_persist`` (#90 fused
                # refactor dropped this). Without it ``local_cache_satisfies``
                # never hits after a fused run — every re-run pays an SSH
                # round-trip — and direct parquet readers (e.g. the
                # restriction-classifier retrain) find nothing on disk.
                # theta_scen is the sampled subset (``self.param_names``
                # order), so it keys the ``param:*`` columns directly.
                sim._persist_local_test_stats(
                    sim._local_test_stats_path(sim._compute_test_stats_hash(aux_samples_csv=aux)),
                    theta_scen,
                    x_scen,
                    sample_index=sample_index_scen,
                    param_names=sim.param_names,
                )
                results[name] = ScenarioResult(
                    theta=theta_scen,
                    x=x_scen,
                    sample_index=sample_index_scen,
                    pool_id=sim.simulation_pool_id,
                    pool_path=f"{pool_root}/{sim.simulation_pool_id}",
                )

        # Preserve scenario insertion order.
        return {name: results[name] for name in self.simulators}

    # ---- Phase 3 fused local posterior-predictive ------------------------

    def _validate_fused_local(self) -> None:
        """Fail fast before a fused local PPC run when simulators disagree
        on anything the single shared scenario-agnostic runner holds fixed.

        A fused batch runs every scenario through **one** ``qsp_sim``
        binary and **one** ``evolve_to_diagnosis`` per theta, then writes
        each scenario from that shared state. So the binary, the param
        template, and the healthy state must match across simulators, and
        ``t_end_days`` / ``min_cadence_hours`` must too —
        :meth:`CppBatchRunner.run_fused` takes one value of each for the
        whole batch. Scenario + drug-metadata YAMLs are *expected* to
        differ (that is the point of fusion) and ride per
        :class:`FusedScenarioSpec`.

        ``evolve_trajectory_dir`` is rejected: a burn-in trajectory dump
        needs the full evolve, which is incompatible with the fused
        cached-state execution model.
        """
        first_name, first = next(iter(self.simulators.items()))
        if first.healthy_state_yaml is None:
            raise ValueError(
                "simulate_with_parameters_all requires healthy_state_yaml — "
                "fused PPC amortizes the evolve_to_diagnosis burn-in across "
                "scenarios, and there is no burn-in without a healthy state."
            )
        for name, sim in self.simulators.items():
            for attr in (
                "binary_path",
                "template_xml",
                "healthy_state_yaml",
                "t_end_days",
                "min_cadence_hours",
                "subtree",
            ):
                a, b = getattr(first, attr), getattr(sim, attr)
                if a != b:
                    raise ValueError(
                        f"simulate_with_parameters_all: simulators[{name!r}].{attr}="
                        f"{b!r} differs from simulators[{first_name!r}].{attr}={a!r}; "
                        f"a fused local PPC runs every scenario through one shared "
                        f"binary + evolve, so these must match."
                    )
            if getattr(sim, "evolve_trajectory_dir", None) is not None:
                raise NotImplementedError(
                    f"simulate_with_parameters_all: simulators[{name!r}] has "
                    f"evolve_trajectory_dir set — a burn-in trajectory dump needs "
                    f"the full evolve and is incompatible with fused cached-state "
                    f"execution. Use the single-scenario "
                    f"CppSimulator.simulate_with_parameters for trajectory dumps."
                )

    def simulate_with_parameters_all(
        self,
        theta: np.ndarray,
        *,
        backend: str = "local",
        pool_suffix: str = "posterior_predictive",
        prediction_targets: Optional[str | Path] = None,
        aux_by_sample_index: Optional[dict[int, dict[str, float]]] = None,
        auxiliary_units: Optional[dict[str, str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, pa.Table]]:
        """Run every scenario at explicit ``theta``, fused (#90 Phase 3).

        The posterior-predictive analogue of :meth:`run_all`: where
        ``run_all`` submits an HPC array over the deterministic theta
        pool, this evaluates a **user-supplied** theta matrix (posterior
        draws). It is the multi-scenario counterpart of
        :meth:`CppSimulator.simulate_with_parameters` — and the reason it
        lives here rather than looping that method N times is fusion:

        - Each scenario's suffix-pool cache is probed first
          (:meth:`CppSimulator._resolve_ppc_context` /
          :meth:`~CppSimulator._ppc_cache_hit`); fully-cached scenarios
          short-circuit and never re-simulate.
        - The scenarios still needing sims go into **one**
          :meth:`CppBatchRunner.run_fused` call. Per theta the
          ``evolve_to_diagnosis`` burn-in (~84%+ of per-sim cost) is
          resolved **once** and every scenario runs from that state — so
          a 4-scenario PPC pays one evolve per theta, not four.
        - The persistent theta-keyed evolve cache (#90 Phase 1) composes
          underneath: fusion shares the evolve across scenarios within
          this call; the cache carries it across calls (repeated PPC
          rounds, or a later single-scenario ``simulate_with_parameters``
          on the same theta).

        Each scenario's result is written to its own suffix-pool cache,
        byte-identical to what :meth:`CppSimulator.simulate_with_parameters`
        would produce — so a later single-scenario call on the same theta
        is a cache hit.

        Args:
            theta: Parameter matrix ``(n_samples, n_params)``, columns
                aligned with the shared ``param_names``. The same theta is
                evaluated under every scenario (joint posterior-predictive).
            backend: ``"local"`` runs the fused C++ batch on this host (the
                default). ``"hpc"`` ships ``theta`` to the cluster as a
                shared samples CSV and submits **one fused array** over the
                uncached scenarios (:meth:`HPCJobManager.submit_cpp_fused_jobs`),
                deriving test stats inline and downloading in one combine.
                HPC mode requires ``job_manager`` and is incompatible with
                ``prediction_targets`` (the merged CSV isn't shipped). The
                ``"hpc"`` backend tags a distinct suffix pool, so HPC and
                local results for the same theta cache separately.
            pool_suffix: Suffix-pool label, applied to every scenario.
            prediction_targets: Optional directory of PredictionTarget
                YAMLs, merged into every scenario's derived columns.
            aux_by_sample_index: Optional per-sim auxiliary draws, keyed by
                ``sample_index``; shared across scenarios.
            auxiliary_units: ``{aux_name: units}`` for the aux draws.

        Returns:
            ``{name: (theta_out, table)}`` in scenario-insertion order —
            each entry exactly the pair
            :meth:`CppSimulator.simulate_with_parameters` returns.
        """
        if backend not in ("local", "hpc"):
            raise ValueError(f"backend must be 'local' or 'hpc'; got {backend!r}")
        self._validate_fused_local()
        if backend == "hpc":
            return self._run_ppc_hpc(
                theta,
                pool_suffix=pool_suffix,
                prediction_targets=prediction_targets,
                aux_by_sample_index=aux_by_sample_index,
                auxiliary_units=auxiliary_units,
            )
        first = next(iter(self.simulators.values()))

        # Per-scenario PPC context + suffix-pool cache probe. A scenario
        # already covered by its cache short-circuits; the rest are fused.
        results: Dict[str, Tuple[np.ndarray, pa.Table]] = {}
        uncached: list[tuple[str, CppSimulator, PpcContext]] = []
        for name, sim in self.simulators.items():
            ctx = sim._resolve_ppc_context(
                theta,
                backend="local",
                prediction_targets=prediction_targets,
                pool_suffix=pool_suffix,
                aux_by_sample_index=aux_by_sample_index,
                auxiliary_units=auxiliary_units,
            )
            hit = sim._ppc_cache_hit(ctx)
            if hit is not None:
                logger.info("MSR PPC: scenario %s — suffix-pool cache hit", name)
                results[name] = hit
            else:
                uncached.append((name, sim, ctx))

        if not uncached:
            logger.info(
                "MSR PPC: all %d scenario(s) satisfied from suffix-pool cache " "— no sims run",
                len(self.simulators),
            )
            return {name: results[name] for name in self.simulators}

        # One fused batch for every scenario still needing sims. Each
        # scenario writes its own species Parquet into its suffix pool;
        # fusion only amortizes the shared per-theta evolve.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fused_specs: list[FusedScenarioSpec] = []
        for name, sim, ctx in uncached:
            write_pool_manifest(ctx.suffix_pool_dir, sim._runner.template_defaults, sim.param_names)
            species_parquet = (
                ctx.suffix_pool_dir
                / f"species_{ts}_{pool_suffix}_{ctx.n_samples}sims_seed{sim.seed}.parquet"
            )
            fused_specs.append(
                FusedScenarioSpec(
                    name=name,
                    output_path=species_parquet,
                    scenario_yaml=sim.scenario_yaml,
                    drug_metadata_yaml=sim.drug_metadata_yaml,
                    start_index=0,  # PPC theta is user-supplied — every scenario runs all rows
                )
            )

        # One scenario-agnostic runner — per-scenario YAMLs ride on the
        # FusedScenarioSpecs. evolve_cache_root threads the persistent
        # Phase 1 cache through so PPC runs accumulate reusable evolves.
        runner = CppBatchRunner(
            binary_path=first.binary_path,
            template_path=first.template_xml,
            subtree=first.subtree,
            default_timeout_s=first.per_sim_timeout_s or 120.0,
            scenario_yaml=None,
            drug_metadata_yaml=None,
            healthy_state_yaml=first.healthy_state_yaml,
            evolve_cache_root=first.evolve_cache_root,
        )
        ref_ctx = uncached[0][2]
        logger.info(
            "MSR PPC: fused batch — %d scenario(s) × %d sims, one evolve per theta",
            len(uncached),
            ref_ctx.n_samples,
        )
        batch_results = runner.run_fused(
            theta_matrix=ref_ctx.theta,
            param_names=list(first.param_names),
            t_end_days=first.t_end_days,
            min_cadence_hours=first.min_cadence_hours,
            scenarios=fused_specs,
            sample_indices=ref_ctx.sample_indices,
            max_workers=first.max_workers,
            per_sim_timeout_s=first.per_sim_timeout_s,
        )

        # Derive per scenario from its species Parquet, write its
        # suffix-pool cache, and assemble the (theta_out, table) pair.
        for (name, sim, ctx), batch in zip(uncached, batch_results):
            if batch is None:
                # start_index=0 ⇒ every scenario receives a parquet; a
                # None here means the fused batch dropped the scenario.
                raise RuntimeError(f"fused PPC produced no output for scenario {name!r}")
            species_df = pd.read_parquet(batch.parquet_path)
            table = sim._derive_test_stats_table(
                species_df,
                ctx.test_stats_df,
                ctx.theta,
                ctx.sample_indices,
                aux_by_sample_index=aux_by_sample_index,
                auxiliary_units=auxiliary_units,
            )
            results[name] = sim._finalize_ppc(ctx, table)

        return {name: results[name] for name in self.simulators}

    # ---- Phase 2 fused HPC posterior-predictive --------------------------

    def _run_ppc_hpc(
        self,
        theta: np.ndarray,
        *,
        pool_suffix: str,
        prediction_targets: Optional[str | Path],
        aux_by_sample_index: Optional[dict[int, dict[str, float]]],
        auxiliary_units: Optional[dict[str, str]],
    ) -> Dict[str, Tuple[np.ndarray, pa.Table]]:
        """HPC backend for :meth:`simulate_with_parameters_all`.

        The cluster counterpart of the local fused PPC: instead of running
        ``CppBatchRunner.run_fused`` on this host, it ships the
        user-supplied ``theta`` to HPC as a shared samples CSV and submits
        **one fused array** (:meth:`HPCJobManager.submit_cpp_fused_jobs`)
        over every scenario still needing sims. Each task evolves a theta
        once and runs every scenario from that state, deriving test stats
        inline on the cluster; results download in one combine
        (:meth:`HPCJobManager.download_test_stats_fused`) and are reshaped +
        cached byte-identically to the local path, so a later
        single-scenario or local call on the same theta is a cache hit.

        Aux draws arrive in the local API's per-sim dict form; on HPC the
        derivation runs on the cluster, so they are materialized to a CSV
        and shipped the same way :meth:`attach_auxiliary_samples` ships them
        for training. ``prediction_targets`` is unsupported — same
        limitation as :meth:`CppSimulator.simulate_with_parameters`
        ``backend='hpc'`` (the merged CSV isn't shipped to the cluster).
        """
        if self.job_manager is None:
            raise RuntimeError(
                "simulate_with_parameters_all(backend='hpc') requires a "
                "job_manager (set it on the simulators or pass it to the runner)."
            )
        if prediction_targets is not None:
            raise NotImplementedError(
                "simulate_with_parameters_all(backend='hpc', prediction_targets=...) "
                "is not supported; the merged calibration+prediction CSV is not "
                "shipped to the cluster. Run locally or split the call."
            )

        # Per-scenario PPC context + suffix-pool cache probe — identical to
        # the local path, but the 'hpc' backend tag keeps HPC and local
        # caches for the same theta distinct. Pre-flight the cluster-only
        # inputs (flat test-stats CSV + model structure) here so a missing
        # one fails before any SSH.
        results: Dict[str, Tuple[np.ndarray, pa.Table]] = {}
        uncached: list[tuple[str, CppSimulator, PpcContext]] = []
        for name, sim in self.simulators.items():
            if sim.test_stats_csv is None:
                raise RuntimeError(
                    f"simulators[{name!r}] has no test_stats_csv — the HPC fused "
                    "PPC ships the flat test-stats CSV to the cluster for inline "
                    "derivation; without it there is nothing to derive."
                )
            if sim.model_structure_file is None:
                raise RuntimeError(
                    f"simulators[{name!r}] has no model_structure_file — the "
                    "derive worker treats every species as dimensionless without it."
                )
            ctx = sim._resolve_ppc_context(
                theta,
                backend="hpc",
                prediction_targets=None,
                pool_suffix=pool_suffix,
                aux_by_sample_index=aux_by_sample_index,
                auxiliary_units=auxiliary_units,
            )
            hit = sim._ppc_cache_hit(ctx)
            if hit is not None:
                logger.info("MSR PPC[hpc]: scenario %s — suffix-pool cache hit", name)
                results[name] = hit
            else:
                uncached.append((name, sim, ctx))

        if not uncached:
            logger.info(
                "MSR PPC[hpc]: all %d scenario(s) satisfied from suffix-pool "
                "cache — no HPC submit",
                len(self.simulators),
            )
            return {name: results[name] for name in self.simulators}

        first = next(iter(self.simulators.values()))
        n_samples = uncached[0][2].n_samples

        # Session prep + shared uploads. Same rail as run_all, but the
        # shared samples CSV is the USER theta (posterior draws), not the
        # deterministic theta pool. The aux CSV is built locally first, then
        # uploaded inside the deferred-upload block with everything else.
        self.prepare_session()
        aux_local = self._write_ppc_aux_csv(aux_by_sample_index)
        self.job_manager.begin_deferred_shared_uploads()
        shared_remote = self._upload_ppc_samples_csv(theta)
        shared_healthy_remote = self.upload_shared_healthy_state()
        shared_model_structure_remote = self.upload_shared_model_structure()
        shared_aux_remote = None
        if aux_local is not None:
            aux_hash = hashlib.sha256(Path(aux_local).read_bytes()).hexdigest()[:12]
            shared_aux_remote = self.job_manager.upload_shared_aux_samples_csv(
                aux_local, f"ppc_aux_shared_{aux_hash}.csv"
            )
        per_scen_remotes = self._preupload_per_scenario_files()
        self.job_manager.flush_shared_uploads()

        # One fused array over the user theta. Each scenario's HPC pool is
        # its theta-hashed suffix-pool dir name, so concurrent PPC calls
        # with distinct thetas never collide; start_offset=0 because a
        # user-supplied theta has no deterministic-pool depth to skip.
        scenarios = []
        for name, sim, ctx in uncached:
            scenarios.append(
                {
                    "name": name,
                    "simulation_pool_id": ctx.suffix_pool_dir.name,
                    "scenario_yaml_remote": per_scen_remotes[name]["scenario_yaml"],
                    "drug_metadata_yaml_remote": per_scen_remotes[name]["drug_metadata_yaml"],
                    "test_stats_csv_remote": per_scen_remotes[name]["test_stats_csv"],
                    "test_stats_hash": sim._compute_test_stats_hash(aux_samples_csv=aux_local),
                    "samples_start_offset": 0,
                }
            )

        remote_binary = first.remote_binary_path or self.job_manager.config.cpp_binary_path
        remote_template = first.remote_template_xml or self.job_manager.config.cpp_template_path

        logger.info(
            "MSR PPC[hpc]: submitting ONE fused array — %d scenario(s) × %d sims, "
            "one evolve per theta",
            len(uncached),
            n_samples,
        )
        job_info = self.job_manager.submit_cpp_fused_jobs(
            scenarios=scenarios,
            samples_csv_remote=shared_remote,
            num_simulations=n_samples,
            samples_start_offset=0,
            t_end_days=first.t_end_days,
            min_cadence_hours=first.min_cadence_hours,
            seed=first.seed,
            binary_path=remote_binary,
            template_path=remote_template,
            subtree=first.subtree,
            healthy_state_yaml_remote=shared_healthy_remote,
            model_structure_remote=shared_model_structure_remote,
            aux_samples_csv_remote=shared_aux_remote,
            auxiliary_units=auxiliary_units,
            evolve_cache=first.evolve_cache_root is not None,
            per_sim_timeout_s=first.per_sim_timeout_s or 300.0,
            discard_trajectories=self.discard_trajectories,
            skip_setup=True,  # prepare_session already ran
        )
        logger.info("MSR PPC[hpc]: fused array submitted (%s) — waiting", job_info.job_ids)
        first._wait_for_jobs(job_info.job_ids)

        # One fused combine + download for every scenario, then reshape each
        # scenario's derived test stats into the (theta_out, table) pair and
        # write its suffix-pool cache (byte-identical to the local path).
        pool_root = self.job_manager.config.simulation_pool_path
        scen_specs = [
            {
                "name": name,
                "pool_path": f"{pool_root}/{ctx.suffix_pool_dir.name}",
                "test_stats_hash": sim._compute_test_stats_hash(aux_samples_csv=aux_local),
            }
            for name, sim, ctx in uncached
        ]
        with tempfile.TemporaryDirectory(prefix="msr_ppc_hpc_dl_") as tmp:
            fused_dl = self.job_manager.download_test_stats_fused(scen_specs, Path(tmp))

        for name, sim, ctx in uncached:
            sample_index_dl, test_stats_dl = fused_dl[name]
            table = self._reshape_hpc_ppc_table(sim, ctx, sample_index_dl, test_stats_dl)
            results[name] = sim._finalize_ppc(ctx, table)

        return {name: results[name] for name in self.simulators}

    def _upload_ppc_samples_csv(self, theta: np.ndarray) -> str:
        """Write the user ``theta`` as a (sample_index, theta) CSV and upload
        it once as the shared samples CSV for an HPC fused PPC.

        Mirrors :meth:`upload_shared_samples_csv` but the rows are the
        caller's posterior draws, not the deterministic theta pool — so the
        pool-indexed local writer (``CppSimulator._write_params_csv``) can't
        be reused. ``sample_index`` is ``arange(n)`` so the downloaded
        parquet round-trips back to caller order.
        """
        first = next(iter(self.simulators.values()))
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        df = pd.DataFrame(theta, columns=list(first.param_names))
        df.insert(0, "sample_index", np.arange(theta.shape[0], dtype=np.int64))
        content_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:12]
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="ppc_samples_"
        )
        df.to_csv(tmp.name, index=False)
        tmp.close()
        # Keep the local file alive for the duration of the call (mirrors
        # upload_shared_samples_csv's _shared_samples_local), then upload.
        self._ppc_samples_local = Path(tmp.name)
        return self.job_manager.upload_shared_samples_csv(
            tmp.name, f"ppc_samples_{content_hash}.csv"
        )

    def _write_ppc_aux_csv(
        self,
        aux_by_sample_index: Optional[dict[int, dict[str, float]]],
    ) -> Optional[str]:
        """Materialize per-sim aux draws (the local API's dict form) to a CSV
        for the cluster derive worker, returning the local path (or None).

        The local fused PPC merges ``aux_by_sample_index`` into
        ``species_dict`` in-process; on HPC, derivation runs on the cluster,
        so the draws must travel as a CSV — one row per ``sample_index``, one
        column per aux name — the same shape :meth:`attach_auxiliary_samples`
        ships for training. The returned path is fed to
        ``_compute_test_stats_hash`` so the cluster-side derive subdir key
        matches.
        """
        if not aux_by_sample_index:
            return None
        rows = sorted(aux_by_sample_index.items())
        aux_names = list(rows[0][1].keys())
        data: dict[str, list] = {"sample_index": [int(i) for i, _ in rows]}
        for nm in aux_names:
            data[nm] = [float(d[nm]) for _, d in rows]
        df = pd.DataFrame(data)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, prefix="ppc_aux_")
        df.to_csv(tmp.name, index=False)
        tmp.close()
        self._ppc_aux_local = Path(tmp.name)  # keep alive for the call
        return tmp.name

    def _reshape_hpc_ppc_table(
        self,
        sim: CppSimulator,
        ctx: PpcContext,
        hpc_sample_index: np.ndarray,
        hpc_test_stats: np.ndarray,
    ) -> pa.Table:
        """Reshape a fused-download ``(sample_index, test_stats)`` pair into
        the ``(sample_index, status, param:*, ts:*)`` table the local fused
        path produces, so :meth:`CppSimulator._finalize_ppc` caches a
        byte-identical parquet regardless of backend.

        Mirrors the reshape in
        :meth:`CppSimulator._simulate_with_parameters_hpc`: rows are
        reordered to the caller's ``sample_index`` and the test-stat columns
        remapped from positional to ``ts:<test_statistic_id>``.

        Column ids come from the **shipped** ``test_stats_csv`` — the exact
        CSV the cluster derived from — not ``ctx.test_stats_df``. The two
        diverge when the scenario carries cross-input rows: the constructor
        appends them into ``test_stats_csv`` (so the cluster derives them),
        but ``_load_test_stats_df`` rebuilds ``ctx.test_stats_df`` from the
        calibration-target YAMLs without them. Keying off the shipped CSV
        keeps the reshape aligned with the downloaded column count + order.
        """
        n_samples = ctx.n_samples
        ts_ids = pd.read_csv(sim.test_stats_csv)["test_statistic_id"].tolist()
        if hpc_test_stats.shape[1] != len(ts_ids):
            raise RuntimeError(
                f"HPC fused PPC returned {hpc_test_stats.shape[1]} test stats but "
                f"the shipped test_stats_csv has {len(ts_ids)} — pool may have been "
                "derived with a different cal-target CSV."
            )
        if hpc_sample_index is None or len(hpc_sample_index) != n_samples:
            raise RuntimeError(
                f"HPC fused PPC returned "
                f"{0 if hpc_sample_index is None else len(hpc_sample_index)} rows "
                f"but caller expects {n_samples} — derivation incomplete?"
            )
        # Reorder rows so they match ctx.sample_indices (arange(n)).
        order = (
            pd.Series(np.arange(len(hpc_sample_index)), index=hpc_sample_index)
            .reindex(ctx.sample_indices)
            .to_numpy()
        )
        if np.any(np.isnan(order.astype(np.float64))):
            raise RuntimeError(
                "HPC fused PPC sample_index missing rows expected by caller "
                "— derivation incomplete?"
            )
        hpc_test_stats = hpc_test_stats[order.astype(np.int64)]

        theta = ctx.theta
        cols: dict[str, pa.Array] = {
            "sample_index": pa.array(ctx.sample_indices.astype(np.int64), type=pa.int64()),
            "status": pa.array(np.zeros(n_samples, dtype=np.int64), type=pa.int64()),
        }
        for j, pname in enumerate(sim.param_names):
            cols[f"param:{pname}"] = pa.array(theta[:, j].astype(np.float64))
        for j, tsid in enumerate(ts_ids):
            cols[f"ts:{tsid}"] = pa.array(hpc_test_stats[:, j].astype(np.float64))
        return pa.table(cols)

    # ---- joint NaN filter (static) ---------------------------------------

    @staticmethod
    def joint_nan_mask(
        results: Mapping[str, "ScenarioResult"],
    ) -> Dict[str, np.ndarray]:
        """Return a per-scenario boolean mask aligned by ``sample_index``.

        Rows where any scenario's ``x`` carries NaN/inf are dropped from
        every scenario's mask, so applying ``mask[name]`` to each
        scenario's ``(theta, x, sample_index)`` produces aligned
        survivors. Caller intersects on sample_index — the return mask is
        per-scenario (not per-shared-index), since each scenario can
        carry a slightly different sample_index ordering after top-up
        merges.
        """
        # Step 1: shared sample_indices (intersection across scenarios).
        sets = [set(int(s) for s in r.sample_index) for r in results.values()]
        shared = sorted(set.intersection(*sets)) if sets else []
        shared_arr = np.array(shared, dtype=np.int64)

        # Step 2: build per-scenario aligned slices and check finiteness.
        ok = np.ones(len(shared_arr), dtype=bool)
        for name, r in results.items():
            si_to_row = {int(s): i for i, s in enumerate(r.sample_index)}
            rows = np.array([si_to_row[s] for s in shared_arr], dtype=np.int64)
            x_aligned = r.x[rows]
            row_ok = np.isfinite(x_aligned).all(axis=1)
            n_drop = int((~row_ok).sum())
            if n_drop:
                logger.info(
                    "MSR.joint_nan_mask: %s drops %d/%d rows",
                    name,
                    n_drop,
                    len(shared_arr),
                )
            ok &= row_ok

        # Step 3: per-scenario mask in each scenario's native row order.
        masks: Dict[str, np.ndarray] = {}
        for name, r in results.items():
            si_to_row = {int(s): i for i, s in enumerate(r.sample_index)}
            keep = np.zeros(len(r.sample_index), dtype=bool)
            for shared_si, ok_i in zip(shared_arr, ok):
                if ok_i:
                    keep[si_to_row[int(shared_si)]] = True
            masks[name] = keep

        logger.info(
            "MSR.joint_nan_mask: %d/%d sample_indices survive joint filter",
            int(ok.sum()),
            len(shared_arr),
        )
        return masks

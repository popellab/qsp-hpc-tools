"""Thin multi-scenario orchestrator over :class:`CppSimulator`.

Joint-shared-θ inference often runs the same theta pool across multiple
scenarios (e.g. baseline + treatment arms). The per-scenario fan-out
already lives in :class:`CppSimulator` (3-tier cache, top-up,
combine + download). This module adds the joint-θ layer on top:

- one upload of the shared (sample_index, theta) CSV before the loop
- per-scenario ``run_hpc(n, samples_csv_remote=..., skip_setup=...)`` so
  scenarios 2..N skip the redundant venv refresh / git pull / cmake
- a :func:`joint_nan_mask` helper for cross-scenario alignment

Inference-agnostic: returns per-scenario ``(theta, x, sample_index)``
arrays the way :meth:`CppSimulator.run_hpc` already produces them.
Tensor packing, copula transforms, and NPE training stay in the caller.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from qsp_hpc.simulation.cpp_simulator import CppSimulator

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
            the first simulator. Useful when you want a single shared
            transport but supplied multiple sims that were built without
            one.

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
    ):
        if not simulators:
            raise ValueError("simulators cannot be empty")
        self.simulators: Dict[str, CppSimulator] = dict(simulators)
        first = next(iter(self.simulators.values()))
        self.job_manager = job_manager or first.job_manager
        if self.job_manager is None:
            raise ValueError(
                "MultiScenarioRunner needs a job_manager (set on the simulators "
                "or passed explicitly)"
            )
        self._validate_alignment()

        self._shared_samples_remote: Optional[str] = None
        self._shared_samples_local: Optional[Path] = None
        self._aux_samples_local: Optional[Path] = None
        self._aux_samples_remote: Optional[str] = None
        self._auxiliary_units: Optional[dict] = None

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
        """Submit each scenario for ``n`` simulations against the shared
        theta pool, wait, return per-scenario ``(theta, x, sample_index)``.

        Strategy:
        - If every scenario is already satisfied from its local
          test-stats cache (:meth:`_all_local`), skip the HPC preamble
          entirely — no venv refresh, no cmake probe, no uploads — and
          go straight to the per-scenario loop, where each
          :meth:`CppSimulator.run_hpc` Tier 1 returns from the local
          Parquet without touching the cluster.
        - Otherwise hoist :meth:`prepare_session` (venv + binary setup)
          before the loop, then submit every scenario with
          ``skip_setup=True``. Decouples HPC setup from any scenario's
          cache-tier resolution.
        - Hoist one ``upload_shared_samples_csv``.
        - Each scenario internally tier-checks (local cache → HPC cache
          → derive → submit), so re-runs hit local cache and the runner
          is safely re-callable.

        Returns ``{name: ScenarioResult}`` in scenario-insertion order.
        """
        if self._all_local(n):
            logger.info(
                "MSR: all %d scenario(s) satisfied from local cache — "
                "skipping HPC session prep + shared uploads",
                len(self.simulators),
            )
            shared_remote = None
            shared_healthy_remote = None
            shared_model_structure_remote = None
            shared_aux_remote = None
            per_scen_remotes = {
                name: {
                    "scenario_yaml": None,
                    "drug_metadata_yaml": None,
                    "test_stats_csv": None,
                }
                for name in self.simulators
            }
        else:
            self.prepare_session()
            shared_remote = self.upload_shared_samples_csv(n)
            shared_healthy_remote = self.upload_shared_healthy_state()
            shared_model_structure_remote = self.upload_shared_model_structure()
            shared_aux_remote = self.upload_shared_aux_samples_csv()
            per_scen_remotes = self._preupload_per_scenario_files()

        results: Dict[str, ScenarioResult] = {}
        for name, sim in self.simulators.items():
            logger.info("MSR: %s run_hpc(n=%d, skip_setup=True)", name, n)
            theta_scen, x_scen = sim.run_hpc(
                n,
                samples_csv_remote=shared_remote,
                healthy_state_yaml_remote=shared_healthy_remote,
                model_structure_remote=shared_model_structure_remote,
                scenario_yaml_remote=per_scen_remotes[name]["scenario_yaml"],
                drug_metadata_yaml_remote=per_scen_remotes[name]["drug_metadata_yaml"],
                test_stats_csv_remote=per_scen_remotes[name]["test_stats_csv"],
                aux_samples_csv=(str(self._aux_samples_local) if self._aux_samples_local else None),
                aux_samples_csv_remote=shared_aux_remote,
                auxiliary_units=self._auxiliary_units,
                skip_setup=True,
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
                pool_path=(
                    f"{self.job_manager.config.simulation_pool_path}/{sim.simulation_pool_id}"
                ),
            )
        return results

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

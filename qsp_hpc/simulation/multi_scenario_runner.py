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

    def run_all(self, n: int) -> Dict[str, ScenarioResult]:
        """Submit each scenario for ``n`` simulations against the shared
        theta pool, wait, return per-scenario ``(theta, x, sample_index)``.

        Strategy:
        - Hoist :meth:`prepare_session` (venv + binary setup) before the
          loop, then submit every scenario with ``skip_setup=True``.
          Decouples HPC setup from any scenario's cache-tier resolution.
        - Hoist one ``upload_shared_samples_csv``.
        - Each scenario internally tier-checks (local cache → HPC cache
          → derive → submit), so re-runs hit local cache and the runner
          is safely re-callable.

        Returns ``{name: ScenarioResult}`` in scenario-insertion order.
        """
        self.prepare_session()
        shared_remote = self.upload_shared_samples_csv(n)
        shared_healthy_remote = self.upload_shared_healthy_state()

        results: Dict[str, ScenarioResult] = {}
        for name, sim in self.simulators.items():
            logger.info("MSR: %s run_hpc(n=%d, skip_setup=True)", name, n)
            theta_scen, x_scen = sim.run_hpc(
                n,
                samples_csv_remote=shared_remote,
                healthy_state_yaml_remote=shared_healthy_remote,
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

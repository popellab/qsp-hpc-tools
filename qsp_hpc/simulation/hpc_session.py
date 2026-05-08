"""Shared HPC session: setup-once orchestration across multi-scenario runs.

Today each :class:`CppSimulator` instance redoes its own setup work —
binary upload, venv check, priors upload, theta_pool generation — at
construction or first ``run_hpc()``. With N scenarios that's N redundant
setups, each costing 30-60s of SSH round-trips. The mitigation in
``sbi_runner.py`` runs the first scenario "to populate the evolve cache"
and the rest sequentially, but the orchestration overhead still repeats
per scenario.

``HPCSession`` factors that work into a session-shared object so it
happens **once** per run::

    session = HPCSession(
        binary_path=...,
        priors_csv=...,
        submodel_priors_yaml=...,
        seed=...,
        job_manager=jm,
    )
    session.ensure_remote()             # binary + venv + priors uploaded; idempotent
    theta = session.sample_theta_pool(N)  # deterministic from (prior, seed)
    batch = session.run_scenario(...)     # cheap parts only

This is the Layer 2.5 scaffold of the local-eval rollout (see
``notes/architecture/local_observable_eval_plan.md`` in pdac-build).
Steps 3-5 of that plan fill in the new pool layout
(``pool_id = sha256(binary | scenario_yaml)``, ``training/`` + ``ppc/``
sub-pools), the SLURM-direct parquet emission path, and the sshfs read
path. Until those land, :meth:`run_scenario` raises
``NotImplementedError`` pointing at the plan; today's callers continue
to use :class:`CppSimulator` directly. ``HPCSession`` is a foundation,
not a behavior change.

Determinism guardrail (D7 in the plan): :meth:`ensure_remote` snapshots
the SHA-256 of the priors-CSV bytes; :meth:`sample_theta_pool` refuses
to draw if the on-disk hash has shifted since the snapshot, since that
would silently mismatch the ``(prior, seed) -> theta_at_sample_index``
contract that cross-scenario alignment relies on.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Union

import numpy as np

from qsp_hpc.simulation.theta_pool import get_theta_pool

if TYPE_CHECKING:
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

logger = logging.getLogger(__name__)


class HPCSession:
    """One-time HPC setup + theta-pool sampling shared across scenarios.

    Per the local-eval plan's Layer 2.5: setup-once orchestration belongs
    here, per-scenario submit/collect belongs in :meth:`run_scenario`
    (NotImplementedError until plan steps 3-5 land).
    """

    def __init__(
        self,
        *,
        binary_path: Union[str, Path],
        priors_csv: Union[str, Path],
        job_manager: "HPCJobManager",
        submodel_priors_yaml: Optional[Union[str, Path]] = None,
        seed: int = 2025,
        theta_pool_size: int = 100_000,
        theta_pool_cache_dir: Union[str, Path] = "cache/theta_pools",
        restriction_classifier_dir: Optional[Union[str, Path]] = None,
        restriction_threshold: float = 0.5,
        classifier_feature_fills: Optional[Mapping[str, float]] = None,
        remote_binary_path: Optional[str] = None,
    ):
        self.binary_path = Path(binary_path).resolve()
        if not self.binary_path.exists():
            raise FileNotFoundError(f"qsp_sim binary not found: {self.binary_path}")
        self.priors_csv = Path(priors_csv).resolve()
        if not self.priors_csv.exists():
            raise FileNotFoundError(f"Priors CSV not found: {self.priors_csv}")
        self.submodel_priors_yaml = (
            Path(submodel_priors_yaml).resolve() if submodel_priors_yaml else None
        )
        self.job_manager = job_manager
        self.seed = int(seed)
        self.theta_pool_size = int(theta_pool_size)
        self.theta_pool_cache_dir = Path(theta_pool_cache_dir)
        self.restriction_classifier_dir = (
            Path(restriction_classifier_dir).resolve() if restriction_classifier_dir else None
        )
        self.restriction_threshold = float(restriction_threshold)
        self.classifier_feature_fills = (
            dict(classifier_feature_fills) if classifier_feature_fills else None
        )
        self.remote_binary_path = remote_binary_path

        # Populated by ensure_remote(); used by sample_theta_pool to refuse
        # drawing if the priors CSV has shifted on disk since setup.
        self._priors_csv_hash: Optional[str] = None
        self._ensured: bool = False

        # Cached theta pool from the most recent sample_theta_pool() call.
        # Keyed by (n,) — repeat calls with the same n return the cached
        # array, larger n triggers a fresh draw (which the underlying
        # get_theta_pool will load from on-disk cache when available).
        self._theta_pool_cache: Optional[tuple[int, np.ndarray]] = None

    # --------------------------------------------------------------
    # Setup: idempotent, runs once per session
    # --------------------------------------------------------------

    def ensure_remote(
        self,
        *,
        skip_git_pull: bool = False,
        skip_build: bool = False,
    ) -> None:
        """Ensure HPC venv + qsp_sim binary are current; snapshot priors hash.

        Idempotent — repeat calls are no-ops once setup has succeeded.
        Per the plan's D7, the priors-CSV content hash is captured here so
        :meth:`sample_theta_pool` can guardrail against drift mid-run.

        Args:
            skip_git_pull: Forwarded to ``job_manager.ensure_cpp_binary``.
                Use when iterating with unpushed local edits in the HPC
                checkout.
            skip_build: Forwarded to ``job_manager.ensure_cpp_binary``.
                Skips cmake/make; only checks the binary exists.
        """
        if self._ensured:
            logger.debug("HPCSession.ensure_remote: already ensured, skipping")
            return

        logger.info("HPCSession.ensure_remote: priming HPC venv + binary")
        self.job_manager.ensure_hpc_venv()
        self.job_manager.ensure_cpp_binary(
            skip_git_pull=skip_git_pull,
            skip_build=skip_build,
            binary_path=self.remote_binary_path,
        )

        # D7 determinism guardrail: snapshot the priors-CSV bytes hash.
        # sample_theta_pool() refuses to draw if the on-disk hash drifts.
        self._priors_csv_hash = hashlib.sha256(self.priors_csv.read_bytes()).hexdigest()
        logger.info(
            "HPCSession.ensure_remote: priors_csv hash snapshot %s...",
            self._priors_csv_hash[:8],
        )
        self._ensured = True

    @property
    def priors_csv_hash(self) -> Optional[str]:
        """SHA-256 of priors-CSV bytes captured at :meth:`ensure_remote`.

        ``None`` until ``ensure_remote()`` has run. Used by the plan's D7
        guardrail to detect mid-run priors edits that would silently
        invalidate the ``(prior, seed) -> theta`` contract.
        """
        return self._priors_csv_hash

    # --------------------------------------------------------------
    # Theta pool: deterministic from (prior, seed)
    # --------------------------------------------------------------

    def sample_theta_pool(self, n: Optional[int] = None) -> np.ndarray:
        """Return the deterministic ``(n, n_params)`` theta matrix.

        Sampling is reproducible from ``(priors_csv, submodel_priors_yaml,
        seed, n)``; calling :meth:`sample_theta_pool` more than once with
        the same ``n`` returns the cached array (and identical results
        with a different ``n`` provided the on-disk theta-pool cache is
        intact, since the underlying generator is content-addressed).

        Args:
            n: Pool size. Defaults to ``theta_pool_size`` from the
                constructor.

        Raises:
            RuntimeError: If :meth:`ensure_remote` has not been called,
                or if the priors-CSV bytes hash has shifted since the
                ensure_remote snapshot (D7 guardrail).
        """
        if not self._ensured:
            raise RuntimeError(
                "HPCSession.sample_theta_pool: call ensure_remote() first "
                "(needed for the D7 priors-hash guardrail)"
            )
        # D7: refuse to draw if priors CSV has shifted since setup. Mid-run
        # prior edits silently break sample_index-keyed cross-scenario
        # alignment because (prior, seed) -> theta_at_sample_index would
        # disagree across scenarios sampled at different times.
        current_hash = hashlib.sha256(self.priors_csv.read_bytes()).hexdigest()
        if current_hash != self._priors_csv_hash:
            raise RuntimeError(
                f"HPCSession: priors_csv {self.priors_csv} has changed since "
                f"ensure_remote() snapshotted it ({self._priors_csv_hash[:8]}... "
                f"-> {current_hash[:8]}...). Mid-run prior edits would break the "
                f"(prior, seed) -> theta_at_sample_index contract that "
                f"cross-scenario alignment relies on."
            )

        n = int(n) if n is not None else self.theta_pool_size
        if self._theta_pool_cache is not None and self._theta_pool_cache[0] == n:
            return self._theta_pool_cache[1]

        theta = get_theta_pool(
            priors_csv=self.priors_csv,
            submodel_priors_yaml=self.submodel_priors_yaml,
            seed=self.seed,
            n_total=n,
            cache_dir=self.theta_pool_cache_dir,
            restriction_classifier_dir=self.restriction_classifier_dir,
            restriction_threshold=self.restriction_threshold,
            classifier_feature_fills=self.classifier_feature_fills,
        )
        self._theta_pool_cache = (n, theta)
        return theta

    # --------------------------------------------------------------
    # Per-scenario run: scaffold only, plan steps 3-5 fill this in
    # --------------------------------------------------------------

    def run_scenario(
        self,
        *,
        scenario_yaml: Union[str, Path],
        n_simulations: int,
        traj_columns: Optional[list[str]] = None,
        kind: str = "training",
        min_cadence_hours: Optional[float] = None,
    ):
        """Submit + collect one scenario's simulations against this session.

        **Not yet implemented.** This is the Layer 4 contract from
        ``notes/architecture/local_observable_eval_plan.md``; landing it
        requires plan steps 3 (content-addressed pool layout with
        ``training/`` + ``ppc/`` sub-pools), 4 (SLURM job script writes
        long-form parquet directly, no ``derive_test_stats_worker``),
        and 5 (sshfs read path). Until those land, callers continue to
        use :class:`CppSimulator` directly.

        Expected signature when complete::

            batch = session.run_scenario(
                scenario_yaml=scen_yaml,
                n_simulations=N,
                traj_columns=cols,                 # species the cal targets need
                kind="training",                    # or "ppc"
                min_cadence_hours=4.0,             # per-scenario override
            )
            # returns a SimulationBatch (theta + traj_df, sample_index-keyed,
            # written to {pool_id}/{kind}/ over sshfs).
        """
        raise NotImplementedError(
            "HPCSession.run_scenario is the Layer 4 contract from "
            "notes/architecture/local_observable_eval_plan.md; plan steps "
            "3-5 (content-addressed pool layout, SLURM-direct parquet, "
            "sshfs read path) must land first. Today's callers should "
            "continue to use CppSimulator.run_hpc() directly."
        )

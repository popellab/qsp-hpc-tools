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

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from qsp_hpc.constants import JOB_QUEUE_TIMEOUT, SLURM_REGISTRATION_DELAY
from qsp_hpc.cpp.batch_runner import (
    POOL_MANIFEST_FILENAME as SUBPOOL_MANIFEST_FILENAME,
)
from qsp_hpc.cpp.batch_runner import (
    POOL_MANIFEST_SCHEMA as SUBPOOL_MANIFEST_SCHEMA,
)
from qsp_hpc.cpp.batch_runner import (
    SUBPOOL_KINDS as _VALID_KINDS,
)
from qsp_hpc.cpp.batch_runner import (
    subpool_dir,
)
from qsp_hpc.cpp.sshfs_reader import sshfs_read_long_form_chunks
from qsp_hpc.simulation.simulation_batch import SimulationBatch
from qsp_hpc.simulation.theta_pool import get_theta_pool
from qsp_hpc.utils.hash_utils import compute_pool_id_hash

if TYPE_CHECKING:
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

logger = logging.getLogger(__name__)


SUBPOOL_KIND = Literal["training", "ppc"]


class _HoldFlocks:
    """Context manager: acquire ``LOCK_EX`` on a list of paths, sorted.

    Used by :meth:`HPCSession.reserve_sample_index_range` to make the
    session-global scan+allocate+broadcast cycle atomic across every
    registered pool's training/ + ppc/ sub-pool. Sorted-path acquisition
    order prevents deadlock between concurrent sessions touching
    overlapping pool sets.
    """

    def __init__(self, paths: list[Path]):
        self._paths = list(paths)
        self._fhs: list = []

    def __enter__(self):
        for p in self._paths:
            fh = open(p, "r+")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            self._fhs.append(fh)
        return self

    def __exit__(self, exc_type, exc, tb):
        for fh in reversed(self._fhs):
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
        self._fhs.clear()
        return False


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
        remote_template_xml: Optional[str] = None,
        model_structure_file: Optional[Union[str, Path]] = None,
        local_pool_root: Union[str, Path] = "cache/scenario_pools",
        t_end_days: float = 180.0,
        poll_interval: float = 30.0,
        max_wait_time: Optional[float] = None,
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
        self.remote_template_xml = remote_template_xml
        self.model_structure_file = (
            Path(model_structure_file).resolve() if model_structure_file else None
        )
        self.local_pool_root = Path(local_pool_root)
        self.t_end_days = float(t_end_days)
        self.poll_interval = float(poll_interval)
        self.max_wait_time = max_wait_time
        self.logger = logger
        # Cache of priors-CSV "name" column, lazily loaded under the
        # ensure_remote priors-hash snapshot. Used by run_scenario to thread
        # ``param_names`` into SimulationBatch in priors-CSV order.
        self._param_names_cache: Optional[list[str]] = None

        # Populated by ensure_remote(); used by sample_theta_pool to refuse
        # drawing if the priors CSV has shifted on disk since setup.
        self._priors_csv_hash: Optional[str] = None
        self._ensured: bool = False

        # Cached theta pool from the most recent sample_theta_pool() call.
        # Keyed by (n,) — repeat calls with the same n return the cached
        # array, larger n triggers a fresh draw (which the underlying
        # get_theta_pool will load from on-disk cache when available).
        self._theta_pool_cache: Optional[tuple[int, np.ndarray]] = None

        # Registered scenario pool dirs (D1: one pool per scenario). Each
        # call to ``reserve_sample_index_range`` broadcasts the allocated
        # range into every registered pool's ``{kind}/pool_manifest.json``.
        # Insertion-ordered for deterministic test output.
        self._scenario_pool_dirs: dict[Path, None] = {}

        # Session-global high-watermark for sample_index allocation. None
        # until first reservation, when we scan all registered pools'
        # training/ + ppc/ manifests for the max end_exclusive seen and
        # cache it. Subsequent reservations bump it locally without
        # rescanning (D1: per-pool flock guarantees that concurrent
        # sessions can't drift below their own cached watermark; if a
        # sibling session writes a higher value while we hold the lock,
        # the next reservation in that session re-derives from disk).
        self._sample_index_watermark: Optional[int] = None

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
    # Sample-index range allocator (D1)
    # --------------------------------------------------------------

    def register_scenario_pool(self, pool_dir: Union[str, Path]) -> Path:
        """Track a scenario pool dir for sample-index reservation broadcasts.

        :meth:`reserve_sample_index_range` writes the allocated range
        into ``{pool_dir}/{kind}/pool_manifest.json`` for every
        registered pool. Idempotent: re-registering the same path is a
        no-op. Pool dir is created if absent (sub-pool dirs are
        materialized lazily on first reservation).

        Once :meth:`reserve_sample_index_range` has run, the in-memory
        watermark is locked in. Registering a *new* pool after that
        invalidates the watermark — the next reservation rescans, since
        the new pool's existing reservations may sit above the old
        max.
        """
        pd = Path(pool_dir).resolve()
        pd.mkdir(parents=True, exist_ok=True)
        if pd not in self._scenario_pool_dirs:
            self._scenario_pool_dirs[pd] = None
            # Force a rescan on next reservation — the new pool may carry
            # reservations from a previous session that we haven't
            # accounted for in our cached watermark.
            self._sample_index_watermark = None
            logger.debug("HPCSession: registered scenario pool %s", pd)
        return pd

    @property
    def registered_pool_dirs(self) -> tuple[Path, ...]:
        return tuple(self._scenario_pool_dirs)

    def reserve_sample_index_range(
        self,
        n: int,
        *,
        kind: SUBPOOL_KIND,
    ) -> tuple[int, int]:
        """Allocate ``[start, end)`` and broadcast to registered pools.

        Session-global allocator (D1, Option A). The same range is
        written into every registered scenario pool's
        ``{kind}/pool_manifest.json`` so cross-scenario alignment can
        rely on ``sample_index`` ≡ theta-index. Concurrent sessions
        colliding on the same pool serialize on a per-pool fcntl flock
        during the reservation-append.

        Args:
            n: Range size. Must be > 0.
            kind: ``"training"`` or ``"ppc"``. Selects the sub-pool
                manifest file to write into. The watermark is shared
                across kinds (a sample_index allocated to ``training``
                is consumed for the whole session — PPC won't reuse
                it).

        Returns:
            ``(start, end_exclusive)`` with ``end_exclusive == start + n``.

        Raises:
            RuntimeError: If :meth:`ensure_remote` hasn't run, or no
                scenario pools are registered.
            ValueError: For non-positive ``n`` or unknown ``kind``.
        """
        if not self._ensured:
            raise RuntimeError(
                "HPCSession.reserve_sample_index_range: call ensure_remote() "
                "first (the (prior, seed) -> theta_at_sample_index "
                "determinism contract is set up there)."
            )
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"reserve_sample_index_range: kind must be one of " f"{_VALID_KINDS}; got {kind!r}"
            )
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"reserve_sample_index_range: n must be a positive int; got {n!r}")
        if not self._scenario_pool_dirs:
            raise RuntimeError(
                "HPCSession.reserve_sample_index_range: no scenario pools "
                "registered. Call register_scenario_pool(pool_dir) for each "
                "scenario before reserving a range."
            )

        # Materialize sub-pool dirs + lockfiles for both kinds (the scan
        # needs to read both kinds; the append writes only the requested
        # one). Holding all locks atomically across scan+write is what
        # makes session-global allocation safe under concurrent writers
        # against the same pool set.
        lock_paths: list[Path] = []
        for pool_dir in self._scenario_pool_dirs:
            for kind_name in _VALID_KINDS:
                sub_dir = subpool_dir(pool_dir, kind_name)
                sub_dir.mkdir(parents=True, exist_ok=True)
                lock_path = sub_dir / ".pool_manifest.lock"
                if not lock_path.exists():
                    lock_path.touch()
                lock_paths.append(lock_path)
        # Sorted path order prevents deadlock between concurrent
        # broadcasters touching overlapping pool sets.
        lock_paths.sort()

        with _HoldFlocks(lock_paths):
            # Rescan under the lock — the cached watermark is best-effort
            # for the no-contention case, but a sibling session may have
            # written between our last scan and this reservation. Trust
            # disk inside the lock.
            disk_watermark = self._scan_high_watermark()
            cached = self._sample_index_watermark or 0
            start = int(max(disk_watermark, cached))
            end = start + int(n)
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for pool_dir in self._scenario_pool_dirs:
                self._append_reservation(
                    pool_dir=pool_dir,
                    kind=kind,
                    start=start,
                    end=end,
                    ts=ts,
                )
            self._sample_index_watermark = end
        logger.info(
            "HPCSession: reserved sample_index [%d, %d) kind=%s across %d pool(s)",
            start,
            end,
            kind,
            len(self._scenario_pool_dirs),
        )
        return start, end

    def _scan_high_watermark(self) -> int:
        """Scan registered pools' training/+ppc/ manifests for max end."""
        max_end = 0
        for pool_dir in self._scenario_pool_dirs:
            for kind_name in _VALID_KINDS:
                manifest_path = subpool_dir(pool_dir, kind_name) / SUBPOOL_MANIFEST_FILENAME
                if not manifest_path.exists():
                    continue
                try:
                    with open(manifest_path) as fh:
                        payload = json.load(fh)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "HPCSession: skipping unreadable manifest %s (%s)",
                        manifest_path,
                        exc,
                    )
                    continue
                for r in payload.get("reservations", []):
                    end = int(r.get("end", 0))
                    if end > max_end:
                        max_end = end
        return max_end

    @staticmethod
    def _append_reservation(
        *,
        pool_dir: Path,
        kind: str,
        start: int,
        end: int,
        ts: str,
    ) -> None:
        """Append a reservation entry to ``{pool_dir}/{kind}/pool_manifest.json``.

        Caller is responsible for holding the per-sub-pool flock during
        this write — :meth:`reserve_sample_index_range` does so via
        :func:`_HoldFlocks` so the scan+broadcast cycle is atomic
        across all registered pools. Atomic rename keeps any concurrent
        lock-free reader from observing a partial JSON.
        """
        sub_dir = subpool_dir(pool_dir, kind)
        sub_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = sub_dir / SUBPOOL_MANIFEST_FILENAME

        if manifest_path.exists():
            with open(manifest_path) as fh:
                payload = json.load(fh)
        else:
            payload = {
                "schema_version": SUBPOOL_MANIFEST_SCHEMA,
                "kind": kind,
                "reservations": [],
            }
        if payload.get("kind", kind) != kind:
            raise RuntimeError(
                f"HPCSession: manifest at {manifest_path} declares "
                f"kind={payload.get('kind')!r} but reservation "
                f"requested kind={kind!r}"
            )
        payload.setdefault("reservations", []).append(
            {"start": int(start), "end": int(end), "ts": ts}
        )
        payload.setdefault("schema_version", SUBPOOL_MANIFEST_SCHEMA)
        payload.setdefault("kind", kind)

        tmp_path = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
        try:
            with open(tmp_path, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            tmp_path.replace(manifest_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # --------------------------------------------------------------
    # Per-scenario run: scaffold only, plan steps 3-5 fill this in
    # --------------------------------------------------------------

    def _load_param_names(self) -> list[str]:
        """Read the ``name`` column of the priors CSV, cached on first use.

        This is the canonical theta column order used by both
        :meth:`sample_theta_pool` (via the composite prior or the
        per-row lognormal sampler) and the samples CSV uploaded to HPC.
        ensure_remote() must have run so the D7 priors-hash guardrail
        catches a mid-session edit.
        """
        if self._param_names_cache is None:
            self._param_names_cache = pd.read_csv(self.priors_csv)["name"].tolist()
        return self._param_names_cache

    def _wait_for_jobs(self, job_ids: list[str]) -> None:
        """Poll ``check_job_status`` until all jobs leave the queue.

        Mirrors :meth:`CppSimulator._wait_for_jobs` — same SLURM
        registration delay, post-empty grace window, and timeout
        semantics. Kept private so the polling cadence stays consistent
        across the two run paths.
        """
        time.sleep(SLURM_REGISTRATION_DELAY)
        start = time.time()
        max_seen = 0
        while True:
            totals = {"completed": 0, "running": 0, "pending": 0, "failed": 0}
            for jid in job_ids:
                try:
                    status = self.job_manager.check_job_status(jid)
                    for k in totals:
                        totals[k] += status[k]
                except Exception as e:  # pragma: no cover — best-effort
                    self.logger.warning(f"Status check failed for job {jid}: {e}")
            total = sum(totals.values())
            max_seen = max(max_seen, total)
            elapsed = time.time() - start
            self.logger.info(
                "  [%dm %ds] %d/%d done | running=%d pending=%d failed=%d",
                int(elapsed // 60),
                int(elapsed % 60),
                totals["completed"],
                total,
                totals["running"],
                totals["pending"],
                totals["failed"],
            )
            active = totals["running"] + totals["pending"]
            if total > 0 and active == 0:
                if totals["failed"] > 0:
                    self.logger.warning(f"{totals['failed']} task(s) failed")
                return
            if total == 0 and max_seen > 0 and elapsed > 30:
                return
            if total == 0 and elapsed > JOB_QUEUE_TIMEOUT:
                self.logger.warning(f"No jobs visible after {JOB_QUEUE_TIMEOUT}s — proceeding")
                return
            if self.max_wait_time and elapsed > self.max_wait_time:
                raise TimeoutError(f"Job wait exceeded {self.max_wait_time}s for {job_ids}")
            time.sleep(self.poll_interval)

    def run_scenario(
        self,
        *,
        scenario_yaml: Union[str, Path],
        n_simulations: int,
        traj_columns: Optional[Sequence[str]] = None,
        kind: SUBPOOL_KIND = "training",
        min_cadence_hours: Optional[float] = None,
        drug_metadata_yaml: Optional[Union[str, Path]] = None,
        healthy_state_yaml: Optional[Union[str, Path]] = None,
        scenario_name: Optional[str] = None,
        seed_override: Optional[int] = None,
        evolve_cache: bool = True,
        sshfs_host: Optional[str] = None,
        filesystem: Any = None,
        _read_helper=sshfs_read_long_form_chunks,
    ) -> SimulationBatch:
        """Submit + collect one scenario's simulations against this session.

        Implements the Layer 4 contract from
        ``notes/architecture/local_observable_eval_plan.md``: compute the
        content-addressed ``pool_id``, register the local manifest dir,
        reserve a sample-index range, write the samples CSV with
        sample_index-keyed theta from the session pool, submit a
        ``submit_cpp_jobs(kind=...)`` array (no test-stats derivation —
        SLURM writes long-form trajectory parquets directly), wait for
        completion, then read the chunks back from
        ``{simulation_pool_path}/{pool_id}/{kind}`` over sshfs.

        Args:
            scenario_yaml: pdac-build scenario YAML driving the run.
                Folded into ``pool_id`` (alongside the binary bytes) so
                edits invalidate the cache.
            n_simulations: Range size to reserve and submit. Mapped to
                theta-pool indices ``[start, start + n_simulations)``.
            traj_columns: Optional species filter pushed down to the
                parquet read. Defaults to ``None`` (every species).
            kind: ``"training"`` or ``"ppc"``. Selects the sub-pool
                dir on disk and on HPC.
            min_cadence_hours: Per-scenario override of the qsp_sim
                output-cadence cap. ``None`` lets the CLI default win.
            drug_metadata_yaml / healthy_state_yaml: Optional companions
                forwarded to ``submit_cpp_jobs`` — same path resolution
                as :class:`CppSimulator`.
            scenario_name: Cosmetic name for parquet filenames. Defaults
                to the YAML stem.
            seed_override: Forwarded to ``submit_cpp_jobs`` as ``seed``.
                Defaults to the session's ``seed``.
            evolve_cache: Forwarded to ``submit_cpp_jobs``.
            sshfs_host: Hostname for the sshfs read connection. Defaults
                to ``job_manager.config.host``.
            filesystem: Optional injected fsspec filesystem (tests).
                When provided, the helper does not open its own sshfs
                connection.
            _read_helper: Test seam — pass a stand-in for
                :func:`sshfs_read_long_form_chunks` to bypass the live
                sshfs path. Kept private to discourage production use.

        Returns:
            :class:`SimulationBatch` with theta sliced to
            ``[start, end)``, ``sample_index`` matching the reservation,
            ``traj_df`` from the SLURM-written long-form parquets, and
            ``species_units`` from ``model_structure.json``.
        """
        if not self._ensured:
            raise RuntimeError(
                "HPCSession.run_scenario: call ensure_remote() first "
                "(needed for the D7 priors-hash guardrail and "
                "sample_theta_pool determinism)."
            )
        if kind not in _VALID_KINDS:
            raise ValueError(f"run_scenario: kind must be one of {_VALID_KINDS}; got {kind!r}")
        if not isinstance(n_simulations, int) or n_simulations <= 0:
            raise ValueError(
                f"run_scenario: n_simulations must be a positive int; got {n_simulations!r}"
            )
        scenario_yaml = Path(scenario_yaml).resolve()
        if not scenario_yaml.exists():
            raise FileNotFoundError(f"scenario_yaml not found: {scenario_yaml}")
        if self.model_structure_file is None or not self.model_structure_file.exists():
            raise RuntimeError(
                "HPCSession.run_scenario: model_structure_file is required so "
                "SimulationBatch.species_units is populated. Pass "
                "model_structure_file=... to HPCSession()."
            )

        # 1. Pool key — content-addressed by binary bytes + scenario YAML.
        pool_id = compute_pool_id_hash(
            binary_path=self.binary_path,
            scenario_yaml=scenario_yaml,
        )
        self.logger.info(
            "HPCSession.run_scenario: pool_id=%s... kind=%s n=%d",
            pool_id[:8],
            kind,
            n_simulations,
        )

        # 2. Register local manifest dir + 3. reserve sample-index range.
        local_pool_dir = self.local_pool_root / pool_id
        self.register_scenario_pool(local_pool_dir)
        start, end = self.reserve_sample_index_range(n_simulations, kind=kind)

        # 4. Pull theta slice from the deterministic pool.
        theta_pool = self.sample_theta_pool(end)
        theta = theta_pool[start:end]
        sample_index = np.arange(start, end, dtype=np.int64)
        param_names = self._load_param_names()
        if theta.shape[1] != len(param_names):
            raise RuntimeError(
                f"run_scenario: theta has {theta.shape[1]} cols but "
                f"priors CSV has {len(param_names)} param names — "
                f"composite-prior / CSV drift."
            )

        # 5. Write samples CSV (sample_index + param cols, mirrors
        # CppSimulator._write_params_csv).
        samples_df = pd.DataFrame(theta, columns=param_names)
        samples_df.insert(0, "sample_index", sample_index)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="hpc_session_params_"
        )
        samples_csv = Path(tmp.name)
        samples_df.to_csv(tmp.name, index=False)
        tmp.close()

        # 6. Submit. derive_test_stats=False per plan §D4: SLURM writes
        # long-form parquet directly, observable evaluation runs locally
        # off the read path.
        remote_binary = self.remote_binary_path or self.job_manager.config.cpp_binary_path
        remote_template = self.remote_template_xml or self.job_manager.config.cpp_template_path
        scen_name = scenario_name or scenario_yaml.stem
        try:
            info = self.job_manager.submit_cpp_jobs(
                samples_csv=str(samples_csv),
                num_simulations=n_simulations,
                simulation_pool_id=pool_id,
                t_end_days=self.t_end_days,
                min_cadence_hours=(min_cadence_hours if min_cadence_hours is not None else 4.0),
                scenario=scen_name,
                seed=int(seed_override if seed_override is not None else self.seed),
                binary_path=remote_binary,
                template_path=remote_template,
                scenario_yaml=str(scenario_yaml),
                drug_metadata_yaml=(str(drug_metadata_yaml) if drug_metadata_yaml else None),
                healthy_state_yaml=(str(healthy_state_yaml) if healthy_state_yaml else None),
                derive_test_stats=False,
                evolve_cache=evolve_cache,
                kind=kind,
            )
        finally:
            samples_csv.unlink(missing_ok=True)

        # 7. Wait for the array to drain.
        self._wait_for_jobs(list(info.job_ids))

        # 8. Read long-form parquets back over sshfs.
        remote_pool_path = f"{self.job_manager.config.simulation_pool_path}/{pool_id}/{kind}"
        host = sshfs_host
        if filesystem is None and host is None:
            host = getattr(self.job_manager.config, "host", None)
        traj_df = _read_helper(
            remote_pool_path,
            sample_indices=sample_index.tolist(),
            traj_columns=list(traj_columns) if traj_columns is not None else None,
            filesystem=filesystem,
            sshfs_host=host,
        )

        # 9. Species units from model_structure.json.
        from qsp_hpc.utils.model_structure_units import load_units_from_model_structure

        species_units = load_units_from_model_structure(self.model_structure_file)

        return SimulationBatch(
            theta=np.asarray(theta),
            sample_index=sample_index,
            traj_df=traj_df,
            species_units=species_units,
            param_names=list(param_names),
            pool_id=pool_id,
        )

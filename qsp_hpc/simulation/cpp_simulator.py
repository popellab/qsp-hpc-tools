"""C++ QSP simulator with local pool-based caching.

Mirrors QSPSimulator's callable interface but uses CppBatchRunner
for execution instead of MATLAB.  Optional HPC integration (M9):
when ``job_manager`` is supplied, :meth:`CppSimulator.run_hpc` runs
the same 3-tier cache walk QSPSimulator uses (local test-stats →
HPC test-stats → HPC full sims + on-cluster derivation → fresh sweep
+ chained derivation), returning ``(theta, test_stats)`` arrays.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.constants import HASH_PREFIX_LENGTH, JOB_QUEUE_TIMEOUT, SLURM_REGISTRATION_DELAY
from qsp_hpc.cpp.batch_runner import CppBatchRunner, write_pool_manifest
from qsp_hpc.utils.logging_config import create_child_logger, format_config, setup_logger

if TYPE_CHECKING:
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

logger = logging.getLogger(__name__)


class CppSimulator:
    """Local C++ QSP simulator with pool-based caching.

    Samples prior parameters, runs simulations via the C++ qsp_sim
    binary, and caches results as Parquet files in a content-addressed
    pool directory.  Designed as a drop-in replacement for the simulation
    step in QSPSimulator's pipeline.

    The pool directory layout mirrors the MATLAB path::

        cache/sbi_simulations/{model_version}_{hash[:8]}_{scenario}/
            batch_YYYYMMDD_HHMMSS_{scenario}_{N}sims_seed{S}.parquet

    The config hash includes the binary's SHA-256 and the template XML
    content so that rebuilding the C++ core invalidates the cache
    automatically.
    """

    def __init__(
        self,
        priors_csv: str | Path,
        binary_path: str | Path,
        template_xml: str | Path,
        model_version: str = "v1",
        scenario: str = "default",
        subtree: str | None = "QSP",
        t_end_days: float = 180.0,
        dt_days: float = 1.0,
        cache_dir: str | Path = "cache/sbi_simulations",
        seed: int = 2025,
        theta_pool_size: int = 100_000,
        restriction_classifier_dir: Optional[str | Path] = None,
        restriction_threshold: float = 0.5,
        max_workers: int | None = None,
        per_sim_timeout_s: float | None = None,
        submodel_priors_yaml: Optional[str | Path] = None,
        scenario_yaml: Optional[str | Path] = None,
        drug_metadata_yaml: Optional[str | Path] = None,
        healthy_state_yaml: Optional[str | Path] = None,
        job_manager: Optional["HPCJobManager"] = None,
        test_stats_csv: Optional[str | Path] = None,
        calibration_targets: Optional[str | Path] = None,
        model_structure_file: Optional[str | Path] = None,
        poll_interval: float = 30.0,
        max_wait_time: Optional[float] = None,
        remote_binary_path: Optional[str] = None,
        remote_template_xml: Optional[str] = None,
        verbose: bool = False,
        evolve_cache: bool = True,
        evolve_cache_root: Optional[str | Path] = None,
    ):
        self.priors_csv = Path(priors_csv)
        if not self.priors_csv.exists():
            raise FileNotFoundError(f"Priors CSV not found: {self.priors_csv}")

        self.binary_path = Path(binary_path).resolve()
        self.template_xml = Path(template_xml).resolve()
        self.model_version = model_version
        self.scenario = scenario
        self.subtree = subtree
        self.t_end_days = t_end_days
        self.dt_days = dt_days
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.theta_pool_size = theta_pool_size
        # Optional classifier-based prior restriction. When set, the theta
        # pool is rejection-sampled against the classifier so only thetas
        # with p(valid) >= restriction_threshold are included. The hash of
        # the classifier dir becomes part of the pool cache key, so a
        # restricted pool lives alongside the unrestricted one on disk.
        self.restriction_classifier_dir = (
            Path(restriction_classifier_dir).resolve()
            if restriction_classifier_dir is not None
            else None
        )
        self.restriction_threshold = float(restriction_threshold)
        self.max_workers = max_workers
        self.per_sim_timeout_s = per_sim_timeout_s
        self.submodel_priors_yaml = Path(submodel_priors_yaml) if submodel_priors_yaml else None
        self.scenario_yaml = Path(scenario_yaml).resolve() if scenario_yaml else None
        self.drug_metadata_yaml = Path(drug_metadata_yaml).resolve() if drug_metadata_yaml else None
        self.healthy_state_yaml = Path(healthy_state_yaml).resolve() if healthy_state_yaml else None
        self.job_manager = job_manager
        # calibration_targets and test_stats_csv are mutually exclusive — the
        # YAML directory is the public-facing API used by pdac-build; the CSV
        # form is the internal/legacy path. Mirrors QSPSimulator.__init__.
        if calibration_targets is not None and test_stats_csv is not None:
            raise ValueError("Provide test_stats_csv OR calibration_targets, not both")
        self._calibration_targets_dir: Optional[Path] = None
        self._temp_test_stats_csv: Optional[Path] = None
        if calibration_targets is not None:
            from qsp_hpc.calibration import load_calibration_targets

            cal_dir = Path(calibration_targets).resolve()
            self._calibration_targets_dir = cal_dir
            df = load_calibration_targets(cal_dir)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix="cpp_cal_targets_"
            )
            tmp.close()
            self._temp_test_stats_csv = Path(tmp.name)
            df.to_csv(self._temp_test_stats_csv, index=False)
            test_stats_csv = self._temp_test_stats_csv
        self.test_stats_csv = Path(test_stats_csv).resolve() if test_stats_csv else None
        self.model_structure_file = (
            Path(model_structure_file).resolve() if model_structure_file else None
        )
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        # HPC-side paths for the binary + template. The laptop-resident
        # self.binary_path / self.template_xml are used for config hashing
        # (read_bytes) and local CppBatchRunner execution; HPC sbatch scripts
        # need the path as it exists on the cluster. When omitted, run_hpc()
        # falls back to credentials.yaml's cpp.binary_path / cpp.template_path.
        self.remote_binary_path = remote_binary_path
        self.remote_template_xml = remote_template_xml
        # Populated by run_hpc() / __call__() with the global theta-pool
        # indices of the rows returned — sbi_runner reads this to align
        # scenarios. None until the first simulate call completes.
        self.last_sample_index: np.ndarray | None = None

        with open(self.priors_csv) as f:
            reader = csv.DictReader(f)
            self.param_names = [row["name"] for row in reader]

        # Evolve-to-diagnosis cache (M13). Keyed on the rendered param-XML
        # hash and shared across scenarios for the same theta, so
        # multi-arm sweeps pay one evolve (~95% of per-sim cost) per theta
        # instead of one per scenario. Location lives OUTSIDE the per-
        # scenario pool directory so baseline + treatment arms of the
        # same theta hit the same cache.
        #
        # Silently inert when healthy_state_yaml is None (nothing to
        # cache without an evolve phase) or evolve_cache=False.
        if evolve_cache and self.healthy_state_yaml is not None:
            self.evolve_cache_root = Path(
                evolve_cache_root
                if evolve_cache_root is not None
                else self.cache_dir / "evolve_cache"
            ).resolve()
        else:
            self.evolve_cache_root = None

        self._runner = CppBatchRunner(
            binary_path=binary_path,
            template_path=template_xml,
            subtree=subtree,
            default_timeout_s=per_sim_timeout_s or 120.0,
            scenario_yaml=self.scenario_yaml,
            drug_metadata_yaml=self.drug_metadata_yaml,
            healthy_state_yaml=self.healthy_state_yaml,
            evolve_cache_root=self.evolve_cache_root,
        )

        self.config_hash = self._compute_config_hash()

        pool_name = f"{model_version}_{self.config_hash[:HASH_PREFIX_LENGTH]}_{scenario}"
        self.pool_dir = self.cache_dir / pool_name
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # LOCAL pool only — CppSimulator writes one batch_*.parquet per
        # top-up directly (no combine step; #43 option A applies only to
        # HPC pools where array tasks shard into batch_*/chunk_*.parquet
        # subdirs).  `_task{N}` and `_{N}sims` segments are optional for
        # backward compatibility with pre-#21 pools; new runs don't emit
        # them and read row counts from parquet metadata instead.
        self._batch_pattern = re.compile(
            r"batch_(\d{8}_\d{6}(?:_\d{6})?)(?:_task\d+)?_(.+?)"
            r"(?:_(\d+)sims)?_seed(\d+)\.parquet"
        )

        base_logger = setup_logger(__name__, verbose=verbose)
        self.logger = create_child_logger(base_logger, scenario)

        self.logger.info(f"Initializing C++ simulator for scenario: {scenario}")
        config_info = {
            "binary": str(self.binary_path),
            "template": str(self.template_xml),
            "priors_csv": str(self.priors_csv),
            "model_version": model_version,
            "scenario": scenario,
            "config_hash": self.config_hash[:8] + "...",
            "pool_dir": str(self.pool_dir),
            "t_end_days": t_end_days,
            "dt_days": dt_days,
            "seed": seed,
            "scenario_yaml": str(self.scenario_yaml) if self.scenario_yaml else "-",
            "drug_metadata_yaml": (
                str(self.drug_metadata_yaml) if self.drug_metadata_yaml else "-"
            ),
            "healthy_state_yaml": (
                str(self.healthy_state_yaml) if self.healthy_state_yaml else "-"
            ),
            "evolve_cache_root": (
                str(self.evolve_cache_root) if self.evolve_cache_root else "disabled"
            ),
        }
        for line in format_config(config_info):
            self.logger.info(line)

    def _compute_config_hash(self) -> str:
        """Hash inputs that affect simulation outputs.

        Extends the standard pool-id hash with the binary's SHA-256 and
        the template XML content so rebuilding the C++ core or editing
        the template invalidates the cache. When a restriction classifier
        is configured, its bytes + threshold are folded in so restricted
        and unrestricted pools (sharing all other config) get distinct
        on-disk pool dirs.
        """
        from qsp_hpc.utils.hash_utils import compute_pool_id_hash

        base_hash = compute_pool_id_hash(
            priors_csv=self.priors_csv,
            model_script="",
            model_version=self.model_version,
            submodel_priors_yaml=self.submodel_priors_yaml,
            seed=self.seed,
        )

        h = hashlib.sha256(base_hash.encode())
        h.update(hashlib.sha256(self.binary_path.read_bytes()).hexdigest().encode())
        h.update(self.template_xml.read_text().encode())
        # Scenario/drug-meta/healthy-state YAMLs change sim outputs but live
        # outside the priors-CSV + XML template hashed above, so fold them
        # in explicitly — edits to any of these must invalidate the pool.
        for yml in (self.scenario_yaml, self.drug_metadata_yaml, self.healthy_state_yaml):
            if yml is not None:
                h.update(yml.read_bytes())
        # Restriction classifier: when set, the theta pool is a rejection-
        # sampled subset of the prior. Fold classifier bytes + threshold
        # into the hash so restricted and unrestricted pools (sharing all
        # other config) get distinct on-disk directories.
        if self.restriction_classifier_dir is not None:
            h.update(b"|restriction|")
            pkl = self.restriction_classifier_dir / "classifier.pkl"
            meta = self.restriction_classifier_dir / "metadata.json"
            if pkl.exists():
                h.update(pkl.read_bytes())
            if meta.exists():
                h.update(meta.read_bytes())
            h.update(f"|tau={self.restriction_threshold:.6f}".encode("utf-8"))
        return h.hexdigest()

    def __del__(self):
        """Clean up the temp CSV serialized from calibration_targets."""
        tmp = getattr(self, "_temp_test_stats_csv", None)
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Pool scanning / loading
    # ------------------------------------------------------------------

    def _scan_pool(self) -> list[dict]:
        """Scan pool directory for cached Parquet batch files.

        Row counts come from parquet metadata (footer-only read) rather
        than the filename's ``_{N}sims`` token — the token was unreliable
        once array-task chunk drops started producing batches with fewer
        rows than originally requested (see #21). The token is still
        accepted by the regex for back-compat but ignored when present.
        """
        batches = []
        for f in sorted(self.pool_dir.glob("batch_*.parquet")):
            m = self._batch_pattern.match(f.name)
            if not m:
                continue
            ts, file_scenario, _legacy_n_sims, file_seed = m.groups()
            if file_scenario != self.scenario:
                continue
            n_sims = pq.read_metadata(str(f)).num_rows
            batches.append(
                {
                    "filepath": f,
                    "timestamp": ts,
                    "n_sims": int(n_sims),
                    "seed": int(file_seed),
                }
            )
        return batches

    def get_available_simulations(self) -> int:
        """Total successful simulations cached in the pool."""
        return sum(b["n_sims"] for b in self._scan_pool())

    def _load_from_pool(self, n_requested: int) -> tuple[np.ndarray, pa.Table]:
        """Load cached simulations, filtering failed rows and sampling
        if the pool is larger than needed."""
        batches = self._scan_pool()
        if not batches:
            raise ValueError(f"No cached simulations for scenario '{self.scenario}'")

        tables = [pq.read_table(str(b["filepath"])) for b in batches]
        combined = pa.concat_tables(tables)

        status_col = combined.column("status").to_numpy()
        ok_mask = status_col == 0
        combined = combined.filter(pa.array(ok_mask))
        n_available = combined.num_rows

        if n_available > n_requested:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(n_available, size=n_requested, replace=False)
            indices.sort()
            combined = combined.take(indices)

        param_cols = sorted(c for c in combined.column_names if c.startswith("param:"))
        theta = np.column_stack([combined.column(c).to_numpy() for c in param_cols])

        return theta, combined

    # ------------------------------------------------------------------
    # Parameter generation (reuses the deterministic theta pool)
    # ------------------------------------------------------------------

    def _generate_parameters(self, indices: np.ndarray) -> np.ndarray:
        from qsp_hpc.simulation.theta_pool import theta_for_indices

        return theta_for_indices(
            indices=indices,
            priors_csv=self.priors_csv,
            submodel_priors_yaml=self.submodel_priors_yaml,
            seed=self.seed,
            n_total=self.theta_pool_size,
            cache_dir=self.cache_dir / "theta_pools",
            restriction_classifier_dir=self.restriction_classifier_dir,
            restriction_threshold=self.restriction_threshold,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, batch_size: Union[int, Tuple[int, ...]]) -> tuple[np.ndarray, pa.Table]:
        """Run simulations, returning cached results when available.

        Args:
            batch_size: Number of simulations (int or tuple — product
                is taken for tuple inputs like BayesFlow's ``(N,)``).

        Returns:
            ``(theta, table)`` where *theta* is ``(n_sims, n_params)``
            and *table* is a pyarrow Table with columns
            ``simulation_id``, ``status``, ``time``, ``param:*``, and
            one list-of-float column per species.
        """
        if isinstance(batch_size, tuple):
            n = int(np.prod(batch_size))
        else:
            n = int(batch_size)

        self.logger.info(f"Simulation request: {n} simulations (seed={self.seed})")

        n_available = self.get_available_simulations()

        if n_available >= n:
            self.logger.info(f"Using pool cache: {n_available} available")
            return self._load_from_pool(n)

        n_needed = n - n_available
        if n_available > 0:
            self.logger.info(f"Pool has {n_available}/{n} — running {n_needed} more")
        else:
            self.logger.info(f"No cached simulations — running {n}")

        start_idx = n_available
        indices = np.arange(start_idx, start_idx + n_needed, dtype=np.int64)
        theta_new = self._generate_parameters(indices)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # No `_{N}sims_` segment — actual row count comes from parquet
        # metadata at scan time (#21). Filename stays stable even if
        # some sims fail and the on-disk row count diverges from n_needed.
        filename = f"batch_{ts}_{self.scenario}_seed{self.seed}.parquet"
        output_path = self.pool_dir / filename

        # #23: sidecar manifest at the pool dir so downstream consumers
        # can resolve non-sampled template defaults without each parquet
        # carrying every one as a broadcast column. Idempotent — reused
        # across every batch into this pool.
        write_pool_manifest(self.pool_dir, self._runner.template_defaults, self.param_names)

        self._runner.run(
            theta_matrix=theta_new,
            param_names=self.param_names,
            sample_indices=indices,
            t_end_days=self.t_end_days,
            dt_days=self.dt_days,
            output_path=output_path,
            scenario=self.scenario,
            seed=self.seed,
            max_workers=self.max_workers,
            per_sim_timeout_s=self.per_sim_timeout_s,
        )

        self.logger.info(f"Batch complete, loading {n} from pool")
        return self._load_from_pool(n)

    # ------------------------------------------------------------------
    # HPC tier (M9) — 3-tier cache walk against an HPC pool
    # ------------------------------------------------------------------

    @property
    def simulation_pool_id(self) -> str:
        """The pool directory name shared between local and HPC.

        Both ``self.pool_dir`` and the HPC pool live at this name so the
        existing :meth:`HPCJobManager.check_hpc_test_stats` /
        :meth:`download_test_stats` (which take a pool path) work
        unchanged.
        """
        return self.pool_dir.name

    def _compute_test_stats_hash(self) -> str:
        """SHA-256 of the test-stats CSV (matches QSPSimulator)."""
        if self.test_stats_csv is None:
            raise RuntimeError("test_stats_csv must be set to compute test_stats_hash")
        from qsp_hpc.utils.hash_utils import compute_test_stats_hash

        return compute_test_stats_hash(self.test_stats_csv)

    def _local_test_stats_path(self, test_stats_hash: str) -> Path:
        """Where downloaded HPC test stats land locally.

        Mirrors the HPC layout (``{pool}/test_stats/{hash}/``) so the
        local cache key is obvious.
        """
        return self.pool_dir / "test_stats" / test_stats_hash / "test_stats.parquet"

    def _load_local_test_stats(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load the cached (params, test_stats) Parquet.

        Returns only the SAMPLED parameter columns (``self.param_names``),
        filtered by exact column match against the stored
        ``param:<name>`` headers. Pre-2026-04 parquets followed the
        "only sampled params are columns" convention; post-2026-04
        parquets include every template default as a broadcast column
        so cal-target functions can reach any model parameter. NPE
        training needs just the sampled set — mixing in constant
        template-defaults creates zero-variance / zero-valued columns
        that blow up np.log(theta) in sbi_runner.

        Also populates ``self.last_sample_index`` when the cache has the
        ``sample_index`` column.
        """
        table = pq.read_table(str(path))
        col_set = set(table.column_names)
        # Project to the sampled-priors subset, preserving priors-CSV order
        # so theta column j always means self.param_names[j] across scenarios.
        selected = [f"param:{n}" for n in self.param_names if f"param:{n}" in col_set]
        if len(selected) != len(self.param_names):
            missing = [n for n in self.param_names if f"param:{n}" not in col_set]
            raise RuntimeError(
                f"Local cache at {path} missing {len(missing)} sampled "
                f"param columns (e.g. {missing[:5]})."
            )
        ts_cols = sorted(
            (c for c in table.column_names if c.startswith("ts:")),
            key=lambda c: int(c.split(":", 1)[1]),
        )
        params = np.column_stack([table.column(c).to_numpy() for c in selected])
        test_stats = np.column_stack([table.column(c).to_numpy() for c in ts_cols])
        if "sample_index" in table.column_names:
            self.last_sample_index = table.column("sample_index").to_numpy().astype(np.int64)
        else:
            self.last_sample_index = None
        return params, test_stats

    def _persist_local_test_stats(
        self,
        path: Path,
        params: np.ndarray,
        test_stats: np.ndarray,
        sample_index: np.ndarray | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        """Persist downloaded HPC test stats next to the local pool.

        When ``param_names`` is supplied (the authoritative list from the
        downloaded params CSV — usually wider than ``self.param_names``
        because HPC parquets include every template parameter as a
        ``param:*`` column, not just the sampled subset), it is used to
        key the parquet columns. Absent that, we fall back to
        ``self.param_names`` sliced to ``params.shape[1]``, which silently
        mislabels columns when the HPC pool has more params than the
        sampled priors — downstream cross-scenario alignment then sees
        mixed column counts and blows up.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        cols: dict[str, np.ndarray] = {}
        if sample_index is not None:
            cols["sample_index"] = np.asarray(sample_index, dtype=np.int64)
        names = param_names if param_names is not None else self.param_names[: params.shape[1]]
        if len(names) != params.shape[1]:
            raise ValueError(
                f"param_names length {len(names)} does not match "
                f"params.shape[1]={params.shape[1]}"
            )
        for i, name in enumerate(names):
            cols[f"param:{name}"] = params[:, i]
        for j in range(test_stats.shape[1]):
            cols[f"ts:{j}"] = test_stats[:, j]
        table = pa.table(cols)
        pq.write_table(table, str(path))

    def _wait_for_jobs(self, job_ids: List[str]) -> None:
        """Poll ``check_job_status`` until all jobs leave the queue."""
        if self.job_manager is None:
            raise RuntimeError("job_manager required for _wait_for_jobs")

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
                except Exception as e:
                    self.logger.warning(f"Status check failed for job {jid}: {e}")
            total = sum(totals.values())
            max_seen = max(max_seen, total)
            elapsed = time.time() - start
            self.logger.info(
                f"  [{int(elapsed // 60)}m {int(elapsed % 60)}s] "
                f"{totals['completed']}/{total} done | "
                f"running={totals['running']} pending={totals['pending']} "
                f"failed={totals['failed']}"
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

    def _download_and_persist(
        self, hpc_pool_path: str, test_stats_hash: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Download HPC test stats and persist a local Parquet cache.

        Uses :meth:`HPCJobManager.download_test_stats_full` so ``sample_index``
        and the downloaded param column order ride on the return value
        rather than instance side-channels (pre-#22 pattern). ``sample_index``
        is stamped onto the local parquet for Tier-1 cache hits and stashed
        on ``self.last_sample_index`` for cross-scenario alignment.
        """
        if self.job_manager is None:
            raise RuntimeError("job_manager required for download")
        with tempfile.TemporaryDirectory() as tmp:
            result = self.job_manager.download_test_stats_full(
                hpc_pool_path, test_stats_hash, Path(tmp)
            )
        if result.params is None:
            raise RuntimeError("HPC has test stats but no params CSV — re-run derivation.")

        params = result.params
        test_stats = result.test_stats
        sample_index = result.sample_index
        param_names: Optional[List[str]] = result.param_names or None

        # Guard against mocks / malformed pools: require sample_index to
        # line up with rows, else treat it as unavailable (legacy pool).
        if sample_index is not None and len(sample_index) == len(params):
            self.last_sample_index = sample_index
        else:
            sample_index = None
            self.last_sample_index = None

        if param_names is not None and len(param_names) != params.shape[1]:
            param_names = None

        # Persist whatever param:* columns HPC sent. Post-#23 this is
        # sampled-only (thin parquets); pre-#23 pools shipped the full
        # template set. Either way, we pass through verbatim — the
        # local cache loader only looks for sampled columns, and
        # cal-target re-derivation is HPC-only (no local path needs
        # template defaults).
        self._persist_local_test_stats(
            self._local_test_stats_path(test_stats_hash),
            params,
            test_stats,
            sample_index=sample_index,
            param_names=param_names,
        )
        # But return ONLY the sampled-priors subset for NPE training —
        # constant-valued broadcast columns (zero flags, dimensionless
        # one-valued toggles) break np.log(theta) downstream.
        if param_names is not None:
            idxs = [param_names.index(n) for n in self.param_names if n in param_names]
            if len(idxs) != len(self.param_names):
                missing = [n for n in self.param_names if n not in param_names]
                raise RuntimeError(
                    f"HPC download missing {len(missing)} sampled param "
                    f"columns (e.g. {missing[:5]}). Check priors CSV vs "
                    f"param_all.xml consistency."
                )
            params = params[:, idxs]
        return params, test_stats

    def _sample_first_n(
        self, params: np.ndarray, test_stats: np.ndarray, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the first ``n`` rows (params + test_stats), or all if fewer.

        Also trims ``self.last_sample_index`` so cross-scenario alignment
        downstream sees the exact indices that match the returned rows.
        """
        if params.shape[0] <= n:
            return params, test_stats
        if self.last_sample_index is not None:
            self.last_sample_index = self.last_sample_index[:n]
        return params[:n], test_stats[:n]

    def _write_params_csv(self, n_sims: int, start_index: int = 0) -> Path:
        """Generate ``n_sims`` thetas via the deterministic theta pool and
        write them to a temp CSV that ``submit_cpp_jobs`` can upload.

        ``start_index`` lets callers request a later slice of the theta
        pool — used by the top-up path in :meth:`run_hpc` when the HPC
        pool already has ``n_hpc`` sims and only ``n - n_hpc`` new draws
        are needed. Theta identity is preserved across submissions because
        the theta pool is seed-deterministic.

        The CSV layout mirrors the MATLAB convention:
        ``sample_index`` is the first column (int64), followed by one
        column per parameter. ``cpp_batch_worker`` peels it off, and the
        written parquet carries the column through — so downstream
        multi-scenario alignment (sbi_runner.py) can intersect pools on
        ``sample_index`` rather than row position.
        """
        indices = np.arange(start_index, start_index + n_sims, dtype=np.int64)
        theta = self._generate_parameters(indices)
        df = pd.DataFrame(theta, columns=self.param_names)
        df.insert(0, "sample_index", indices)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="cpp_hpc_params_"
        )
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return Path(tmp.name)

    def validate(self) -> None:
        """Run pre-flight checks that don't require HPC access.

        Catches bugs that would otherwise burn ~2 minutes of HPC submit
        time + a full array's worth of doomed compute before surfacing
        as a worker crash. Failures raise immediately with messages
        pointing at the offending file (see #31).

        Currently checks: priors CSV column names ⊆ XML template
        parameter names. Future checks (test_stats observables ⊆
        model_structure species, scenario-yaml initialization_function
        when evolve-to-diagnosis is required) can be appended as
        ``_validate_*`` helpers and invoked from here.
        """
        self._validate_priors_in_xml()

    def _validate_priors_in_xml(self) -> None:
        """Every priors CSV name must exist in the XML template.

        Concrete failure mode: 50 array tasks each crash at
        CppBatchRunner.run with ParamNotFoundError because the priors
        CSV carries 40 orphan rows (left over from an older model
        version) that aren't in param_all.xml. Without set -e in the
        sbatch (#27) this used to look like 50/50 COMPLETED, then the
        derivation found zero parquets and reported "no batches".
        Surfacing this locally takes ~ms and saves the round-trip.
        """
        unknown = sorted(set(self.param_names) - self._runner.parameter_names)
        if unknown:
            raise ValueError(
                f"{len(unknown)} priors CSV column(s) not in XML template "
                f"({self.template_xml.name}): {unknown[:10]}"
                + (f" ...(+{len(unknown) - 10} more)" if len(unknown) > 10 else "")
                + f". Drop these rows from {self.priors_csv} or add them to the "
                "template before submitting."
            )

    def run_hpc(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run ``n`` simulations through the 3-tier HPC cache.

        Returns ``(theta, test_stats)`` arrays — *not* full trajectories.
        Mirrors :meth:`QSPSimulator.__call__`'s flow:

        1. Local test-stats cache hit → return immediately.
        2. HPC test-stats hit → download + cache locally.
        3. HPC has full sims (no derived stats) → submit derivation,
           wait, download.
        4. Otherwise → submit a fresh C++ array with chained derivation,
           wait for both, download.

        Requires ``job_manager`` and ``test_stats_csv`` set in the
        constructor.
        """
        if self.job_manager is None:
            raise RuntimeError("run_hpc() requires job_manager")
        if self.test_stats_csv is None:
            raise RuntimeError("run_hpc() requires test_stats_csv")
        # Pre-flight (#31): surface priors/template mismatches before any
        # ssh / sbatch round-trip. Local tier hits skip HPC entirely so
        # validation runs only on the path that would actually submit.
        self.validate()

        # Resolve HPC-side paths: explicit ctor arg → credentials → error.
        # Laptop-resident self.binary_path / self.template_xml are unsuitable
        # for the cluster (the sbatch worker and ensure_cpp_binary check both
        # need paths that exist on HPC).
        remote_binary_path = self.remote_binary_path or self.job_manager.config.cpp_binary_path
        remote_template_xml = self.remote_template_xml or self.job_manager.config.cpp_template_path
        if not remote_binary_path:
            raise RuntimeError(
                "HPC binary path unset — pass remote_binary_path to CppSimulator "
                "or set cpp.binary_path in credentials.yaml"
            )
        if not remote_template_xml:
            raise RuntimeError(
                "HPC template path unset — pass remote_template_xml to CppSimulator "
                "or set cpp.template_path in credentials.yaml"
            )
        if self.model_structure_file is None:
            raise RuntimeError(
                "run_hpc() requires model_structure_file — without it the "
                "derivation worker treats every species as dimensionless "
                "and most cal-target unit conversions silently NaN out."
            )

        test_stats_hash = self._compute_test_stats_hash()
        local_cache = self._local_test_stats_path(test_stats_hash)

        self.logger.info(f"HPC request: {n} simulations (scenario={self.scenario})")
        self.logger.info(f"  pool id: {self.simulation_pool_id}")
        self.logger.info(f"  test_stats_hash: {test_stats_hash[:8]}...")

        # Tier 1: local test-stats cache
        if local_cache.exists():
            params, test_stats = self._load_local_test_stats(local_cache)
            if params.shape[0] >= n:
                self.logger.info(f"✓ Local test-stats cache hit ({params.shape[0]} available)")
                return self._sample_first_n(params, test_stats, n)
            self.logger.info(f"Local cache has {params.shape[0]}/{n} — checking HPC")

        hpc_pool_path = f"{self.job_manager.config.simulation_pool_path}/{self.simulation_pool_id}"
        self.logger.info(f"  HPC pool path: {hpc_pool_path}")

        # Tier 2: HPC pre-derived test stats
        self.logger.info("Checking HPC for pre-derived test statistics...")
        if self.job_manager.check_hpc_test_stats(hpc_pool_path, test_stats_hash, expected_n_sims=n):
            self.logger.info("✓ HPC test stats found — downloading")
            params, test_stats = self._download_and_persist(hpc_pool_path, test_stats_hash)
            if params.shape[0] >= n:
                return self._sample_first_n(params, test_stats, n)
            # Partial — check_hpc_test_stats returned True because some
            # stats exist, but we have fewer than n. _download_and_persist
            # has already cached what's there locally (Tier 1 will pick
            # it up next run). Fall through to Tier 3 / 3.5 so the pool
            # can be topped up and a fresh derivation covers all rows.
            # Without this fall-through, run_hpc(1000) over a 40-sim pool
            # silently returns (40, ...) — caller sees N-too-small arrays
            # and the "caller can top up" comment in check_hpc_test_stats
            # goes unenforced.
            self.logger.info(
                "HPC test stats partial (%d/%d) — falling through to Tier 3 for top-up",
                params.shape[0],
                n,
            )
        else:
            self.logger.info("No pre-derived test stats on HPC")

        # Tier 3: HPC full sims exist — derive on cluster, then download
        has_pool = self.job_manager.result_collector.check_pool_directory_exists(hpc_pool_path)
        n_hpc = (
            self.job_manager.result_collector.count_pool_simulations(hpc_pool_path)
            if has_pool
            else 0
        )
        if n_hpc >= n:
            self.logger.info(f"✓ HPC pool has {n_hpc} full sims — submitting derivation")
            derive_id = self.job_manager.submit_derivation_job(
                pool_path=hpc_pool_path,
                test_stats_csv=str(self.test_stats_csv),
                test_stats_hash=test_stats_hash,
                model_structure_file=(
                    str(self.model_structure_file) if self.model_structure_file else None
                ),
            )
            self.logger.info(f"Derivation job: {derive_id}")
            self._wait_for_jobs([derive_id])
            params, test_stats = self._download_and_persist(hpc_pool_path, test_stats_hash)
            return self._sample_first_n(params, test_stats, n)

        # Tier 3.5: partial pool — run only the delta.
        # Mirrors QSPSimulator._run_new_simulations(n_needed) — if the pool
        # has some sims but not enough, submit only (n - n_hpc) new draws
        # with theta indices [n_hpc, n) from the deterministic theta pool.
        # Each submission drops a new batch_{ts}_*/chunk_*.parquet subdir
        # in the pool (#43 option A), so the next derivation walks old +
        # new transparently.
        if n_hpc > 0:
            n_needed = n - n_hpc
            start_index = n_hpc
            self.logger.info(
                "HPC pool has %d/%d sims — submitting delta of %d (indices [%d, %d))",
                n_hpc,
                n,
                n_needed,
                start_index,
                start_index + n_needed,
            )
        else:
            # Tier 4: empty pool — submit a fresh full sweep.
            n_needed = n
            start_index = 0
            self.logger.info("HPC pool empty — submitting fresh sweep of %d sims", n_needed)

        params_csv = self._write_params_csv(n_needed, start_index=start_index)
        try:
            info = self.job_manager.submit_cpp_jobs(
                samples_csv=str(params_csv),
                num_simulations=n_needed,
                simulation_pool_id=self.simulation_pool_id,
                t_end_days=self.t_end_days,
                dt_days=self.dt_days,
                scenario=self.scenario,
                seed=self.seed,
                binary_path=remote_binary_path,
                template_path=remote_template_xml,
                subtree=self.subtree,
                scenario_yaml=(str(self.scenario_yaml) if self.scenario_yaml else None),
                drug_metadata_yaml=(
                    str(self.drug_metadata_yaml) if self.drug_metadata_yaml else None
                ),
                healthy_state_yaml=(
                    str(self.healthy_state_yaml) if self.healthy_state_yaml else None
                ),
                derive_test_stats=True,
                test_stats_csv=str(self.test_stats_csv),
                test_stats_hash=test_stats_hash,
                model_structure_file=(
                    str(self.model_structure_file) if self.model_structure_file else None
                ),
                evolve_cache=self.evolve_cache_root is not None,
            )
        finally:
            params_csv.unlink(missing_ok=True)

        self.logger.info(f"Submitted jobs: {info.job_ids}")
        self._wait_for_jobs(info.job_ids)
        params, test_stats = self._download_and_persist(hpc_pool_path, test_stats_hash)
        return self._sample_first_n(params, test_stats, n)

    # ------------------------------------------------------------------
    # Posterior predictive: run at user-supplied thetas + cache
    # ------------------------------------------------------------------

    def simulate_with_parameters(
        self,
        theta: np.ndarray,
        *,
        prediction_targets: Optional[str | Path] = None,
        pool_suffix: str = "posterior_predictive",
    ) -> Tuple[np.ndarray, pa.Table]:
        """Run C++ simulations at explicit thetas and return derived test stats.

        Mirrors :meth:`QSPSimulator.simulate_with_parameters` (local path):
        the theta matrix is user-supplied (typically posterior draws), not
        drawn from the prior theta pool. Each unique theta matrix maps to a
        dedicated suffix pool dir keyed by SHA-256 of ``theta.tobytes()``
        plus the calibration / prediction target hashes — so identical
        inputs hit the cache without row-by-row matching, and edits to
        either target set invalidate the cache automatically.

        Test stats are derived locally after the batch runs: the parquet
        from :meth:`CppBatchRunner.run` is fed through
        :func:`compute_test_statistics_batch` along with the merged
        calibration + prediction DataFrame. Output columns are named
        ``ts:<test_statistic_id>`` (not positional ``ts:0 ts:1…``) so
        callers can identify the 12 new prediction columns by id rather
        than by ordering.

        Args:
            theta: Parameter matrix ``(n_samples, n_params)``, columns
                aligned with ``self.param_names``.
            prediction_targets: Optional directory of PredictionTarget
                YAMLs (``prediction_target_id`` schema). When given, the
                prediction rows are concatenated with calibration rows
                before derivation and contribute extra ``ts:*`` columns.
            pool_suffix: Label combined with the theta hash to build the
                suffix-pool directory name. Only change this when you want
                two logically distinct posterior-predictive runs to stay
                cache-isolated even when the thetas happen to collide.

        Returns:
            ``(theta_out, table)`` where ``theta_out`` has the same shape
            as the input (failed rows stay NaN-filled but are not dropped
            — caller decides how to handle them) and ``table`` is a
            pyarrow Table with columns:
                - ``sample_index`` (int64) — ``arange(n_samples)``
                - ``status`` (int64)       — 0 ok / nonzero failed
                - ``param:<name>``         — one per sampled param
                - ``ts:<target_id>``       — one per test-stat, NaN on fail
        """
        if self.test_stats_csv is None and self._calibration_targets_dir is None:
            raise RuntimeError(
                "simulate_with_parameters() requires test_stats_csv or "
                "calibration_targets at construction; without them there is "
                "nothing to derive."
            )
        if theta.ndim != 2:
            raise ValueError(f"theta must be 2-D; got shape {theta.shape}")
        if theta.shape[1] != len(self.param_names):
            raise ValueError(
                f"theta has {theta.shape[1]} columns but priors CSV has "
                f"{len(self.param_names)} parameters"
            )

        theta = np.ascontiguousarray(theta, dtype=np.float64)
        n_samples = theta.shape[0]

        pred_dir: Optional[Path] = (
            Path(prediction_targets).resolve() if prediction_targets is not None else None
        )

        # Cache key: theta + calibration targets + prediction targets.
        # Without the target hashes, editing a YAML silently hits a stale
        # pool and the caller sees endpoint columns derived from the old
        # observable code.
        theta_hash = hashlib.sha256(theta.tobytes()).hexdigest()
        cal_hash = self._calibration_targets_hash()
        pred_hash = self._prediction_targets_hash(pred_dir)
        key_hash = hashlib.sha256(
            (theta_hash + "|" + cal_hash + "|" + pred_hash).encode()
        ).hexdigest()[:HASH_PREFIX_LENGTH]

        suffix_pool_dir = self.pool_dir.parent / f"{self.pool_dir.name}_{pool_suffix}_{key_hash}"
        suffix_pool_dir.mkdir(parents=True, exist_ok=True)
        cache_path = suffix_pool_dir / "test_stats.parquet"

        self.logger.info(
            f"simulate_with_parameters: n={n_samples}, "
            f"theta_hash={theta_hash[:8]}, "
            f"cal_hash={cal_hash[:8]}, pred_hash={pred_hash[:8]}"
        )
        self.logger.info(f"  suffix pool: {suffix_pool_dir.name}")

        if cache_path.exists():
            cached = pq.read_table(str(cache_path))
            if cached.num_rows >= n_samples:
                self.logger.info(f"✓ suffix-pool cache hit ({cached.num_rows} rows)")
                cached = cached.slice(0, n_samples)
                theta_out = self._theta_from_table(cached)
                return theta_out, cached
            self.logger.info(
                f"suffix-pool cache has {cached.num_rows}/{n_samples} — "
                "recomputing (partial caches are discarded)"
            )

        # Resolve the merged test-stats DataFrame. load_calibration_targets
        # only runs when the caller didn't pre-flatten via test_stats_csv.
        test_stats_df = self._load_test_stats_df(pred_dir)

        # Run the sweep at the user's thetas. Sample indices are local
        # (arange) — the suffix pool is isolated by theta_hash so there's
        # no cross-scenario alignment to worry about.
        sample_indices = np.arange(n_samples, dtype=np.int64)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        species_parquet = (
            suffix_pool_dir / f"species_{ts}_{pool_suffix}_{n_samples}sims_seed{self.seed}.parquet"
        )
        # #23: the pool manifest is what lets derive look up non-sampled
        # template defaults (e.g. parameters referenced by a calibration
        # target but not varied by the priors CSV). Without it, those
        # targets silently NaN out.
        write_pool_manifest(suffix_pool_dir, self._runner.template_defaults, self.param_names)

        self._runner.run(
            theta_matrix=theta,
            param_names=self.param_names,
            sample_indices=sample_indices,
            t_end_days=self.t_end_days,
            dt_days=self.dt_days,
            output_path=species_parquet,
            scenario=pool_suffix,
            seed=self.seed,
            max_workers=self.max_workers,
            per_sim_timeout_s=self.per_sim_timeout_s,
        )

        species_df = pd.read_parquet(species_parquet)
        table = self._derive_test_stats_table(species_df, test_stats_df, theta, sample_indices)
        pq.write_table(table, str(cache_path))
        self.logger.info(
            f"simulate_with_parameters complete: {table.num_rows} rows × "
            f"{len(test_stats_df)} test stats → {cache_path.name}"
        )
        theta_out = self._theta_from_table(table)
        return theta_out, table

    # ------------------------------------------------------------------
    # Helpers for simulate_with_parameters
    # ------------------------------------------------------------------

    def _calibration_targets_hash(self) -> str:
        """Hash of the calibration-target directory, or of the flat CSV
        when the caller passed ``test_stats_csv`` directly."""
        if self._calibration_targets_dir is not None:
            from qsp_hpc.calibration import hash_calibration_targets

            return hash_calibration_targets(self._calibration_targets_dir)
        if self.test_stats_csv is not None:
            return hashlib.sha256(self.test_stats_csv.read_bytes()).hexdigest()
        return ""

    def _prediction_targets_hash(self, pred_dir: Optional[Path]) -> str:
        """Hash of the prediction-target directory. Empty string when
        prediction targets are not requested — keeps the cache key stable
        for calibration-only callers."""
        if pred_dir is None:
            return ""
        from qsp_hpc.calibration import hash_prediction_targets

        return hash_prediction_targets(pred_dir)

    def _load_test_stats_df(self, pred_dir: Optional[Path]) -> pd.DataFrame:
        """Merge calibration + prediction rows into the single DataFrame
        passed to ``compute_test_statistics_batch``.

        When constructed with ``calibration_targets=`` (directory), we
        reload from YAMLs so the ``is_prediction_only`` flag lines up with
        the prediction rows. When constructed with the legacy
        ``test_stats_csv=`` path we read that CSV and assume every row is
        a calibration target (``is_prediction_only=False``) — the common
        case for SBI training flows that don't care about predictions.
        """
        if self._calibration_targets_dir is not None:
            from qsp_hpc.calibration import load_calibration_targets

            cal_df = load_calibration_targets(self._calibration_targets_dir)
        else:
            cal_df = pd.read_csv(self.test_stats_csv)

        # Calibration rows intentionally lack ``is_prediction_only`` — the
        # column is dropped from the canonical CSV schema so
        # compute_test_stats_hash stays byte-stable across HPC caches.
        # Back-fill here to let the downstream concat produce a
        # rectangular frame.
        if "is_prediction_only" not in cal_df.columns:
            cal_df = cal_df.copy()
            cal_df["is_prediction_only"] = False

        if pred_dir is None:
            return cal_df

        from qsp_hpc.calibration import load_prediction_targets

        pred_df = load_prediction_targets(pred_dir)
        # Guard against id collisions — calibration and prediction share
        # the ``test_statistic_id`` column, and compute_test_statistics_batch
        # keys its registry on that id. A collision would silently overwrite
        # one function with the other.
        dup = set(cal_df["test_statistic_id"]) & set(pred_df["test_statistic_id"])
        if dup:
            raise ValueError(
                f"Prediction target id(s) collide with calibration target ids: {sorted(dup)}"
            )
        return pd.concat([cal_df, pred_df], ignore_index=True)

    def _derive_test_stats_table(
        self,
        species_df: pd.DataFrame,
        test_stats_df: pd.DataFrame,
        theta: np.ndarray,
        sample_indices: np.ndarray,
    ) -> pa.Table:
        """Build the output pa.Table: sample_index / status / param:* / ts:<id>."""
        from qsp_hpc.batch.derive_test_stats_worker import (
            build_test_stat_registry,
            compute_test_statistics_batch,
        )

        registry = build_test_stat_registry(test_stats_df)

        species_units: dict = {}
        if self.model_structure_file is not None and self.model_structure_file.exists():
            from qsp_hpc.utils.model_structure_units import load_units_from_model_structure

            species_units = load_units_from_model_structure(self.model_structure_file)

        template_defaults = dict(self._runner.template_defaults or {})

        test_stats_matrix = compute_test_statistics_batch(
            species_df,
            test_stats_df,
            registry,
            species_units,
            template_defaults=template_defaults,
        )

        status_np = (
            species_df["status"].to_numpy().astype(np.int64)
            if "status" in species_df.columns
            else np.zeros(theta.shape[0], dtype=np.int64)
        )

        cols: dict[str, pa.Array] = {
            "sample_index": pa.array(sample_indices, type=pa.int64()),
            "status": pa.array(status_np, type=pa.int64()),
        }
        for j, name in enumerate(self.param_names):
            cols[f"param:{name}"] = pa.array(theta[:, j].astype(np.float64))
        for j, tsid in enumerate(test_stats_df["test_statistic_id"].tolist()):
            cols[f"ts:{tsid}"] = pa.array(test_stats_matrix[:, j].astype(np.float64))
        return pa.table(cols)

    def _theta_from_table(self, table: pa.Table) -> np.ndarray:
        """Recover the theta matrix from a cached suffix-pool table.

        Reads ``param:<name>`` columns in priors-CSV order — not sorted —
        so downstream callers see the same column semantics whether they
        just ran a fresh sweep or hit the cache.
        """
        arrays = [table.column(f"param:{n}").to_numpy() for n in self.param_names]
        return np.column_stack(arrays).astype(np.float64)

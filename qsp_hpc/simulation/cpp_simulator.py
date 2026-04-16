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
from qsp_hpc.cpp.batch_runner import CppBatchRunner
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
        verbose: bool = False,
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

        with open(self.priors_csv) as f:
            reader = csv.DictReader(f)
            self.param_names = [row["name"] for row in reader]

        self._runner = CppBatchRunner(
            binary_path=binary_path,
            template_path=template_xml,
            subtree=subtree,
            default_timeout_s=per_sim_timeout_s or 120.0,
            scenario_yaml=self.scenario_yaml,
            drug_metadata_yaml=self.drug_metadata_yaml,
            healthy_state_yaml=self.healthy_state_yaml,
        )

        self.config_hash = self._compute_config_hash()

        pool_name = f"{model_version}_{self.config_hash[:HASH_PREFIX_LENGTH]}_{scenario}"
        self.pool_dir = self.cache_dir / pool_name
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # Optional `_task{N}` segment is emitted by the HPC worker
        # (cpp_batch_worker.py) to make concurrent array tasks' filenames
        # unique when timestamp resolution is only 1 second.  Local-only
        # CppSimulator runs don't use it.
        self._batch_pattern = re.compile(
            r"batch_(\d{8}_\d{6})(?:_task\d+)?_(.+?)_(\d+)sims_seed(\d+)\.parquet"
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
        }
        for line in format_config(config_info):
            self.logger.info(line)

    def _compute_config_hash(self) -> str:
        """Hash inputs that affect simulation outputs.

        Extends the standard pool-id hash with the binary's SHA-256 and
        the template XML content so rebuilding the C++ core or editing
        the template invalidates the cache.
        """
        from qsp_hpc.utils.hash_utils import compute_pool_id_hash

        base_hash = compute_pool_id_hash(
            priors_csv=self.priors_csv,
            model_script="",
            model_version=self.model_version,
            submodel_priors_yaml=self.submodel_priors_yaml,
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
        """Scan pool directory for cached Parquet batch files."""
        batches = []
        for f in sorted(self.pool_dir.glob("batch_*.parquet")):
            m = self._batch_pattern.match(f.name)
            if not m:
                continue
            ts, file_scenario, n_sims, file_seed = m.groups()
            if file_scenario != self.scenario:
                continue
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
        filename = f"batch_{ts}_{self.scenario}_{n_needed}sims_seed{self.seed}.parquet"
        output_path = self.pool_dir / filename

        self._runner.run(
            theta_matrix=theta_new,
            param_names=self.param_names,
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
        """Load the cached (params, test_stats) Parquet."""
        table = pq.read_table(str(path))
        param_cols = sorted(c for c in table.column_names if c.startswith("param:"))
        ts_cols = sorted(
            (c for c in table.column_names if c.startswith("ts:")),
            key=lambda c: int(c.split(":", 1)[1]),
        )
        params = np.column_stack([table.column(c).to_numpy() for c in param_cols])
        test_stats = np.column_stack([table.column(c).to_numpy() for c in ts_cols])
        return params, test_stats

    def _persist_local_test_stats(
        self, path: Path, params: np.ndarray, test_stats: np.ndarray
    ) -> None:
        """Persist downloaded HPC test stats next to the local pool."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cols: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.param_names[: params.shape[1]]):
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
        """Download HPC test stats and persist a local Parquet cache."""
        if self.job_manager is None:
            raise RuntimeError("job_manager required for download")
        with tempfile.TemporaryDirectory() as tmp:
            params, test_stats = self.job_manager.download_test_stats(
                hpc_pool_path, test_stats_hash, Path(tmp)
            )
        if params is None:
            raise RuntimeError("HPC has test stats but no params CSV — re-run derivation.")
        self._persist_local_test_stats(
            self._local_test_stats_path(test_stats_hash), params, test_stats
        )
        return params, test_stats

    def _sample_first_n(
        self, params: np.ndarray, test_stats: np.ndarray, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the first ``n`` rows (params + test_stats), or all if fewer."""
        if params.shape[0] <= n:
            return params, test_stats
        return params[:n], test_stats[:n]

    def _write_params_csv(self, n_sims: int) -> Path:
        """Generate ``n_sims`` thetas via the deterministic theta pool and
        write them to a temp CSV that ``submit_cpp_jobs`` can upload."""
        indices = np.arange(0, n_sims, dtype=np.int64)
        theta = self._generate_parameters(indices)
        df = pd.DataFrame(theta, columns=self.param_names)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="cpp_hpc_params_"
        )
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return Path(tmp.name)

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
            return self._sample_first_n(params, test_stats, n)
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

        # Tier 4: submit fresh C++ array + chained derivation
        self.logger.info(
            f"HPC pool has {n_hpc}/{n} sims — submitting fresh sweep + chained derivation"
        )
        params_csv = self._write_params_csv(n)
        try:
            info = self.job_manager.submit_cpp_jobs(
                samples_csv=str(params_csv),
                num_simulations=n,
                simulation_pool_id=self.simulation_pool_id,
                t_end_days=self.t_end_days,
                dt_days=self.dt_days,
                scenario=self.scenario,
                seed=self.seed,
                binary_path=str(self.binary_path),
                template_path=str(self.template_xml),
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
            )
        finally:
            params_csv.unlink(missing_ok=True)

        self.logger.info(f"Submitted jobs: {info.job_ids}")
        self._wait_for_jobs(info.job_ids)
        params, test_stats = self._download_and_persist(hpc_pool_path, test_stats_hash)
        return self._sample_first_n(params, test_stats, n)

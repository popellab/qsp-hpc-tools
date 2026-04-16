"""C++ QSP simulator with local pool-based caching.

Mirrors QSPSimulator's callable interface but uses CppBatchRunner
for execution instead of MATLAB. No HPC integration (see M6).
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.constants import HASH_PREFIX_LENGTH
from qsp_hpc.cpp.batch_runner import CppBatchRunner
from qsp_hpc.utils.logging_config import create_child_logger, format_config, setup_logger

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

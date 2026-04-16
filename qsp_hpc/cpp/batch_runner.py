"""Run a batch of C++ QSP simulations and write one MATLAB-compatible Parquet.

This layer sits between the single-sim CppRunner (M3) and the top-level
CppSimulator (M5). Given a `theta_matrix` of parameter samples plus the
priors column names, it fans out over a ProcessPoolExecutor, collects
trajectories, and writes a Parquet whose schema matches what the MATLAB
pipeline has been emitting — so downstream caching / test-stat
derivation code keeps working without edits.

Schema (one row per simulation):
    simulation_id: int64     # zero-indexed within this batch
    status:        int64     # 0 = success, 1 = qsp_sim failure
    time:          list<float64>    # length n_times (same for all rows)
    param:<name>:  float64   # one column per priors-CSV param
    <species>:     list<float64>   # one column per qsp_sim species
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qsp_hpc.cpp.param_xml import ParamNotFoundError
from qsp_hpc.cpp.runner import CppRunner, QspSimError, SimResult

logger = logging.getLogger(__name__)


STATUS_OK = 0
STATUS_FAILED = 1


@dataclass
class BatchResult:
    """Summary returned after a batch completes."""

    parquet_path: Path
    n_sims: int
    n_failed: int
    species_names: list[str]
    n_times: int


# --- Worker (module-level so ProcessPoolExecutor can pickle it) -------------

# Each worker process holds one CppRunner for its lifetime; we initialize it
# in an executor `initializer` and stash it as a module global. The hot path
# then avoids repaying ParamXMLRenderer parse + binary-path validation on
# every sim.
_WORKER_RUNNER: CppRunner | None = None
_WORKER_WORKDIR: Path | None = None


def _worker_init(
    binary_path: str,
    template_path: str,
    subtree: str | None,
    workdir: str,
    default_timeout_s: float,
    scenario_yaml: str | None,
    drug_metadata_yaml: str | None,
    healthy_state_yaml: str | None,
) -> None:
    global _WORKER_RUNNER, _WORKER_WORKDIR
    _WORKER_RUNNER = CppRunner(
        binary_path=binary_path,
        template_path=template_path,
        subtree=subtree,
        default_timeout_s=default_timeout_s,
        scenario_yaml=scenario_yaml,
        drug_metadata_yaml=drug_metadata_yaml,
        healthy_state_yaml=healthy_state_yaml,
    )
    # Per-process subdir so concurrent workers don't fight over the same
    # `failed/` folder or race on UUID collisions (unlikely but cheap).
    _WORKER_WORKDIR = Path(workdir) / f"worker_{os.getpid()}"
    _WORKER_WORKDIR.mkdir(parents=True, exist_ok=True)


def _run_one_in_worker(
    sim_id: int,
    params: dict[str, float],
    t_end_days: float,
    dt_days: float,
    timeout_s: float | None,
) -> tuple[int, int, np.ndarray | None, list[str] | None, str | None]:
    """Return (sim_id, status, trajectory_or_None, species_or_None, err_msg_or_None)."""
    assert _WORKER_RUNNER is not None, "_worker_init must be called first"
    assert _WORKER_WORKDIR is not None
    try:
        result: SimResult = _WORKER_RUNNER.run_one(
            params=params,
            t_end_days=t_end_days,
            dt_days=dt_days,
            workdir=_WORKER_WORKDIR,
            timeout_s=timeout_s,
        )
        return sim_id, STATUS_OK, result.trajectory, result.species_names, None
    except (QspSimError, ParamNotFoundError) as e:
        return sim_id, STATUS_FAILED, None, None, str(e)


# --- Public batch runner ----------------------------------------------------


class CppBatchRunner:
    """Run many sims in parallel; emit one MATLAB-schema Parquet."""

    def __init__(
        self,
        binary_path: str | Path,
        template_path: str | Path,
        subtree: str | None = "QSP",
        default_timeout_s: float = 120.0,
        scenario_yaml: str | Path | None = None,
        drug_metadata_yaml: str | Path | None = None,
        healthy_state_yaml: str | Path | None = None,
    ):
        # Validate eagerly so callers fail fast, before we fork workers.
        probe = CppRunner(
            binary_path=binary_path,
            template_path=template_path,
            subtree=subtree,
            default_timeout_s=default_timeout_s,
            scenario_yaml=scenario_yaml,
            drug_metadata_yaml=drug_metadata_yaml,
            healthy_state_yaml=healthy_state_yaml,
        )
        self.binary_path = probe.binary_path
        self.template_path = Path(template_path).resolve()
        self.subtree = subtree
        self.default_timeout_s = default_timeout_s
        self.scenario_yaml = probe.scenario_yaml
        self.drug_metadata_yaml = probe.drug_metadata_yaml
        self.healthy_state_yaml = probe.healthy_state_yaml
        self.parameter_names = probe.parameter_names

    def run(
        self,
        theta_matrix: np.ndarray,
        param_names: Sequence[str],
        t_end_days: float,
        dt_days: float,
        output_path: str | Path,
        scenario: str = "default",
        seed: int = 0,
        workdir: str | Path | None = None,
        max_workers: int | None = None,
        per_sim_timeout_s: float | None = None,
    ) -> BatchResult:
        """Execute a batch and write the Parquet.

        Args:
            theta_matrix: shape (n_sims, n_params). Row i is the parameter
                vector for sim i.
            param_names: length n_params; the priors-CSV column names that
                line up with theta_matrix's columns.
            t_end_days, dt_days: passed through to qsp_sim.
            output_path: Parquet destination. Parent dirs created.
            scenario, seed: metadata embedded in the Parquet filename
                schema used elsewhere in the codebase. Not written into
                the file itself (Parquet layout is scenario-agnostic).
            workdir: scratch dir for per-sim XML + binary outputs.
                Defaults to a sibling of `output_path`.
            max_workers: process-pool size. Default = CPU count.
            per_sim_timeout_s: overrides default_timeout_s for this batch.

        Returns:
            BatchResult with the written path, counts, and schema info.
        """
        n_sims, n_params = theta_matrix.shape
        if len(param_names) != n_params:
            raise ValueError(
                f"theta_matrix has {n_params} columns but {len(param_names)} "
                f"param_names were given"
            )

        unknown = set(param_names) - self.parameter_names
        if unknown:
            raise ParamNotFoundError(
                f"{len(unknown)} priors column(s) not in XML template: " f"{sorted(unknown)[:10]}"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if workdir is None:
            workdir = output_path.parent / f".workdir_{output_path.stem}"
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        # Submit.
        logger.info(
            "Starting batch: %d sims × %d params, %d workers, scenario=%s seed=%d",
            n_sims,
            n_params,
            max_workers or os.cpu_count(),
            scenario,
            seed,
        )
        trajectories: list[np.ndarray | None] = [None] * n_sims
        statuses: list[int] = [STATUS_FAILED] * n_sims
        errors: list[str | None] = [None] * n_sims
        species_names: list[str] | None = None

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(
                str(self.binary_path),
                str(self.template_path),
                self.subtree,
                str(workdir),
                self.default_timeout_s,
                str(self.scenario_yaml) if self.scenario_yaml else None,
                str(self.drug_metadata_yaml) if self.drug_metadata_yaml else None,
                str(self.healthy_state_yaml) if self.healthy_state_yaml else None,
            ),
        ) as pool:
            futures = []
            for i in range(n_sims):
                params = {name: float(theta_matrix[i, j]) for j, name in enumerate(param_names)}
                futures.append(
                    pool.submit(
                        _run_one_in_worker,
                        i,
                        params,
                        t_end_days,
                        dt_days,
                        per_sim_timeout_s,
                    )
                )
            for fut in as_completed(futures):
                sim_id, status, traj, sp, err = fut.result()
                statuses[sim_id] = status
                if status == STATUS_OK:
                    trajectories[sim_id] = traj
                    if species_names is None:
                        species_names = sp
                else:
                    errors[sim_id] = err
                    logger.warning("sim %d failed: %s", sim_id, err)

        n_failed = sum(1 for s in statuses if s != STATUS_OK)
        if species_names is None:
            # Every sim failed. We still need a schema to write — raise
            # rather than fabricate one, because an empty-species Parquet
            # would silently break downstream consumers expecting columns
            # by name.
            raise QspSimError(
                f"All {n_sims} sims failed; cannot infer species schema.\n"
                f"First error: {next((e for e in errors if e), '(none)')}"
            )

        n_times = trajectories[next(i for i, t in enumerate(trajectories) if t is not None)].shape[
            0
        ]

        parquet_path = _write_batch_parquet(
            output_path=output_path,
            theta_matrix=theta_matrix,
            param_names=list(param_names),
            statuses=statuses,
            trajectories=trajectories,
            species_names=species_names,
            t_end_days=t_end_days,
            dt_days=dt_days,
            n_times=n_times,
        )

        logger.info(
            "Batch complete: %d/%d succeeded, wrote %s",
            n_sims - n_failed,
            n_sims,
            parquet_path,
        )
        return BatchResult(
            parquet_path=parquet_path,
            n_sims=n_sims,
            n_failed=n_failed,
            species_names=species_names,
            n_times=n_times,
        )


# --- Parquet writer ---------------------------------------------------------


def _write_batch_parquet(
    output_path: Path,
    theta_matrix: np.ndarray,
    param_names: list[str],
    statuses: list[int],
    trajectories: list[np.ndarray | None],
    species_names: list[str],
    t_end_days: float,
    dt_days: float,
    n_times: int,
) -> Path:
    """Build one pyarrow Table matching MATLAB's Parquet schema, write it."""
    n_sims = len(statuses)

    # Time column is the same for every row; reconstruct from dt × i.
    time_axis = (np.arange(n_times) * dt_days).tolist()
    time_lists = [time_axis for _ in range(n_sims)]

    # Failed rows get NaN arrays so downstream can filter on status==0.
    nan_row = np.full(n_times, np.nan, dtype=np.float64)

    columns: dict[str, pa.Array] = {
        "simulation_id": pa.array(np.arange(n_sims, dtype=np.int64)),
        "status": pa.array(np.asarray(statuses, dtype=np.int64)),
        "time": pa.array(time_lists, type=pa.list_(pa.float64())),
    }
    for j, name in enumerate(param_names):
        columns[f"param:{name}"] = pa.array(theta_matrix[:, j].astype(np.float64))
    # Stack trajectories into (n_sims, n_times, n_species). Can't hold all
    # of this as one ndarray if n_sims × n_times × n_species is huge, but
    # for typical sweep shapes (≤10k sims × ≤1800 times × 164 species ≈
    # 2.4 GB) it's fine on a single node. The Parquet writer holds the
    # same memory anyway.
    for k, sp in enumerate(species_names):
        per_sim_series: list[list[float]] = []
        for i, traj in enumerate(trajectories):
            if traj is None:
                per_sim_series.append(nan_row.tolist())
            else:
                per_sim_series.append(traj[:, k].tolist())
        columns[sp] = pa.array(per_sim_series, type=pa.list_(pa.float64()))

    table = pa.Table.from_pydict(columns)
    pq.write_table(table, str(output_path), compression="snappy")
    return output_path


def batch_filename(
    batch_index: int,
    n_sims: int,
    scenario: str,
    seed: int,
    timestamp: datetime | None = None,
) -> str:
    """Produce the filename the existing SimulationPoolManager expects."""
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"batch_{batch_index:03d}_{ts}_{n_sims}sims_seed{seed}.parquet"

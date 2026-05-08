"""Orchestrator for a single scenario's HPC submit → fetch → SimulationBatch.

Replaces the per-scenario inline orchestration that grew inside
sbi_runner. Generic enough that any consumer of qsp-hpc-tools can call
it with their own theta + sample_index allocation policy. Stateless:
no shared session object, no reservation allocator, no pool layout
opinions beyond what the C++ batch worker already enforces.

Pipeline:

1. Hash a content-addressed pool_id from (binary bytes, scenario_yaml).
2. Write the per-scenario samples CSV (sample_index + theta cols).
   Caller can hoist the upload above the loop with
   ``samples_csv_remote=`` (see HPCJobManager.upload_shared_samples_csv);
   when set, this step is byte-only-on-disk for the local pool.
3. Submit the SLURM array (``submit_cpp_jobs(skip_setup=True,
   derive_test_stats=False)``).
4. Wait for the array to drain via ``_wait_for_array_completion`` (logs
   per-poll progress).
5. Run ``concat_trajectory_chunks`` on HPC to produce one combined
   parquet per pool.
6. SCP the combined parquet locally + load it
   (``fetch_combined_trajectory``).
7. Filter theta + sample_index to the surviving sample_indices (sims
   whose qsp_sim run produced trajectory rows). Build a
   :class:`SimulationBatch`.

Setup-once orchestration (binary upload + venv probe) belongs above
this call — typically the caller does ``job_manager.ensure_hpc_venv()``
+ ``job_manager.ensure_cpp_binary()`` once per session and then calls
``run_scenario`` per scenario with ``skip_setup=True`` baked in here.

Top-up support is via the ``sample_index`` argument: callers pass only
the deficit (computed via :func:`existing_sample_indices`), so already-
simulated sample_indices aren't redundantly re-run. The downstream
fetch reads back every sample_index requested (including the ones that
existed before this call) so the returned SimulationBatch has the full
training set.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from qsp_hpc.simulation.fetch_combined import fetch_combined_trajectory
from qsp_hpc.simulation.simulation_batch import SimulationBatch
from qsp_hpc.utils.hash_utils import compute_pool_id_hash

if TYPE_CHECKING:
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

logger = logging.getLogger(__name__)


@dataclass
class ScenarioRunResult:
    """Returned by :func:`run_scenario`.

    Carries the SimulationBatch plus a ``dropped_indices`` list of
    sample_indices that the simulator failed for (no trajectory rows
    in the combined parquet). Callers managing per-sim aux records
    use ``dropped_indices`` (or the survivor sample_index in
    ``batch.sample_index``) to filter their auxiliary records to the
    same row order.
    """

    batch: SimulationBatch
    dropped_indices: list[int]
    pool_id: str


def run_scenario(
    job_manager: "HPCJobManager",
    *,
    scenario_yaml: Union[str, Path],
    binary_path: Union[str, Path],
    theta: np.ndarray,
    sample_index: np.ndarray,
    param_names: Sequence[str],
    species_units: Mapping[str, str],
    traj_columns: Optional[Sequence[str]] = None,
    scenario_name: Optional[str] = None,
    seed: int = 2025,
    t_end_days: float = 180.0,
    min_cadence_hours: float = 4.0,
    drug_metadata_yaml: Optional[Union[str, Path]] = None,
    healthy_state_yaml: Optional[Union[str, Path]] = None,
    samples_csv_remote: Optional[str] = None,
    samples_csv_local: Optional[Union[str, Path]] = None,
    evolve_cache: bool = True,
    fetch_keep_local: bool = False,
) -> ScenarioRunResult:
    """End-to-end orchestrator for one scenario's HPC run.

    Args:
        job_manager: Configured HPCJobManager. Caller must already have
            run ``ensure_hpc_venv()`` + ``ensure_cpp_binary()`` once for
            the session.
        scenario_yaml: Local path to the scenario YAML.
        binary_path: Local path to the ``qsp_sim`` binary. Used only
            for the content hash that determines ``pool_id``; the HPC
            side rebuilds its own binary at the same git ref.
        theta: ``(n, n_params)`` matrix of QSP parameters to submit.
        sample_index: ``(n,)`` int64 array of sample_index values to
            label these sims with. Caller is responsible for choosing
            non-overlapping ranges across scenarios when joint inference
            requires shared theta-index keying.
        param_names: Column names matching ``theta``'s columns. Written
            into the samples CSV header.
        species_units: Per-species canonical unit string. Carried into
            the returned SimulationBatch for downstream observable eval.
        traj_columns: Optional species filter pushed down to the
            trajectory read. ``None`` returns every species.
        scenario_name: Cosmetic name for the SLURM array + parquet
            filenames. Defaults to the YAML stem.
        seed: Random seed forwarded to ``submit_cpp_jobs``.
        t_end_days / min_cadence_hours: Forwarded to qsp_sim.
        drug_metadata_yaml / healthy_state_yaml: Optional companions.
        samples_csv_remote: When set, skip the per-pool samples upload
            and reference this remote path from the worker config.
            Caller is expected to upload once via
            ``job_manager.upload_shared_samples_csv`` and pass the
            returned remote path here.
        samples_csv_local: Optional local path for the samples CSV.
            Defaults to a tempfile that's deleted on return.
        evolve_cache: Forward to ``submit_cpp_jobs``.
        fetch_keep_local: Pass-through to fetch_combined_trajectory.

    Returns:
        :class:`ScenarioRunResult` with:
          * ``batch``: SimulationBatch where ``theta`` / ``sample_index``
            are filtered to the surviving sims (those that produced
            trajectory rows). The full ``traj_df`` is included.
          * ``dropped_indices``: sample_indices that produced no
            trajectory rows.
          * ``pool_id``: content-addressed pool hash.
    """
    scenario_yaml = Path(scenario_yaml).resolve()
    if not scenario_yaml.exists():
        raise FileNotFoundError(f"scenario_yaml not found: {scenario_yaml}")
    binary_path_p = Path(binary_path).resolve()
    if not binary_path_p.exists():
        raise FileNotFoundError(f"binary_path not found: {binary_path_p}")
    if theta.shape[0] != sample_index.shape[0]:
        raise ValueError(
            f"theta has {theta.shape[0]} rows but sample_index has "
            f"{sample_index.shape[0]} entries; they must align row-for-row."
        )
    if theta.shape[1] != len(param_names):
        raise ValueError(
            f"theta has {theta.shape[1]} columns but {len(param_names)} " f"param_names supplied."
        )

    n_simulations = int(theta.shape[0])
    if n_simulations == 0:
        raise ValueError(
            "run_scenario: theta is empty; nothing to submit. "
            "Caller (top-up path) should short-circuit before this call."
        )

    pool_id = compute_pool_id_hash(
        binary_path=binary_path_p,
        scenario_yaml=scenario_yaml,
    )
    scen_name = scenario_name or scenario_yaml.stem
    remote_pool_path = f"{job_manager.config.simulation_pool_path}/{pool_id}"
    logger.info("run_scenario(%s): pool_id=%s n=%d", scen_name, pool_id[:8], n_simulations)

    # 1. Write samples CSV.
    samples_df = pd.DataFrame(theta, columns=list(param_names))
    samples_df.insert(0, "sample_index", sample_index.astype(np.int64))
    owns_local_csv = samples_csv_local is None
    if owns_local_csv:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="qsp_run_scenario_"
        )
        local_csv = Path(tmp.name)
        tmp.close()
    else:
        local_csv = Path(samples_csv_local)
        local_csv.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(local_csv, index=False)

    try:
        # 2. Submit (skip_setup=True, derive_test_stats=False).
        info = job_manager.submit_cpp_jobs(
            samples_csv=str(local_csv),
            num_simulations=n_simulations,
            simulation_pool_id=pool_id,
            t_end_days=t_end_days,
            min_cadence_hours=min_cadence_hours,
            scenario=scen_name,
            seed=seed,
            binary_path=job_manager.config.cpp_binary_path,
            template_path=job_manager.config.cpp_template_path,
            scenario_yaml=str(scenario_yaml),
            drug_metadata_yaml=str(drug_metadata_yaml) if drug_metadata_yaml else None,
            healthy_state_yaml=str(healthy_state_yaml) if healthy_state_yaml else None,
            derive_test_stats=False,
            evolve_cache=evolve_cache,
            skip_setup=True,
            samples_csv_remote=samples_csv_remote,
        )

        # 3. Wait for the array to drain. _wait_for_array_completion
        # logs per-poll progress (running/pending/failed).
        for job_id in info.job_ids:
            job_manager._wait_for_array_completion(job_id)

        # 4. Concat per-chunk parquets on HPC into one combined file.
        # No-ops gracefully if no chunks were produced (e.g. the array
        # was scancel'd before any task wrote output).
        concat_t0 = time.time()
        job_manager.concat_trajectory_chunks(pool_id, kind="training")
        logger.info("run_scenario(%s): concat in %.1fs", scen_name, time.time() - concat_t0)

        # 5. SCP combined locally + load it.
        traj_df = fetch_combined_trajectory(
            job_manager,
            remote_pool_path,
            kind="training",
            traj_columns=list(traj_columns) if traj_columns is not None else None,
            keep_local=fetch_keep_local,
        )
    finally:
        if owns_local_csv:
            local_csv.unlink(missing_ok=True)

    # 6. Survivor handling: which of the requested sample_indices
    # actually produced trajectory rows? (Cancelled tasks, qsp_sim
    # non-zero exits, etc. drop sims silently.)
    requested = set(int(s) for s in sample_index.tolist())
    if len(traj_df):
        present = set(int(s) for s in traj_df["sample_index"].unique() if int(s) in requested)
    else:
        present = set()
    dropped = sorted(requested - present)

    if dropped:
        logger.info(
            "run_scenario(%s): %d of %d sims dropped (no trajectory rows)",
            scen_name,
            len(dropped),
            n_simulations,
        )
        # Filter theta + sample_index to survivors, in ascending
        # sample_index order to match traj_df groupby downstream.
        order = np.argsort(sample_index)
        sample_index_sorted = sample_index[order]
        theta_sorted = theta[order]
        keep_mask = np.array([int(s) in present for s in sample_index_sorted], dtype=bool)
        theta_out = theta_sorted[keep_mask]
        sample_index_out = sample_index_sorted[keep_mask]
    else:
        order = np.argsort(sample_index)
        theta_out = theta[order]
        sample_index_out = sample_index[order]

    batch = SimulationBatch(
        theta=np.asarray(theta_out),
        sample_index=np.asarray(sample_index_out, dtype=np.int64),
        traj_df=traj_df,
        species_units=dict(species_units),
        param_names=list(param_names),
        pool_id=pool_id,
    )
    return ScenarioRunResult(batch=batch, dropped_indices=dropped, pool_id=pool_id)

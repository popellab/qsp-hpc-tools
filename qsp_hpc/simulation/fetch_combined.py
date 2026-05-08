"""Fetch a pool's combined trajectory parquet locally and load it.

Compared to :func:`qsp_hpc.cpp.sshfs_reader.sshfs_read_long_form_chunks`,
which reads the parquet over sshfs (each pyarrow row-group / footer
read pays an SSH round-trip), this helper:

1. SCPs the single ``combined.trajectory.parquet`` to a local temp path
   (one TCP stream, no per-range RTT amplification).
2. Reads it locally with pyarrow column projection + filter pushdown.
3. Cleans up the temp file.

For typical sweep sizes (20-50 MB combined parquet) this runs in
~3-5 s vs ~60-90 s for the sshfs path.

Falls back to the sshfs reader when the combined file is missing
(e.g. concat hasn't run yet, or the layout is the legacy per-chunk one).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd

    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

logger = logging.getLogger(__name__)

_COMBINED_NAME = "combined.trajectory.parquet"


def fetch_combined_trajectory(
    job_manager: "HPCJobManager",
    remote_pool_path: str,
    *,
    kind: str = "",
    sample_indices: Optional[Sequence[int]] = None,
    traj_columns: Optional[Sequence[str]] = None,
    local_cache_dir: Optional[Path] = None,
    keep_local: bool = False,
) -> "pd.DataFrame":
    """SCP ``{remote_pool_path}/{kind}/combined.trajectory.parquet`` to
    a local file, then load + filter it locally.

    Falls back to :func:`qsp_hpc.cpp.sshfs_reader.sshfs_read_long_form_chunks`
    when the combined file is absent (e.g. the orchestrator skipped concat,
    or no chunks were produced).

    Returns a long-form DataFrame with the canonical schema
    ``[sample_index, t_to_diagnosis_days, column, value]``. Empty when
    the pool has no chunks.

    Args:
        job_manager: HPCJobManager whose transport is used for the SCP.
        remote_pool_path: Absolute remote path of the pool dir
            (e.g. ``{simulation_pool_path}/{pool_id}``).
        kind: ``"training"`` or ``"ppc"`` — selects the sub-pool.
        sample_indices: Optional row filter pushed down to parquet.
        traj_columns: Optional species / column filter pushed down.
        local_cache_dir: Optional dir to drop the SCP'd file in. Caller
            is expected to manage retention. ``None`` (default) uses a
            ``mkdtemp`` directory cleaned up on return unless
            ``keep_local`` is ``True``.
        keep_local: If ``True``, keep the SCP'd file at the returned
            path for inspection; otherwise delete it before returning.
    """

    remote_subdir = f"{remote_pool_path}/{kind}" if kind else remote_pool_path
    remote_combined = f"{remote_subdir}/{_COMBINED_NAME}"

    # Probe for existence cheaply via ssh exec; avoids paying the SCP
    # cost when concat hasn't run.
    rc, _ = job_manager.transport.exec(
        f'test -f "{remote_combined}" && echo y || echo n', timeout=15
    )
    # The exec helper folds stderr in; use rc as the source of truth.
    # (Note: most SSHTransport implementations return rc=0 even when
    # ``test`` is false because the inner echo always succeeds. So
    # parse the stdout instead.)
    rc, out = job_manager.transport.exec(
        f'test -f "{remote_combined}" && echo y || echo n', timeout=15
    )
    has_combined = rc == 0 and out.strip().endswith("y")
    if not has_combined:
        logger.info(
            "fetch_combined_trajectory: %s not present, falling back to sshfs reader",
            remote_combined,
        )
        from qsp_hpc.cpp.sshfs_reader import sshfs_read_long_form_chunks

        return sshfs_read_long_form_chunks(
            remote_subdir,
            sample_indices=sample_indices,
            traj_columns=traj_columns,
            sshfs_host=job_manager.config.ssh_host,
        )

    owns_local_dir = local_cache_dir is None
    local_dir = (
        Path(local_cache_dir) if local_cache_dir else Path(tempfile.mkdtemp(prefix="qsp_combined_"))
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    local_combined = local_dir / _COMBINED_NAME

    try:
        scp_t0 = time.time()
        job_manager.transport.download(remote_combined, str(local_dir))
        scp_dt = time.time() - scp_t0
        size_mb = local_combined.stat().st_size / 1e6 if local_combined.exists() else 0.0
        logger.info(
            "fetch_combined_trajectory: scp'd %.1f MB in %.1fs (%.1f MB/s)",
            size_mb,
            scp_dt,
            size_mb / max(scp_dt, 1e-3),
        )

        # Build pushdown filter for sample_indices / traj_columns.
        filters: list[tuple] = []
        if sample_indices is not None:
            si_list = [int(x) for x in sample_indices]
            if not si_list:
                return _empty_traj_df()
            filters.append(("sample_index", "in", si_list))
        if traj_columns is not None:
            tc_list = list(traj_columns)
            if not tc_list:
                return _empty_traj_df()
            # Writer-side column name is "species"; canonical post-rename
            # is "column". Filter against the on-disk name.
            filters.append(("species", "in", tc_list))

        read_t0 = time.time()
        dataset = pq.ParquetDataset(str(local_combined), filters=filters or None)
        table = dataset.read(columns=["sample_index", "time", "species", "value"])
        if pa.types.is_dictionary(table.schema.field("species").type):
            table = table.set_column(
                table.schema.get_field_index("species"),
                "species",
                table.column("species").cast(pa.string()),
            )
        df = table.to_pandas()
        read_dt = time.time() - read_t0
        logger.info(
            "fetch_combined_trajectory: parsed %d rows in %.1fs",
            len(df),
            read_dt,
        )
        return df.rename(columns={"time": "t_to_diagnosis_days", "species": "column"})
    finally:
        if owns_local_dir and not keep_local:
            shutil.rmtree(local_dir, ignore_errors=True)
        elif keep_local:
            logger.info("fetch_combined_trajectory: kept local copy at %s", local_combined)


def _empty_traj_df() -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame(
        {
            "sample_index": pd.Series(dtype="int64"),
            "t_to_diagnosis_days": pd.Series(dtype="float64"),
            "column": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )

"""Enumerate sample_indices already present in a remote pool.

Used by the run_scenario orchestrator to support top-up: when re-running
the same (priors, seed, n_total, scenario_yaml) tuple, sample_indices
that were already simulated stay on disk in the pool's combined parquet
or per-chunk parquets. This helper reads just the ``sample_index``
column to figure out which sample_indices are already covered, so the
caller submits only the deficit.

Cheap when the combined file exists (one column projection over scp'd
parquet). Falls back to a light sshfs read against per-chunk parquets
when the combined hasn't been produced yet.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager

logger = logging.getLogger(__name__)

_COMBINED_NAME = "combined.trajectory.parquet"


def existing_sample_indices(
    job_manager: "HPCJobManager",
    remote_pool_path: str,
    *,
    kind: str = "training",
) -> set[int]:
    """Return the set of ``sample_index`` values already present in
    ``{remote_pool_path}/{kind}/``.

    Resolution order:

    1. If ``combined.trajectory.parquet`` exists, scp it locally and
       read just the ``sample_index`` column (cheap; ~MB-scale even for
       5k×21day pools).
    2. Otherwise return an empty set. (We could fall back to the sshfs
       reader walking ``batch_*/`` but that's expensive and the
       orchestrator typically runs concat before calling here.)

    Returns an empty set when the pool dir doesn't exist, the combined
    file isn't present, or the parquet has no rows. Callers treat that
    as "submit everything."
    """
    remote_combined = f"{remote_pool_path}/{kind}/{_COMBINED_NAME}"

    rc, out = job_manager.transport.exec(
        f'test -f "{remote_combined}" && echo y || echo n', timeout=15
    )
    if rc != 0 or not out.strip().endswith("y"):
        return set()

    tmp_dir = Path(tempfile.mkdtemp(prefix="qsp_scan_existing_"))
    try:
        job_manager.transport.download(remote_combined, str(tmp_dir))
        local = tmp_dir / _COMBINED_NAME
        if not local.exists():
            logger.warning("existing_sample_indices: scp reported success but %s missing", local)
            return set()
        # Single-column read so we don't pay the cost of materializing
        # the trajectory + species + value columns.
        table = pq.read_table(str(local), columns=["sample_index"])
        if table.num_rows == 0:
            return set()
        return {int(s) for s in table.column("sample_index").to_pylist()}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

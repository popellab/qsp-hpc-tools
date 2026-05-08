"""Read long-form trajectory chunks from a remote pool over ``sshfs``.

This is the Layer 4 read path from
``notes/architecture/local_observable_eval_plan.md`` (pdac-build). Per
the 2026-05-06 spike documented under "Read-path performance" /
``D5. sshfs``, ``sshfs.SSHFileSystem`` (asyncssh-backed) is the
chosen glue: it pipelines small parquet byte-range reads where the
naive paramiko shape stalls. Column projection over sshfs hits ~3.4 s
on a 90 MB trajectory parquet and extrapolates to ~40 s for
20k × 3 scenarios — well under the simulation phase.

Used by :meth:`HPCSession.run_scenario` to pull the long-form chunks
written by SLURM into a ``SimulationBatch.traj_df``. Failed sims emit
zero trajectory rows on the writer side, so a status filter is not
needed at read time; the params sidecar carries the per-sim status.

The helper accepts an injected ``filesystem=`` for tests so the unit
tests can substitute :class:`fsspec.implementations.local.LocalFileSystem`
without spinning up a real SSH server. The plan's D5 mount lifecycle —
"open inside the read helper, unmount immediately after the parquet
read returns" — applies only when the helper opens its own
``sshfs.SSHFileSystem``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


_TRAJECTORY_SUFFIX = ".trajectory.parquet"


def _list_trajectory_chunks(filesystem: Any, remote_pool_path: str) -> list[str]:
    """Return the absolute paths of all ``*.trajectory.parquet`` chunks
    directly under ``remote_pool_path`` (non-recursive).

    Uses :meth:`fsspec.AbstractFileSystem.ls`; both ``sshfs.SSHFileSystem``
    and ``fsspec.implementations.local.LocalFileSystem`` honor that API.
    """
    entries = filesystem.ls(remote_pool_path, detail=False)
    return sorted(p for p in entries if str(p).endswith(_TRAJECTORY_SUFFIX))


def _open_default_sshfs(host: str, **sshfs_kwargs: Any) -> Any:
    """Lazy-open an ``sshfs.SSHFileSystem`` against ``host``.

    Lazy-imported so the helper itself is testable on machines that
    haven't installed ``sshfs`` (the test fixture passes its own
    ``filesystem=`` and never trips this path).
    """
    try:
        import sshfs  # type: ignore
    except ImportError as exc:  # pragma: no cover — install-time guard
        raise RuntimeError(
            "sshfs_read_long_form_chunks: opening a default sshfs "
            "filesystem requires the `sshfs` package. Install it with "
            "`uv pip install sshfs`, or pass an existing fsspec "
            "filesystem via `filesystem=`."
        ) from exc
    return sshfs.SSHFileSystem(host, **sshfs_kwargs)


def sshfs_read_long_form_chunks(
    remote_pool_path: str,
    *,
    sample_indices: Optional[Sequence[int]] = None,
    traj_columns: Optional[Sequence[str]] = None,
    filesystem: Any = None,
    sshfs_host: Optional[str] = None,
    sshfs_kwargs: Optional[dict[str, Any]] = None,
) -> "pd.DataFrame":
    """Read long-form trajectory chunks under a remote pool sub-directory.

    Args:
        remote_pool_path: Absolute remote path of the pool sub-directory
            holding the chunks (e.g.
            ``/.../qsp_simulations/{pool_id}/training``). The helper reads
            every ``chunk_*.trajectory.parquet`` directly inside it; nested
            directories are not walked.
        sample_indices: Optional row filter — only rows whose
            ``sample_index`` appears in this list are returned. Pushed
            down to parquet so unrelated reservation ranges in the same
            pool are skipped via row-group statistics. ``None`` (the
            default) returns every row.
        traj_columns: Optional species filter — only rows whose
            ``species`` appears in this list are returned. The on-disk
            ``species`` column is dictionary-encoded and combines real
            species with compartments + assignment rules; the filter
            matches against the decoded string value. Pushed down to
            parquet. ``None`` returns every species.
        filesystem: Optional fsspec-shaped filesystem. When provided,
            the helper does not open its own connection — useful for
            tests with :class:`fsspec.implementations.local.LocalFileSystem`,
            or for callers that want to share an open ``sshfs.SSHFileSystem``
            across multiple reads.
        sshfs_host: Hostname for the default sshfs connection. Required
            when ``filesystem`` is ``None``.
        sshfs_kwargs: Forwarded to :class:`sshfs.SSHFileSystem` when
            ``filesystem`` is ``None``. ``None`` defaults to ``{}``.

    Returns:
        A long-form ``pd.DataFrame`` with columns
        ``[sample_index, time, species, value]``. ``species`` is a plain
        string column (dictionary decoded). The frame is the
        concatenation of every chunk under ``remote_pool_path``,
        post-filter; row order is chunk-stable but not otherwise
        sorted.

    Raises:
        FileNotFoundError: If ``remote_pool_path`` doesn't exist.
        ValueError: If ``filesystem`` is ``None`` and ``sshfs_host`` is
            missing.
    """
    import pandas as pd

    owns_filesystem = filesystem is None
    if owns_filesystem:
        if sshfs_host is None:
            raise ValueError(
                "sshfs_read_long_form_chunks: pass either filesystem= or "
                "sshfs_host= (the helper needs at least one route to the "
                "remote pool)."
            )
        filesystem = _open_default_sshfs(sshfs_host, **(sshfs_kwargs or {}))

    try:
        if not filesystem.exists(remote_pool_path):
            raise FileNotFoundError(
                f"sshfs_read_long_form_chunks: remote pool path does not "
                f"exist on the supplied filesystem: {remote_pool_path}"
            )
        chunk_paths = _list_trajectory_chunks(filesystem, remote_pool_path)
        logger.debug(
            "sshfs_read_long_form_chunks: %d chunk(s) under %s",
            len(chunk_paths),
            remote_pool_path,
        )

        if not chunk_paths:
            return pd.DataFrame(
                {
                    "sample_index": pd.Series(dtype="int64"),
                    "time": pd.Series(dtype="float64"),
                    "species": pd.Series(dtype="object"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        # Build a pyarrow filter expression. Predicate pushdown skips row
        # groups whose statistics rule out a match — this is what makes
        # other reservation ranges in the same pool free to read past.
        filters: list[tuple] = []
        if sample_indices is not None:
            sample_indices_list = [int(x) for x in sample_indices]
            if not sample_indices_list:
                # Empty filter intentionally returns no rows; pyarrow's
                # ``in`` filter against an empty list raises, so short
                # circuit.
                return pd.DataFrame(
                    {
                        "sample_index": pd.Series(dtype="int64"),
                        "time": pd.Series(dtype="float64"),
                        "species": pd.Series(dtype="object"),
                        "value": pd.Series(dtype="float64"),
                    }
                )
            filters.append(("sample_index", "in", sample_indices_list))
        if traj_columns is not None:
            traj_cols_list = list(traj_columns)
            if not traj_cols_list:
                return pd.DataFrame(
                    {
                        "sample_index": pd.Series(dtype="int64"),
                        "time": pd.Series(dtype="float64"),
                        "species": pd.Series(dtype="object"),
                        "value": pd.Series(dtype="float64"),
                    }
                )
            filters.append(("species", "in", traj_cols_list))

        # ``ParquetDataset`` accepts a list of file paths or a directory.
        # We pass the explicit chunk list so it doesn't accidentally walk
        # into nested sub-pools (e.g. ``training/`` next to ``ppc/``).
        # ``filters`` lives on the constructor; ``.read()`` only takes
        # ``columns``.
        dataset = pq.ParquetDataset(
            chunk_paths,
            filesystem=filesystem,
            filters=filters or None,
        )
        table = dataset.read(columns=["sample_index", "time", "species", "value"])

        # Decode the dictionary-encoded species column to plain strings
        # so consumers can filter / pivot without pulling in the
        # dictionary index.
        if pa.types.is_dictionary(table.schema.field("species").type):
            table = table.set_column(
                table.schema.get_field_index("species"),
                "species",
                table.column("species").cast(pa.string()),
            )
        return table.to_pandas()
    finally:
        if owns_filesystem:
            close = getattr(filesystem, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover — best-effort unmount
                    logger.debug(
                        "sshfs_read_long_form_chunks: filesystem close raised",
                        exc_info=True,
                    )

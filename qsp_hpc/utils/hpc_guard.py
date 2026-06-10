"""Kill switch for accidental HPC writes from a dev machine.

Set ``QSP_HPC_NO_SUBMIT`` to a truthy value (``1``/``true``/``yes``/``on``)
to make any cluster-*mutating* operation raise instead of executing:

- codebase sync (``rsync -avz --delete`` onto ``remote_project_path``), and
- SLURM job submission (``sbatch``).

Read-only operations (``squeue``, ``ls``, status checks) are unaffected, so
inspection and debugging still work with the switch on.

Rationale: a machine with live ``~/.config/qsp-hpc/credentials.yaml`` will
reach the real cluster the moment a simulator cache-misses (``sim()`` /
``run_hpc`` / ``submit_*``). Exporting ``QSP_HPC_NO_SUBMIT=1`` on such a
machine makes that physically impossible, complementing the existing
same-host ``--delete`` guard in ``sync_codebase``.
"""

from __future__ import annotations

import os

#: Environment variable that, when truthy, blocks all remote-mutating HPC ops.
NO_SUBMIT_ENV = "QSP_HPC_NO_SUBMIT"

_TRUTHY = {"1", "true", "yes", "on"}


class HPCSubmitBlockedError(RuntimeError):
    """Raised when a remote-mutating HPC op runs while the kill switch is set."""


def no_submit_enabled() -> bool:
    """Return True if ``QSP_HPC_NO_SUBMIT`` is set to a truthy value."""
    return os.environ.get(NO_SUBMIT_ENV, "").strip().lower() in _TRUTHY


def ensure_remote_writes_allowed(operation: str) -> None:
    """Raise :class:`HPCSubmitBlockedError` if the kill switch is set.

    Call at the top of any cluster-mutating action (codebase sync, sbatch
    submission) so the block happens before any side effect (upload, script
    generation, job creation).

    Args:
        operation: Short name of the blocked operation, surfaced in the error
            (e.g. ``"sync_codebase"``, ``"submit_job"``).
    """
    if no_submit_enabled():
        raise HPCSubmitBlockedError(
            f"{operation!r} blocked: {NO_SUBMIT_ENV} is set. "
            f"Unset it to allow HPC submission/sync, or run this from a "
            f"machine intended to reach the cluster. "
            f"(Read-only checks are unaffected.)"
        )

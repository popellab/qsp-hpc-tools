"""A failed sim says which kind of failure it was."""

from __future__ import annotations

import pytest

from qsp_hpc.cpp.batch_runner import (
    INIT_REJECT_EXIT_CODE,
    STATUS_FAILED,
    STATUS_INIT_REJECTED,
    STATUS_INIT_REJECTED_ALL,
    STATUS_INIT_REJECTED_DEGENERATE,
    STATUS_INIT_REJECTED_FAST,
    STATUS_INIT_REJECTED_SLOW,
    STATUS_LABELS,
    STATUS_OK,
    classify_failure,
)
from qsp_hpc.cpp.runner import QspSimError


def _rejected(reason: str) -> QspSimError:
    return QspSimError(
        "qsp_sim exited 2",
        returncode=INIT_REJECT_EXIT_CODE,
        stderr=f"[evolve] rejected: {reason}\n",
    )


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("EVOLVE_TOO_FAST: target reached too fast (12 d < 30 d min)", STATUS_INIT_REJECTED_FAST),
        (
            "EVOLVE_TOO_SLOW: target diameter 2 cm not reached by 7300 d (max=0.4 cm)",
            STATUS_INIT_REJECTED_SLOW,
        ),
        ("EVOLVE_DEGENERATE: initial diameter already >= target", STATUS_INIT_REJECTED_DEGENERATE),
    ],
)
def test_tokens_classify(reason, expected):
    assert classify_failure(_rejected(reason)) == expected


def test_rejection_without_a_token_is_still_a_rejection():
    """A model that emits no token is unclassified, not a solver failure.

    Guessing STATUS_FAILED here would move a rejected patient into the bucket
    that means "we could not integrate it", which is the exact confusion the
    codes exist to remove.
    """
    assert classify_failure(_rejected("some model's own wording")) == STATUS_INIT_REJECTED


def test_crash_is_a_solver_failure():
    """Death by signal is ours, not the patient's."""
    exc = QspSimError("qsp_sim exited -6", returncode=-6, stderr="terminate called")
    assert classify_failure(exc) == STATUS_FAILED


def test_other_nonzero_exit_is_a_solver_failure():
    assert classify_failure(QspSimError("boom", returncode=1, stderr="")) == STATUS_FAILED


def test_exception_with_no_exit_code_is_a_solver_failure():
    """Timeouts and unreadable output raise without a returncode."""
    assert classify_failure(QspSimError("timed out")) == STATUS_FAILED
    assert classify_failure(RuntimeError("unrelated")) == STATUS_FAILED


def test_token_in_stderr_but_wrong_exit_code_does_not_classify():
    """Only the driver's rejection exit means a rejection.

    A crash whose stderr happens to quote a token is a crash.
    """
    exc = QspSimError("boom", returncode=-6, stderr="EVOLVE_TOO_SLOW: ...")
    assert classify_failure(exc) == STATUS_FAILED


def test_codes_are_distinct_and_nonzero():
    """No rejection code may collide with OK, or a filter would train on one."""
    codes = set(STATUS_INIT_REJECTED_ALL)
    assert len(codes) == 4
    assert STATUS_OK not in codes
    assert STATUS_FAILED not in codes


def test_every_code_has_a_label():
    assert STATUS_INIT_REJECTED_ALL <= set(STATUS_LABELS)
    assert len(set(STATUS_LABELS.values())) == len(STATUS_LABELS)

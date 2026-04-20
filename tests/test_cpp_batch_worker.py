"""Tests for qsp_hpc.batch.cpp_batch_worker helpers."""

from __future__ import annotations

from qsp_hpc.batch.cpp_batch_worker import _resolve_max_workers


def test_resolve_max_workers_config_wins(monkeypatch):
    """Explicit config value overrides SLURM env and default."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert _resolve_max_workers(4) == 4


def test_resolve_max_workers_falls_back_to_slurm_env(monkeypatch):
    """SLURM_CPUS_PER_TASK used when config is None — the HPC path.

    Regression: Python 3.11's ProcessPoolExecutor(max_workers=None) uses
    os.cpu_count() which returns the NODE's physical cores (64 on
    Rockfish), spawning that many workers on the 1 CPU SLURM actually
    allocated. Oversubscription → context-switch thrashing → ~2-3×
    slower sims than necessary. Reading SLURM_CPUS_PER_TASK directly
    gives the cgroup-correct count.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert _resolve_max_workers(None) == 4


def test_resolve_max_workers_none_without_slurm(monkeypatch):
    """No SLURM env + no config → None, letting downstream use os.cpu_count()
    (correct for local runs where the process owns the whole machine)."""
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert _resolve_max_workers(None) is None


def test_resolve_max_workers_config_zero_passes_through(monkeypatch):
    """Explicit 0 is a caller choice (pathological, but theirs to make)
    — don't silently promote it via the SLURM fallback."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert _resolve_max_workers(0) == 0

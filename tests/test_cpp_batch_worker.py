"""Tests for qsp_hpc.batch.cpp_batch_worker helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qsp_hpc.batch.cpp_batch_worker import _resolve_max_workers, run_fused_chunk


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


# --- Fused multi-scenario chunk (#90 Phase 2) ------------------------------

_MINI_TEMPLATE = b"""<Param>
  <QSP>
    <A>1.0</A>
    <B>2.0</B>
  </QSP>
</Param>
"""


def _make_fake_fused_binary(tmp_path: Path) -> Path:
    """Fake qsp_sim: --dump-state writes an opaque blob; a scenario run
    writes a v3 trajectory whose spA row-0 value is the --scenario YAML
    body (so each scenario's parquet is identifiable)."""
    script = tmp_path / "fake_fused_qsp_sim.sh"
    script.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        set -e
        DUMP_STATE=""
        SCENARIO=""
        while [ $# -gt 0 ]; do
          case "$1" in
            --binary-out) BIN_OUT="$2"; shift 2 ;;
            --species-out) SP_OUT="$2"; shift 2 ;;
            --compartments-out) COMP_OUT="$2"; shift 2 ;;
            --rules-out) RULES_OUT="$2"; shift 2 ;;
            --t-end-days) TEND="$2"; shift 2 ;;
            --min-cadence-hours) DT="$2"; shift 2 ;;
            --dump-state) DUMP_STATE="$2"; shift 2 ;;
            --scenario) SCENARIO="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        if [ -n "$DUMP_STATE" ]; then
          printf 'FAKE_EVOLVE_STATE' > "$DUMP_STATE"
          exit 0
        fi
        SCEN_VAL=0.0
        if [ -n "$SCENARIO" ]; then SCEN_VAL=$(cat "$SCENARIO"); fi
        python3 - <<PY
        import struct
        header = struct.pack('<IIQQQQdddQQ', 0x51535042, 3, 2, 3, 0, 0, float("$DT"), float("$TEND"), 0.0, 0, 0)
        body = struct.pack('<8d', 0.0, float("$SCEN_VAL"), 2.0, 3.0, 0.1, 4.0, 5.0, 6.0)
        open("$BIN_OUT", 'wb').write(header + body)
        open("$SP_OUT", 'w').write("spA\\nspB\\nspC\\n")
        open("$COMP_OUT", 'w').write('')
        open("$RULES_OUT", 'w').write('')
        PY
    """))
    script.chmod(0o755)
    return script


def _fused_config(tmp_path: Path) -> dict:
    """Build a fused cpp_job_config for a 3-theta, 2-scenario chunk."""
    binary = _make_fake_fused_binary(tmp_path)
    template = tmp_path / "template.xml"
    template.write_bytes(_MINI_TEMPLATE)
    healthy = tmp_path / "healthy.yaml"
    healthy.write_text("# dummy\n")

    csv = tmp_path / "params.csv"
    pd.DataFrame({"sample_index": [0, 1, 2], "A": [1.1, 1.2, 1.3], "B": [2.1, 2.2, 2.3]}).to_csv(
        csv, index=False
    )

    scen_a = tmp_path / "scen_a.yaml"
    scen_a.write_text("11.0")
    drug_a = tmp_path / "drug_a.yaml"
    drug_a.write_text("# drug\n")
    scen_b = tmp_path / "scen_b.yaml"
    scen_b.write_text("22.0")
    drug_b = tmp_path / "drug_b.yaml"
    drug_b.write_text("# drug\n")

    pool_base = tmp_path / "pools"
    return {
        "binary_path": str(binary),
        "template_path": str(template),
        "subtree": "QSP",
        "param_csv": str(csv),
        "n_simulations": 3,
        "samples_start_offset": 0,
        "seed": 0,
        "jobs_per_chunk": 10,
        "t_end_days": 0.2,
        "min_cadence_hours": 0.1,
        "simulation_pool_path": str(pool_base),
        "max_workers": 2,
        "per_sim_timeout_s": 60,
        "healthy_state_yaml": str(healthy),
        "evolve_cache_root": None,
        "discard_trajectories": False,
        "model_structure_file": None,
        "aux_samples_csv": None,
        "auxiliary_units": {},
        "scenarios": [
            {
                "name": "drugA",
                "simulation_pool_id": "pool_drugA",
                "scenario_yaml": str(scen_a),
                "drug_metadata_yaml": str(drug_a),
                "test_stats_csv": None,
                "test_stats_hash": None,
                "batch_subdir": "batch_T_drugA_seed0",
                "samples_start_offset": 0,
            },
            {
                "name": "drugB",
                "simulation_pool_id": "pool_drugB",
                "scenario_yaml": str(scen_b),
                "drug_metadata_yaml": str(drug_b),
                "test_stats_csv": None,
                "test_stats_hash": None,
                "batch_subdir": "batch_T_drugB_seed0",
                "samples_start_offset": 0,
            },
        ],
    }


def test_run_fused_chunk_writes_one_parquet_per_scenario(tmp_path):
    """One fused task → one chunk parquet per scenario, each in its own
    pool dir, each carrying that scenario's trajectory."""
    config = _fused_config(tmp_path)
    run_fused_chunk(config, array_idx=0)

    pool_base = Path(config["simulation_pool_path"])
    a_parquet = pool_base / "pool_drugA" / "batch_T_drugA_seed0" / "chunk_000.parquet"
    b_parquet = pool_base / "pool_drugB" / "batch_T_drugB_seed0" / "chunk_000.parquet"
    assert a_parquet.exists() and b_parquet.exists()

    a_table = pq.read_table(a_parquet)
    b_table = pq.read_table(b_parquet)
    assert a_table.num_rows == 3 and b_table.num_rows == 3
    np.testing.assert_array_equal(a_table.column("sample_index").to_numpy(), [0, 1, 2])
    # Each scenario's parquet carries its own scenario marker.
    assert a_table.column("spA").to_pylist()[0] == [11.0, 4.0]
    assert b_table.column("spA").to_pylist()[0] == [22.0, 4.0]
    # Per-scenario pool manifest written.
    assert (pool_base / "pool_drugA" / "pool_manifest.json").exists()
    assert (pool_base / "pool_drugB" / "pool_manifest.json").exists()


def test_run_fused_chunk_per_scenario_deficit(tmp_path):
    """A scenario with a non-zero samples_start_offset only sims its
    deficit tail; the fused offset is the min across scenarios."""
    config = _fused_config(tmp_path)
    # drugA fully fresh from 0; drugB already has [0, 2) → deficit [2, 3).
    config["scenarios"][1]["samples_start_offset"] = 2
    run_fused_chunk(config, array_idx=0)

    pool_base = Path(config["simulation_pool_path"])
    a_table = pq.read_table(pool_base / "pool_drugA" / "batch_T_drugA_seed0" / "chunk_000.parquet")
    b_table = pq.read_table(pool_base / "pool_drugB" / "batch_T_drugB_seed0" / "chunk_000.parquet")
    assert a_table.num_rows == 3
    # drugB only ran sample_index >= 2.
    assert b_table.num_rows == 1
    np.testing.assert_array_equal(b_table.column("sample_index").to_numpy(), [2])


def test_run_fused_chunk_discard_trajectories_noop_without_test_stats(tmp_path):
    """discard_trajectories only fires after an inline derive — with no
    test_stats_csv the parquet is the deliverable and is kept."""
    config = _fused_config(tmp_path)
    config["discard_trajectories"] = True
    run_fused_chunk(config, array_idx=0)
    pool_base = Path(config["simulation_pool_path"])
    # No test_stats_csv → no derive → parquet survives.
    assert (pool_base / "pool_drugA" / "batch_T_drugA_seed0" / "chunk_000.parquet").exists()

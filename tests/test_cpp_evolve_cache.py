"""Tests for :mod:`qsp_hpc.cpp.evolve_cache` — the persistent evolve cache.

Pure-Python tests build synthetic QSTH blobs and exercise the namespace
keying, shard write/read, manifest + compaction, and miss handling
without a qsp_sim binary. Integration tests run the real binary when
available, proving a blob written through the cache by one batch is
reused (``--initial-state``) by a later batch under a different scenario.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from qsp_hpc.cpp.evolve_cache import (
    EvolveCache,
    EvolveCacheWriter,
    compute_namespace,
)
from qsp_hpc.cpp.qsth import (
    _QSTH_HASH_LEN,
    _QSTH_HEADER_SIZE,
    _QSTH_MAGIC,
    _QSTH_VERSION,
    theta_hash_for_xml,
    wire_hash,
)

# --- synthetic QSTH blob helpers ------------------------------------------


def _pack_qsth(params_hash: str, *, trailing: bytes = b"\x00" * 96) -> bytes:
    """A synthetic QSTH blob: 128-byte header + opaque payload."""
    hash_bytes = params_hash.encode("ascii")[:_QSTH_HASH_LEN].ljust(_QSTH_HASH_LEN, b"\x00")
    fixed = struct.pack(
        f"<IIQdd{_QSTH_HASH_LEN}s",
        _QSTH_MAGIC,
        _QSTH_VERSION,
        164,
        857.0,
        3.2,
        hash_bytes,
    )
    return fixed + b"\x00" * (_QSTH_HEADER_SIZE - len(fixed)) + trailing


def _theta_hash(seed: str) -> str:
    """A deterministic 64-char sha256 hex theta_hash for tests."""
    return theta_hash_for_xml(f"<xml seed={seed!r}/>".encode())


def _blob_for(theta_hash: str, *, trailing: bytes = b"\x00" * 96) -> bytes:
    """A QSTH blob whose header params_hash matches wire_hash(theta_hash)."""
    return _pack_qsth(wire_hash(theta_hash), trailing=trailing)


@pytest.fixture
def binary_file(tmp_path: Path) -> Path:
    p = tmp_path / "qsp_sim"
    p.write_bytes(b"fake qsp_sim binary bytes")
    return p


@pytest.fixture
def healthy_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "healthy_state.yaml"
    p.write_text("diagnosis_target: 3.2\n")
    return p


# --- namespace keying ------------------------------------------------------


def test_namespace_is_deterministic(binary_file: Path, healthy_yaml: Path):
    a = compute_namespace(binary_file, healthy_yaml)
    b = compute_namespace(binary_file, healthy_yaml)
    assert a == b
    assert len(a) == 16


def test_namespace_changes_with_binary(binary_file: Path, healthy_yaml: Path, tmp_path: Path):
    other = tmp_path / "qsp_sim_v2"
    other.write_bytes(b"a rebuilt qsp_sim binary")
    assert compute_namespace(binary_file, healthy_yaml) != compute_namespace(other, healthy_yaml)


def test_namespace_changes_with_healthy_state(
    binary_file: Path, healthy_yaml: Path, tmp_path: Path
):
    other = tmp_path / "healthy_v2.yaml"
    other.write_text("diagnosis_target: 4.0\n")
    assert compute_namespace(binary_file, healthy_yaml) != compute_namespace(binary_file, other)


def test_namespace_extra_disambiguates(binary_file: Path, healthy_yaml: Path):
    base = compute_namespace(binary_file, healthy_yaml)
    with_extra = compute_namespace(binary_file, healthy_yaml, extra=b"evolve-config-v2")
    assert base != with_extra


def test_for_run_uses_compute_namespace(binary_file: Path, healthy_yaml: Path, tmp_path: Path):
    cache = EvolveCache.for_run(
        tmp_path / "evolve_cache", binary_path=binary_file, healthy_state_yaml=healthy_yaml
    )
    assert cache.namespace == compute_namespace(binary_file, healthy_yaml)
    assert cache.namespace_dir == (tmp_path / "evolve_cache" / cache.namespace).resolve()


# --- shard write / read round trip ----------------------------------------


def test_writer_flush_then_get(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("a")
    blob = _blob_for(th)
    writer = cache.writer()
    writer.add(th, blob)
    shard = writer.flush()
    assert shard is not None and shard.exists()
    assert shard.parent == cache.namespace_dir

    # A fresh cache over the same namespace finds the written state.
    fresh = EvolveCache(tmp_path / "cache", "ns0")
    assert fresh.get(th) == blob
    assert th in fresh
    assert len(fresh) == 1
    assert fresh.stats == {"hits": 1, "misses": 0}


def test_empty_writer_flushes_nothing(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    writer = cache.writer()
    assert writer.flush() is None
    assert list(cache.namespace_dir.glob("shard_*.qsep")) == []


def test_writer_context_manager_flushes_on_clean_exit(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("ctx")
    with cache.writer() as writer:
        writer.add(th, _blob_for(th))
    assert EvolveCache(tmp_path / "cache", "ns0").get(th) is not None


def test_writer_context_manager_skips_flush_on_exception(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("boom")
    with pytest.raises(RuntimeError, match="deliberate"):
        with cache.writer() as writer:
            writer.add(th, _blob_for(th))
            raise RuntimeError("deliberate")
    assert list(cache.namespace_dir.glob("shard_*.qsep")) == []


def test_get_miss_returns_none(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    assert cache.get(_theta_hash("absent")) is None
    assert cache.stats == {"hits": 0, "misses": 1}


def test_multiple_shards_layer_into_one_index(tmp_path: Path):
    """Two writers (two tasks) each flush their own shard; one cache reads
    both — the NFS-safe append-only-shard property."""
    cache = EvolveCache(tmp_path / "cache", "ns0")
    a, b = _theta_hash("A"), _theta_hash("B")
    w1 = cache.writer()
    w1.add(a, _blob_for(a))
    w1.flush()
    w2 = cache.writer()
    w2.add(b, _blob_for(b))
    w2.flush()
    assert len(list(cache.namespace_dir.glob("shard_*.qsep"))) == 2

    fresh = EvolveCache(tmp_path / "cache", "ns0")
    assert fresh.get(a) == _blob_for(a)
    assert fresh.get(b) == _blob_for(b)
    assert len(fresh) == 2


def test_namespaces_are_isolated(tmp_path: Path):
    th = _theta_hash("shared-theta")
    w = EvolveCache(tmp_path / "cache", "ns_A").writer()
    w.add(th, _blob_for(th))
    w.flush()
    # Same theta_hash, different namespace → not visible.
    assert EvolveCache(tmp_path / "cache", "ns_B").get(th) is None
    assert EvolveCache(tmp_path / "cache", "ns_A").get(th) is not None


def test_materialize_writes_blob_to_disk(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("m")
    w = cache.writer()
    w.add(th, _blob_for(th))
    w.flush()

    fresh = EvolveCache(tmp_path / "cache", "ns0")
    out = fresh.materialize(th, tmp_path / "wd")
    assert out is not None and out.exists()
    assert out.read_bytes() == _blob_for(th)
    assert fresh.materialize(_theta_hash("absent"), tmp_path / "wd") is None


# --- corrupt-blob handling -------------------------------------------------


def test_get_treats_wrong_params_hash_as_miss(tmp_path: Path):
    """A blob filed under theta A but carrying theta B's params_hash is a
    miss — the caller re-evolves rather than feeding qsp_sim a wrong
    state. (Forced by writing the shard with validation off.)"""
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("real")
    mismatched = _blob_for(_theta_hash("other"))  # header hash is for a different theta
    writer = cache.writer()
    writer.add(th, mismatched, validate=False)
    writer.flush()

    fresh = EvolveCache(tmp_path / "cache", "ns0")
    assert fresh.get(th) is None
    assert fresh.stats["misses"] == 1


def test_get_treats_bad_header_as_miss(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("bad")
    writer = cache.writer()
    writer.add(th, b"not a qsth blob, just opaque junk bytes" * 4, validate=False)
    writer.flush()

    fresh = EvolveCache(tmp_path / "cache", "ns0")
    assert fresh.get(th) is None


def test_load_skips_unreadable_shard(tmp_path: Path, caplog):
    """A half-written shard (crashed task) or a foreign file is skipped,
    never aborts the load."""
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("good")
    w = cache.writer()
    w.add(th, _blob_for(th))
    w.flush()
    # A garbage file matching the shard glob.
    (cache.namespace_dir / "shard_garbage.qsep").write_bytes(b"\x00" * 4)

    fresh = EvolveCache(tmp_path / "cache", "ns0")
    assert fresh.get(th) is not None  # the good shard still loads
    assert len(fresh) == 1


# --- compaction / manifest -------------------------------------------------


def test_compact_writes_manifest(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    hashes = [_theta_hash(f"t{i}") for i in range(5)]
    for th in hashes:
        w = cache.writer()
        w.add(th, _blob_for(th))
        w.flush()

    manifest = cache.compact()
    assert manifest is not None and manifest.name == "manifest.json"
    data = json.loads(manifest.read_text())
    assert data["schema"] == "evolve-cache-manifest-v1"
    assert set(data["entries"]) == set(hashes)
    assert len(data["shards"]) == 5


def test_compact_no_shards_returns_none(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    cache.namespace_dir.mkdir(parents=True, exist_ok=True)
    assert cache.compact() is None


def test_load_uses_manifest_and_scans_new_shards(tmp_path: Path, caplog):
    """After compaction, load() serves manifest-covered thetas without
    re-scanning their shards, and still picks up shards added afterwards."""
    import logging

    cache = EvolveCache(tmp_path / "cache", "ns0")
    old = [_theta_hash(f"old{i}") for i in range(3)]
    for th in old:
        w = cache.writer()
        w.add(th, _blob_for(th))
        w.flush()
    cache.compact()

    # A shard added after the manifest.
    new_th = _theta_hash("new")
    w = cache.writer()
    w.add(new_th, _blob_for(new_th))
    w.flush()

    fresh = EvolveCache(tmp_path / "cache", "ns0")
    with caplog.at_level(logging.INFO, logger="qsp_hpc.cpp.evolve_cache"):
        fresh.load()
    # 3 thetas from the manifest, exactly 1 shard scanned (the new one).
    assert len(fresh) == 4
    for th in old + [new_th]:
        assert fresh.get(th) == _blob_for(th)
    assert any("3 from manifest, 1 shard(s) scanned" in r.message for r in caplog.records)


def test_corrupt_manifest_falls_back_to_full_scan(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("survivor")
    w = cache.writer()
    w.add(th, _blob_for(th))
    w.flush()
    (cache.namespace_dir / "manifest.json").write_text("{ this is not valid json")

    fresh = EvolveCache(tmp_path / "cache", "ns0")
    assert fresh.get(th) == _blob_for(th)  # full scan still finds it


def test_compact_leaves_no_tmp_files(tmp_path: Path):
    cache = EvolveCache(tmp_path / "cache", "ns0")
    th = _theta_hash("x")
    w = cache.writer()
    w.add(th, _blob_for(th))
    w.flush()
    cache.compact()
    leftovers = list(cache.namespace_dir.glob("manifest.json.tmp*"))
    assert leftovers == [], f"compact left temp files: {leftovers}"


def test_evolve_cache_writer_standalone(tmp_path: Path):
    """EvolveCacheWriter can be used directly against a namespace dir."""
    ns_dir = tmp_path / "cache" / "nsX"
    th = _theta_hash("direct")
    writer = EvolveCacheWriter(ns_dir)
    writer.add(th, _blob_for(th))
    assert len(writer) == 1
    assert th in writer
    shard = writer.flush()
    assert shard is not None and shard.parent == ns_dir
    assert writer.flushed_path == shard


# --- Integration with the real qsp_sim binary -----------------------------
#
# Opt-in: env vars point at a prebuilt qsp_sim + param template + healthy
# state YAML. Tests skip cleanly if any are unset (keeps CI green when the
# C++ side isn't built).


def _real_binary_path() -> Path | None:
    env = os.environ.get("QSP_SIM_BINARY")
    return Path(env) if env and Path(env).exists() else None


def _real_template_path() -> Path | None:
    env = os.environ.get("QSP_SIM_TEMPLATE")
    return Path(env) if env and Path(env).exists() else None


def _real_healthy_yaml() -> Path | None:
    env = os.environ.get("QSP_SIM_HEALTHY_YAML")
    return Path(env) if env and Path(env).exists() else None


_SKIP_INTEGRATION = (
    _real_binary_path() is None or _real_template_path() is None or _real_healthy_yaml() is None
)
_SKIP_REASON = (
    "qsp_sim binary / param template / healthy_state.yaml not found; "
    "set QSP_SIM_BINARY, QSP_SIM_TEMPLATE, QSP_SIM_HEALTHY_YAML"
)


@pytest.fixture
def real_runner():
    from qsp_hpc.cpp.runner import CppRunner

    return CppRunner(
        binary_path=_real_binary_path(),
        template_path=_real_template_path(),
        healthy_state_yaml=_real_healthy_yaml(),
        default_timeout_s=60.0,
    )


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_cached_state_round_trips_through_qsp_sim(real_runner, tmp_path: Path):
    """A blob written through the cache, then read back and materialized,
    drives ``--initial-state`` to a trajectory byte-identical to a fresh
    ``--evolve-to-diagnosis`` run."""
    cache = EvolveCache(tmp_path / "evolve_cache", "ns_real")

    xml = real_runner._renderer.render({})
    th = theta_hash_for_xml(xml)

    # Populate: evolve once via --dump-state, write the blob through.
    dump_path = tmp_path / "dump.bin"
    real_runner.dump_evolve_state(
        params={}, params_hash=wire_hash(th), state_out=dump_path, workdir=tmp_path
    )
    writer = cache.writer()
    writer.add(th, dump_path.read_bytes())
    writer.flush()

    # Read back from a fresh cache and run the scenario from the cached state.
    fresh = EvolveCache(tmp_path / "evolve_cache", "ns_real")
    state_path = fresh.materialize(th, tmp_path / "wd")
    assert state_path is not None

    cached = real_runner.run_one(
        params={},
        t_end_days=1.0,
        min_cadence_hours=0.1,
        workdir=tmp_path,
        evolve_state_path=state_path,
        params_hash=th,
    )
    fresh_run = real_runner.run_one(
        params={}, t_end_days=1.0, min_cadence_hours=0.1, workdir=tmp_path
    )
    np.testing.assert_array_equal(cached.trajectory, fresh_run.trajectory)


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_batch_runner_reuses_cache_across_scenarios(tmp_path: Path):
    """Two batches over the same theta under different scenarios: the
    first writes a cache shard, the second hits it and writes no shard."""
    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    cache_root = tmp_path / "evolve_cache"
    br = CppBatchRunner(
        binary_path=_real_binary_path(),
        template_path=_real_template_path(),
        healthy_state_yaml=_real_healthy_yaml(),
        default_timeout_s=60.0,
        evolve_cache_root=cache_root,
    )
    theta = np.empty((2, 0))  # two sims, the shared template-default theta

    r1 = br.run(
        theta_matrix=theta,
        param_names=[],
        t_end_days=1.0,
        min_cadence_hours=0.1,
        output_path=tmp_path / "batch_baseline.parquet",
        scenario="baseline",
        max_workers=1,
    )
    r2 = br.run(
        theta_matrix=theta,
        param_names=[],
        t_end_days=1.0,
        min_cadence_hours=0.1,
        output_path=tmp_path / "batch_treatment.parquet",
        scenario="treatment",
        max_workers=1,
    )
    assert r1.n_failed == 0 and r2.n_failed == 0

    # Batch 1 evolved a cold theta and wrote a shard; batch 2 hit the
    # cache for every sim and wrote nothing.
    assert r1.evolve_shard_path is not None and r1.evolve_shard_path.exists()
    assert r2.evolve_shard_path is None

    # Exactly one namespace, holding exactly one shared theta state.
    ns_dirs = [p for p in cache_root.iterdir() if p.is_dir()]
    assert len(ns_dirs) == 1, f"expected one namespace dir, got {ns_dirs}"
    cache = EvolveCache(cache_root, ns_dirs[0].name)
    assert len(cache) == 1, f"expected one cached evolve state, got {len(cache)}"


@pytest.mark.skipif(_SKIP_INTEGRATION, reason=_SKIP_REASON)
def test_batch_runner_warns_when_cache_requested_without_healthy_state(tmp_path: Path, caplog):
    """Regression guard for #34: passing evolve_cache_root without
    healthy_state_yaml WARNs and disables the cache (rather than a silent
    DEBUG-level disable that left HPC workers writing 0 states)."""
    import logging

    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    with caplog.at_level(logging.WARNING, logger="qsp_hpc.cpp.batch_runner"):
        br = CppBatchRunner(
            binary_path=_real_binary_path(),
            template_path=_real_template_path(),
            # Intentionally omit healthy_state_yaml.
            default_timeout_s=60.0,
            evolve_cache_root=tmp_path / "evolve_cache",
        )
    assert br.evolve_cache_root is None
    assert any(
        "evolve_cache_root" in rec.message and "ignored" in rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
    ), f"expected a WARNING about evolve_cache_root being ignored; got {caplog.records}"

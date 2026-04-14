"""Deterministic, indexable theta pool shared across scenarios.

The QSP simulator originally drew parameters from a stateful
:class:`numpy.random.Generator`, which meant scenarios that consumed batches
of different sizes from the same nominal seed produced *different* theta
matrices. Joint multi-scenario inference relies on theta being identical
across scenarios at the same row index — drift here destroys that property
and tanks the joint NaN-filter retention.

This module pre-generates ``n_total`` rows of theta deterministically given
``(priors_csv, submodel_priors_yaml, seed, n_total)`` and caches them as a
``.npy`` file. Callers ask for theta by ``sample_index`` and always get the
same row.

The ``sample_index`` is propagated downstream through the simulator, MATLAB
worker, parquet outputs, derivation worker, and result loader so that
multi-scenario alignment becomes an integer-set intersection rather than a
positional join.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np


def theta_pool_cache_path(
    cache_dir: Union[str, Path],
    priors_csv: Union[str, Path],
    submodel_priors_yaml: Optional[Union[str, Path]],
    seed: int,
    n_total: int,
) -> Path:
    """Deterministic on-disk path for a cached theta pool.

    Hash includes priors CSV content + submodel priors YAML content (when
    present) + seed + n_total. Pool layout drift (e.g. different priors
    revisions) thus produces a different file rather than silently reusing
    a stale pool.
    """
    h = hashlib.sha256()
    h.update(Path(priors_csv).read_text().encode("utf-8"))
    if submodel_priors_yaml is not None:
        smp = Path(submodel_priors_yaml)
        if smp.exists():
            h.update(smp.read_text().encode("utf-8"))
    h.update(str(seed).encode("utf-8"))
    h.update(str(n_total).encode("utf-8"))
    return Path(cache_dir) / f"theta_pool_{h.hexdigest()[:16]}_n{n_total}.npy"


def get_theta_pool(
    priors_csv: Union[str, Path],
    submodel_priors_yaml: Optional[Union[str, Path]],
    seed: int,
    n_total: int,
    cache_dir: Union[str, Path] = "cache/theta_pools",
) -> np.ndarray:
    """Return a deterministic ``(n_total, n_params)`` theta matrix.

    First call generates and caches; subsequent calls with the same inputs
    load from cache. Sampling uses the composite copula prior when a
    submodel YAML is provided, falling back to per-parameter lognormal
    sampling from the CSV otherwise.
    """
    pool_path = theta_pool_cache_path(cache_dir, priors_csv, submodel_priors_yaml, seed, n_total)
    if pool_path.exists():
        return np.load(pool_path)

    use_submodel = submodel_priors_yaml is not None and Path(submodel_priors_yaml).exists()
    if use_submodel:
        import torch
        from qsp_inference.priors.copula_prior import load_composite_prior_log

        prior_log, _ = load_composite_prior_log(str(submodel_priors_yaml), str(priors_csv))
        torch.manual_seed(int(seed))
        with torch.no_grad():
            log_samples = prior_log.sample((n_total,)).numpy()
        theta = np.exp(log_samples)
    else:
        import pandas as pd

        rng = np.random.default_rng(seed)
        priors_df = pd.read_csv(priors_csv)
        n_params = len(priors_df)
        theta = np.zeros((n_total, n_params))
        for i in range(n_params):
            row = priors_df.iloc[i]
            if row["distribution"] == "lognormal":
                theta[:, i] = rng.lognormal(
                    mean=row["dist_param1"], sigma=row["dist_param2"], size=n_total
                )
            else:
                raise ValueError(f"Unsupported distribution: {row['distribution']}")

    pool_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pool_path, theta)
    return theta


def theta_for_indices(
    indices: np.ndarray,
    priors_csv: Union[str, Path],
    submodel_priors_yaml: Optional[Union[str, Path]],
    seed: int,
    n_total: int,
    cache_dir: Union[str, Path] = "cache/theta_pools",
) -> np.ndarray:
    """Slice the theta pool by integer ``sample_index`` array.

    ``indices`` may be unordered or contain gaps; the returned matrix is in
    the same order as ``indices``.
    """
    pool = get_theta_pool(
        priors_csv=priors_csv,
        submodel_priors_yaml=submodel_priors_yaml,
        seed=seed,
        n_total=n_total,
        cache_dir=cache_dir,
    )
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= n_total):
        raise IndexError(
            f"sample_index out of range: min={indices.min()} max={indices.max()} n_total={n_total}"
        )
    return pool[indices]

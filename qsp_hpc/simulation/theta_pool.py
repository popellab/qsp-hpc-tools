"""Deterministic, indexable theta pool shared across scenarios.

The QSP simulator originally drew parameters from a stateful
:class:`numpy.random.Generator`, which meant scenarios that consumed batches
of different sizes from the same nominal seed produced *different* theta
matrices. Joint multi-scenario inference relies on theta being identical
across scenarios at the same row index — drift here destroys that property
and tanks the joint NaN-filter retention.

This module pre-generates ``n_total`` rows of theta deterministically given
``(priors_csv, submodel_priors_yaml, seed, n_total[, restriction])`` and
caches them as a ``.npy`` file. Callers ask for theta by ``sample_index``
and always get the same row.

The ``sample_index`` is propagated downstream through the simulator, MATLAB
worker, parquet outputs, derivation worker, and result loader so that
multi-scenario alignment becomes an integer-set intersection rather than a
positional join.

Optional classifier-based prior restriction: pass a
``restriction_classifier_dir`` (pointing at a
``qsp_inference.inference.RestrictionClassifier`` serialization — i.e. a
directory containing ``classifier.pkl`` + ``metadata.json``) and a
``restriction_threshold``. The pool is then built by rejection sampling
against the classifier, yielding thetas from the (approximate) viable
region of the prior so downstream sim jobs don't waste compute on
draws that always fail.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np


def _classifier_hash_suffix(
    restriction_classifier_dir: Optional[Union[str, Path]],
    restriction_threshold: float,
    classifier_feature_fills: Optional[Mapping[str, float]] = None,
) -> bytes:
    """Content-hash bytes identifying a classifier dir + threshold + fills."""
    if restriction_classifier_dir is None:
        return b""
    d = Path(restriction_classifier_dir)
    pkl = d / "classifier.pkl"
    meta = d / "metadata.json"
    buf = b"|classifier|"
    if pkl.exists():
        buf += pkl.read_bytes()
    if meta.exists():
        buf += meta.read_bytes()
    buf += f"|tau={restriction_threshold:.6f}".encode("utf-8")
    if classifier_feature_fills:
        # Sort to keep hash insensitive to dict ordering.
        fills_str = ",".join(
            f"{k}={float(v):.12g}" for k, v in sorted(classifier_feature_fills.items())
        )
        buf += f"|fills={fills_str}".encode("utf-8")
    return buf


def theta_pool_cache_path(
    cache_dir: Union[str, Path],
    priors_csv: Union[str, Path],
    submodel_priors_yaml: Optional[Union[str, Path]],
    seed: int,
    n_total: int,
    restriction_classifier_dir: Optional[Union[str, Path]] = None,
    restriction_threshold: float = 0.5,
    classifier_feature_fills: Optional[Mapping[str, float]] = None,
) -> Path:
    """Deterministic on-disk path for a cached theta pool.

    Hash includes priors CSV content + submodel priors YAML content (when
    present) + seed + n_total + restriction classifier bytes (when
    restricted). Pool layout drift (e.g. different priors revisions, or a
    different classifier) thus produces a different file rather than
    silently reusing a stale pool.
    """
    h = hashlib.sha256()
    h.update(Path(priors_csv).read_text().encode("utf-8"))
    if submodel_priors_yaml is not None:
        smp = Path(submodel_priors_yaml)
        if smp.exists():
            h.update(smp.read_text().encode("utf-8"))
    h.update(str(seed).encode("utf-8"))
    h.update(str(n_total).encode("utf-8"))
    h.update(
        _classifier_hash_suffix(
            restriction_classifier_dir,
            restriction_threshold,
            classifier_feature_fills,
        )
    )
    suffix = "_restricted" if restriction_classifier_dir is not None else ""
    return Path(cache_dir) / f"theta_pool_{h.hexdigest()[:16]}_n{n_total}{suffix}.npy"


def _sample_prior_batch(
    priors_csv: Path,
    submodel_priors_yaml: Optional[Path],
    n: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Draw ``n`` theta rows from the composite prior (copula or lognormal).

    Returns ``(theta, param_names)``. Caller is responsible for threading
    ``seed`` deterministically across multiple calls — each call seeds its
    own RNG identically to how ``get_theta_pool`` did originally.
    """
    use_submodel = submodel_priors_yaml is not None and submodel_priors_yaml.exists()
    if use_submodel:
        import torch
        from qsp_inference.priors.copula_prior import load_composite_prior_log

        prior_log, param_names = load_composite_prior_log(
            str(submodel_priors_yaml), str(priors_csv)
        )
        torch.manual_seed(int(seed))
        with torch.no_grad():
            log_samples = prior_log.sample((n,)).numpy()
        return np.exp(log_samples), list(param_names)

    import pandas as pd

    rng = np.random.default_rng(seed)
    priors_df = pd.read_csv(priors_csv)
    n_params = len(priors_df)
    theta = np.zeros((n, n_params))
    for i in range(n_params):
        row = priors_df.iloc[i]
        if row["distribution"] == "lognormal":
            theta[:, i] = rng.lognormal(mean=row["dist_param1"], sigma=row["dist_param2"], size=n)
        else:
            raise ValueError(f"Unsupported distribution: {row['distribution']}")
    return theta, priors_df["name"].tolist()


def get_theta_pool(
    priors_csv: Union[str, Path],
    submodel_priors_yaml: Optional[Union[str, Path]],
    seed: int,
    n_total: int,
    cache_dir: Union[str, Path] = "cache/theta_pools",
    restriction_classifier_dir: Optional[Union[str, Path]] = None,
    restriction_threshold: float = 0.5,
    restriction_oversample_factor: float = 2.5,
    restriction_max_oversample: int = 8,
    classifier_feature_fills: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Return a deterministic ``(n_total, n_params)`` theta matrix.

    First call generates and caches; subsequent calls with the same inputs
    load from cache. Sampling uses the composite copula prior when a
    submodel YAML is provided, falling back to per-parameter lognormal
    sampling from the CSV otherwise.

    When ``restriction_classifier_dir`` is provided, the pool is built by
    rejection sampling: each batch oversamples the prior by
    ``restriction_oversample_factor`` and keeps draws that score
    ``>= restriction_threshold`` under the classifier. If the first batch
    yields fewer than ``n_total`` accepted thetas, the batch size is
    doubled (up to ``restriction_max_oversample`` × baseline) and resampled
    with a fresh seed derived from ``seed`` — guaranteeing termination but
    keeping the cache key deterministic on the input args.
    """
    priors_csv = Path(priors_csv)
    submodel_priors_yaml = Path(submodel_priors_yaml) if submodel_priors_yaml else None
    pool_path = theta_pool_cache_path(
        cache_dir,
        priors_csv,
        submodel_priors_yaml,
        seed,
        n_total,
        restriction_classifier_dir=restriction_classifier_dir,
        restriction_threshold=restriction_threshold,
        classifier_feature_fills=classifier_feature_fills,
    )
    if pool_path.exists():
        return np.load(pool_path)

    if restriction_classifier_dir is None:
        theta, _ = _sample_prior_batch(priors_csv, submodel_priors_yaml, n_total, seed)
    else:
        from qsp_inference.inference.restriction import RestrictionClassifier

        clf = RestrictionClassifier.load(restriction_classifier_dir)
        # Oversample the prior; keep only classifier-accepted rows. If we
        # fall short, bump the oversample factor deterministically via a
        # seed offset and retry.
        # When the live prior has drifted relative to the classifier
        # (params added/retired), we project caller-side theta onto the
        # classifier's feature_order via accept_named, dropping live-only
        # columns and filling classifier-only columns from
        # ``classifier_feature_fills``.
        accepted = []
        n_accepted = 0
        factor = float(restriction_oversample_factor)
        attempt = 0
        while n_accepted < n_total:
            attempt += 1
            if attempt > 1 and factor >= restriction_oversample_factor * restriction_max_oversample:
                raise RuntimeError(
                    f"restricted pool: could not reach {n_total} accepted at "
                    f"τ={restriction_threshold}; got {n_accepted} after "
                    f"oversample factor {factor:.1f}"
                )
            batch_n = int(factor * n_total)
            # Deterministic offset per attempt so cache is reproducible.
            batch_seed = int(seed) + attempt - 1
            theta_batch, batch_names = _sample_prior_batch(
                priors_csv, submodel_priors_yaml, batch_n, batch_seed
            )
            if list(batch_names) == list(clf.feature_order):
                keep = clf.accept(theta_batch, threshold=restriction_threshold)
            else:
                keep = clf.accept_named(
                    theta_batch,
                    batch_names,
                    fills=classifier_feature_fills,
                    threshold=restriction_threshold,
                )
            accepted.append(theta_batch[keep])
            n_accepted += int(keep.sum())
            factor *= 2.0
        theta = np.concatenate(accepted, axis=0)[:n_total]

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
    restriction_classifier_dir: Optional[Union[str, Path]] = None,
    restriction_threshold: float = 0.5,
    classifier_feature_fills: Optional[Mapping[str, float]] = None,
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
        restriction_classifier_dir=restriction_classifier_dir,
        restriction_threshold=restriction_threshold,
        classifier_feature_fills=classifier_feature_fills,
    )
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= n_total):
        raise IndexError(
            f"sample_index out of range: min={indices.min()} max={indices.max()} n_total={n_total}"
        )
    return pool[indices]

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


def _load_vary_list(vary_policy: Optional[Union[str, Path]]) -> Optional[list[str]]:
    """Read the top-level ``vary:`` allowlist from a policy YAML.

    Returns ``None`` when no policy is given. Raises if the file has no
    non-empty ``vary:`` list — an empty policy would silently pin every
    param, which is never the intent.
    """
    if vary_policy is None:
        return None
    import yaml

    with open(vary_policy) as f:
        data = yaml.safe_load(f)
    vary = list((data or {}).get("vary") or [])
    if not vary:
        raise ValueError(f"vary policy {vary_policy} has no non-empty `vary:` list")
    return vary


def theta_pool_cache_path(
    cache_dir: Union[str, Path],
    priors_csv: Union[str, Path],
    submodel_priors_yaml: Optional[Union[str, Path]],
    seed: int,
    n_total: int,
    restriction_classifier_dir: Optional[Union[str, Path]] = None,
    restriction_threshold: float = 0.5,
    classifier_feature_fills: Optional[Mapping[str, float]] = None,
    vary_policy: Optional[Union[str, Path]] = None,
    derived_yaml: Optional[Union[str, Path]] = None,
) -> Path:
    """Deterministic on-disk path for a cached theta pool.

    Hash includes priors CSV content + submodel priors YAML content (when
    present) + seed + n_total + restriction classifier bytes (when
    restricted) + vary-policy content (when a σ-overlay policy is applied) +
    derived-parameter policy content (when derived params are injected). Pool
    layout drift (e.g. different priors revisions, a different classifier, or a
    change to the vary/pin or derived policy) thus produces a different file
    rather than silently reusing a stale pool.
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
    overlay_suffix = ""
    if vary_policy is not None:
        vp = Path(vary_policy)
        if vp.exists():
            h.update(b"|vary_policy|")
            h.update(vp.read_text().encode("utf-8"))
        overlay_suffix = "_overlay"
    derived_suffix = ""
    if derived_yaml is not None:
        dp = Path(derived_yaml)
        if dp.exists():
            h.update(b"|derived_yaml|")
            h.update(dp.read_text().encode("utf-8"))
        derived_suffix = "_derived"
    suffix = (
        ("_restricted" if restriction_classifier_dir is not None else "")
        + overlay_suffix
        + derived_suffix
    )
    return Path(cache_dir) / f"theta_pool_{h.hexdigest()[:16]}_n{n_total}{suffix}.npy"


def _sample_prior_batch(
    priors_csv: Path,
    submodel_priors_yaml: Optional[Path],
    n: int,
    seed: int,
    vary_policy: Optional[Path] = None,
    derived_yaml: Optional[Path] = None,
) -> tuple[np.ndarray, list[str]]:
    """Draw ``n`` theta rows from the composite prior (copula or lognormal).

    Returns ``(theta, param_names)``. Caller is responsible for threading
    ``seed`` deterministically across multiple calls — each call seeds its
    own RNG identically to how ``get_theta_pool`` did originally.

    When ``vary_policy`` is given, the cloud is drawn from the σ-overlay
    prior (``load_overlay_prior_log``) instead of the plain center composite:
    center μ everywhere, population σ where an ``observed_distribution`` is
    declared, and every param outside the policy's ``vary:`` allowlist pinned
    to its center (σ→~0, copula-decoupled). This is what makes a vary/pin
    policy actually reach the training cloud rather than only the Python-side
    embedding prior.

    When ``derived_yaml`` is given, listed params are replaced by power-law
    children of their parents (``load_derived_specs`` /
    ``apply_derived_priors``), applied AFTER any σ-overlay so a derived child
    tracks its post-overlay parents.
    """
    use_submodel = submodel_priors_yaml is not None and submodel_priors_yaml.exists()
    if use_submodel:
        import torch

        vary = _load_vary_list(vary_policy)
        if vary is not None:
            from qsp_inference.priors.copula_prior import load_overlay_prior_log

            prior_log, param_names = load_overlay_prior_log(
                str(submodel_priors_yaml),
                str(priors_csv),
                vary_params=vary,
                derived_yaml=str(derived_yaml) if derived_yaml else None,
            )
        else:
            from qsp_inference.priors.copula_prior import load_composite_prior_log

            prior_log, param_names = load_composite_prior_log(
                str(submodel_priors_yaml),
                str(priors_csv),
                derived_yaml=str(derived_yaml) if derived_yaml else None,
            )
        torch.manual_seed(int(seed))
        with torch.no_grad():
            log_samples = prior_log.sample((n,)).numpy()
        return np.exp(log_samples), list(param_names)

    if vary_policy is not None:
        raise ValueError(
            "vary_policy σ-overlay requires a submodel_priors.yaml (it overlays "
            "the composite center prior); none was provided."
        )
    if derived_yaml is not None:
        raise ValueError(
            "derived_yaml requires a submodel_priors.yaml (derived params inject "
            "into the composite copula prior); none was provided."
        )

    import pandas as pd

    rng = np.random.default_rng(seed)
    priors_df = pd.read_csv(priors_csv)
    n_params = len(priors_df)
    theta = np.zeros((n, n_params))
    for i in range(n_params):
        row = priors_df.iloc[i]
        dist = row["distribution"]
        p1 = float(row["dist_param1"])
        p2 = float(row["dist_param2"])
        if dist == "lognormal":
            theta[:, i] = rng.lognormal(mean=p1, sigma=p2, size=n)
        elif dist == "normal":
            theta[:, i] = rng.normal(loc=p1, scale=p2, size=n)
        elif dist == "uniform":
            theta[:, i] = rng.uniform(low=p1, high=p2, size=n)
        elif dist == "beta":
            # dist_param1 = alpha (concentration1), dist_param2 = beta (concentration0)
            # Matches qsp-inference's load_sbi_priors convention.
            theta[:, i] = rng.beta(a=p1, b=p2, size=n)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
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
    vary_policy: Optional[Union[str, Path]] = None,
    derived_yaml: Optional[Union[str, Path]] = None,
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
    vary_policy = Path(vary_policy) if vary_policy else None
    derived_yaml = Path(derived_yaml) if derived_yaml else None
    pool_path = theta_pool_cache_path(
        cache_dir,
        priors_csv,
        submodel_priors_yaml,
        seed,
        n_total,
        restriction_classifier_dir=restriction_classifier_dir,
        restriction_threshold=restriction_threshold,
        classifier_feature_fills=classifier_feature_fills,
        vary_policy=vary_policy,
        derived_yaml=derived_yaml,
    )
    if pool_path.exists():
        return np.load(pool_path)

    if restriction_classifier_dir is None:
        theta, _ = _sample_prior_batch(
            priors_csv,
            submodel_priors_yaml,
            n_total,
            seed,
            vary_policy=vary_policy,
            derived_yaml=derived_yaml,
        )
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
                priors_csv,
                submodel_priors_yaml,
                batch_n,
                batch_seed,
                vary_policy=vary_policy,
                derived_yaml=derived_yaml,
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
    vary_policy: Optional[Union[str, Path]] = None,
    derived_yaml: Optional[Union[str, Path]] = None,
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
        vary_policy=vary_policy,
        derived_yaml=derived_yaml,
    )
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= n_total):
        raise IndexError(
            f"sample_index out of range: min={indices.min()} max={indices.max()} n_total={n_total}"
        )
    return pool[indices]

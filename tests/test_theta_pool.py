"""Tests for theta_pool caching + classifier-restricted sampling."""

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.simulation.theta_pool import (
    get_theta_pool,
    theta_for_indices,
    theta_pool_cache_path,
)


def _make_priors_csv(tmp_path, n_params=4):
    rows = []
    for i in range(n_params):
        rows.append(
            {
                "name": f"p{i}",
                "distribution": "lognormal",
                "dist_param1": 0.0,
                "dist_param2": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    p = tmp_path / "priors.csv"
    df.to_csv(p, index=False)
    return p


def test_unrestricted_pool_deterministic(tmp_path):
    priors = _make_priors_csv(tmp_path)
    pool1 = get_theta_pool(priors, None, seed=42, n_total=500, cache_dir=tmp_path / "c1")
    pool2 = get_theta_pool(priors, None, seed=42, n_total=500, cache_dir=tmp_path / "c2")
    assert pool1.shape == (500, 4)
    np.testing.assert_array_equal(pool1, pool2)
    # theta_for_indices slices the cached pool in order.
    idx = np.array([0, 42, 100])
    sliced = theta_for_indices(idx, priors, None, seed=42, n_total=500, cache_dir=tmp_path / "c1")
    np.testing.assert_array_equal(sliced, pool1[idx])


def test_cache_path_changes_with_classifier(tmp_path):
    priors = _make_priors_csv(tmp_path)
    # Fake classifier dir with different contents → different hash.
    cdir_a = tmp_path / "a"
    cdir_a.mkdir()
    (cdir_a / "classifier.pkl").write_bytes(b"A-bytes")
    (cdir_a / "metadata.json").write_text('{"v": 1}')
    cdir_b = tmp_path / "b"
    cdir_b.mkdir()
    (cdir_b / "classifier.pkl").write_bytes(b"B-bytes")
    (cdir_b / "metadata.json").write_text('{"v": 2}')

    p_none = theta_pool_cache_path(tmp_path, priors, None, 1, 100)
    p_a = theta_pool_cache_path(tmp_path, priors, None, 1, 100, cdir_a, 0.5)
    p_b = theta_pool_cache_path(tmp_path, priors, None, 1, 100, cdir_b, 0.5)
    p_a_tau9 = theta_pool_cache_path(tmp_path, priors, None, 1, 100, cdir_a, 0.9)
    # All four paths distinct.
    assert len({p_none, p_a, p_b, p_a_tau9}) == 4
    # Restricted paths carry the suffix.
    assert str(p_a).endswith("_restricted.npy")
    assert str(p_none).endswith(f"_n{100}.npy")


def test_restricted_pool_uses_classifier(tmp_path):
    """Classifier-restricted pool should only contain accepted thetas."""
    pytest.importorskip("sklearn")
    from qsp_inference.inference.restriction import train_restriction_classifier

    priors = _make_priors_csv(tmp_path, n_params=3)

    # Train a classifier on a big prior-sample: valid iff p0 > 1 (i.e.
    # log p0 > 0). That keeps ~50% of draws.
    train_pool = get_theta_pool(
        priors, None, seed=1, n_total=4000, cache_dir=tmp_path / "train_cache"
    )
    valid = train_pool[:, 0] > 1.0
    clf = train_restriction_classifier(train_pool, valid, ["p0", "p1", "p2"], cv_folds=0)
    clf_dir = tmp_path / "clf"
    clf.save(clf_dir)

    # Build a restricted pool of 300 thetas at τ=0.5.
    restricted = get_theta_pool(
        priors,
        None,
        seed=2,
        n_total=300,
        cache_dir=tmp_path / "rcache",
        restriction_classifier_dir=clf_dir,
        restriction_threshold=0.5,
    )
    assert restricted.shape == (300, 3)
    # Every row should pass the classifier.
    scores = clf.score(restricted)
    assert (scores >= 0.5).all(), f"restricted pool has {(scores < 0.5).sum()} rows below τ"
    # Deterministic: call again, same bytes.
    again = get_theta_pool(
        priors,
        None,
        seed=2,
        n_total=300,
        cache_dir=tmp_path / "rcache",
        restriction_classifier_dir=clf_dir,
        restriction_threshold=0.5,
    )
    np.testing.assert_array_equal(restricted, again)

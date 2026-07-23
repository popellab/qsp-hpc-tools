"""Tests for theta_pool caching + classifier-restricted sampling."""

import numpy as np
import pandas as pd
import pytest

from qsp_hpc.simulation.theta_pool import (
    _load_vary_list,
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


def test_cache_path_changes_with_vary_policy(tmp_path):
    """A σ-overlay vary policy keys a distinct pool with an ``_overlay`` suffix,
    and editing the policy content changes the hash."""
    priors = _make_priors_csv(tmp_path)
    pol_a = tmp_path / "policy_a.yaml"
    pol_a.write_text("vary:\n  - p0\n  - p1\n")
    pol_b = tmp_path / "policy_b.yaml"
    pol_b.write_text("vary:\n  - p0\n")  # different allowlist → different hash

    p_none = theta_pool_cache_path(tmp_path, priors, None, 1, 100)
    p_a = theta_pool_cache_path(tmp_path, priors, None, 1, 100, vary_policy=pol_a)
    p_b = theta_pool_cache_path(tmp_path, priors, None, 1, 100, vary_policy=pol_b)
    assert len({p_none, p_a, p_b}) == 3
    assert str(p_a).endswith("_overlay.npy")
    assert str(p_none).endswith("_n100.npy")


def test_vary_policy_requires_submodel_yaml(tmp_path):
    """The overlay overlays the composite center prior, so a vary policy with no
    submodel YAML is a misconfiguration — fail loudly, don't silently ignore it."""
    priors = _make_priors_csv(tmp_path)
    pol = tmp_path / "policy.yaml"
    pol.write_text("vary:\n  - p0\n")
    with pytest.raises(ValueError, match="requires a submodel_priors.yaml"):
        get_theta_pool(priors, None, seed=1, n_total=10, cache_dir=tmp_path / "c", vary_policy=pol)


def test_load_vary_list_rejects_empty(tmp_path):
    assert _load_vary_list(None) is None
    empty = tmp_path / "empty.yaml"
    empty.write_text("vary: []\n")
    with pytest.raises(ValueError, match="no non-empty"):
        _load_vary_list(empty)


def test_cache_path_changes_with_derived_yaml(tmp_path):
    """A derived-parameter policy keys a distinct pool with a ``_derived`` suffix,
    and editing the policy content changes the hash."""
    priors = _make_priors_csv(tmp_path)
    d_a = tmp_path / "derived_a.yaml"
    d_a.write_text(
        "parameters:\n  p1:\n    parents: {p0: 1.0}\n    log_coeff: -0.5\n    sigma_coeff: 0.4\n"
    )
    d_b = tmp_path / "derived_b.yaml"
    d_b.write_text(
        "parameters:\n  p1:\n    parents: {p0: 1.0}\n    log_coeff: -1.0\n    sigma_coeff: 0.4\n"
    )

    p_none = theta_pool_cache_path(tmp_path, priors, None, 1, 100)
    p_a = theta_pool_cache_path(tmp_path, priors, None, 1, 100, derived_yaml=d_a)
    p_b = theta_pool_cache_path(tmp_path, priors, None, 1, 100, derived_yaml=d_b)
    assert len({p_none, p_a, p_b}) == 3
    assert str(p_a).endswith("_derived.npy")


def test_derived_yaml_requires_submodel_yaml(tmp_path):
    """Derived params inject into the composite copula prior, so a derived policy
    with no submodel YAML is a misconfiguration — fail loudly."""
    priors = _make_priors_csv(tmp_path)
    d = tmp_path / "derived.yaml"
    d.write_text(
        "parameters:\n  p1:\n    parents: {p0: 1.0}\n    log_coeff: -0.5\n    sigma_coeff: 0.4\n"
    )
    with pytest.raises(ValueError, match="requires a submodel_priors.yaml"):
        get_theta_pool(priors, None, seed=1, n_total=10, cache_dir=tmp_path / "c", derived_yaml=d)


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

import numpy as np
import pandas as pd
import pytest

import core.clustering as cl
from core.clustering import ClusterResult, cluster_merchants, cluster_stats
from core.features import FEATURE_COLUMNS


def _features(n_merchants, seed=42):
    rng = np.random.default_rng(seed)
    # two obvious blobs so real clustering succeeds
    half = n_merchants // 2
    a = rng.normal(0.0, 0.1, size=(half, len(FEATURE_COLUMNS)))
    b = rng.normal(5.0, 0.1, size=(n_merchants - half, len(FEATURE_COLUMNS)))
    data = np.vstack([a, b])
    return pd.DataFrame(data, columns=FEATURE_COLUMNS,
                        index=[f"M{i}" for i in range(n_merchants)])


def test_clusters_two_blobs():
    r = cluster_merchants(_features(12), n_transactions=50)
    assert not r.used_fallback
    assert r.k == 2
    assert r.silhouette > 0.5


def test_fallback_too_few_transactions():
    r = cluster_merchants(_features(12), n_transactions=9)
    assert r.used_fallback and r.reason == "not_enough_transactions"


def test_fallback_too_few_merchants():
    r = cluster_merchants(_features(5), n_transactions=50)
    assert r.used_fallback and r.reason == "not_enough_merchants"


def test_fallback_weak_silhouette(monkeypatch):
    monkeypatch.setattr(cl, "silhouette_score", lambda X, labels: 0.05)
    r = cluster_merchants(_features(12), n_transactions=50)
    assert r.used_fallback and r.reason == "weak_clusters"


def test_fallback_uses_categories_when_available():
    feats = _features(5)
    cats = pd.Series(["Food", "Food", "Transport", "Transport", "Food"],
                     index=feats.index)
    r = cluster_merchants(feats, n_transactions=50, categories=cats)
    assert r.used_fallback
    assert r.labels.nunique() == 2


def test_cluster_stats_aggregates():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-05"]),
        "amount": [10.0, 20.0, 12.99],
        "description": ["SHOP A 1", "SHOP A 2", "NETFLIX"],
        "category": ["", "", ""],
        "merchant": ["SHOP A", "SHOP A", "NETFLIX"],
    })
    labels = pd.Series({"SHOP A": 0, "NETFLIX": 1})
    stats = cluster_stats(df, labels)
    c0 = next(s for s in stats if s["cluster_id"] == 0)
    assert c0["n_transactions"] == 2
    assert c0["total_spend"] == pytest.approx(30.0)
    assert c0["top_merchants"] == ["SHOP A"]
    assert "2026-01" in c0["monthly_totals"]

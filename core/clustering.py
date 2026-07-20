"""Per-merchant feature matrix -> K-means clusters and aggregate statistics."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_TRANSACTIONS = 10
MIN_MERCHANTS = 6
MIN_SILHOUETTE = 0.15
MAX_K = 5


@dataclass
class ClusterResult:
    """Result of clustering the merchant feature matrix."""

    labels: pd.Series
    k: int
    silhouette: float | None
    used_fallback: bool
    reason: str | None


def _fallback(
    features: pd.DataFrame,
    reason: str,
    categories: pd.Series | None,
) -> ClusterResult:
    """Use provided categories, or one neutral group, when K-means is unsuitable."""
    if categories is not None and categories.replace("", pd.NA).notna().any():
        aligned_categories = categories.reindex(features.index).fillna("")
        # A fallback must respect the same maximum as modelled clusters. Keep the
        # most common categories and group the long tail into one neutral bucket.
        if aligned_categories.nunique() > MAX_K:
            largest_categories = aligned_categories.value_counts().nlargest(MAX_K - 1).index
            aligned_categories = aligned_categories.where(
                aligned_categories.isin(largest_categories), "Other"
            )
        codes, _ = pd.factorize(aligned_categories)
        labels = pd.Series(codes, index=features.index, dtype="int64")
    else:
        labels = pd.Series(0, index=features.index, dtype="int64")

    return ClusterResult(
        labels=labels,
        k=int(labels.nunique()),
        silhouette=None,
        used_fallback=True,
        reason=reason,
    )


def cluster_merchants(
    features: pd.DataFrame,
    n_transactions: int,
    categories: pd.Series | None = None,
) -> ClusterResult:
    """Cluster merchants with silhouette-selected K-means and safe fallbacks."""
    n_merchants = len(features)
    if n_transactions < MIN_TRANSACTIONS:
        return _fallback(features, "not_enough_transactions", categories)
    if n_merchants < MIN_MERCHANTS:
        return _fallback(features, "not_enough_merchants", categories)

    k_max = min(MAX_K, n_merchants // 3)
    if k_max < 2:
        return _fallback(features, "no_valid_k", categories)

    scaled_features = StandardScaler().fit_transform(features.to_numpy())
    best_k: int | None = None
    best_score = -1.0
    best_labels = None

    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        if score > best_score:
            best_k = k
            best_score = float(score)
            best_labels = labels

    if best_score < MIN_SILHOUETTE or best_k is None or best_labels is None:
        return _fallback(features, "weak_clusters", categories)

    return ClusterResult(
        labels=pd.Series(best_labels, index=features.index, dtype="int64"),
        k=best_k,
        silhouette=best_score,
        used_fallback=False,
        reason=None,
    )


def cluster_stats(df: pd.DataFrame, labels: pd.Series) -> list[dict]:
    """Return explainable, aggregate statistics for each merchant cluster."""
    work = df.assign(cluster=df["merchant"].map(labels)).dropna(subset=["cluster"])
    statistics: list[dict] = []

    for cluster_id, group in work.groupby("cluster"):
        monthly = group.groupby(group["date"].dt.to_period("M"))["amount"].sum()
        top_merchant_spend = group.groupby("merchant")["amount"].sum().nlargest(5)
        statistics.append(
            {
                "cluster_id": int(cluster_id),
                "n_transactions": int(len(group)),
                "n_merchants": int(group["merchant"].nunique()),
                "total_spend": round(float(group["amount"].sum()), 2),
                "avg_amount": round(float(group["amount"].mean()), 2),
                "top_merchants": top_merchant_spend.index.tolist(),
                "top_merchant_items": [
                    [merchant, round(float(amount), 2)]
                    for merchant, amount in top_merchant_spend.items()
                ],
                "example_descriptions": group["description"].drop_duplicates().head(3).tolist(),
                "weekend_share": round(
                    float((group["date"].dt.dayofweek >= 5).mean()), 2
                ),
                "monthly_totals": {
                    str(period): round(float(amount), 2)
                    for period, amount in monthly.items()
                },
            }
        )

    return sorted(statistics, key=lambda item: -item["total_spend"])

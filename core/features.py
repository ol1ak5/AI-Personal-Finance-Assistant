"""Normalized expense DataFrame -> per-merchant feature matrix.

Volume features are normalized per month so different upload periods are
comparable. Recurrence is computed across the whole period (a monthly
subscription is invisible inside a single month).
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["monthly_spend", "tx_per_month", "avg_amount", "amount_cv",
                   "log_avg_amount", "weekend_share", "interval_regularity",
                   "amount_stability"]


def analysis_period(df: pd.DataFrame):
    """Months = calendar months spanned (Jan 10 - Feb 20 counts as 2), so
    monthly rates aren't inflated for uploads starting mid-month."""
    start, end = df["date"].min(), df["date"].max()
    months = (end.year * 12 + end.month) - (start.year * 12 + start.month) + 1
    return start, end, max(months, 1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    _, _, months = analysis_period(df)
    rows = []
    for merchant, g in df.groupby("merchant"):
        amounts, n = g["amount"], len(g)
        avg = float(amounts.mean())
        std = float(amounts.std(ddof=0)) if n > 1 else 0.0
        cv = std / avg if avg > 0 else 0.0
        gaps = g["date"].sort_values().diff().dt.days.dropna()
        if len(gaps) >= 2 and gaps.mean() > 0:
            regularity = 1.0 / (1.0 + float(gaps.std(ddof=0)) / float(gaps.mean()))
        else:
            regularity = 0.0
        rows.append({
            "merchant": merchant,
            "monthly_spend": float(amounts.sum()) / months,
            "tx_per_month": n / months,
            "avg_amount": avg,
            "amount_cv": cv,
            "log_avg_amount": float(np.log1p(avg)),
            "weekend_share": float((g["date"].dt.dayofweek >= 5).mean()),
            "interval_regularity": regularity,
            "amount_stability": 1.0 / (1.0 + cv),
        })
    return pd.DataFrame(rows).set_index("merchant")[FEATURE_COLUMNS]


def recurring_charges(feats: pd.DataFrame) -> list[list]:
    """[merchant, exact monthly price] for every charge with a regular
    monthly rhythm and a stable amount — the candidate subscription list
    handed to the AI summary as given facts (it filters out rent/bills)."""
    mask = (
        (feats["interval_regularity"] >= 0.8)
        & (feats["amount_stability"] >= 0.97)
        & feats["tx_per_month"].between(0.8, 1.3)
    )
    selected = feats[mask].sort_values("avg_amount", ascending=False)
    return [[merchant, round(float(row["avg_amount"]), 2)]
            for merchant, row in selected.iterrows()]


def format_frequency(tx_per_month: float) -> str:
    """Human wording for a purchase rate; never a raw '0.53/month'."""
    if tx_per_month >= 1.5:
        return f"{round(tx_per_month)}/month"
    if tx_per_month >= 0.8:
        return "monthly"
    if tx_per_month >= 0.4:
        return "every 2 months"
    if tx_per_month >= 0.28:
        return "every 3 months"
    return "1-2 times in 6 months"


def behaviour_label(tx_per_month: float, avg_amount: float,
                    interval_regularity: float, amount_stability: float) -> str:
    """Rule-based habit phrase; the local fallback for AI-written behaviours."""
    if tx_per_month >= 3.5:
        return "frequent small purchases" if avg_amount < 15 else "frequent shopping"
    if tx_per_month >= 1.5:
        return "regular shopping" if interval_regularity >= 0.6 else "repeat purchases"
    if 0.8 <= tx_per_month <= 1.3 and interval_regularity >= 0.8:
        if avg_amount >= 100:
            return "regular fixed payment"  # rent-sized, not a "subscription"
        return ("recurring subscription" if amount_stability >= 0.97
                else "steady monthly purchase")
    if 0.4 <= tx_per_month < 0.8 and interval_regularity >= 0.9:
        return "recurring bimonthly bill"
    if 0.4 <= tx_per_month < 0.8 and interval_regularity >= 0.6:
        return "occasional shopping trips"
    if avg_amount >= 100:
        return "occasional big-ticket purchase"
    return "one-off purchases"

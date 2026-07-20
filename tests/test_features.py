import numpy as np
import pandas as pd
import pytest

from core.features import analysis_period, behaviour_label, build_features, format_frequency


def _tx(merchant, dates, amount):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "amount": [amount] * len(dates),
        "description": [merchant] * len(dates),
        "category": [""] * len(dates),
        "merchant": [merchant] * len(dates),
    })


def test_period_months():
    df = _tx("A", ["2026-01-01", "2026-03-01"], 10.0)
    _, _, months = analysis_period(df)
    assert months == 3  # Jan, Feb, Mar — calendar months covered


def test_period_partial_months_not_inflated():
    df = _tx("A", ["2026-01-10", "2026-02-20"], 10.0)
    _, _, months = analysis_period(df)
    assert months == 2


def test_per_month_normalization():
    """Same monthly behaviour over 2 vs 12 months -> comparable features."""
    two = _tx("COFFEE", ["2026-01-10", "2026-01-20", "2026-02-10", "2026-02-20"], 5.0)
    twelve = _tx("COFFEE", [f"2026-{m:02d}-{d}" for m in range(1, 13) for d in (10, 20)], 5.0)
    f2, f12 = build_features(two), build_features(twelve)
    assert f2.loc["COFFEE", "monthly_spend"] == pytest.approx(
        f12.loc["COFFEE", "monthly_spend"], rel=0.01)
    assert f2.loc["COFFEE", "avg_amount"] == f12.loc["COFFEE", "avg_amount"]


def test_subscription_has_high_recurrence():
    sub = _tx("NETFLIX", ["2026-01-05", "2026-02-05", "2026-03-05", "2026-04-05"], 12.99)
    random_dates = ["2026-01-03", "2026-01-09", "2026-02-27", "2026-03-02"]
    noise = _tx("RESTAURANT", random_dates, 12.99)
    noise["amount"] = [8.0, 55.0, 23.0, 90.0]
    f = build_features(pd.concat([sub, noise], ignore_index=True))
    assert f.loc["NETFLIX", "interval_regularity"] > f.loc["RESTAURANT", "interval_regularity"]
    assert f.loc["NETFLIX", "amount_stability"] > f.loc["RESTAURANT", "amount_stability"]


def test_single_transaction_merchant_no_nan():
    f = build_features(_tx("ONEOFF", ["2026-01-15"], 99.0))
    assert not f.isna().any().any()
    assert f.loc["ONEOFF", "interval_regularity"] == 0.0


def test_format_frequency_buckets():
    assert format_frequency(4.0) == "4/month"
    assert format_frequency(1.0) == "monthly"
    assert format_frequency(0.5) == "every 2 months"
    assert format_frequency(0.33) == "every 3 months"
    assert format_frequency(0.17) == "1-2 times in 6 months"


def test_behaviour_label_rules():
    # order: tx_per_month, avg_amount, interval_regularity, amount_stability
    assert behaviour_label(4.0, 10.0, 0.9, 0.9) == "frequent small purchases"
    assert behaviour_label(4.0, 69.0, 0.88, 0.91) == "frequent shopping"
    assert behaviour_label(2.0, 47.7, 0.92, 0.88) == "regular shopping"
    assert behaviour_label(2.0, 47.7, 0.3, 0.88) == "repeat purchases"
    assert behaviour_label(1.0, 13.99, 0.96, 1.0) == "recurring subscription"
    assert behaviour_label(1.0, 58.65, 0.96, 0.94) == "steady monthly purchase"
    assert behaviour_label(1.0, 840.0, 0.96, 1.0) == "regular fixed payment"
    assert behaviour_label(0.5, 18.6, 0.98, 1.0) == "recurring bimonthly bill"
    assert behaviour_label(0.5, 132.3, 0.87, 0.91) == "occasional shopping trips"
    assert behaviour_label(0.33, 242.0, 0.0, 0.85) == "occasional big-ticket purchase"
    assert behaviour_label(0.17, 14.2, 0.0, 1.0) == "one-off purchases"


def test_demo_merchants_get_sensible_behaviours():
    from pathlib import Path

    from core import merchants, parser

    data = (Path(__file__).parent.parent / "data" / "spending_demo.csv").read_bytes()
    frames = parser.load_frames(data, "spending_demo.csv")
    df = list(frames.values())[0]
    result = parser.apply_mapping(df, parser.guess_mapping(df))
    feats = build_features(merchants.add_merchant_column(result.df))

    labels = {
        merchant: behaviour_label(
            row["tx_per_month"], row["avg_amount"],
            row["interval_regularity"], row["amount_stability"],
        )
        for merchant, row in feats.iterrows()
    }
    assert set(labels) == set(feats.index)          # every merchant labeled
    assert labels["MERCADONA"] == "frequent shopping"
    assert labels["NETFLIX COM"] == "recurring subscription"
    assert labels["CANAL ISABEL II"] == "recurring bimonthly bill"
    assert labels["BOOKING COM"] == "occasional big-ticket purchase"
    assert labels["RENT PAYMENT"] == "regular fixed payment"


def test_recurring_charges_from_demo_data():
    from pathlib import Path

    from core import merchants, parser
    from core.features import recurring_charges

    data = (Path(__file__).parent.parent / "data" / "spending_demo.csv").read_bytes()
    frames = parser.load_frames(data, "spending_demo.csv")
    df = list(frames.values())[0]
    result = parser.apply_mapping(df, parser.guess_mapping(df))
    feats = build_features(merchants.add_merchant_column(result.df))

    charges = {merchant: price for merchant, price in recurring_charges(feats)}
    assert charges["WELLHUB"] == 22.99          # exact price, not "about 23"
    assert charges["GLOVO PRIME"] == 7.99
    assert charges["RENT PAYMENT"] == 840.00    # included; prompt filters it
    assert "MERCADONA" not in charges           # frequent, not recurring

import pandas as pd

from core.merchants import add_merchant_column, normalize_merchant


def test_strips_reference_numbers():
    assert normalize_merchant("SUPERMARKET 1234 CITY") == "SUPERMARKET CITY"


def test_same_merchant_variants_collapse():
    a = normalize_merchant("NETFLIX.COM 12/07 REF98765")
    b = normalize_merchant("NETFLIX.COM 15/08 REF11111")
    assert a == b == "NETFLIX COM"


def test_punctuation_and_case():
    assert normalize_merchant("uber *trip-4412") == "UBER TRIP"


def test_keeps_full_rent_a_car_merchant_name():
    assert normalize_merchant("Sixt Rent a Car 2026-04-08") == "SIXT RENT A CAR"


def test_empty_becomes_unknown():
    assert normalize_merchant("  123456  ") == "UNKNOWN"


def test_add_merchant_column_preserves_description():
    df = pd.DataFrame({"description": ["SHOP 99 MAIN ST"]})
    out = add_merchant_column(df)
    assert out["merchant"].iloc[0] == "SHOP MAIN ST"
    assert out["description"].iloc[0] == "SHOP 99 MAIN ST"

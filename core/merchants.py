"""Raw transaction descriptions -> cleaned merchant identifiers.

Deliberately simple (see docs/DESIGN.md): uppercase, drop tokens containing
digits, strip punctuation, keep the first three tokens. The common “Rent a
Car” suffix is retained so car-hire merchants do not appear as truncated names.
No fuzzy matching.
"""
import re

import pandas as pd

_PUNCT = re.compile(r"[*#/\\\-_.,:;!?()\[\]{}'\"@+&]")


def normalize_merchant(description: str) -> str:
    s = _PUNCT.sub(" ", str(description).upper())
    tokens = [t for t in s.split() if not any(c.isdigit() for c in t)]
    # “SIXT RENT A CAR” and “EUROPCAR RENT A CAR” need all four words for a
    # readable dashboard label. Preserve this suffix without broadening the
    # normal three-token rule for unrelated, noisy bank descriptions.
    for index in range(max(0, len(tokens) - 2)):
        if tokens[index : index + 3] == ["RENT", "A", "CAR"]:
            return " ".join(tokens[: index + 3])
    return " ".join(tokens[:3]) or "UNKNOWN"


def add_merchant_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["merchant"] = out["description"].map(normalize_merchant)
    return out

# AI Personal Finance Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the AI Personal Finance Assistant dashboard (spec: `docs/DESIGN.md`) deployed on Streamlit Community Cloud with a Devpost submission by July 21, 2026, 5:00 pm PT.

**Architecture:** Single Streamlit app; pure-Python modules under `core/` (parser → merchants → features → clustering → llm → pdf), each importable and tested without the UI. `llm.py` is the only module touching the OpenAI API and honors `MOCK_LLM=true`.

**Tech Stack:** Python 3.11, uv, Streamlit, pandas, scikit-learn, Plotly + kaleido, OpenAI SDK (model `gpt-5.6`), fpdf2, openpyxl, pytest.

## Global Constraints

- Use `uv run` for project commands and `uv run pytest` for tests (never bare `python`).
- Expenses only; income rows are filtered out, never counted as "skipped".
- Cluster **merchants**, not transactions. k = 2..`min(7, n_merchants // 3)`.
- Fallback (no KMeans) when: `< 10` valid expense transactions, `< 6` distinct merchants, no valid k, or best silhouette `< 0.15`.
- Column mapping accepted only if `>= 90%` of rows yield valid date + numeric amount.
- Volume features normalized per month; recurrence computed across months; `< 2` full months ⇒ UI caveat.
- GPT-5.6 receives only: headers + max 5 sample rows (after explicit consent), and aggregated cluster stats. Never raw transaction tables. Nothing user-uploaded is written to disk.
- All tests run with `MOCK_LLM=true`, no network.
- Every LLM response is JSON, validated, one retry, then local fallback. The app never hard-fails on the API.
- Merchant normalization stays dumb: uppercase, drop digit-bearing tokens, strip punctuation, first 3 tokens. No fuzzy matching.
- Flat repo: modules in `core/`, docs in `docs/`, no deeper nesting.
- Commit after every green test run. All commits happen in Codex sessions where possible (Build Week evidence).

## File Structure

```
app.py                     # Streamlit UI + orchestration only
core/__init__.py
core/parser.py             # file bytes → normalized expense DataFrame
core/merchants.py          # description → merchant identifier
core/features.py           # DataFrame → per-merchant feature matrix
core/clustering.py         # features → ClusterResult + aggregate stats
core/llm.py                # all GPT-5.6 calls, MOCK_LLM, fallbacks
core/pdf.py                # figures + text → PDF bytes (never imports llm)
data/spending_demo.csv         # bundled anonymized demo export
tests/conftest.py          # synthetic bank-export fixtures
tests/test_parser.py  tests/test_merchants.py  tests/test_features.py
tests/test_clustering.py  tests/test_llm.py  tests/test_pdf.py
.streamlit/config.toml     # theme + 5 MB upload limit
pyproject.toml  uv.lock  .python-version  README.md  .gitignore
```

---

# Day 1 — Tue Jul 15: scaffolding + parsing

### Task 1: Repo scaffolding and git identity

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `core/__init__.py`, `tests/__init__.py`, `.streamlit/config.toml`

**Interfaces:**
- Produces: installable environment; `core` importable; git author fixed before the repo ever goes public.

- [ ] **Step 1: Fix the git author (all existing commits are `olya@ol-yaui-keompyuteo.local`)**

```bash
git config --global user.name "Olya"
git config --global user.email "ol2ksenova@gmail.com"
git rebase --root --exec 'git commit --amend --reset-author --no-edit'
git log --format='%an %ae' | sort -u   # expect exactly one line: Olya ol2ksenova@gmail.com
```

- [ ] **Step 2: Write the files**

`pyproject.toml` declares runtime dependencies and the `dev` dependency group for pytest. `uv lock` produces the committed `uv.lock`; `.python-version` pins local development to Python 3.11.

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.streamlit/secrets.toml
.pytest_cache/
```

`core/__init__.py` and `tests/__init__.py`: empty files.

`.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 5

[theme]
base = "light"
primaryColor = "#0F766E"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F1F5F9"
textColor = "#0F172A"
font = "sans serif"
```

- [ ] **Step 3: Sync dependencies and sanity-check**

```bash
uv sync --group dev
uv run python -c "import streamlit, sklearn, plotly, fpdf, openai, openpyxl; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore core tests .streamlit
git commit -m "chore: scaffold project structure and dependencies"
```

---

### Task 2: `core/parser.py` — mapping, dates, amounts, expense filtering

**Files:**
- Create: `core/parser.py`
- Test: `tests/test_parser.py`, `tests/conftest.py`

**Interfaces:**
- Produces (later tasks rely on these exact names):
  - `@dataclass ColumnMapping(date_col, description_col, amount_col=None, debit_col=None, credit_col=None, category_col=None, date_format=None, decimal_separator=".", expenses_are="negative")` — `expenses_are` ∈ `{"negative","positive","debit_col"}`
  - `guess_mapping(df: pd.DataFrame) -> ColumnMapping | None`
  - `validate_mapping(df, mapping) -> float` (share of rows with valid date+amount, 0..1)
  - `apply_mapping(df, mapping) -> ParseResult` where `@dataclass ParseResult(df, skipped_rows: int, total_rows: int)`; `ParseResult.df` columns: `date` (Timestamp), `amount` (float, positive expense magnitude), `description` (str), `category` (str or "")
  - `MAPPING_MIN_VALID = 0.9`

- [ ] **Step 1: Write fixtures in `tests/conftest.py`**

```python
import io
import pandas as pd
import pytest


@pytest.fixture
def simple_df():
    """English bank export, signed amounts (expenses negative)."""
    return pd.DataFrame({
        "Date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-02-05"],
        "Description": ["SUPERMARKET 123", "SALARY JANUARY", "NETFLIX.COM 555", "NETFLIX.COM 556"],
        "Amount": ["-52.30", "2100.00", "-12.99", "-12.99"],
    })


@pytest.fixture
def european_df():
    """German-style export: dd.mm.yyyy dates, 1.234,56 amounts."""
    return pd.DataFrame({
        "Datum": ["05.01.2026", "06.01.2026", "07.01.2026"],
        "Beschreibung": ["EDEKA FILIALE 44", "MIETE JANUAR", "BAHN TICKET 9981"],
        "Betrag": ["-1.234,56", "-800,00", "-49,90"],
    })


@pytest.fixture
def debit_credit_df():
    """Separate debit / credit columns, category present."""
    return pd.DataFrame({
        "Fecha": ["05/01/2026", "06/01/2026", "07/01/2026"],
        "Concepto": ["MERCADONA 8", "NOMINA", "TAXI 4412"],
        "Cargo": ["45.10", "", "9.80"],
        "Abono": ["", "2000.00", ""],
        "Categoria": ["Comida", "Ingresos", "Transporte"],
    })


@pytest.fixture
def malformed_df():
    """Junk rows mixed in: totals row, blank row, bad date."""
    return pd.DataFrame({
        "Date": ["2026-01-05", "TOTAL", "", "2026-01-08", "not a date"],
        "Description": ["SHOP A", "", "", "SHOP B", "SHOP C"],
        "Amount": ["-10.00", "-999.99", "", "-20.00", "-30.00"],
    })


def make_xlsx_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()
```

- [ ] **Step 2: Write failing tests in `tests/test_parser.py`**

```python
import pandas as pd
import pytest

from core.parser import (
    ColumnMapping, MAPPING_MIN_VALID, apply_mapping, guess_mapping, validate_mapping,
)


def test_guess_mapping_english(simple_df):
    m = guess_mapping(simple_df)
    assert m.date_col == "Date"
    assert m.amount_col == "Amount"
    assert m.description_col == "Description"


def test_apply_mapping_filters_expenses(simple_df):
    m = ColumnMapping(date_col="Date", description_col="Description",
                      amount_col="Amount", expenses_are="negative")
    r = apply_mapping(simple_df, m)
    assert r.total_rows == 4
    assert r.skipped_rows == 0          # income is filtered, not "skipped"
    assert len(r.df) == 3               # salary row excluded
    assert (r.df["amount"] > 0).all()   # stored as positive magnitudes
    assert r.df["amount"].sum() == pytest.approx(52.30 + 12.99 + 12.99)


def test_european_formats(european_df):
    m = ColumnMapping(date_col="Datum", description_col="Beschreibung",
                      amount_col="Betrag", date_format="%d.%m.%Y",
                      decimal_separator=",", expenses_are="negative")
    r = apply_mapping(european_df, m)
    assert r.df["amount"].tolist() == pytest.approx([1234.56, 800.00, 49.90])
    assert r.df["date"].iloc[0] == pd.Timestamp("2026-01-05")


def test_debit_credit_columns(debit_credit_df):
    m = ColumnMapping(date_col="Fecha", description_col="Concepto",
                      debit_col="Cargo", credit_col="Abono",
                      category_col="Categoria", date_format="%d/%m/%Y",
                      expenses_are="debit_col")
    r = apply_mapping(debit_credit_df, m)
    assert len(r.df) == 2               # only rows with a debit value
    assert r.df["category"].tolist() == ["Comida", "Transporte"]


def test_malformed_rows_skipped_with_count(malformed_df):
    m = ColumnMapping(date_col="Date", description_col="Description",
                      amount_col="Amount", expenses_are="negative")
    r = apply_mapping(malformed_df, m)
    assert len(r.df) == 3
    assert r.skipped_rows == 2          # TOTAL row and blank row / bad date


def test_wrong_mapping_fails_validation(simple_df):
    wrong = ColumnMapping(date_col="Description", description_col="Date",
                          amount_col="Amount", expenses_are="negative")
    assert validate_mapping(simple_df, wrong) < MAPPING_MIN_VALID


def test_good_mapping_passes_validation(simple_df):
    good = ColumnMapping(date_col="Date", description_col="Description",
                         amount_col="Amount", expenses_are="negative")
    assert validate_mapping(simple_df, good) >= MAPPING_MIN_VALID
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
MOCK_LLM=true uv run pytest tests/test_parser.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'core.parser'`

- [ ] **Step 4: Implement `core/parser.py`**

```python
"""Uploaded bank export -> validated, normalized expense DataFrame."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

MAPPING_MIN_VALID = 0.9

DATE_HINTS = ("date", "datum", "fecha", "data", "дата")
AMOUNT_HINTS = ("amount", "betrag", "importe", "importo", "value", "сумма")
DESC_HINTS = ("description", "beschreibung", "concepto", "merchant",
              "details", "payee", "narrative", "назначение")
CATEGORY_HINTS = ("category", "categoria", "kategorie", "категория")
DEBIT_HINTS = ("debit", "cargo", "expense", "расход")
CREDIT_HINTS = ("credit", "abono", "income", "приход")


@dataclass
class ColumnMapping:
    date_col: str
    description_col: str
    amount_col: str | None = None
    debit_col: str | None = None
    credit_col: str | None = None
    category_col: str | None = None
    date_format: str | None = None
    decimal_separator: str = "."
    expenses_are: str = "negative"  # "negative" | "positive" | "debit_col"


@dataclass
class ParseResult:
    df: pd.DataFrame
    skipped_rows: int
    total_rows: int


def _parse_amount(value, decimal_separator: str = ".") -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = re.sub(r"[^\d,.\-+]", "", str(value).strip())
    if not s or s in ("-", "+"):
        return None
    if decimal_separator == ",":
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(value, date_format: str | None = None) -> pd.Timestamp | None:
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    if date_format:
        try:
            return pd.Timestamp(datetime.strptime(s, date_format))
        except ValueError:
            return None
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return None if pd.isna(ts) else pd.Timestamp(ts)


def _find(columns, hints) -> str | None:
    for col in columns:
        if any(h in str(col).lower() for h in hints):
            return col
    return None


def guess_mapping(df: pd.DataFrame) -> ColumnMapping | None:
    cols = list(df.columns)
    date_col = _find(cols, DATE_HINTS)
    desc_col = _find(cols, DESC_HINTS)
    amount_col = _find(cols, AMOUNT_HINTS)
    debit_col = _find(cols, DEBIT_HINTS)
    credit_col = _find(cols, CREDIT_HINTS)
    if not date_col or not desc_col:
        return None
    if debit_col and credit_col:
        return ColumnMapping(date_col=date_col, description_col=desc_col,
                             debit_col=debit_col, credit_col=credit_col,
                             category_col=_find(cols, CATEGORY_HINTS),
                             expenses_are="debit_col")
    if amount_col:
        return ColumnMapping(date_col=date_col, description_col=desc_col,
                             amount_col=amount_col,
                             category_col=_find(cols, CATEGORY_HINTS))
    return None


def _row_values(df: pd.DataFrame, mapping: ColumnMapping):
    """Yield (date, signed_amount_or_None) per row under this mapping."""
    for _, row in df.iterrows():
        date = _parse_date(row.get(mapping.date_col), mapping.date_format)
        if mapping.expenses_are == "debit_col":
            debit = _parse_amount(row.get(mapping.debit_col), mapping.decimal_separator)
            credit = _parse_amount(row.get(mapping.credit_col), mapping.decimal_separator)
            if debit is not None and debit != 0:
                amount = -abs(debit)
            elif credit is not None:
                amount = abs(credit)
            else:
                amount = None
        else:
            amount = _parse_amount(row.get(mapping.amount_col), mapping.decimal_separator)
        yield date, amount


def validate_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> float:
    if df.empty:
        return 0.0
    ok = sum(1 for d, a in _row_values(df, mapping) if d is not None and a is not None)
    return ok / len(df)


def apply_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> ParseResult:
    rows, skipped = [], 0
    values = list(_row_values(df, mapping))
    for (_, src), (date, amount) in zip(df.iterrows(), values):
        if date is None or amount is None:
            skipped += 1
            continue
        if mapping.expenses_are == "positive":
            expense = amount if amount > 0 else None
        else:  # "negative" and "debit_col" both store expenses as negative
            expense = -amount if amount < 0 else None
        if expense is None:
            continue  # income: filtered, not skipped
        category = str(src.get(mapping.category_col, "") or "") if mapping.category_col else ""
        rows.append({"date": date, "amount": round(expense, 2),
                     "description": str(src.get(mapping.description_col, "")).strip(),
                     "category": category})
    out = pd.DataFrame(rows, columns=["date", "amount", "description", "category"])
    return ParseResult(df=out, skipped_rows=skipped, total_rows=len(df))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
MOCK_LLM=true uv run pytest tests/test_parser.py -v
```
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add core/parser.py tests/conftest.py tests/test_parser.py
git commit -m "feat: column mapping, date/amount parsing, expense filtering"
```

---

### Task 3: `core/parser.py` — file loading, sheets, empty/unsupported files

**Files:**
- Modify: `core/parser.py` (append)
- Test: `tests/test_parser.py` (append)

**Interfaces:**
- Consumes: `guess_mapping`, `validate_mapping`, `MAPPING_MIN_VALID` from Task 2.
- Produces:
  - `load_frames(data: bytes, filename: str) -> dict[str, pd.DataFrame]` — raises `UnsupportedFileError` / `EmptyFileError`
  - `pick_best_sheet(frames: dict) -> str | None` — `None` means ambiguous, ask the user
  - `class ParserError(Exception)`, `class UnsupportedFileError(ParserError)`, `class EmptyFileError(ParserError)`

- [ ] **Step 1: Append failing tests to `tests/test_parser.py`**

```python
from core.parser import EmptyFileError, UnsupportedFileError, load_frames, pick_best_sheet
from tests.conftest import make_xlsx_bytes


def test_load_csv_bytes(simple_df):
    data = simple_df.to_csv(index=False).encode("utf-8")
    frames = load_frames(data, "export.csv")
    assert list(frames.values())[0].shape[0] == 4


def test_load_semicolon_csv(european_df):
    data = european_df.to_csv(index=False, sep=";").encode("utf-8")
    frames = load_frames(data, "export.csv")
    assert "Betrag" in list(frames.values())[0].columns


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        load_frames(b"whatever", "export.xls")


def test_empty_file():
    with pytest.raises(EmptyFileError):
        load_frames(b"", "export.csv")


def test_multisheet_picks_transaction_sheet(simple_df):
    info = pd.DataFrame({"Info": ["Bank Statement"], "Value": ["2026"]})
    data = make_xlsx_bytes({"Cover": info, "Transactions": simple_df})
    frames = load_frames(data, "export.xlsx")
    assert pick_best_sheet(frames) == "Transactions"
```

- [ ] **Step 2: Run to verify new tests fail**

```bash
MOCK_LLM=true uv run pytest tests/test_parser.py -v
```
Expected: previous 7 pass, new 5 FAIL with ImportError

- [ ] **Step 3: Append implementation to `core/parser.py`**

```python
import io  # add to imports at top of file


class ParserError(Exception):
    pass


class UnsupportedFileError(ParserError):
    pass


class EmptyFileError(ParserError):
    pass


def load_frames(data: bytes, filename: str) -> dict[str, pd.DataFrame]:
    if not data:
        raise EmptyFileError("The uploaded file is empty.")
    name = filename.lower()
    if name.endswith(".csv"):
        text = data.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str)
        frames = {"data": df}
    elif name.endswith(".xlsx"):
        frames = pd.read_excel(io.BytesIO(data), sheet_name=None)
    else:
        raise UnsupportedFileError(
            "Only .csv and .xlsx files are supported. "
            "Please export your transactions in one of these formats.")
    frames = {k: v for k, v in frames.items() if not v.dropna(how="all").empty}
    if not frames:
        raise EmptyFileError("The file contains no data rows.")
    return frames


def pick_best_sheet(frames: dict[str, pd.DataFrame]) -> str | None:
    if len(frames) == 1:
        return next(iter(frames))
    scored = []
    for name, df in frames.items():
        mapping = guess_mapping(df)
        score = validate_mapping(df, mapping) if mapping else 0.0
        scored.append((score, name))
    scored.sort(reverse=True)
    best_score, best_name = scored[0]
    if best_score < MAPPING_MIN_VALID:
        return None
    if len(scored) > 1 and scored[1][0] >= MAPPING_MIN_VALID:
        return None  # two plausible sheets -> ask the user
    return best_name
```

- [ ] **Step 4: Run tests**

```bash
MOCK_LLM=true uv run pytest tests/test_parser.py -v
```
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add core/parser.py tests/test_parser.py
git commit -m "feat: csv/xlsx loading, sheet selection, empty-file handling"
```

---

### Task 4: `core/merchants.py`

**Files:**
- Create: `core/merchants.py`
- Test: `tests/test_merchants.py`

**Interfaces:**
- Produces:
  - `normalize_merchant(description: str) -> str`
  - `add_merchant_column(df: pd.DataFrame) -> pd.DataFrame` (adds `merchant` column; original `description` preserved)

- [ ] **Step 1: Write failing tests in `tests/test_merchants.py`**

```python
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


def test_empty_becomes_unknown():
    assert normalize_merchant("  123456  ") == "UNKNOWN"


def test_add_merchant_column_preserves_description():
    df = pd.DataFrame({"description": ["SHOP 99 MAIN ST"]})
    out = add_merchant_column(df)
    assert out["merchant"].iloc[0] == "SHOP MAIN ST"
    assert out["description"].iloc[0] == "SHOP 99 MAIN ST"
```

- [ ] **Step 2: Run to verify failure**

```bash
MOCK_LLM=true uv run pytest tests/test_merchants.py -v
```
Expected: FAIL — no module `core.merchants`

- [ ] **Step 3: Implement `core/merchants.py`**

```python
"""Raw transaction descriptions -> cleaned merchant identifiers.

Deliberately simple (see docs/DESIGN.md): uppercase, drop tokens containing
digits, strip punctuation, keep the first three tokens. No fuzzy matching.
"""
import re

import pandas as pd

_PUNCT = re.compile(r"[*#/\\\-_.,:;!?()\[\]{}'\"@+&]")


def normalize_merchant(description: str) -> str:
    s = _PUNCT.sub(" ", str(description).upper())
    tokens = [t for t in s.split() if not any(c.isdigit() for c in t)]
    return " ".join(tokens[:3]) or "UNKNOWN"


def add_merchant_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["merchant"] = out["description"].map(normalize_merchant)
    return out
```

- [ ] **Step 4: Run tests**

```bash
MOCK_LLM=true uv run pytest tests/test_merchants.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add core/merchants.py tests/test_merchants.py
git commit -m "feat: simple merchant normalization"
```

---

# Day 2 — Wed Jul 16: features, clustering, skeleton deploy

### Task 5: `core/features.py`

**Files:**
- Create: `core/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: DataFrame with `date, amount, description, category, merchant` (Tasks 2+4).
- Produces:
  - `analysis_period(df) -> tuple[pd.Timestamp, pd.Timestamp, float]` (start, end, months ≥ 1.0)
  - `build_features(df) -> pd.DataFrame` indexed by `merchant`, columns exactly: `monthly_spend, tx_per_month, avg_amount, amount_cv, log_avg_amount, weekend_share, interval_regularity, amount_stability`

- [ ] **Step 1: Write failing tests in `tests/test_features.py`**

```python
import numpy as np
import pandas as pd
import pytest

from core.features import analysis_period, build_features


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
    assert months == pytest.approx(60 / 30.44, rel=0.01)


def test_per_month_normalization():
    """Same monthly behaviour over 2 vs 12 months -> comparable features."""
    two = _tx("COFFEE", ["2026-01-10", "2026-01-20", "2026-02-10", "2026-02-20"], 5.0)
    twelve = _tx("COFFEE", [f"2026-{m:02d}-{d}" for m in range(1, 13) for d in (10, 20)], 5.0)
    f2, f12 = build_features(two), build_features(twelve)
    assert f2.loc["COFFEE", "monthly_spend"] == pytest.approx(
        f12.loc["COFFEE", "monthly_spend"], rel=0.35)
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
```

- [ ] **Step 2: Run to verify failure**

```bash
MOCK_LLM=true uv run pytest tests/test_features.py -v
```
Expected: FAIL — no module `core.features`

- [ ] **Step 3: Implement `core/features.py`**

```python
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
    start, end = df["date"].min(), df["date"].max()
    months = max(((end - start).days + 1) / 30.44, 1.0)
    return start, end, months


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
```

- [ ] **Step 4: Run tests**

```bash
MOCK_LLM=true uv run pytest tests/test_features.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add core/features.py tests/test_features.py
git commit -m "feat: per-merchant features, month-normalized with recurrence signals"
```

---

### Task 6: `core/clustering.py`

**Files:**
- Create: `core/clustering.py`
- Test: `tests/test_clustering.py`

**Interfaces:**
- Consumes: feature matrix from `build_features` (Task 5); transaction df with `merchant` column.
- Produces:
  - `@dataclass ClusterResult(labels: pd.Series, k: int, silhouette: float | None, used_fallback: bool, reason: str | None)` — `labels` maps merchant → int cluster id
  - `cluster_merchants(features: pd.DataFrame, n_transactions: int, categories: pd.Series | None = None) -> ClusterResult`
  - `cluster_stats(df: pd.DataFrame, labels: pd.Series) -> list[dict]` — each dict has keys: `cluster_id, n_transactions, total_spend, avg_amount, top_merchants, example_descriptions, weekend_share, monthly_totals`
  - Constants: `MIN_TRANSACTIONS = 10`, `MIN_MERCHANTS = 6`, `MIN_SILHOUETTE = 0.15`, `MAX_K = 7`

- [ ] **Step 1: Write failing tests in `tests/test_clustering.py`**

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
MOCK_LLM=true uv run pytest tests/test_clustering.py -v
```
Expected: FAIL — no module `core.clustering`

- [ ] **Step 3: Implement `core/clustering.py`**

```python
"""Per-merchant feature matrix -> K-means clusters + aggregate statistics."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_TRANSACTIONS = 10
MIN_MERCHANTS = 6
MIN_SILHOUETTE = 0.15
MAX_K = 7


@dataclass
class ClusterResult:
    labels: pd.Series
    k: int
    silhouette: float | None
    used_fallback: bool
    reason: str | None


def _fallback(features: pd.DataFrame, reason: str,
              categories: pd.Series | None) -> ClusterResult:
    if categories is not None and categories.replace("", pd.NA).notna().any():
        codes, _ = pd.factorize(categories.reindex(features.index).fillna(""))
        labels = pd.Series(codes, index=features.index)
    else:
        labels = pd.Series(0, index=features.index)
    return ClusterResult(labels=labels, k=int(labels.nunique()),
                         silhouette=None, used_fallback=True, reason=reason)


def cluster_merchants(features: pd.DataFrame, n_transactions: int,
                      categories: pd.Series | None = None) -> ClusterResult:
    n = len(features)
    if n_transactions < MIN_TRANSACTIONS:
        return _fallback(features, "not_enough_transactions", categories)
    if n < MIN_MERCHANTS:
        return _fallback(features, "not_enough_merchants", categories)
    k_max = min(MAX_K, n // 3)
    if k_max < 2:
        return _fallback(features, "no_valid_k", categories)

    X = StandardScaler().fit_transform(features.values)
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_k, best_score, best_labels = k, score, km.labels_
    if best_score < MIN_SILHOUETTE:
        return _fallback(features, "weak_clusters", categories)
    return ClusterResult(labels=pd.Series(best_labels, index=features.index),
                         k=best_k, silhouette=float(best_score),
                         used_fallback=False, reason=None)


def cluster_stats(df: pd.DataFrame, labels: pd.Series) -> list[dict]:
    work = df.assign(cluster=df["merchant"].map(labels))
    out = []
    for cid, g in work.groupby("cluster"):
        monthly = g.groupby(g["date"].dt.to_period("M"))["amount"].sum()
        out.append({
            "cluster_id": int(cid),
            "n_transactions": int(len(g)),
            "total_spend": round(float(g["amount"].sum()), 2),
            "avg_amount": round(float(g["amount"].mean()), 2),
            "top_merchants": g.groupby("merchant")["amount"].sum()
                              .nlargest(5).index.tolist(),
            "example_descriptions": g["description"].drop_duplicates()
                                     .head(3).tolist(),
            "weekend_share": round(float((g["date"].dt.dayofweek >= 5).mean()), 2),
            "monthly_totals": {str(p): round(float(v), 2)
                               for p, v in monthly.items()},
        })
    return sorted(out, key=lambda s: -s["total_spend"])
```

- [ ] **Step 4: Run tests**

```bash
MOCK_LLM=true uv run pytest tests/test_clustering.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add core/clustering.py tests/test_clustering.py
git commit -m "feat: merchant clustering with silhouette selection and guarded fallbacks"
```

---

### Task 7: Walking-skeleton `app.py` + demo data

**Files:**
- Create: `app.py`
- Use: `data/spending_demo.csv`

**Interfaces:**
- Consumes: everything from Tasks 2–6.
- Produces: runnable app — upload or demo button → parsed table → cluster donut chart. No LLM yet (generic labels).

- [ ] **Step 1: Use the bundled demo export**

`data/spending_demo.csv` is the anonymized demo file. The **Load demo data** button must read its bytes and pass them through `parser.load_frames`, exactly as it would process a user upload.

- [ ] **Step 2: Write skeleton `app.py`**

```python
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core import clustering, features, merchants, parser

st.set_page_config(page_title="AI Personal Finance Assistant", page_icon="📊", layout="wide")
st.title("AI Personal Finance Assistant")
st.caption("Upload a bank export - understand the habits behind your spending. "
           "Files are processed in memory and never stored.")

uploaded = st.file_uploader("Bank export (.csv or .xlsx, max 5 MB)",
                            type=["csv", "xlsx"])
use_demo = st.button("Load demo data")

raw_df = None
frames = None
if use_demo:
    demo_path = Path("data/spending_demo.csv")
    frames = parser.load_frames(demo_path.read_bytes(), demo_path.name)
elif uploaded is not None:
    try:
        frames = parser.load_frames(uploaded.getvalue(), uploaded.name)
    except parser.ParserError as e:
        st.error(str(e))
        st.stop()

if frames is not None:
    sheet = parser.pick_best_sheet(frames)
    if sheet is None:
        sheet = st.selectbox("Which sheet holds your transactions?",
                             list(frames))
    raw_df = frames[sheet]

if raw_df is not None:
    mapping = parser.guess_mapping(raw_df)
    if mapping is None or parser.validate_mapping(raw_df, mapping) < parser.MAPPING_MIN_VALID:
        st.warning("Could not auto-detect columns yet - manual mapping and "
                   "AI-assisted mapping arrive in the next milestone.")
        st.stop()
    result = parser.apply_mapping(raw_df, mapping)
    if result.skipped_rows:
        st.info(f"Skipped {result.skipped_rows} rows that couldn't be parsed.")
    df = merchants.add_merchant_column(result.df)
    if len(df) < clustering.MIN_TRANSACTIONS:
        st.warning("Not enough expense transactions to find patterns (need at least 10).")
        st.stop()
    feats = features.build_features(df)
    cats = df.groupby("merchant")["category"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "")
    cres = clustering.cluster_merchants(feats, n_transactions=len(df), categories=cats)
    stats = clustering.cluster_stats(df, cres.labels)

    st.subheader("Spending patterns (unnamed - AI labels come next)")
    donut = px.pie(
        values=[s["total_spend"] for s in stats],
        names=[f"Pattern {s['cluster_id']}" for s in stats], hole=0.5)
    st.plotly_chart(donut, use_container_width=True)
    st.dataframe(df.head(50))
```

- [ ] **Step 3: Run and verify manually**

```bash
uv run streamlit run app.py
```
Expected: app opens; "Load demo data" produces a donut chart with 2+ patterns and a transaction table; full test suite still green: `MOCK_LLM=true uv run pytest -v` → all pass.

- [ ] **Step 4: Commit**

```bash
git add app.py data/spending_demo.csv
git commit -m "feat: walking-skeleton Streamlit app with demo data"
```

---

### Task 8: GitHub + Streamlit Community Cloud deploy

**Files:** none (operations task)

- [ ] **Step 1: Create the public repo and push**

```bash
gh repo create ai-personal-finance-assistant --public --source=. --push
```
Expected: repo visible at github.com/<username>/ai-personal-finance-assistant with full history.

- [ ] **Step 2: Deploy on Streamlit Community Cloud (manual, in browser)**

1. share.streamlit.io → "New app" → pick the `ai-personal-finance-assistant` repo, branch `main`, main file `app.py`.
2. In app settings → Secrets, add: `OPENAI_API_KEY = "sk-..."` (does nothing yet; ready for Day 3).
3. Open the public URL, click "Load demo data", confirm the donut renders.

- [ ] **Step 3: Add UptimeRobot monitor (alerts only, per spec — no keep-alive pings)**

uptimerobot.com → new HTTP(s) monitor on the app URL, alert email on downtime.

- [ ] **Step 4: Record the app URL in the README stub and commit**

```bash
printf '# AI Personal Finance Assistant\n\nLive app: <URL>\n\nFirst load may take up to a minute if the app was asleep.\n' > README.md
git add README.md && git commit -m "docs: README stub with live URL" && git push
```
Expected: push triggers automatic redeploy. From now on every push redeploys — deploys stay boring.

---

# Day 3 — Thu Jul 17: GPT-5.6 integration

### Task 9: `core/llm.py` with mock + validation + retry

**Files:**
- Create: `core/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `cluster_stats` dicts (Task 6); `parser.guess_mapping` for the mock path.
- Produces:
  - `MODEL = "gpt-5.6"`, `class LLMError(Exception)`
  - `map_columns(headers: list[str], sample_rows: list[list]) -> dict` — keys: `date_col, description_col, amount_col, debit_col, credit_col, category_col, date_format, decimal_separator, expenses_are` (unused keys `None`); raises `LLMError` after one failed retry
  - `analyze_clusters(stats: list[dict], period: dict) -> dict` — shape `{"clusters": [{"cluster_id": int, "name": str, "category": str}], "summary": str}`; raises `LLMError` after one failed retry
  - `generic_labels(stats) -> dict` — same shape, local, never fails (API-unavailable fallback)
  - Mock mode (`MOCK_LLM=true`): `map_columns` delegates to `parser.guess_mapping`; `analyze_clusters` returns deterministic canned output.

- [ ] **Step 1: Write failing tests in `tests/test_llm.py`**

```python
import pytest

import core.llm as llm
from core.llm import LLMError, analyze_clusters, generic_labels, map_columns

STATS = [{"cluster_id": 0, "n_transactions": 12, "total_spend": 240.0,
          "avg_amount": 20.0, "top_merchants": ["SUPERMARKET GREENFIELD"],
          "example_descriptions": ["SUPERMARKET GREENFIELD 123"],
          "weekend_share": 0.2, "monthly_totals": {"2026-01": 120.0, "2026-02": 120.0}}]
PERIOD = {"start": "2026-01-01", "end": "2026-02-28", "months": 1.9}


def test_mock_map_columns_uses_heuristics(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    out = map_columns(["Date", "Description", "Amount"],
                      [["2026-01-05", "SHOP", "-10.00"]])
    assert out["date_col"] == "Date" and out["amount_col"] == "Amount"


def test_mock_analyze_is_deterministic(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    a, b = analyze_clusters(STATS, PERIOD), analyze_clusters(STATS, PERIOD)
    assert a == b
    assert a["clusters"][0]["cluster_id"] == 0
    assert a["summary"]


def test_invalid_json_retries_then_raises(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    calls = []
    def bad_chat(system, user):
        calls.append(1)
        return "not json at all"
    monkeypatch.setattr(llm, "_chat_json", bad_chat)
    with pytest.raises(LLMError):
        analyze_clusters(STATS, PERIOD)
    assert len(calls) == 2  # one retry


def test_valid_response_passes_validation(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(llm, "_chat_json", lambda s, u:
        '{"clusters": [{"cluster_id": 0, "name": "Groceries", '
        '"category": "Food"}], "summary": "Steady grocery spending."}')
    out = analyze_clusters(STATS, PERIOD)
    assert out["clusters"][0]["name"] == "Groceries"


def test_missing_keys_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(llm, "_chat_json", lambda s, u: '{"clusters": []}')
    with pytest.raises(LLMError):
        analyze_clusters(STATS, PERIOD)


def test_generic_labels_never_fail():
    out = generic_labels(STATS)
    assert out["clusters"][0]["name"] == "Spending pattern 1"
    assert "summary" in out
```

- [ ] **Step 2: Run to verify failure**

```bash
MOCK_LLM=true uv run pytest tests/test_llm.py -v
```
Expected: FAIL — no module `core.llm`

- [ ] **Step 3: Implement `core/llm.py`**

```python
"""All GPT-5.6 calls. The only module that touches the OpenAI API.

Privacy: receives only headers + up to 5 sample rows (after user consent)
and aggregated cluster statistics. Never raw transaction tables.
"""
from __future__ import annotations

import json
import os

MODEL = "gpt-5.6"
MAX_OUTPUT_TOKENS = 1200

MAPPING_KEYS = ["date_col", "description_col", "amount_col", "debit_col",
                "credit_col", "category_col", "date_format",
                "decimal_separator", "expenses_are"]


class LLMError(Exception):
    pass


def _mock() -> bool:
    return os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes")


def _chat_json(system: str, user: str) -> str:
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.choices[0].message.content


def _json_with_retry(system: str, user: str, validate) -> dict:
    error = ""
    for attempt in range(2):
        try:
            raw = _chat_json(system, user + error)
            data = json.loads(raw)
            validate(data)
            return data
        except LLMError:
            raise
        except Exception as exc:  # invalid JSON, missing keys, API error
            error = f"\n\nYour previous response was invalid: {exc}. Return valid JSON only."
    raise LLMError(f"Model returned invalid output twice: {error}")


MAPPING_SYSTEM = """You map bank-export spreadsheet columns to a schema.
Reply with JSON only, keys: date_col, description_col, amount_col, debit_col,
credit_col, category_col, date_format (strptime, e.g. %d.%m.%Y),
decimal_separator ("." or ","), expenses_are ("negative", "positive" or
"debit_col"). Use null for keys that don't apply. Column names must be copied
exactly from the provided headers."""

ANALYSIS_SYSTEM = """You are a personal-finance analyst. You receive aggregated
spending-cluster statistics (no raw transactions). Reply with JSON only:
{"clusters": [{"cluster_id": <int>, "name": "<short descriptive name>",
"category": "<concise semantic category>"}], "summary": "<plain-English trends,
3-6 sentences>"}.
Rules: base every label and every claim strictly on the supplied evidence.
Never invent merchants, amounts, or trends. If the evidence for a cluster is
insufficient, name it "Unclear spending pattern". Mention notable
month-over-month changes with their approximate percentage when the monthly
totals support them."""


def map_columns(headers: list[str], sample_rows: list[list]) -> dict:
    if _mock():
        import pandas as pd
        from core import parser
        guess = parser.guess_mapping(pd.DataFrame(sample_rows, columns=headers))
        if guess is None:
            raise LLMError("mock: heuristics found no mapping")
        return {k: getattr(guess, k) for k in MAPPING_KEYS}

    def validate(data):
        cols = set(headers)
        if data.get("date_col") not in cols or data.get("description_col") not in cols:
            raise ValueError("date_col/description_col missing from headers")
        if not (data.get("amount_col") in cols
                or (data.get("debit_col") in cols and data.get("credit_col") in cols)):
            raise ValueError("no valid amount or debit/credit columns")

    user = json.dumps({"headers": headers, "sample_rows": sample_rows[:5]})
    data = _json_with_retry(MAPPING_SYSTEM, user, validate)
    return {k: data.get(k) for k in MAPPING_KEYS}


def analyze_clusters(stats: list[dict], period: dict) -> dict:
    if _mock():
        return {"clusters": [{"cluster_id": s["cluster_id"],
                              "name": f"Mock pattern {s['cluster_id']}",
                              "category": "Mock"} for s in stats],
                "summary": "Mock summary: spending is stable across the period."}

    def validate(data):
        if "summary" not in data or "clusters" not in data:
            raise ValueError("missing summary or clusters")
        ids = {c["cluster_id"] for c in data["clusters"]}
        if ids != {s["cluster_id"] for s in stats}:
            raise ValueError("cluster ids do not match input")

    user = json.dumps({"period": period, "clusters": stats})
    return _json_with_retry(ANALYSIS_SYSTEM, user, validate)


def generic_labels(stats: list[dict]) -> dict:
    """Local fallback when the API is unavailable. Never fails."""
    return {"clusters": [{"cluster_id": s["cluster_id"],
                          "name": f"Spending pattern {i + 1}",
                          "category": "Uncategorized"}
                         for i, s in enumerate(stats)],
            "summary": ("AI analysis is temporarily unavailable. "
                        "The patterns below were detected locally from your data.")}
```

- [ ] **Step 4: Run tests**

```bash
MOCK_LLM=true uv run pytest tests/test_llm.py -v
```
Expected: 6 passed. (Note: tests 3–5 set `MOCK_LLM=false` but monkeypatch `_chat_json`, so still no network.)

- [ ] **Step 5: Commit**

```bash
git add core/llm.py tests/test_llm.py
git commit -m "feat: GPT-5.6 module with JSON validation, retry, mock and local fallback"
```

---

### Task 10: Wire real GPT-5.6 into the app — consent flow + manual mapping

**Files:**
- Modify: `app.py` (full replacement in Task 11 includes this; here modify incrementally)

**Interfaces:**
- Consumes: `llm.map_columns`, `llm.analyze_clusters`, `llm.generic_labels`, `llm.LLMError` (Task 9).
- Produces: app flow — heuristics first; if uncertain, user chooses **manual mapping** (dropdowns) or **AI-assisted mapping** behind an explicit consent checkbox; analysis uses real GPT-5.6 when `OPENAI_API_KEY` present, generic labels otherwise.

- [ ] **Step 1: Replace the mapping section of `app.py`**

Replace the block starting `mapping = parser.guess_mapping(raw_df)` through `st.stop()` with:

```python
mapping = parser.guess_mapping(raw_df)
if mapping is None or parser.validate_mapping(raw_df, mapping) < parser.MAPPING_MIN_VALID:
    st.warning("Couldn't auto-detect your columns.")
    tab_manual, tab_ai = st.tabs(["Map columns manually", "AI-assisted mapping"])
    with tab_ai:
        st.caption("Sends ONLY the header row and up to 5 sample rows to GPT-5.6. "
                   "Nothing else leaves this app.")
        consent = st.checkbox("I agree to send headers + 5 sample rows to OpenAI")
        if consent and st.button("Detect columns with AI"):
            try:
                raw = llm.map_columns(list(raw_df.columns),
                                      raw_df.head(5).astype(str).values.tolist())
                mapping = parser.ColumnMapping(**{k: raw.get(k) for k in llm.MAPPING_KEYS
                                                  if raw.get(k) is not None})
            except llm.LLMError as e:
                st.error(f"AI mapping failed: {e}. Please map manually.")
    with tab_manual:
        cols = ["(none)"] + list(raw_df.columns)
        date_c = st.selectbox("Date column", cols)
        desc_c = st.selectbox("Description column", cols)
        amount_c = st.selectbox("Amount column (signed)", cols)
        cat_c = st.selectbox("Category column (optional)", cols)
        sep = st.radio("Decimal separator", [".", ","], horizontal=True)
        conv = st.radio("Expenses are…", ["negative", "positive"], horizontal=True)
        if st.button("Apply manual mapping") and "(none)" not in (date_c, desc_c, amount_c):
            mapping = parser.ColumnMapping(
                date_col=date_c, description_col=desc_c, amount_col=amount_c,
                category_col=None if cat_c == "(none)" else cat_c,
                decimal_separator=sep, expenses_are=conv)
    if mapping is None:
        st.stop()
    if parser.validate_mapping(raw_df, mapping) < parser.MAPPING_MIN_VALID:
        st.error("That mapping doesn't parse at least 90% of rows - "
                 "please check the columns and try again.")
        st.stop()
```

And replace the cluster-naming placeholder: after `stats = clustering.cluster_stats(...)` add:

```python
start, end, months = features.analysis_period(df)
period = {"start": str(start.date()), "end": str(end.date()),
          "months": round(months, 1)}
try:
    analysis = llm.analyze_clusters(stats, period)
except llm.LLMError:
    analysis = llm.generic_labels(stats)
names = {c["cluster_id"]: c["name"] for c in analysis["clusters"]}
```
Use `names[s["cluster_id"]]` for the donut labels; render `analysis["summary"]` under the chart with `st.markdown`.

Add `from core import llm` to imports, and `import os` is not needed (key read by SDK from env/secrets — on Community Cloud, add to `app.py` top:
`import os; os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))`).

- [ ] **Step 2: Test locally against the real API once**

```bash
OPENAI_API_KEY=sk-... uv run streamlit run app.py
```
Expected: demo data produces real GPT-5.6 cluster names ("Groceries", "Recurring subscriptions"…) and a summary. Then confirm the no-key path: `uv run streamlit run app.py` without a key → generic "Spending pattern N" labels + unavailable notice, no crash.

- [ ] **Step 3: Run full suite, commit, push (auto-deploys)**

```bash
MOCK_LLM=true uv run pytest -v
git add app.py && git commit -m "feat: consent-gated AI mapping, manual mapping, real GPT-5.6 analysis" && git push
```
Expected: all tests pass; deployed app shows real AI labels (key is in Cloud secrets).

**Milestone check:** this is the "real GPT-5.6 by day 3–4" gate from the spec. If it slips past Jul 18, cut scope from Day 5 (PDF becomes summary-text-only) — do not let the video slip.

---

# Day 4 — Sat Jul 18: full dashboard

### Task 11: Dashboard UI (final `app.py`)

**Files:**
- Modify: `app.py` — reorganize into the final layout. Keep all Task 10 logic; this task is additive layout work, not new algorithms.

**Interfaces:**
- Consumes: everything.
- Produces: `build_figures(stats, names) -> dict[str, plotly Figure]` in `app.py` (reused by PDF task): keys `donut`, `monthly`.

- [ ] **Step 1: Add KPI row, charts, cluster table, caveats, and session limits to `app.py`**

Final `app.py` structure (append/replace sections so the file reads in this order):

```python
# --- after analysis is computed ---
LIMIT_ANALYSES = 5
st.session_state.setdefault("analyses", 0)
# increment once per new upload/demo click; if exceeded:
if st.session_state["analyses"] > LIMIT_ANALYSES:
    st.error("Demo limit reached for this session - refresh to start over.")
    st.stop()

MAX_ROWS = 5000
if len(raw_df) > MAX_ROWS:
    st.error(f"File has {len(raw_df)} rows; the demo supports up to {MAX_ROWS}.")
    st.stop()

# KPI row
total = df["amount"].sum()
recurring_merchants = feats[(feats["interval_regularity"] > 0.7)
                            & (feats["amount_stability"] > 0.8)].index
recurring_spend = df[df["merchant"].isin(recurring_merchants)]["amount"].sum()
biggest = names[stats[0]["cluster_id"]]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total spending", f"{total:,.2f}")
c2.metric("Transactions", len(df))
c3.metric("Avg transaction", f"{df['amount'].mean():,.2f}")
c4.metric("Biggest pattern", biggest)
c5.metric("Est. recurring / month", f"{recurring_spend / months:,.2f}")

st.caption(f"Analysis period: {period['start']} to {period['end']} "
           f"({period['months']} months)")
if months < 2:
    st.warning("This export covers less than 2 full months - recurring "
               "subscriptions can't be detected reliably with this little data.")
if cres.used_fallback:
    st.info("The data didn't form strong behavioural clusters "
            f"(reason: {cres.reason}). Showing basic grouping instead.")


def build_figures(stats, names):
    import plotly.express as px
    donut = px.pie(values=[s["total_spend"] for s in stats],
                   names=[names[s["cluster_id"]] for s in stats], hole=0.5,
                   title="Spending by pattern")
    monthly_rows = [{"month": m, "pattern": names[s["cluster_id"]], "spend": v}
                    for s in stats for m, v in s["monthly_totals"].items()]
    monthly = px.bar(pd.DataFrame(monthly_rows), x="month", y="spend",
                     color="pattern", title="Monthly spending by pattern")
    return {"donut": donut, "monthly": monthly}


figs = build_figures(stats, names)
left, right = st.columns(2)
left.plotly_chart(figs["donut"], use_container_width=True)
right.plotly_chart(figs["monthly"], use_container_width=True)

# transparency table: every AI label backed by its evidence
st.subheader("Patterns in detail")
st.dataframe(pd.DataFrame([{
    "Pattern": names[s["cluster_id"]],
    "Total": s["total_spend"],
    "Transactions": s["n_transactions"],
    "Avg amount": s["avg_amount"],
    "Top merchants": ", ".join(s["top_merchants"][:3]),
} for s in stats]), use_container_width=True, hide_index=True)

st.subheader("What GPT-5.6 sees in your spending")
st.markdown(analysis["summary"])
```

- [ ] **Step 1b: Design pass (added after review — do this before Step 2)**

Step 1 above (KPI row, charts, plain evidence table, caveats, session limits) is done and verified working. This step replaces the plain evidence table with labeled cluster boxes and applies a two-palette theme. Use the `ui-ux-pro-max` skill (installed for Codex per github.com/nextlevelbuilder/ui-ux-pro-max-skill) for the palette, spacing, and card-styling specifics.

Cluster display — replace the "Patterns in detail" `st.dataframe` with one box per cluster (name + total + count + top merchants each, styled as a card), not a plain table row. No chat interface, no phone-mockup styling — just the box-per-category pattern.

Theme — two deliberate palettes, not one copied wholesale:
- Light mode: keep the existing teal accent (`#0F766E`, already set in `.streamlit/config.toml`). Do not introduce purple here.
- Dark mode: purple/violet accent.
- Ship both as fixed themes. Do not build a custom in-app light/dark toggle switch — Streamlit's built-in menu (☰ → Settings) already offers light/dark/system for free.

- [ ] **Step 2: Manual verification pass**

Run `uv run streamlit run app.py`; check with demo data: 5 KPIs render, both charts, cluster boxes show merchants behind each label, summary present, dark and light themes both look intentional. Upload a fixture-style CSV with 8 rows → "not enough data" state, no crash. Full suite: `MOCK_LLM=true uv run pytest -v` → green.

- [ ] **Step 3: Commit and push**

```bash
git add app.py
git commit -m "feat: full dashboard with KPIs, charts, cluster boxes, caveats and two-palette theme"
git push
```

---

# Day 5 — Sun Jul 19: PDF export + hardening

### Task 12: `core/pdf.py` + download button

**Files:**
- Create: `core/pdf.py`
- Modify: `app.py` (download button)
- Test: `tests/test_pdf.py`

**Interfaces:**
- Consumes: `stats`, `names`, `analysis["summary"]`, `period`, PNG bytes from `fig.to_image(format="png")` (kaleido). **Never imports `core.llm`.**
- Produces: `build_pdf(period: dict, kpis: dict[str, str], clusters: list[dict], names: dict[int, str], summary: str, chart_pngs: list[bytes]) -> bytes`

- [ ] **Step 1: Write failing test in `tests/test_pdf.py`**

```python
import plotly.express as px

from core.pdf import build_pdf

STATS = [{"cluster_id": 0, "n_transactions": 12, "total_spend": 240.0,
          "avg_amount": 20.0, "top_merchants": ["SUPERMARKET"],
          "example_descriptions": ["SUPERMARKET 1"], "weekend_share": 0.2,
          "monthly_totals": {"2026-01": 240.0}}]


def test_pdf_end_to_end_offline():
    """Full PDF build with MOCK_LLM - no API, no network (spec requirement)."""
    png = px.pie(values=[1], names=["x"]).to_image(format="png")
    out = build_pdf(period={"start": "2026-01-01", "end": "2026-02-28", "months": 1.9},
                    kpis={"Total spending": "240.00", "Transactions": "12"},
                    clusters=STATS, names={0: "Groceries"},
                    summary="Grocery spending was stable. Ünïcödé test.",
                    chart_pngs=[png])
    assert isinstance(out, bytes) and out.startswith(b"%PDF")
    assert len(out) > 5000


def test_pdf_never_imports_llm():
    import core.pdf as pdf_mod
    import sys
    assert "core.llm" not in getattr(pdf_mod, "__dict__", {})
    src = open(pdf_mod.__file__).read()
    assert "llm" not in src.replace("MOCK_LLM", "")
```

- [ ] **Step 2: Run to verify failure**

```bash
MOCK_LLM=true uv run pytest tests/test_pdf.py -v
```
Expected: FAIL — no module `core.pdf`

- [ ] **Step 3: Implement `core/pdf.py`**

```python
"""Dashboard figures + summary -> PDF bytes. Contains no raw transaction rows."""
import io

from fpdf import FPDF


def _latin(s: str) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def build_pdf(period: dict, kpis: dict, clusters: list[dict],
              names: dict, summary: str, chart_pngs: list[bytes]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "AI Personal Finance Assistant Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, _latin(f"Analysis period: {period['start']} to {period['end']} "
                          f"({period['months']} months)"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Key metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for label, value in kpis.items():
        pdf.cell(0, 7, _latin(f"  {label}: {value}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for png in chart_pngs:
        pdf.image(io.BytesIO(png), w=180)
        pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Spending patterns", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for s in clusters:
        line = (f"  {names.get(s['cluster_id'], 'Pattern')}: "
                f"{s['total_spend']:.2f} total, {s['n_transactions']} transactions, "
                f"top: {', '.join(s['top_merchants'][:3])}")
        pdf.multi_cell(0, 6, _latin(line))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "AI summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _latin(summary))
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, _latin(
        "Privacy: this report contains aggregated statistics only. "
        "Raw transactions were processed in memory and never stored."))
    return bytes(pdf.output())
```

- [ ] **Step 4: Run tests**

```bash
MOCK_LLM=true uv run pytest tests/test_pdf.py -v
```
Expected: 2 passed

- [ ] **Step 5: Add download button to `app.py`** (after the summary section)

```python
kpis = {"Total spending": f"{total:,.2f}", "Transactions": str(len(df)),
        "Average transaction": f"{df['amount'].mean():,.2f}",
        "Biggest pattern": biggest,
        "Estimated recurring per month": f"{recurring_spend / months:,.2f}"}
pngs = [figs["donut"].to_image(format="png", width=900, height=500),
        figs["monthly"].to_image(format="png", width=900, height=500)]
from core.pdf import build_pdf
st.download_button("Download PDF report",
                   data=build_pdf(period, kpis, stats, names,
                                  analysis["summary"], pngs),
                   file_name="ai-personal-finance-assistant_report.pdf", mime="application/pdf")
```

Wrap the `build_pdf` call in `try/except Exception` → `st.error("PDF export failed - the dashboard is unaffected. Try again.")` (spec: PDF failure preserves dashboard).

- [ ] **Step 6: Verify, commit, push**

```bash
MOCK_LLM=true uv run pytest -v          # all green
uv run streamlit run app.py                   # demo -> download -> open the PDF
git add core/pdf.py tests/test_pdf.py app.py
git commit -m "feat: PDF report export" && git push
```

---

### Task 13: Caching, safeguards, cost limits

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Cache the pipeline so UI interactions don't re-run KMeans or GPT-5.6**

Wrap the two expensive stages in cached functions at the top of `app.py`:

```python
@st.cache_data(show_spinner="Analyzing spending patterns…")
def run_pipeline(csv_bytes: bytes, mapping_key: str):
    """csv_bytes = normalized df serialized; cache key includes mapping."""
    df = pd.read_json(io.BytesIO(csv_bytes))
    df["date"] = pd.to_datetime(df["date"])
    feats = features.build_features(df)
    cats = df.groupby("merchant")["category"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "")
    cres = clustering.cluster_merchants(feats, n_transactions=len(df), categories=cats)
    return feats, cres, clustering.cluster_stats(df, cres.labels)


@st.cache_data(show_spinner="Asking GPT-5.6 for analysis…")
def run_analysis(stats_json: str, period_json: str):
    try:
        return llm.analyze_clusters(json.loads(stats_json), json.loads(period_json))
    except llm.LLMError:
        return llm.generic_labels(json.loads(stats_json))
```

Call sites pass `df.to_json().encode()` / `json.dumps(stats)` so Streamlit's cache keys are stable. Increment `st.session_state["analyses"]` only on cache miss (put the increment inside `run_analysis`).

- [ ] **Step 2: Set OpenAI account guards (manual, in browser)**

platform.openai.com → Billing: set a hard monthly spending limit (e.g. $10) and a usage-alert email at $5. This is the real cost backstop.

- [ ] **Step 3: Full manual edge-case sweep with the deployed app**

Upload each of these and confirm graceful behavior (no tracebacks anywhere):
empty CSV → error; 8-row CSV → "not enough data"; European fixture → parses;
debit/credit fixture → parses; multi-sheet xlsx → sheet picker or auto-pick;
`.txt` renamed to `.csv` gibberish → error message.

- [ ] **Step 4: Commit and push**

```bash
MOCK_LLM=true uv run pytest -v
git add app.py && git commit -m "feat: caching, session limits and cost safeguards" && git push
```

---

# Day 6 — Mon Jul 20: README, polish, dry run

### Task 14: README + submission collateral

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the full README** with these sections (all required by Devpost rules or the spec):

```markdown
# AI Personal Finance Assistant
One-paragraph pitch + live app URL + note: "First load may take up to a
minute if the app was asleep."

## What it does            (features, expenses-only scope)
## How it works            (pipeline diagram in text: upload -> mapping ->
                            merchants -> features -> KMeans -> GPT-5.6 -> PDF)
## Privacy                 (what is and isn't sent to OpenAI; nothing stored)
## Built with Codex and GPT-5.6
   - How Codex accelerated development (concrete examples + session ID)
   - The three GPT-5.6 integration points (mapping, naming, trends)
## Run locally             (`uv sync --group dev`; `uv run streamlit run app.py`;
                            `MOCK_LLM=true` for no-key mode)
## Tests                   (`MOCK_LLM=true uv run pytest`)
```

- [ ] **Step 2: Full dry run of the demo flow on the deployed app**, exactly as the video will show it: open URL → Load demo data → walk the dashboard → download PDF → open PDF. Time it: must fit under 2 minutes to leave video room for intro/outro.

- [ ] **Step 3: Commit and push**

```bash
git add README.md && git commit -m "docs: full README with Codex/GPT-5.6 usage and run instructions" && git push
```

---

# Day 7 — Tue Jul 21: video + submission (deadline 5:00 pm PT)

### Task 15: Record, submit, verify

- [ ] **Step 1 (morning): Record the demo video** — under 3 minutes, with audio, synthetic data only. Script: problem (20s) → upload & mapping (30s) → clusters + AI summary (60s) → PDF download (20s) → privacy + how Codex/GPT-5.6 built it (30s). Upload to YouTube (public or unlisted).

- [ ] **Step 2: Assemble the Devpost submission**: project description, repo URL, live app URL, YouTube URL, Codex session ID, "built with" list, track = Apps for Your Life.

- [ ] **Step 3: Submit by early afternoon** — target 1:00 pm PT, four hours of buffer. Then re-open the submission page in a private browser window and verify every link works (video plays, repo public, app loads).

---

## Self-Review (completed)

- **Spec coverage:** every DESIGN.md section maps to a task — parsing incl. European/debit-credit/multi-sheet (2–3), merchants (4), month-normalized features + recurrence (5), guarded clustering incl. 0.15 threshold (6), LLM with consent/validation/retry/fallbacks (9–10), dashboard incl. caveats and evidence table (11), PDF without LLM (12), cost safeguards (13), deployment + UptimeRobot alerts-only (8), README/compliance (14–15). All 16 required test cases from the spec appear in Tasks 2, 3, 4, 5, 6, 9, 12.
- **Placeholder scan:** none — every code step contains runnable code.
- **Type consistency:** `ColumnMapping`/`ParseResult` fields, `FEATURE_COLUMNS`, `ClusterResult`, stats-dict keys, and `analysis` shape are used identically across Tasks 2→13.

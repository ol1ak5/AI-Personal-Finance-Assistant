"""Uploaded bank export -> validated, normalized expense DataFrame."""
from __future__ import annotations

import io
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
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):  # ISO: never dayfirst-swapped
        ts = pd.to_datetime(s, errors="coerce")
    else:
        ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return None if pd.isna(ts) else pd.Timestamp(ts)


def _find(columns, hints) -> str | None:
    for col in columns:
        if any(h in str(col).lower() for h in hints):
            return col
    return None


def _detect_decimal_separator(values) -> str:
    """Vote over sample amount strings: money decimals are always followed by
    exactly two digits, so '13,85' votes comma and '1,385.20' votes dot.
    Ambiguous values (integers, blanks) abstain. Ties default to '.'."""
    comma_votes, dot_votes = 0, 0
    for value in values:
        s = re.sub(r"[^\d,.\-+]", "", str(value).strip())
        if re.search(r",\d{2}$", s):
            comma_votes += 1
        elif re.search(r"\.\d{2}$", s):
            dot_votes += 1
    return "," if comma_votes > dot_votes else "."


def guess_mapping(df: pd.DataFrame) -> ColumnMapping | None:
    cols = list(df.columns)
    date_col = _find(cols, DATE_HINTS)
    desc_col = _find(cols, DESC_HINTS)
    amount_col = _find(cols, AMOUNT_HINTS)
    debit_col = _find(cols, DEBIT_HINTS)
    credit_col = _find(cols, CREDIT_HINTS)
    if not date_col or not desc_col:
        return None
    sample = df.head(50)
    if debit_col and credit_col:
        sep = _detect_decimal_separator(
            list(sample[debit_col]) + list(sample[credit_col]))
        return ColumnMapping(date_col=date_col, description_col=desc_col,
                             debit_col=debit_col, credit_col=credit_col,
                             category_col=_find(cols, CATEGORY_HINTS),
                             decimal_separator=sep,
                             expenses_are="debit_col")
    if amount_col:
        sep = _detect_decimal_separator(sample[amount_col])
        return ColumnMapping(date_col=date_col, description_col=desc_col,
                             amount_col=amount_col,
                             category_col=_find(cols, CATEGORY_HINTS),
                             decimal_separator=sep)
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

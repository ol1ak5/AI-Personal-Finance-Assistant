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
    assert len(r.df) == 2
    assert r.skipped_rows == 3          # TOTAL row, blank row, bad-date row


def test_wrong_mapping_fails_validation(simple_df):
    wrong = ColumnMapping(date_col="Description", description_col="Date",
                          amount_col="Amount", expenses_are="negative")
    assert validate_mapping(simple_df, wrong) < MAPPING_MIN_VALID


def test_good_mapping_passes_validation(simple_df):
    good = ColumnMapping(date_col="Date", description_col="Description",
                         amount_col="Amount", expenses_are="negative")
    assert validate_mapping(simple_df, good) >= MAPPING_MIN_VALID


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


def test_guess_detects_comma_decimal(european_df):
    m = guess_mapping(european_df)
    assert m is not None
    assert m.decimal_separator == ","


def test_guess_detects_dot_decimal(simple_df):
    m = guess_mapping(simple_df)
    assert m is not None
    assert m.decimal_separator == "."


def test_real_demo_file_parses_at_correct_scale():
    """End-to-end on the bundled anonymized bank export (CSV): dd/mm/yyyy
    dates, comma decimals, EUR suffix. Guards against the 100x corruption
    bug (13,85 read as 1385)."""
    from pathlib import Path
    data = (Path(__file__).parent.parent / "data" / "spending_demo.csv").read_bytes()
    frames = load_frames(data, "spending_demo.csv")
    df = list(frames.values())[0]
    m = guess_mapping(df)
    assert m is not None
    assert m.decimal_separator == ","
    assert validate_mapping(df, m) >= MAPPING_MIN_VALID
    r = apply_mapping(df, m)
    assert len(r.df) == 181                       # every expense row survives
    assert r.df["amount"].max() == pytest.approx(840.00)    # monthly rent
    assert r.df["amount"].min() < 5               # small coffee amounts intact
    assert r.df["date"].min() == pd.Timestamp("2026-01-01")
    assert r.df["date"].max() == pd.Timestamp("2026-06-29")

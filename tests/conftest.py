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

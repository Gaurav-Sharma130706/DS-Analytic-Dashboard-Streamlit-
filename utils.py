"""Reusable data functions: loading, validation, cleaning, statistics, KPIs."""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

NUMERIC_COLS = ["Sales", "Quantity", "Discount", "Profit"]
DATE_COLS = ["Order Date", "Ship Date"]
TEXT_COLS = [
    "Ship Mode", "Segment", "Country", "City", "State", "Region",
    "Category", "Sub-Category", "Product Name",
]
REQUIRED_COLS = ["Order ID", "Order Date", "Sales", "Quantity", "Profit"]

# Maps for fixing inconsistent spellings / labels
CANONICAL = {
    "Ship Mode": {"sameday": "Same Day", "same day": "Same Day",
                  "standard class": "Standard Class", "first class": "First Class",
                  "second class": "Second Class"},
    "Segment": {"consumer": "Consumer", "corporate": "Corporate",
                "home office": "Home Office"},
    "Region": {"east": "East", "west": "West", "south": "South",
               "central": "Central", "north": "North"},
    "Category": {"technology": "Technology", "furniture": "Furniture",
                 "office supplies": "Office Supplies",
                 "office supply": "Office Supplies"},
}
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}


# ---------------------------------------------------------------- loading
def load_file(uploaded_file) -> pd.DataFrame:
    """Load CSV / XLSX / JSON from a Streamlit UploadedFile (or any file-like)."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".json"):
        raw = json.load(uploaded_file)
        if isinstance(raw, dict):
            # accept {"data": [...]} or column-oriented dicts
            for key in ("data", "records", "rows"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
        return pd.json_normalize(raw) if isinstance(raw, list) else pd.DataFrame(raw)
    raise ValueError("Unsupported file type. Please upload .csv, .xlsx or .json")


# ------------------------------------------------------------- validation
def validate_data(df: pd.DataFrame) -> Dict:
    """Return dimensions, dtypes, missing values and duplicate info."""
    # treat blank / whitespace-only strings as missing for reporting
    blanks = df.replace(r"^\s*$", np.nan, regex=True)
    missing = blanks.isna().sum()
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": pd.DataFrame({"Column": df.columns,
                                "Data Type": df.dtypes.astype(str).values}),
        "missing": pd.DataFrame({"Column": missing.index,
                                 "Missing": missing.values,
                                 "Missing %": (missing.values / max(len(df), 1) * 100).round(2)}),
        "duplicate_count": int(df.duplicated().sum()),
        "duplicates": df[df.duplicated(keep=False)],
        "missing_required": [c for c in REQUIRED_COLS if c not in df.columns],
    }


# ------------------------------------------------------------ preprocessing
def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace, fix casing, map known variants to canonical labels."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    for col, mapping in CANONICAL.items():
        if col in df.columns:
            lowered = df[col].str.lower()
            df[col] = lowered.map(mapping).fillna(df[col])
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse mixed date formats (ISO, dd/mm/yyyy, dd-Mon-yyyy, dd-mm-yyyy)."""
    df = df.copy()
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d-%m-%Y"]
    for col in DATE_COLS:
        if col not in df.columns:
            continue
        s = df[col].astype("string").str.strip()
        out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for fmt in formats:
            mask = out.isna() & s.notna()
            out[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
        df[col] = out
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns; words like 'five' -> 5; junk -> NaN; inf -> NaN."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = df[col].astype("string").str.strip().str.lower()
        s = s.replace(WORD_NUMBERS)
        df[col] = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flag impossible values as missing (negative sales/qty, discount outside 0-1)."""
    df = df.copy()
    if "Sales" in df:
        df.loc[df["Sales"] < 0, "Sales"] = np.nan
    if "Quantity" in df:
        df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
    if "Discount" in df:
        df.loc[(df["Discount"] < 0) | (df["Discount"] > 1), "Discount"] = np.nan
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols=("Sales", "Quantity"), k: float = 3.0):
    """Drop rows beyond k*IQR (default 3 = extreme only). Returns (df, n_removed)."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask &= df[col].isna() | df[col].between(q1 - k * iqr, q3 + k * iqr)
    return df[mask], int((~mask).sum())


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median for Sales/Quantity/Discount, 'Unknown' for text, drop rows lacking
    Order ID or Order Date (cannot be analysed), derive Profit when missing."""
    df = df.copy()
    df = df.dropna(subset=[c for c in ["Order ID", "Order Date"] if c in df.columns])
    for col in ["Sales", "Quantity", "Discount"]:
        if col in df:
            df[col] = df[col].fillna(df[col].median())
    if "Profit" in df:
        # profit can't be guessed reliably: use the median margin of known rows
        known = df.dropna(subset=["Profit", "Sales"])
        margin = (known["Profit"].sum() / known["Sales"].sum()) if len(known) else 0
        df["Profit"] = df["Profit"].fillna(df["Sales"] * margin)
    for col in TEXT_COLS + ["Customer ID", "Product ID"]:
        if col in df:
            df[col] = df[col].fillna("Unknown")
    return df


def preprocess(df: pd.DataFrame, drop_outliers: bool = True):
    """Full pipeline. Returns (clean_df, log) where log lists every step taken."""
    log = {"rows_in": len(df)}
    df = standardize_text(df)
    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df = parse_dates(df)
    df = clean_numeric(df)
    df = handle_invalid_values(df)
    n_before = len(df)
    df = df.dropna(subset=[c for c in ["Order ID", "Order Date"] if c in df.columns])
    log["rows_dropped_no_id_or_date"] = n_before - len(df)
    log["outliers_removed"] = 0
    if drop_outliers:
        df, log["outliers_removed"] = remove_outliers_iqr(df)
    df = fill_missing(df)
    if "Ship Date" in df and "Order Date" in df:
        bad = df["Ship Date"] < df["Order Date"]
        log["ship_before_order_fixed"] = int(bad.sum())
        df.loc[bad, "Ship Date"] = pd.NaT
        df["Days to Ship"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log


# --------------------------------------------------------------- statistics
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    summary = df[cols].describe().T
    summary["median"] = df[cols].median()
    summary["skew"] = df[cols].skew()
    return summary.round(2)


def categorical_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (df.groupby(col)
              .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                   Orders=("Order ID", "nunique"), Quantity=("Quantity", "sum"))
              .assign(**{"Margin %": lambda x: (x["Profit"] / x["Sales"] * 100).round(2)})
              .round(2).sort_values("Sales", ascending=False))


# --------------------------------------------------------------------- KPIs
def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    sales = float(df["Sales"].sum())
    profit = float(df["Profit"].sum())
    return {
        "Total Sales": sales,
        "Total Profit": profit,
        "Total Orders": int(df["Order ID"].nunique()),
        "Total Quantity": int(df["Quantity"].sum()),
        "Profit Margin (%)": (profit / sales * 100) if sales else 0.0,
    }


def apply_filters(df: pd.DataFrame, date_range=None, regions=None, categories=None,
                  segments=None, ship_modes=None) -> pd.DataFrame:
    out = df
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[out["Order Date"].between(start, end)]
    for col, vals in [("Region", regions), ("Category", categories),
                      ("Segment", segments), ("Ship Mode", ship_modes)]:
        if vals is not None and col in out:
            out = out[out[col].isin(vals)]
    return out
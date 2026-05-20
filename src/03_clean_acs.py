from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


def _std_col(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def build_community_area_lookup(boundaries_csv: Path) -> pd.DataFrame:
    b = pd.read_csv(boundaries_csv, usecols=["AREA_NUMBE", "COMMUNITY"], low_memory=False)
    b = b.rename(columns={"AREA_NUMBE": "community_area", "COMMUNITY": "community"})
    b["community_area"] = pd.to_numeric(b["community_area"], errors="coerce").astype("Int64")
    b["community"] = b["community"].astype("string").str.strip().str.upper()
    b = b.dropna(subset=["community_area", "community"]).drop_duplicates(subset=["community"])
    b = b[(b["community_area"] >= 1) & (b["community_area"] <= 77)]
    return b[["community_area", "community"]]


def weighted_median_income(row: pd.Series, bracket_cols: list[str]) -> float:
    counts = row[bracket_cols].to_numpy(dtype=float)
    if np.isnan(counts).all() or counts.sum() <= 0:
        return np.nan

    bounds = np.array(
        [
            (0.0, 25_000.0),
            (25_000.0, 50_000.0),
            (50_000.0, 75_000.0),
            (75_000.0, 125_000.0),
            (125_000.0, 200_000.0),
        ],
        dtype=float,
    )

    total = counts.sum()
    target = total / 2.0
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), len(counts) - 1)

    lower, upper = bounds[idx]
    prev_cum = cum[idx - 1] if idx > 0 else 0.0
    in_bin = counts[idx]
    if in_bin <= 0:
        return float((lower + upper) / 2.0)

    frac = float(np.clip((target - prev_cum) / in_bin, 0.0, 1.0))
    return float(lower + frac * (upper - lower))


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype="float64")
    return (series - mean) / std


def clean_acs(acs_csv: Path, boundaries_csv: Path, output_parquet: Path) -> None:
    raw = pd.read_csv(acs_csv, low_memory=False)
    raw = raw.rename(columns={c: _std_col(c) for c in raw.columns})

    needed = {
        "acs_year",
        "community_area",
        "under_25_000",
        "25_000_to_49_999",
        "50_000_to_74_999",
        "75_000_to_125_000",
        "125_000",
        "total_population",
        "black_or_african_american",
        "hispanic_or_latino",
    }
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise SystemExit(f"ACS file missing expected columns: {missing}")

    raw["acs_year"] = pd.to_numeric(raw["acs_year"], errors="coerce")
    latest_year = int(raw["acs_year"].dropna().max())
    df = raw.loc[raw["acs_year"] == latest_year].copy()

    df["community"] = df["community_area"].astype("string").str.strip().str.upper()
    lookup = build_community_area_lookup(boundaries_csv)
    df = df.merge(lookup, how="left", on="community")

    df["community_area"] = df["community_area_y"]
    df = df.drop(columns=["community_area_x", "community_area_y"])

    income_cols = [
        "under_25_000",
        "25_000_to_49_999",
        "50_000_to_74_999",
        "75_000_to_125_000",
        "125_000",
    ]
    df[income_cols] = df[income_cols].apply(_to_number)
    df["population"] = _to_number(df["total_population"])
    df["black_count"] = _to_number(df["black_or_african_american"])
    df["hispanic_count"] = _to_number(df["hispanic_or_latino"])

    df["income_households"] = df[income_cols].sum(axis=1)
    df["median_income"] = df.apply(lambda r: weighted_median_income(r, income_cols), axis=1)

    # Proxies based on this ACS extract (it doesn't include vehicle ownership directly).
    df["poverty_rate"] = (df["under_25_000"] / df["income_households"]) * 100.0
    df["pct_black"] = (df["black_count"] / df["population"]) * 100.0
    df["pct_hispanic"] = (df["hispanic_count"] / df["population"]) * 100.0
    df["pct_no_vehicle"] = np.nan

    out = df[
        [
            "community_area",
            "median_income",
            "poverty_rate",
            "pct_no_vehicle",
            "pct_black",
            "pct_hispanic",
            "population",
        ]
    ].copy()

    out["community_area"] = pd.to_numeric(out["community_area"], errors="coerce").astype("Int64")
    out = out[(out["community_area"] >= 1) & (out["community_area"] <= 77)]

    numeric_cols = [
        "median_income",
        "poverty_rate",
        "pct_no_vehicle",
        "pct_black",
        "pct_hispanic",
        "population",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in numeric_cols:
        mean = out[col].mean()
        # If the entire column is missing (NaN mean), fall back to 0 so downstream
        # z-scoring and clustering can proceed (the variable becomes non-informative).
        if pd.isna(mean):
            mean = 0.0
        out[col] = out[col].fillna(mean)

    income_z = zscore(out["median_income"])
    no_vehicle_z = zscore(out["pct_no_vehicle"])
    poverty_z = zscore(out["poverty_rate"])
    out["transit_deprivation_score"] = (-income_z + no_vehicle_z + poverty_z) / 3.0

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"acs_year used: {latest_year}")
    print(f"saved: {output_parquet}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean ACS demographics to parquet.")
    parser.add_argument(
        "--acs",
        type=Path,
        default=Path("/Users/minseojung/Downloads/ACS_5_Year_Data_by_Community_Area_20260519.csv"),
        help="Path to ACS CSV.",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=Path("/Users/minseojung/Downloads/Boundaries_-_Community_Areas_20260517.csv"),
        help="Path to community area boundaries CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/acs_clean.parquet"),
        help="Output parquet path.",
    )
    args = parser.parse_args()
    clean_acs(args.acs, args.boundaries, args.output)


if __name__ == "__main__":
    main()

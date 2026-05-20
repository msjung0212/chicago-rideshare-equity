from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


RENAME_MAP = {
    "trip_start_timestamp": "start_time",
    "start_timestamp": "start_time",
    "start_time": "start_time",
    "fare": "fare",
    "tip": "tip",
    "tips": "tip",
    "additional_charges": "additional_charges",
    "extras": "additional_charges",
    "trip_total": "trip_total",
    "total": "trip_total",
    "trip_miles": "distance_miles",
    "distance_miles": "distance_miles",
    "trip_seconds": "duration_sec",
    "duration_sec": "duration_sec",
    "pickup_community_area": "pickup_ca",
    "pickup_ca": "pickup_ca",
    "dropoff_community_area": "dropoff_ca",
    "dropoff_ca": "dropoff_ca",
    "company": "company",
}

REQUIRED_COLUMNS = {
    "start_time",
    "fare",
    "distance_miles",
    "duration_sec",
    "pickup_ca",
    "dropoff_ca",
}

OUTPUT_COLUMNS = [
    "start_time",
    "fare",
    "tip",
    "additional_charges",
    "trip_total",
    "total_cost",
    "distance_miles",
    "duration_sec",
    "duration_min",
    "pickup_ca",
    "dropoff_ca",
    "company",
    "hour",
    "day_of_week",
    "month",
    "year",
    "late_night",
    "fare_per_mile",
    "log_fare_per_mile",
    "log_fare",
    "log_total",
    "log_distance",
    "log_duration",
    "trip_type",
]


def standardize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def money_to_float(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_start_time(series: pd.Series) -> pd.Series:
    # Handles both Chicago open data format and ISO-like timestamps (df3cleaned.csv).
    dt = pd.to_datetime(series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if dt.notna().mean() < 0.95:
        dt = pd.to_datetime(series, errors="coerce")
    return dt


def get_source_columns(input_path: Path) -> list[str]:
    header = pd.read_csv(input_path, nrows=0)
    usecols: list[str] = []
    for original_name in header.columns:
        standardized = standardize_column_name(original_name)
        if standardized in RENAME_MAP:
            usecols.append(original_name)
    return usecols


def empty_output_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": pd.Series(dtype="datetime64[ns]"),
            "fare": pd.Series(dtype="float64"),
            "tip": pd.Series(dtype="float64"),
            "additional_charges": pd.Series(dtype="float64"),
            "trip_total": pd.Series(dtype="float64"),
            "total_cost": pd.Series(dtype="float64"),
            "distance_miles": pd.Series(dtype="float64"),
            "duration_sec": pd.Series(dtype="int32"),
            "duration_min": pd.Series(dtype="float64"),
            "pickup_ca": pd.Series(dtype="int16"),
            "dropoff_ca": pd.Series(dtype="int16"),
            "company": pd.Series(dtype="string"),
            "hour": pd.Series(dtype="int8"),
            "day_of_week": pd.Series(dtype="int8"),
            "month": pd.Series(dtype="int8"),
            "year": pd.Series(dtype="int16"),
            "late_night": pd.Series(dtype="int8"),
            "fare_per_mile": pd.Series(dtype="float64"),
            "log_fare_per_mile": pd.Series(dtype="float64"),
            "log_fare": pd.Series(dtype="float64"),
            "log_total": pd.Series(dtype="float64"),
            "log_distance": pd.Series(dtype="float64"),
            "log_duration": pd.Series(dtype="float64"),
            "trip_type": pd.Series(dtype="string"),
        }
    )


def clean_chunk(chunk: pd.DataFrame, year: Optional[int], month: Optional[int]) -> pd.DataFrame:
    chunk = chunk.rename(columns={col: standardize_column_name(col) for col in chunk.columns})
    chunk = chunk.rename(
        columns={col: RENAME_MAP[col] for col in chunk.columns if col in RENAME_MAP}
    )

    missing = sorted(REQUIRED_COLUMNS - set(chunk.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    cleaned = pd.DataFrame()
    cleaned["start_time"] = parse_start_time(chunk["start_time"])
    cleaned["fare"] = money_to_float(chunk["fare"])
    cleaned["tip"] = money_to_float(chunk["tip"]) if "tip" in chunk.columns else 0.0
    cleaned["additional_charges"] = (
        money_to_float(chunk["additional_charges"]) if "additional_charges" in chunk.columns else 0.0
    )
    cleaned["trip_total"] = (
        money_to_float(chunk["trip_total"]) if "trip_total" in chunk.columns else np.nan
    )
    cleaned["distance_miles"] = pd.to_numeric(chunk["distance_miles"], errors="coerce")
    cleaned["duration_sec"] = pd.to_numeric(chunk["duration_sec"], errors="coerce")
    cleaned["pickup_ca"] = pd.to_numeric(chunk["pickup_ca"], errors="coerce")
    cleaned["dropoff_ca"] = pd.to_numeric(chunk["dropoff_ca"], errors="coerce")

    # Keep totals consistent even if some components are missing.
    cleaned["tip"] = cleaned["tip"].fillna(0.0)
    cleaned["additional_charges"] = cleaned["additional_charges"].fillna(0.0)
    cleaned["total_cost"] = cleaned["fare"] + cleaned["tip"] + cleaned["additional_charges"]

    if "company" in chunk.columns:
        cleaned["company"] = chunk["company"].astype("string").fillna("unknown")
    else:
        cleaned["company"] = "unknown"

    valid = (
        cleaned["start_time"].notna()
        & cleaned["pickup_ca"].notna()
        & cleaned["dropoff_ca"].notna()
        & cleaned["distance_miles"].between(0.5, 40, inclusive="both")
        & cleaned["fare"].between(3, 150, inclusive="both")
        & cleaned["duration_sec"].between(60, 7200, inclusive="both")
    )
    cleaned = cleaned.loc[valid].copy()

    if cleaned.empty:
        return empty_output_frame()

    cleaned["duration_sec"] = cleaned["duration_sec"].astype("int32")
    cleaned["duration_min"] = cleaned["duration_sec"] / 60.0
    cleaned["pickup_ca"] = cleaned["pickup_ca"].astype("int16")
    cleaned["dropoff_ca"] = cleaned["dropoff_ca"].astype("int16")

    cleaned["hour"] = cleaned["start_time"].dt.hour.astype("int8")
    cleaned["day_of_week"] = cleaned["start_time"].dt.dayofweek.astype("int8")
    cleaned["month"] = cleaned["start_time"].dt.month.astype("int8")
    cleaned["year"] = cleaned["start_time"].dt.year.astype("int16")
    cleaned["late_night"] = ((cleaned["hour"] >= 22) | (cleaned["hour"] <= 5)).astype("int8")

    if year is not None:
        cleaned = cleaned.loc[cleaned["year"] == year].copy()
        if cleaned.empty:
            return empty_output_frame()
    if month is not None:
        cleaned = cleaned.loc[cleaned["month"] == month].copy()
        if cleaned.empty:
            return empty_output_frame()

    cleaned["fare_per_mile"] = cleaned["fare"] / cleaned["distance_miles"]
    cleaned["log_fare_per_mile"] = np.log(cleaned["fare_per_mile"])
    cleaned["log_fare"] = np.log(cleaned["fare"])
    cleaned["log_total"] = np.log(cleaned["total_cost"])
    cleaned["log_distance"] = np.log(cleaned["distance_miles"])
    cleaned["log_duration"] = np.log(cleaned["duration_min"])
    cleaned["trip_type"] = "rideshare"

    return cleaned[OUTPUT_COLUMNS]


def write_parquet(
    input_path: Path,
    output_path: Path,
    chunksize: int,
    nrows: Optional[int],
    year: Optional[int],
    month: Optional[int],
) -> None:
    usecols = get_source_columns(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(
        input_path,
        chunksize=chunksize,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )

    rows_in = 0
    rows_out = 0
    writer = None

    try:
        for chunk_number, chunk in enumerate(reader, start=1):
            rows_in += len(chunk)
            cleaned = clean_chunk(chunk, year=year, month=month)
            rows_out += len(cleaned)

            if cleaned.empty:
                continue

            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            print(f"chunk {chunk_number}: kept {len(cleaned):,} rows")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty_table = pa.Table.from_pandas(empty_output_frame(), preserve_index=False)
        pq.write_table(empty_table, output_path, compression="snappy")

    print(f"input rows: {rows_in:,}")
    print(f"clean rows: {rows_out:,}")
    print(f"saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean rideshare trips to a standard schema.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/minseojung/Downloads/df3cleaned.csv"),
        help="Path to the rideshare CSV (df3cleaned.csv by default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tnc_trips_clean.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_parquet(
        args.input,
        args.output,
        chunksize=args.chunksize,
        nrows=args.nrows,
        year=args.year,
        month=args.month,
    )


if __name__ == "__main__":
    main()

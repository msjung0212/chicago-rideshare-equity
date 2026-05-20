from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ACS_FEATURE_COLS = [
    "median_income",
    "poverty_rate",
    "pct_no_vehicle",
    "pct_black",
    "pct_hispanic",
    "population",
    "transit_deprivation_score",
]


def merge_one(trips: pd.DataFrame, acs: pd.DataFrame) -> pd.DataFrame:
    acs = acs.copy()
    acs["community_area"] = pd.to_numeric(acs["community_area"], errors="coerce").astype("Int64")

    acs_pickup = acs.rename(
        columns={c: f"pickup_{c}" for c in ACS_FEATURE_COLS} | {"community_area": "pickup_ca"}
    )
    acs_dropoff = acs.rename(
        columns={c: f"dropoff_{c}" for c in ACS_FEATURE_COLS} | {"community_area": "dropoff_ca"}
    )

    merged = trips.merge(acs_pickup, how="left", on="pickup_ca").merge(
        acs_dropoff, how="left", on="dropoff_ca"
    )
    return merged


def report_match_quality(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"{label}: 0 rows (skipping match quality)")
        return

    pickup_fail = df["pickup_median_income"].isna().mean() * 100.0
    dropoff_fail = df["dropoff_median_income"].isna().mean() * 100.0
    print(f"{label}: pickup unmatched = {pickup_fail:.2f}%, dropoff unmatched = {dropoff_fail:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ACS demographics onto trips.")
    parser.add_argument(
        "--tnc",
        type=Path,
        default=Path("data/processed/tnc_trips_clean.parquet"),
        help="Cleaned rideshare parquet.",
    )
    parser.add_argument(
        "--taxi",
        type=Path,
        default=Path("data/processed/taxi_trips_clean.parquet"),
        help="Cleaned taxi parquet.",
    )
    parser.add_argument(
        "--acs",
        type=Path,
        default=Path("data/processed/acs_clean.parquet"),
        help="Cleaned ACS parquet.",
    )
    parser.add_argument("--out-tnc", type=Path, default=Path("data/processed/tnc_merged.parquet"))
    parser.add_argument("--out-taxi", type=Path, default=Path("data/processed/taxi_merged.parquet"))
    parser.add_argument(
        "--out-combined", type=Path, default=Path("data/processed/combined_merged.parquet")
    )
    args = parser.parse_args()

    tnc = pd.read_parquet(args.tnc)
    taxi = pd.read_parquet(args.taxi)
    acs = pd.read_parquet(args.acs)

    tnc_merged = merge_one(tnc, acs)
    taxi_merged = merge_one(taxi, acs)

    report_match_quality(tnc_merged, "tnc")
    report_match_quality(taxi_merged, "taxi")

    args.out_tnc.parent.mkdir(parents=True, exist_ok=True)
    tnc_merged.to_parquet(args.out_tnc, index=False)
    taxi_merged.to_parquet(args.out_taxi, index=False)

    combined = pd.concat([tnc_merged, taxi_merged], ignore_index=True)
    combined.to_parquet(args.out_combined, index=False)

    print(f"saved: {args.out_tnc}")
    print(f"saved: {args.out_taxi}")
    print(f"saved: {args.out_combined}")


if __name__ == "__main__":
    main()


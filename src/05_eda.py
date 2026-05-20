from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    # Set before importing matplotlib/seaborn to avoid ~/.matplotlib writes.
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "mplconfig"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    fig_dir = Path("outputs/figures")
    tab_dir = Path("outputs/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    def savefig(name: str) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        print("saved:", path)

    df = pd.read_parquet("data/processed/combined_merged.parquet")

    # Community area number -> neighborhood name lookup (Chicago community areas).
    boundaries_path = Path("/Users/minseojung/Downloads/Boundaries_-_Community_Areas_20260517.csv")
    ca_lookup = None
    if boundaries_path.exists():
        b = pd.read_csv(boundaries_path, usecols=["AREA_NUMBE", "COMMUNITY"], low_memory=False)
        b = b.rename(columns={"AREA_NUMBE": "community_area", "COMMUNITY": "community"})
        b["community_area"] = pd.to_numeric(b["community_area"], errors="coerce").astype("Int64")
        b["community"] = b["community"].astype("string").str.strip()
        ca_lookup = dict(
            b.dropna(subset=["community_area", "community"])
            .drop_duplicates(subset=["community_area"])
            .set_index("community_area")["community"]
            .to_dict()
        )

    # 1) Histograms of fare_per_mile + log_fare_per_mile per mode
    for trip_type in ["rideshare", "taxi"]:
        sub = df.loc[df["trip_type"] == trip_type]

        plt.figure(figsize=(8, 4))
        sns.histplot(sub["fare_per_mile"], bins=60)
        if trip_type == "rideshare":
            plt.title("Distribution of Rideshare Fare per Mile, January 2024")
            plt.xlim(0, 25)
        else:
            plt.title("Distribution of Taxi Fare per Mile, January 2024")
            plt.xlim(0, 20)
        plt.xlabel("Fare per Mile ($/mile)")
        savefig(f"hist_fare_per_mile_{trip_type}.png")
        plt.close()

        plt.figure(figsize=(8, 4))
        sns.histplot(sub["log_fare_per_mile"], bins=60)
        if trip_type == "rideshare":
            plt.title("Distribution of Log Rideshare Fare per Mile, January 2024")
        else:
            plt.title("Distribution of Log Taxi Fare per Mile, January 2024")
            plt.figtext(
                0.02,
                0.01,
                "Note: small cluster of high log outliers (~3.8–4.0) may reflect rare expensive/short trips.",
                ha="left",
                va="bottom",
                fontsize=9,
            )
        plt.xlabel("Log Fare per Mile")
        savefig(f"hist_log_fare_per_mile_{trip_type}.png")
        plt.close()

        print(trip_type, "fare_per_mile skew:", float(sub["fare_per_mile"].skew()))
        print(trip_type, "log_fare_per_mile skew:", float(sub["log_fare_per_mile"].skew()))

    # 2) Trip count by hour
    hour_counts = df.groupby(["hour", "trip_type"]).size().reset_index(name="trip_count")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=hour_counts, x="hour", y="trip_count", hue="trip_type", marker="o")
    plt.title("Trip Counts by Hour (January 2024)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.legend(title="Mode", loc="best")
    savefig("trip_count_by_hour.png")
    plt.close()

    late_counts = df.groupby(["trip_type", "late_night"]).size().unstack(fill_value=0)
    print("late-night counts (0=not late, 1=late):")
    print(late_counts)

    # 3) Mean fare_per_mile per pickup community area for each mode
    pickup_means = (
        df.groupby(["pickup_ca", "trip_type"], as_index=False)
        .agg(
            mean_fare_per_mile=("fare_per_mile", "mean"),
            n_trips=("fare_per_mile", "size"),
            pickup_deprivation=("pickup_transit_deprivation_score", "mean"),
        )
    )

    # 4) Bar charts: top/bottom 10 by mean rideshare fare/mile
    rideshare_ca = pickup_means.loc[pickup_means["trip_type"] == "rideshare"].copy()
    rideshare_ca = rideshare_ca.sort_values("mean_fare_per_mile")
    bottom10 = rideshare_ca.head(10)
    top10 = rideshare_ca.tail(10)

    plt.figure(figsize=(10, 4))
    sns.barplot(
        data=top10.sort_values("mean_fare_per_mile", ascending=False),
        x="pickup_ca",
        y="mean_fare_per_mile",
    )
    plt.title("Top 10 Pickup Community Areas by Mean Rideshare Fare per Mile, January 2024")
    if ca_lookup is not None:
        labels = [
            ca_lookup.get(int(x), str(int(x))) if pd.notna(x) else "" for x in top10.sort_values("mean_fare_per_mile", ascending=False)["pickup_ca"]
        ]
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=35, ha="right")
        plt.xlabel("Pickup Community Area")
    else:
        plt.xlabel("pickup_ca")
    plt.ylabel("Mean Fare per Mile ($/mile)")
    savefig("top10_rideshare_mean_fare_per_mile_by_pickup_ca.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=bottom10, x="pickup_ca", y="mean_fare_per_mile")
    plt.title("Bottom 10 Pickup Community Areas by Mean Rideshare Fare per Mile, January 2024")
    if ca_lookup is not None:
        labels = [
            ca_lookup.get(int(x), str(int(x))) if pd.notna(x) else "" for x in bottom10["pickup_ca"]
        ]
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=35, ha="right")
        plt.xlabel("Pickup Community Area")
    else:
        plt.xlabel("pickup_ca")
    plt.ylabel("Mean Fare per Mile ($/mile)")
    savefig("bottom10_rideshare_mean_fare_per_mile_by_pickup_ca.png")
    plt.close()

    # 5) Correlation across community areas (rideshare pickup means)
    corr_df = rideshare_ca.dropna(subset=["mean_fare_per_mile", "pickup_deprivation"])
    corr = corr_df["mean_fare_per_mile"].corr(corr_df["pickup_deprivation"])
    print(
        "Correlation (rideshare mean fare_per_mile vs pickup deprivation score):",
        float(corr),
    )

    # 6) Time-of-day gap chart (mean fare/mile by hour and mode)
    hour_means = (
        df.groupby(["hour", "trip_type"], as_index=False)
        .agg(mean_fare_per_mile=("fare_per_mile", "mean"), n_trips=("fare_per_mile", "size"))
    )
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=hour_means, x="hour", y="mean_fare_per_mile", hue="trip_type", marker="o")
    plt.title("Mean Fare per Mile by Hour, January 2024")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Fare per Mile ($/mile)")
    plt.legend(
        title="Mode",
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )
    plt.figtext(
        0.01,
        0.01,
        "Note: Raw taxi fare/mile is higher across hours; this can reflect airport-heavy and metered pricing.\n"
        "In regressions, controlling for distance and duration helps separate mode effects from trip mix.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    savefig("mean_fare_per_mile_by_hour_trip_type.png")
    plt.close()

    # 7) Matched pairs table: deprivation quartiles (by pickup community area)
    tmp = df.dropna(subset=["pickup_transit_deprivation_score"]).copy()
    ca_scores = (
        tmp.groupby("pickup_ca", as_index=False)
        .agg(deprivation=("pickup_transit_deprivation_score", "mean"))
        .dropna()
    )
    ca_scores["deprivation_quartile"] = pd.qcut(
        ca_scores["deprivation"],
        4,
        labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"],
    )
    tmp = tmp.merge(ca_scores[["pickup_ca", "deprivation_quartile"]], on="pickup_ca", how="left")

    quart_table = (
        tmp.groupby(["deprivation_quartile", "trip_type"], as_index=False)
        .agg(
            mean_fare_per_mile=("fare_per_mile", "mean"),
            mean_distance_miles=("distance_miles", "mean"),
            mean_duration_sec=("duration_sec", "mean"),
            n_trips=("fare_per_mile", "size"),
        )
        .sort_values(["deprivation_quartile", "trip_type"])
    )

    out_path = tab_dir / "matched_pairs_by_deprivation_quartile.csv"
    quart_table.to_csv(out_path, index=False)
    print("saved:", out_path)
    print(quart_table)


if __name__ == "__main__":
    main()

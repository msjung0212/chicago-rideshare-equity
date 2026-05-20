from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")


# Keep this consistent with the language in your writeup; we also compute the observed value from the data.
AFFLUENT_BENCHMARK_RIDESHARE_TRIPS_PER_1000 = 2180.0

# For Step 8D (commuter case study): Englewood vs Lincoln Park community area numbers.
ENGLEWOOD_CA = 68
LINCOLN_PARK_CA = 7


CLUSTER_ORDER = [
    "High poverty / transit deserts",
    "Mixed / higher deprivation",
    "Mixed / lower deprivation",
    "Affluent / low deprivation",
]


def _setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    trips = pd.read_parquet(
        "data/processed/combined_merged_clustered.parquet",
        columns=[
            "pickup_ca",
            "trip_type",
            "fare",
            "total_cost",
            "distance_miles",
            "pickup_cluster_label",
        ],
    )
    acs = pd.read_parquet("data/processed/acs_clean.parquet")
    acs_clusters = pd.read_parquet("data/processed/acs_clustered.parquet")

    # ---- 8A: Access burden (community-area level) ----
    ca_pop = acs[["community_area", "population", "median_income"]].rename(
        columns={"community_area": "pickup_ca"}
    )
    ca_cluster = acs_clusters.rename(columns={"community_area": "pickup_ca"})[
        ["pickup_ca", "cluster_label"]
    ]

    # Start from the full set of community areas (77 rows) so areas with zero trips are included.
    ca_base = ca_pop.merge(ca_cluster, on="pickup_ca", how="left")

    # Trip counts per community area and mode (pickup side).
    ca_counts = (
        trips.groupby(["pickup_ca", "trip_type"], dropna=False)
        .size()
        .reset_index(name="trip_count")
    )
    ca_counts_piv = ca_counts.pivot_table(
        index="pickup_ca",
        columns="trip_type",
        values="trip_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    ca_counts_piv.columns.name = None
    if "rideshare" not in ca_counts_piv.columns:
        ca_counts_piv["rideshare"] = 0
    if "taxi" not in ca_counts_piv.columns:
        ca_counts_piv["taxi"] = 0
    ca_counts_piv = ca_counts_piv.rename(
        columns={"rideshare": "rideshare_trip_count", "taxi": "taxi_trip_count"}
    )

    ca_piv = ca_base.merge(ca_counts_piv, on="pickup_ca", how="left")
    ca_piv["rideshare_trip_count"] = ca_piv["rideshare_trip_count"].fillna(0).astype(float)
    ca_piv["taxi_trip_count"] = ca_piv["taxi_trip_count"].fillna(0).astype(float)
    ca_piv["rideshare_trips_per_1000"] = (ca_piv["rideshare_trip_count"] / ca_piv["population"]) * 1000.0
    ca_piv["taxi_trips_per_1000"] = (ca_piv["taxi_trip_count"] / ca_piv["population"]) * 1000.0

    # Observed affluent-cluster benchmark (January 2024).
    # IMPORTANT: we use the unweighted mean across community areas (to match the interpretation/benchmark used in Step 7).
    observed_affluent_benchmark = float(
        ca_piv.loc[ca_piv["cluster_label"] == "Affluent / low deprivation", "rideshare_trips_per_1000"].mean()
    )

    # Access gap uses the (paper-facing) benchmark of 2,180.
    ca_piv["access_gap_trips_per_1000"] = (
        AFFLUENT_BENCHMARK_RIDESHARE_TRIPS_PER_1000 - ca_piv["rideshare_trips_per_1000"]
    )
    ca_piv["access_gap_trips_per_1000"] = ca_piv["access_gap_trips_per_1000"].clip(lower=0.0)
    ca_piv["foregone_trips_month"] = (ca_piv["access_gap_trips_per_1000"] / 1000.0) * ca_piv["population"]
    ca_piv["foregone_trips_year"] = ca_piv["foregone_trips_month"] * 12.0

    ca_out = TAB_DIR / "step8_access_gap_by_community_area.csv"
    ca_piv.sort_values(["cluster_label", "pickup_ca"]).to_csv(ca_out, index=False)
    print("saved:", ca_out)

    # Headline: sum foregone trips in the High poverty / transit deserts cluster.
    hp = ca_piv.loc[ca_piv["cluster_label"] == "High poverty / transit deserts"].copy()
    hp_foregone_year = float(hp["foregone_trips_year"].sum())

    headline_out = TAB_DIR / "step8_access_gap_headline.csv"
    pd.DataFrame(
        [
            {
                "affluent_benchmark_used": AFFLUENT_BENCHMARK_RIDESHARE_TRIPS_PER_1000,
                "affluent_benchmark_observed": observed_affluent_benchmark,
                "high_poverty_transit_deserts_foregone_trips_year": hp_foregone_year,
            }
        ]
    ).to_csv(headline_out, index=False)
    print("saved:", headline_out)

    # ---- 8F: Cluster-level summary table (paper Table 1) ----
    # Trips/1k residents by cluster and mode (population-weighted).
    # Cluster-level trips/1k residents: unweighted mean across community areas in the cluster.
    cluster_rates = (
        ca_piv.groupby("cluster_label", dropna=False)
        .agg(
            rideshare_trips_per_1000=("rideshare_trips_per_1000", "mean"),
            taxi_trips_per_1000=("taxi_trips_per_1000", "mean"),
        )
        .reset_index()
    )

    # Mean fare and mean distance by cluster + mode (trip-weighted; pickup side).
    fare_stats = (
        trips.groupby(["pickup_cluster_label", "trip_type"], dropna=False)
        .agg(mean_total_fare=("fare", "mean"), mean_total_cost=("total_cost", "mean"), mean_distance=("distance_miles", "mean"))
        .reset_index()
        .rename(columns={"pickup_cluster_label": "cluster_label"})
    )

    # Median household income per cluster (population-weighted mean of CA median income).
    # Median household income per cluster: unweighted mean across community areas (neighborhood-typical).
    inc_df = ca_pop.merge(ca_cluster, on="pickup_ca", how="left").copy()
    acs_income = (
        inc_df.groupby("cluster_label", dropna=False)["median_income"]
        .mean()
        .reset_index(name="median_household_income")
    )

    # Build Table 1 with one row per cluster, with rideshare stats as the default "Mean total fare" / "Mean distance".
    rideshare_stats = fare_stats.loc[fare_stats["trip_type"] == "rideshare"].copy()
    taxi_stats = fare_stats.loc[fare_stats["trip_type"] == "taxi"].copy()
    rideshare_stats = rideshare_stats.rename(
        columns={
            "mean_total_fare": "mean_total_fare_rideshare",
            "mean_total_cost": "mean_total_cost_rideshare",
            "mean_distance": "mean_distance_rideshare",
        }
    )
    taxi_stats = taxi_stats.rename(
        columns={
            "mean_total_fare": "mean_total_fare_taxi",
            "mean_total_cost": "mean_total_cost_taxi",
            "mean_distance": "mean_distance_taxi",
        }
    )

    table1 = (
        cluster_rates.merge(acs_income, on="cluster_label", how="left")
        .merge(rideshare_stats[["cluster_label", "mean_total_fare_rideshare", "mean_distance_rideshare"]], on="cluster_label", how="left")
        .merge(taxi_stats[["cluster_label", "mean_total_fare_taxi"]], on="cluster_label", how="left")
    )

    # Annual rideshare fare per resident (January -> annualized) as % of median household income.
    table1["annual_rideshare_fare_per_resident"] = (
        (table1["rideshare_trips_per_1000"] / 1000.0) * 12.0 * table1["mean_total_fare_rideshare"]
    )
    table1["annual_fare_as_pct_of_income"] = (
        (table1["annual_rideshare_fare_per_resident"] / table1["median_household_income"]) * 100.0
    )

    # Paper-facing column names (match Step 8F).
    table1_out = pd.DataFrame(
        {
            "Cluster": table1["cluster_label"],
            "Trips/1k residents (rideshare)": table1["rideshare_trips_per_1000"],
            "Trips/1k residents (taxi)": table1["taxi_trips_per_1000"],
            "Mean total fare ($)": table1["mean_total_fare_rideshare"],
            "Mean distance (miles)": table1["mean_distance_rideshare"],
            "Median household income ($)": table1["median_household_income"],
            "Annual fare as % of income": table1["annual_fare_as_pct_of_income"],
        }
    )
    table1_out["Cluster"] = pd.Categorical(table1_out["Cluster"], categories=CLUSTER_ORDER, ordered=True)
    table1_out = table1_out.sort_values("Cluster")

    out_csv = TAB_DIR / "equity_burden_summary.csv"
    table1_out.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    # ---- 8C/8E: Derived comparisons (cluster-level) ----
    # Taxi vs rideshare dominance ratios by cluster.
    ratios = table1_out.copy()
    ratios["rideshare_to_taxi_ratio"] = ratios["Trips/1k residents (rideshare)"] / ratios["Trips/1k residents (taxi)"]
    ratio_out = TAB_DIR / "step8_taxi_vs_rideshare_access_ratio_by_cluster.csv"
    ratios[["Cluster", "rideshare_to_taxi_ratio"]].to_csv(ratio_out, index=False)
    print("saved:", ratio_out)

    # Total cost burden via trip length: compare High poverty / transit deserts vs Affluent / low deprivation.
    # (This is descriptive, not causal: it summarizes cluster-level averages.)
    def _row(cluster: str) -> pd.Series:
        r = table1_out.loc[table1_out["Cluster"] == cluster]
        return r.iloc[0] if len(r) else pd.Series(dtype=float)

    hp_row = _row("High poverty / transit deserts")
    aff_row = _row("Affluent / low deprivation")
    if not hp_row.empty and not aff_row.empty:
        cost_gap_per_trip = float(hp_row["Mean total fare ($)"] - aff_row["Mean total fare ($)"])
        dist_gap = float(hp_row["Mean distance (miles)"] - aff_row["Mean distance (miles)"])
        # Annual trips per resident (Jan -> annualized).
        hp_annual_trips_per_res = float((hp_row["Trips/1k residents (rideshare)"] / 1000.0) * 12.0)
        aff_annual_trips_per_res = float((aff_row["Trips/1k residents (rideshare)"] / 1000.0) * 12.0)
        annual_spend_hp = float(hp_annual_trips_per_res * hp_row["Mean total fare ($)"])
        annual_spend_aff = float(aff_annual_trips_per_res * aff_row["Mean total fare ($)"])
        annual_spend_gap = float(annual_spend_hp - annual_spend_aff)
        burden_out = TAB_DIR / "step8_total_cost_burden_high_poverty_vs_affluent.csv"
        pd.DataFrame(
            [
                {
                    "high_poverty_mean_fare": float(hp_row["Mean total fare ($)"]),
                    "affluent_mean_fare": float(aff_row["Mean total fare ($)"]),
                    "cost_gap_per_trip_high_minus_affluent": cost_gap_per_trip,
                    "high_poverty_mean_distance": float(hp_row["Mean distance (miles)"]),
                    "affluent_mean_distance": float(aff_row["Mean distance (miles)"]),
                    "distance_gap_miles_high_minus_affluent": dist_gap,
                    "high_poverty_annual_trips_per_resident_est": hp_annual_trips_per_res,
                    "affluent_annual_trips_per_resident_est": aff_annual_trips_per_res,
                    "high_poverty_annual_spend_per_resident_est": annual_spend_hp,
                    "affluent_annual_spend_per_resident_est": annual_spend_aff,
                    "annual_spend_per_resident_gap_high_minus_affluent": annual_spend_gap,
                }
            ]
        ).to_csv(burden_out, index=False)
        print("saved:", burden_out)

    # ---- 8G: Figure — mean total fare per trip by cluster (rideshare vs taxi) ----
    try:
        _setup_matplotlib()
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        plot_df = fare_stats.copy()
        plot_df["cluster_label"] = pd.Categorical(plot_df["cluster_label"], categories=CLUSTER_ORDER, ordered=True)
        plot_df = plot_df.sort_values("cluster_label")

        plt.figure(figsize=(11, 4.3))
        ax = sns.barplot(
            data=plot_df,
            x="cluster_label",
            y="mean_total_fare",
            hue="trip_type",
            palette={"rideshare": "#d97706", "taxi": "#2563eb"},
        )
        for patch in ax.patches:
            h = patch.get_height()
            if not np.isfinite(h):
                continue
            x = patch.get_x() + patch.get_width() / 2.0
            ax.text(x, h + 0.25, f"${h:,.2f}", ha="center", va="bottom", fontsize=8, color="#111827")

        plt.xticks(rotation=15, ha="right")
        plt.ylabel("Mean Total Fare ($)")
        plt.xlabel("")
        plt.title("Mean Total Fare per Trip by Neighborhood Cluster, January 2024")
        plt.legend(title="Mode", loc="best")
        plt.tight_layout()
        fig_path = FIG_DIR / "mean_total_fare_by_cluster.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print("saved:", fig_path)
    except Exception as e:
        print("mean total fare figure skipped:", repr(e))

    # ---- 8D: Commuter case study — Englewood vs Lincoln Park ----
    rs = trips.loc[trips["trip_type"] == "rideshare"].copy()
    ca_case = rs.loc[rs["pickup_ca"].isin([ENGLEWOOD_CA, LINCOLN_PARK_CA])].copy()
    ca_case = ca_case.merge(
        ca_pop.rename(columns={"pickup_ca": "pickup_ca"}), on="pickup_ca", how="left"
    )
    by_ca = (
        ca_case.groupby("pickup_ca", dropna=False)
        .agg(
            mean_fare=("fare", "mean"),
            mean_total_cost=("total_cost", "mean"),
            mean_distance=("distance_miles", "mean"),
            median_income=("median_income", "first"),
            population=("population", "first"),
            n_trips=("fare", "size"),
        )
        .reset_index()
    )
    by_ca["annual_spend_2_trips_per_week"] = 2.0 * 52.0 * by_ca["mean_fare"]
    by_ca["annual_spend_pct_income"] = (by_ca["annual_spend_2_trips_per_week"] / by_ca["median_income"]) * 100.0
    by_ca["community_area"] = by_ca["pickup_ca"].map({ENGLEWOOD_CA: "Englewood", LINCOLN_PARK_CA: "Lincoln Park"})
    by_ca = by_ca[
        [
            "community_area",
            "pickup_ca",
            "n_trips",
            "mean_distance",
            "mean_fare",
            "annual_spend_2_trips_per_week",
            "median_income",
            "annual_spend_pct_income",
        ]
    ]

    case_out = TAB_DIR / "step8_commuter_case_study_englewood_vs_lincoln_park.csv"
    by_ca.to_csv(case_out, index=False)
    print("saved:", case_out)

    # ---- 8H: Key sentences draft ----
    # Use the cluster-level rates (population-weighted).
    rates_map = {
        r["Cluster"]: float(r["Trips/1k residents (rideshare)"]) for _, r in table1_out.iterrows()
    }
    rs_aff = rates_map.get("Affluent / low deprivation", float("nan"))
    rs_hp = rates_map.get("High poverty / transit deserts", float("nan"))
    ratio = (rs_aff / rs_hp) if (np.isfinite(rs_aff) and np.isfinite(rs_hp) and rs_hp > 0) else float("nan")

    sentences = []
    sentences.append(
        f"Residents of Chicago's most transit-deprived neighborhoods took rideshare at less than one-fifth the rate "
        f"of affluent neighborhood residents ({rs_hp:,.0f} vs {rs_aff:,.0f} trips per 1,000 residents; ~{ratio:.1f}x gap), "
        f"despite facing no statistically significant per-mile price premium across six model specifications."
    )
    sentences.append(
        'The primary dimension of rideshare inequity in our data is not price but access. '
        f'The ~{ratio:.1f}x gap in rideshare utilization between affluent and transit-desert neighborhoods suggests that '
        'structural barriers — whether platform accessibility, driver availability, or affordability of total trip cost — '
        'prevent the platform from fulfilling its stated equity mission in the neighborhoods that need alternative transportation most.'
    )
    sentences.append(
        'Our analysis is limited to January 2024, a single winter month, and may not generalize to other seasons when demand patterns '
        'and trip characteristics differ. Additionally, our access gap measure reflects observed trips rather than latent demand — '
        'we cannot distinguish between residents who choose not to use rideshare and those who face barriers to access.'
    )

    sent_out = TAB_DIR / "step8_key_sentences.txt"
    with open(sent_out, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.strip() + "\n")
    print("saved:", sent_out)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


FEATURES = [
    "median_income",
    "poverty_rate",  # renamed to pct_poverty for clustering output
    "pct_black",
    "pct_hispanic",
    "transit_deprivation_score",
]

HEATMAP_LABELS = {
    "median_income": "Median Income",
    "pct_poverty": "% Poverty",
    "pct_black": "% Black",
    "pct_hispanic": "% Hispanic",
    "transit_deprivation_score": "Deprivation Score",
}


def _ensure_dirs() -> Tuple[Path, Path]:
    fig_dir = Path("outputs/figures")
    tab_dir = Path("outputs/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, tab_dir


def _load_ca_names() -> Dict[int, str]:
    boundaries_path = Path("/Users/minseojung/Downloads/Boundaries_-_Community_Areas_20260517.csv")
    if not boundaries_path.exists():
        return {}
    b = pd.read_csv(boundaries_path, usecols=["AREA_NUMBE", "COMMUNITY"], low_memory=False)
    b = b.rename(columns={"AREA_NUMBE": "community_area", "COMMUNITY": "community"})
    b["community_area"] = pd.to_numeric(b["community_area"], errors="coerce").astype("Int64")
    b["community"] = b["community"].astype("string").str.strip()
    b = b.dropna(subset=["community_area", "community"]).drop_duplicates(subset=["community_area"])
    out: Dict[int, str] = {}
    for k, v in b.set_index("community_area")["community"].to_dict().items():
        if pd.isna(k) or pd.isna(v):
            continue
        out[int(k)] = str(v)
    return out


def _select_acs_features(acs: pd.DataFrame) -> pd.DataFrame:
    acs = acs.copy()
    if "community_area" not in acs.columns:
        raise SystemExit("acs parquet missing `community_area` column")

    for col in FEATURES:
        if col not in acs.columns:
            raise SystemExit(f"acs parquet missing `{col}` column")

    acs["community_area"] = pd.to_numeric(acs["community_area"], errors="coerce").astype("Int64")
    acs = acs.dropna(subset=["community_area"])
    acs = acs[(acs["community_area"] >= 1) & (acs["community_area"] <= 77)]

    # Enforce one row per community area.
    if acs["community_area"].duplicated().any():
        acs = (
            acs.groupby("community_area", as_index=False)[FEATURES]
            .mean(numeric_only=True)
            .merge(acs[["community_area"]].drop_duplicates(), on="community_area", how="right")
        )

    # Build clustering frame.
    X = acs[["community_area"] + FEATURES].copy()
    X = X.rename(columns={"poverty_rate": "pct_poverty"})
    return X


def _kmeans_inertia_curve(Xz: np.ndarray, ks: List[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(Xz)
        rows.append({"k": k, "inertia": float(km.inertia_)})
    return pd.DataFrame(rows)


def _choose_k_from_elbow(curve: pd.DataFrame) -> int:
    # Simple knee heuristic: maximize distance to line between endpoints.
    ks = curve["k"].to_numpy(dtype=float)
    ys = curve["inertia"].to_numpy(dtype=float)
    # Normalize for numerical stability
    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    p1 = np.array([ks_n[0], ys_n[0]])
    p2 = np.array([ks_n[-1], ys_n[-1]])
    v = p2 - p1
    v_norm = np.linalg.norm(v) + 1e-12
    dists = []
    for x, y in zip(ks_n, ys_n):
        p = np.array([x, y])
        # distance from p to line through p1->p2 (2D cross product magnitude)
        w = p - p1
        cross_mag = v[0] * w[1] - v[1] * w[0]
        dist = np.abs(cross_mag) / v_norm
        dists.append(dist)
    k_star = int(curve.loc[int(np.argmax(dists)), "k"])

    # Per instructions, choose 4 or 5 (pick the closer).
    return 4 if abs(k_star - 4) <= abs(k_star - 5) else 5


def _label_clusters(centroids: pd.DataFrame) -> Dict[int, str]:
    # centroids contains one row per cluster with means in original units.
    # We label clusters in order of deprivation (low -> high).
    ordered = centroids.sort_values("transit_deprivation_score").reset_index(drop=True)
    labels: Dict[int, str] = {}
    k = len(ordered)

    base = []
    if k == 4:
        base = [
            "Affluent / low deprivation",
            "Mixed / lower deprivation",
            "Mixed / higher deprivation",
            "High poverty / transit deserts",
        ]
    elif k == 5:
        base = [
            "Affluent / low deprivation",
            "Mixed / lower deprivation",
            "Mixed / moderate deprivation",
            "Lower income / higher deprivation",
            "High poverty / transit deserts",
        ]
    else:
        base = [f"Cluster {i}" for i in range(k)]

    for i, row in ordered.iterrows():
        labels[int(row["cluster_id"])] = base[i] if i < len(base) else f"Cluster {i}"
    return labels


def _plot_elbow(curve: pd.DataFrame, selected_k: int, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=curve, x="k", y="inertia", marker="o")
    plt.title("K-Means Elbow Curve (ACS Community Areas)")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.axvline(selected_k, linestyle="--", color="black", alpha=0.7)
    y_at_k = float(curve.loc[curve["k"] == selected_k, "inertia"].iloc[0])
    plt.annotate(
        f"selected k={selected_k}",
        xy=(selected_k, y_at_k),
        xytext=(selected_k + 0.15, y_at_k * 1.02),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def _plot_dendrogram(Xz: np.ndarray, labels: List[str], k: int, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    Z = linkage(Xz, method="ward")
    plt.figure(figsize=(12, 5))
    # Color by a horizontal cut that yields k clusters.
    color_threshold = float(Z[-(k - 1), 2]) if k > 1 else None
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=color_threshold,
        above_threshold_color="grey",
    )
    plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
    plt.xlabel("Community Area")
    plt.ylabel("Distance")
    plt.figtext(
        0.01,
        0.01,
        f"Note: Leaf labels are Chicago Community Area numbers. Colors show groups at the k={k} cut.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def _plot_centroid_heatmap(centroids_z: pd.DataFrame, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    pretty = centroids_z.copy()
    pretty = pretty.rename(columns=HEATMAP_LABELS)
    # Prefer showing clusters ordered by deprivation if labels already reflect ordering.
    plt.figure(figsize=(10, 4))
    sns.heatmap(
        pretty.set_index("cluster_label"),
        cmap="RdBu_r",
        center=0.0,
        annot=False,
        cbar_kws={"label": "Standardized value (z-score)"},
    )
    plt.title("Cluster Centroid Heatmap (Standardized ACS Variables)")
    plt.xlabel("Variable")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def _plot_violin(df: pd.DataFrame, trip_type: str, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    sub = df.loc[df["trip_type"] == trip_type].copy()
    sub = sub.dropna(subset=["pickup_cluster_label"])
    plt.figure(figsize=(10, 4))
    order = (
        sub.groupby("pickup_cluster_label")["pickup_transit_deprivation_score"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    sns.violinplot(
        data=sub,
        x="pickup_cluster_label",
        y="fare_per_mile",
        order=order,
        cut=0,
        inner="quartile",
    )
    plt.title(f"{trip_type.title()} Fare per Mile by Pickup Neighborhood Cluster")
    plt.xlabel("Pickup Cluster")
    plt.ylabel("Fare per Mile ($/mile)")
    if trip_type == "rideshare":
        plt.ylim(0, 20)
    else:
        plt.ylim(0, 20)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster Chicago community areas using ACS features.")
    parser.add_argument(
        "--acs",
        type=Path,
        default=Path("data/processed/acs_clean.parquet"),
        help="Cleaned ACS parquet path.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters (if omitted, choose 4 or 5 using elbow heuristic).",
    )
    parser.add_argument(
        "--out-acs",
        type=Path,
        default=Path("data/processed/acs_clustered.parquet"),
        help="Output ACS parquet with cluster labels.",
    )
    parser.add_argument(
        "--out-tnc",
        type=Path,
        default=Path("data/processed/tnc_merged_clustered.parquet"),
        help="Output rideshare trips parquet with cluster labels.",
    )
    parser.add_argument(
        "--out-taxi",
        type=Path,
        default=Path("data/processed/taxi_merged_clustered.parquet"),
        help="Output taxi trips parquet with cluster labels.",
    )
    parser.add_argument(
        "--out-combined",
        type=Path,
        default=Path("data/processed/combined_merged_clustered.parquet"),
        help="Output stacked trips parquet with cluster labels.",
    )
    args = parser.parse_args()

    fig_dir, tab_dir = _ensure_dirs()
    ca_names = _load_ca_names()

    acs = pd.read_parquet(args.acs)
    X = _select_acs_features(acs)
    print("acs rows (community areas):", len(X))

    cluster_vars = [
        "median_income",
        "pct_poverty",
        "pct_black",
        "pct_hispanic",
        "transit_deprivation_score",
    ]

    before = len(X)
    X_clean = X.dropna(subset=cluster_vars).copy()
    dropped = before - len(X_clean)
    print("dropped community areas due to missing clustering vars:", dropped)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_clean[cluster_vars].to_numpy(dtype=float))

    # 6B: Elbow curve
    ks = list(range(2, 9))
    curve = _kmeans_inertia_curve(Xz, ks)
    curve.to_csv(tab_dir / "kmeans_inertia_curve.csv", index=False)
    # Choose k from elbow + enforce 4/5.
    k = args.k if args.k is not None else _choose_k_from_elbow(curve)
    if k not in (4, 5):
        print("WARNING: forcing k to 4 or 5 per instructions; requested k =", k)
        k = 4 if k < 5 else 5
    _plot_elbow(curve, selected_k=k, out_path=fig_dir / "kmeans_elbow_curve.png")

    # 6B: Dendrogram (Ward linkage)
    _plot_dendrogram(
        Xz,
        labels=[str(int(x)) for x in X_clean["community_area"]],
        k=k,
        out_path=fig_dir / "dendrogram.png",
    )

    print("chosen k:", k)

    # 6C: Final KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    X_clean["cluster_id_raw"] = km.fit_predict(Xz).astype(int)

    # Re-number clusters by mean deprivation (low -> high) for stable ordering.
    tmp = X_clean.groupby("cluster_id_raw", as_index=False)["transit_deprivation_score"].mean()
    order = tmp.sort_values("transit_deprivation_score")["cluster_id_raw"].tolist()
    remap = {old: new for new, old in enumerate(order)}
    X_clean["cluster_id"] = X_clean["cluster_id_raw"].map(remap).astype(int)
    X_clean = X_clean.drop(columns=["cluster_id_raw"])

    centroids = (
        X_clean.groupby("cluster_id", as_index=False)[cluster_vars]
        .mean()
        .rename(columns={"cluster_id": "cluster_id"})
    )
    label_map = _label_clusters(centroids.assign(cluster_id=centroids["cluster_id"]))
    X_clean["cluster_label"] = X_clean["cluster_id"].map(label_map)

    # Quick check: top/bottom 10 by deprivation score (with names, if available).
    depr_rank = X_clean[["community_area", "transit_deprivation_score"]].copy()
    depr_rank["community_area"] = depr_rank["community_area"].astype(int)
    depr_rank["community_name"] = depr_rank["community_area"].map(ca_names).fillna("")
    depr_rank = depr_rank.sort_values("transit_deprivation_score")
    bottom10 = depr_rank.head(10).copy()
    top10 = depr_rank.tail(10).copy()
    print("Bottom 10 community areas by transit_deprivation_score (lowest deprivation):")
    print(bottom10.to_string(index=False))
    print("Top 10 community areas by transit_deprivation_score (highest deprivation):")
    print(top10.to_string(index=False))
    pd.concat(
        [bottom10.assign(rank_group="bottom10"), top10.assign(rank_group="top10")],
        ignore_index=True,
    ).to_csv(tab_dir / "deprivation_top_bottom_10.csv", index=False)

    # Centroid tables
    centroids_labeled = centroids.copy()
    centroids_labeled["cluster_label"] = centroids_labeled["cluster_id"].map(label_map)
    centroids_labeled.to_csv(tab_dir / "cluster_centroids.csv", index=False)

    # Standardized centroid heatmap
    centroids_z = pd.DataFrame(
        scaler.transform(centroids[cluster_vars].to_numpy(dtype=float)),
        columns=cluster_vars,
    )
    centroids_z["cluster_id"] = centroids["cluster_id"]
    centroids_z["cluster_label"] = centroids_labeled["cluster_label"]
    centroids_z.to_csv(tab_dir / "cluster_centroids_z.csv", index=False)
    _plot_centroid_heatmap(
        centroids_z[["cluster_label"] + cluster_vars],
        fig_dir / "cluster_centroid_heatmap.png",
    )

    # Note: pct_no_vehicle is intentionally excluded because the current ACS extract does not include it.

    # 6F: deprivation ordering check
    depr_table = (
        X_clean.groupby(["cluster_id", "cluster_label"], as_index=False)["transit_deprivation_score"]
        .mean()
        .sort_values("transit_deprivation_score")
    )
    print("mean transit_deprivation_score per cluster (low -> high):")
    print(depr_table.to_string(index=False))
    depr_table.to_csv(tab_dir / "cluster_deprivation_order.csv", index=False)

    # Correlation check: cluster ordering vs deprivation (should be strongly positive after re-numbering).
    rho, pval = spearmanr(X_clean["cluster_id"], X_clean["transit_deprivation_score"])
    print(f"Spearman corr(cluster_id, transit_deprivation_score) = {rho:.3f} (p={pval:.3g})")
    pd.DataFrame(
        [{"spearman_rho": float(rho), "p_value": float(pval), "k": int(k)}]
    ).to_csv(tab_dir / "cluster_deprivation_spearman.csv", index=False)

    # Save clustered ACS
    acs_out = X_clean[["community_area", "cluster_id", "cluster_label"]].copy()
    args.out_acs.parent.mkdir(parents=True, exist_ok=True)
    acs_out.to_parquet(args.out_acs, index=False)
    print("saved:", args.out_acs)

    # 6D: Merge onto trips (pickup + dropoff)
    def merge_trips(trips_path: Path, out_path: Path, label: str) -> pd.DataFrame:
        trips = pd.read_parquet(trips_path)
        m = acs_out.rename(
            columns={
                "community_area": "pickup_ca",
                "cluster_id": "pickup_cluster_id",
                "cluster_label": "pickup_cluster_label",
            }
        )
        trips = trips.merge(m, how="left", on="pickup_ca")
        m2 = acs_out.rename(
            columns={
                "community_area": "dropoff_ca",
                "cluster_id": "dropoff_cluster_id",
                "cluster_label": "dropoff_cluster_label",
            }
        )
        trips = trips.merge(m2, how="left", on="dropoff_ca")

        pickup_ok = trips["pickup_cluster_id"].notna().mean() * 100.0 if len(trips) else 0.0
        dropoff_ok = trips["dropoff_cluster_id"].notna().mean() * 100.0 if len(trips) else 0.0
        print(f"{label}: pickup clustered = {pickup_ok:.2f}%, dropoff clustered = {dropoff_ok:.2f}%")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        trips.to_parquet(out_path, index=False)
        print("saved:", out_path)
        return trips

    tnc_clustered = merge_trips(Path("data/processed/tnc_merged.parquet"), args.out_tnc, "tnc")
    taxi_clustered = merge_trips(Path("data/processed/taxi_merged.parquet"), args.out_taxi, "taxi")
    combined_clustered = pd.concat([tnc_clustered, taxi_clustered], ignore_index=True)
    args.out_combined.parent.mkdir(parents=True, exist_ok=True)
    combined_clustered.to_parquet(args.out_combined, index=False)
    print("saved:", args.out_combined)

    # Quick check: mean distance by pickup cluster (compositional explanation).
    dist_by_cluster = (
        combined_clustered.dropna(subset=["pickup_cluster_label"])
        .groupby("pickup_cluster_label", as_index=False)
        .agg(mean_distance_miles=("distance_miles", "mean"), n_trips=("distance_miles", "size"))
        .sort_values("mean_distance_miles")
    )
    print("Mean distance_miles by pickup cluster (all trips):")
    print(dist_by_cluster.to_string(index=False))
    dist_by_cluster.to_csv(tab_dir / "mean_distance_by_cluster_all.csv", index=False)

    # 6E: Violin plots (pickup cluster)
    _plot_violin(combined_clustered, "rideshare", fig_dir / "violin_fare_per_mile_by_cluster_rideshare.png")
    _plot_violin(combined_clustered, "taxi", fig_dir / "violin_fare_per_mile_by_cluster_taxi.png")

    # Extra Step-7 readiness check: mean fare/mile vs deprivation across community areas.
    rideshare = combined_clustered.loc[combined_clustered["trip_type"] == "rideshare"].copy()
    if len(rideshare):
        ca = (
            rideshare.groupby("pickup_ca", as_index=False)
            .agg(
                mean_fare_per_mile=("fare_per_mile", "mean"),
                deprivation=("pickup_transit_deprivation_score", "mean"),
                n_trips=("fare_per_mile", "size"),
            )
            .dropna(subset=["mean_fare_per_mile", "deprivation"])
        )
        pearson = ca["mean_fare_per_mile"].corr(ca["deprivation"])
        rho2, p2 = spearmanr(ca["mean_fare_per_mile"], ca["deprivation"])
        print(
            "Correlation across pickup community areas (rideshare): "
            f"Pearson={pearson:.3f}, Spearman={rho2:.3f} (p={p2:.3g})"
        )
        ca.to_csv(tab_dir / "ca_mean_fare_per_mile_vs_deprivation_rideshare.csv", index=False)
        pd.DataFrame(
            [
                {
                    "pearson_r": float(pearson),
                    "spearman_rho": float(rho2),
                    "spearman_p": float(p2),
                    "n_community_areas": int(len(ca)),
                }
            ]
        ).to_csv(tab_dir / "ca_mean_fare_per_mile_vs_deprivation_corr.csv", index=False)


if __name__ == "__main__":
    main()

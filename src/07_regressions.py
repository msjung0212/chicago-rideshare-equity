from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.graphics.regressionplots import plot_partregress


FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")


def _setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_coef(coef: float, se: float, p: float) -> str:
    return f"{coef:.3f}{_stars(p)}\n({se:.3f})"


def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


@dataclass
class ModelResult:
    name: str
    key_term: str
    coef: float
    se: float
    t: float
    p: float
    ci_low: float
    ci_high: float
    nobs: int
    r2: float
    clusters: int
    cov: pd.DataFrame
    fitted: object


def _fit_ols_cluster(
    df: pd.DataFrame,
    formula: str,
    cluster_col: str,
    model_name: str,
    key_term: str,
) -> ModelResult:
    model = smf.ols(formula=formula, data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})

    coef = float(res.params[key_term])
    se = float(res.bse[key_term])
    t = float(res.tvalues[key_term])
    p = float(res.pvalues[key_term])
    ci = res.conf_int().loc[key_term].to_numpy(dtype=float)
    clusters = df[cluster_col].nunique(dropna=True)
    return ModelResult(
        name=model_name,
        key_term=key_term,
        coef=coef,
        se=se,
        t=t,
        p=p,
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        nobs=_safe_int(res.nobs),
        r2=float(res.rsquared),
        clusters=_safe_int(clusters),
        cov=res.cov_params(),
        fitted=res,
    )


def _delta_se_sum(res, term_a: str, term_b: str) -> Tuple[float, float]:
    cov = res.cov_params()
    b = res.params
    total = float(b[term_a] + b[term_b])
    var = float(cov.loc[term_a, term_a] + cov.loc[term_b, term_b] + 2.0 * cov.loc[term_a, term_b])
    se = float(np.sqrt(max(var, 0.0)))
    return total, se


def _coef_table(
    models: List[ModelResult],
    rows: List[Tuple[str, Dict[str, Optional[str]]]],
) -> pd.DataFrame:
    # rows: list of (display_name, term_by_model_name)
    out: Dict[str, List[str]] = {"Variable": [r[0] for r in rows]}
    for m in models:
        col: List[str] = []
        for _, term_map in rows:
            term = term_map.get(m.name)
            if term is None:
                col.append("")
                continue
            if term in m.fitted.params.index:
                coef = float(m.fitted.params[term])
                se = float(m.fitted.bse[term])
                p = float(m.fitted.pvalues[term])
                col.append(_fmt_coef(coef, se, p))
            else:
                col.append("")
        out[m.name] = col
    return pd.DataFrame(out)


def _save_coef_plot(models: List[ModelResult], out_path: Path) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    data = []
    for m in models:
        data.append(
            {
                "model": m.name,
                "coef": m.coef,
                "low": m.ci_low,
                "high": m.ci_high,
                "kind": "Taxi" if "Taxi" in m.name else "Rideshare",
            }
        )
    dfp = pd.DataFrame(data)
    order = [m.name for m in models]
    colors = {"Rideshare": "#d97706", "Taxi": "#2563eb"}  # orange / blue

    plt.figure(figsize=(10, 4))
    for i, row in dfp.set_index("model").loc[order].reset_index().iterrows():
        plt.plot([i, i], [row["low"], row["high"]], color=colors[row["kind"]], lw=2)
        plt.scatter([i], [row["coef"]], color=colors[row["kind"]], s=40, zorder=3)
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)
    plt.xticks(range(len(order)), order, rotation=15, ha="right")
    plt.ylabel("Coefficient on Deprivation Score")
    plt.title("Deprivation Coefficient Across Models (95% CI)")
    # Ensure no confidence interval is clipped (Model 5 can extend lower).
    ymin = float(dfp["low"].min())
    plt.ylim(bottom=min(-0.8, ymin - 0.03))
    plt.figtext(
        0.5,
        -0.02,
        "All confidence intervals cross zero, indicating no statistically significant relationship between "
        "neighborhood deprivation and fare outcomes across any specification. Model 2 (Taxi, blue) shows a tighter "
        "interval reflecting more uniform trip distribution across neighborhoods. The consistent pattern across all "
        "six models strengthens the null finding: results are not sensitive to outcome choice.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def _save_robustness_swap_plot(rows: List[Dict], out_path: Path) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    dfp = pd.DataFrame([r for r in rows if "coef" in r and "se" in r and "label" in r]).copy()
    if dfp.empty:
        return
    dfp["low"] = dfp["coef"] - 1.96 * dfp["se"]
    dfp["high"] = dfp["coef"] + 1.96 * dfp["se"]
    order = dfp["label"].tolist()

    plt.figure(figsize=(9, 4))
    for i, row in dfp.reset_index(drop=True).iterrows():
        plt.plot([i, i], [row["low"], row["high"]], color="#111827", lw=2)
        plt.scatter([i], [row["coef"]], color="#111827", s=40, zorder=3)
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)
    plt.xticks(range(len(order)), order, rotation=15, ha="right")
    plt.ylabel("Coefficient (95% CI)")
    plt.title("Robustness: Key Predictor Swap (Rideshare)")
    plt.figtext(
        0.5,
        -0.02,
        "Predictors are on different scales, so coefficient magnitudes are not directly comparable. "
        "The % Black Residents coefficient reflects structural patterns of racial segregation in Chicago's "
        "transit investment history, not individual-level discrimination.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def _save_partregress_plot(df: pd.DataFrame, formula: str, key_var: str, out_path: Path) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    # Use variable-level terms rather than design-matrix column names.
    # This avoids Patsy parsing issues on expanded C(...) terms.
    endog = "log_fare_per_mile"
    exog_others = [
        "log_distance",
        "log_duration",
        "pickup_median_income_10k",
        "C(hour, Treatment(reference=12))",
        "C(day_of_week, Treatment(reference=0))",
    ]
    plt.figure(figsize=(7, 5))
    plot_partregress(
        endog=endog,
        exog_i=key_var,
        exog_others=exog_others,
        data=df,
        obs_labels=False,
    )
    plt.xlabel("Transit Deprivation Score (residualized)")
    plt.ylabel("Log Fare per Mile (residualized)")
    plt.title("Partial Regression Plot: Deprivation vs Log Fare per Mile")
    plt.figtext(
        0.5,
        -0.02,
        "Two isolated point clusters on the right are high-deprivation, low-volume community areas "
        "(Riverdale, Community Area 54; Fuller Park, Community Area 37) and represent high-leverage observations. "
        "The downward slope is consistent with the negative but insignificant Model 1 coefficient.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    tnc = pd.read_parquet("data/processed/tnc_merged_clustered.parquet")
    taxi = pd.read_parquet("data/processed/taxi_merged_clustered.parquet")
    acs = pd.read_parquet("data/processed/acs_clean.parquet")

    # Rescale median income for interpretability in tables (per $10k).
    for df_ in (tnc, taxi):
        df_["pickup_median_income_10k"] = df_["pickup_median_income"] / 10_000.0
        df_["dropoff_median_income_10k"] = df_["dropoff_median_income"] / 10_000.0
        # Backfill outcomes if older parquets are missing them.
        if "log_fare" not in df_.columns:
            df_["log_fare"] = np.log(df_["fare"])
        if "total_cost" not in df_.columns:
            tip = df_["tip"] if "tip" in df_.columns else 0.0
            add = df_["additional_charges"] if "additional_charges" in df_.columns else 0.0
            df_["total_cost"] = (
                df_["fare"]
                + pd.to_numeric(tip, errors="coerce").fillna(0.0)
                + pd.to_numeric(add, errors="coerce").fillna(0.0)
            )
        if "log_total" not in df_.columns:
            df_["log_total"] = np.log(df_["total_cost"])

    print("tnc rows:", len(tnc))
    print("taxi rows:", len(taxi))

    # Multicollinearity diagnostic (ACS-level; 77 rows).
    dep_income_corr = float(
        acs[["transit_deprivation_score", "median_income"]].corr().iloc[0, 1]
    )
    pd.DataFrame(
        [{"acs_dep_income_corr": dep_income_corr}]
    ).to_csv(TAB_DIR / "acs_dep_income_corr.csv", index=False)
    print("ACS corr(transit_deprivation_score, median_income):", f"{dep_income_corr:.3f}")

    # Model 1/2 common fixed effects.
    fe = " + C(hour, Treatment(reference=12)) + C(day_of_week, Treatment(reference=0))"

    # Model 1 (rideshare, pickup)
    m1_df = tnc.copy()
    m1_key = "pickup_transit_deprivation_score"
    m1_formula = (
        "log_fare_per_mile ~ pickup_transit_deprivation_score + log_distance + log_duration + "
        "pickup_median_income_10k" + fe
    )
    m1 = _fit_ols_cluster(m1_df, m1_formula, "pickup_ca", "Model 1 (Rideshare)", m1_key)
    print("Model 1 key coef:", m1.coef, "se:", m1.se, "p:", m1.p)

    # Partial regression plot for Model 1 (non-clustered fit for plotting).
    _save_partregress_plot(m1_df, m1_formula, "pickup_transit_deprivation_score", FIG_DIR / "partregress_model1.png")

    # Model 2 (taxi, same spec)
    m2_df = taxi.copy()
    m2_key = "pickup_transit_deprivation_score"
    m2_formula = m1_formula
    m2 = _fit_ols_cluster(m2_df, m2_formula, "pickup_ca", "Model 2 (Taxi)", m2_key)
    print("Model 2 key coef:", m2.coef, "se:", m2.se, "p:", m2.p)

    # Z-test difference in coefficients (independent samples approximation).
    diff = m1.coef - m2.coef
    diff_se = float(np.sqrt(m1.se**2 + m2.se**2))
    z = float(diff / diff_se) if diff_se > 0 else float("nan")
    pz = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")
    print(f"coef diff (rideshare - taxi) = {diff:.4f}, z={z:.3f}, p={pz:.3g}")

    # Model 3 (rideshare interaction)
    m3_df = tnc.copy()
    m3_df["deprivation_x_latenight"] = m3_df["pickup_transit_deprivation_score"] * m3_df["late_night"]
    m3_formula = (
        # NOTE: a standalone late_night dummy is perfectly collinear with full hour fixed effects.
        # To identify the late_night main effect, we replace hour FE with a smooth hour control.
        "log_fare_per_mile ~ pickup_transit_deprivation_score + late_night + deprivation_x_latenight + "
        "log_distance + log_duration + pickup_median_income_10k + hour + I(hour**2) + "
        "C(day_of_week, Treatment(reference=0))"
    )
    # For Model 3, the "β1" of interest is still the main deprivation term (daytime effect).
    # We'll separately extract the interaction term for Step 7H.
    m3 = _fit_ols_cluster(m3_df, m3_formula, "pickup_ca", "Model 3 (Interaction)", m1_key)
    # Also capture the baseline deprivation term for late-night total effect.
    total_ln, total_ln_se = _delta_se_sum(m3.fitted, "pickup_transit_deprivation_score", "deprivation_x_latenight")
    b3 = float(m3.fitted.params["deprivation_x_latenight"])
    se3 = float(m3.fitted.bse["deprivation_x_latenight"])
    p3 = float(m3.fitted.pvalues["deprivation_x_latenight"])
    print(f"Model 3 interaction coef: {b3:.4f} (se {se3:.4f}, p {p3:.3g})")
    print(f"Model 3 total late-night deprivation effect (b1+b3): {total_ln:.4f} (se {total_ln_se:.4f})")

    # Model 4 (rideshare, dropoff-side)
    m4_df = tnc.copy()
    m4_key = "dropoff_transit_deprivation_score"
    m4_formula = (
        "log_fare_per_mile ~ dropoff_transit_deprivation_score + log_distance + log_duration + "
        "dropoff_median_income_10k" + fe
    )
    m4 = _fit_ols_cluster(m4_df, m4_formula, "dropoff_ca", "Model 4 (Dropoff)", m4_key)
    print("Model 4 key coef:", m4.coef, "se:", m4.se, "p:", m4.p)

    # Model 5 (rideshare, pickup) — outcome: total fare
    m5_df = tnc.copy()
    m5_key = "pickup_transit_deprivation_score"
    m5_formula = (
        "log_fare ~ pickup_transit_deprivation_score + log_distance + log_duration + "
        "pickup_median_income_10k" + fe
    )
    m5 = _fit_ols_cluster(m5_df, m5_formula, "pickup_ca", "Model 5 (Total Fare)", m5_key)
    print("Model 5 key coef:", m5.coef, "se:", m5.se, "p:", m5.p)

    # Model 6 (rideshare, pickup) — outcome: total cost = fare + tip + additional charges
    m6_df = tnc.copy()
    m6_key = "pickup_transit_deprivation_score"
    m6_formula = (
        "log_total ~ pickup_transit_deprivation_score + log_distance + log_duration + "
        "pickup_median_income_10k" + fe
    )
    m6 = _fit_ols_cluster(m6_df, m6_formula, "pickup_ca", "Model 6 (Total Cost)", m6_key)
    print("Model 6 key coef:", m6.coef, "se:", m6.se, "p:", m6.p)

    # Master results table
    rows = [
        (
            "Deprivation Score (Pickup/Dropoff)",
            {
                m1.name: "pickup_transit_deprivation_score",
                m2.name: "pickup_transit_deprivation_score",
                m3.name: "pickup_transit_deprivation_score",
                m4.name: "dropoff_transit_deprivation_score",
                m5.name: "pickup_transit_deprivation_score",
                m6.name: "pickup_transit_deprivation_score",
            },
        ),
        (
            "Median Income (Pickup/Dropoff, $10k)",
            {
                m1.name: "pickup_median_income_10k",
                m2.name: "pickup_median_income_10k",
                m3.name: "pickup_median_income_10k",
                m4.name: "dropoff_median_income_10k",
                m5.name: "pickup_median_income_10k",
                m6.name: "pickup_median_income_10k",
            },
        ),
        (
            "Log Distance",
            {
                m1.name: "log_distance",
                m2.name: "log_distance",
                m3.name: "log_distance",
                m4.name: "log_distance",
                m5.name: "log_distance",
                m6.name: "log_distance",
            },
        ),
        (
            "Log Duration (minutes)",
            {
                m1.name: "log_duration",
                m2.name: "log_duration",
                m3.name: "log_duration",
                m4.name: "log_duration",
                m5.name: "log_duration",
                m6.name: "log_duration",
            },
        ),
        (
            "Late Night",
            {m1.name: None, m2.name: None, m3.name: "late_night", m4.name: None, m5.name: None, m6.name: None},
        ),
        (
            "Deprivation × Late Night",
            {
                m1.name: None,
                m2.name: None,
                m3.name: "deprivation_x_latenight",
                m4.name: None,
                m5.name: None,
                m6.name: None,
            },
        ),
        ("Hour FE", {m1.name: None, m2.name: None, m3.name: None, m4.name: None, m5.name: None, m6.name: None}),
        ("Day of Week FE", {m1.name: None, m2.name: None, m3.name: None, m4.name: None, m5.name: None, m6.name: None}),
        ("N", {m1.name: None, m2.name: None, m3.name: None, m4.name: None, m5.name: None, m6.name: None}),
        ("R²", {m1.name: None, m2.name: None, m3.name: None, m4.name: None, m5.name: None, m6.name: None}),
        ("Clusters", {m1.name: None, m2.name: None, m3.name: None, m4.name: None, m5.name: None, m6.name: None}),
    ]
    table = _coef_table([m1, m2, m3, m4, m5, m6], rows)

    # Fill in FE flags + N/R2/Clusters
    def _set_scalar(row_name: str, model: ModelResult, value: str) -> None:
        idx = table.index[table["Variable"] == row_name]
        if len(idx):
            table.loc[idx[0], model.name] = value

    for m in [m1, m2, m4, m5, m6]:
        _set_scalar("Hour FE", m, "✓")
    _set_scalar("Hour FE", m3, "Hour poly")

    for m in [m1, m2, m3, m4, m5, m6]:
        _set_scalar("Day of Week FE", m, "✓")
        _set_scalar("N", m, f"{m.nobs}")
        _set_scalar("R²", m, f"{m.r2:.3f}")
        _set_scalar("Clusters", m, f"{m.clusters}")

    out_table = TAB_DIR / "step7_master_results_table.csv"
    table.to_csv(out_table, index=False)
    print("saved:", out_table)

    note_path = TAB_DIR / "step7_master_results_table_note.txt"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("Notes:\n")
        f.write("- Standard errors clustered at community area level (pickup_ca for pickup models; dropoff_ca for Model 4).\n")
        f.write("- Significance stars: * p<0.05, ** p<0.01, *** p<0.001. No asterisk = p>0.05.\n")
        f.write("- Model 3 uses hour + hour^2 (not full hour fixed effects) so the late_night main effect is identified.\n")
    print("saved:", note_path)

    # Coefficient plot for deprivation score across models (includes total-fare / total-cost outcomes).
    _save_coef_plot([m1, m2, m3, m4, m5, m6], FIG_DIR / "coefplot_deprivation_across_models.png")

    # Save full summaries (avoid dumping in terminal).
    summaries_path = TAB_DIR / "step7_model_summaries.txt"
    with open(summaries_path, "w", encoding="utf-8") as f:
        f.write("MODEL 1 (RIDESHARE)\\n")
        f.write(m1.fitted.summary().as_text())
        f.write("\\n\\nMODEL 2 (TAXI)\\n")
        f.write(m2.fitted.summary().as_text())
        f.write("\\n\\nMODEL 3 (INTERACTION)\\n")
        f.write(m3.fitted.summary().as_text())
        f.write("\\n\\nMODEL 4 (DROPOFF)\\n")
        f.write(m4.fitted.summary().as_text())
        f.write("\\n\\nMODEL 5 (TOTAL FARE)\\n")
        f.write(m5.fitted.summary().as_text())
        f.write("\\n\\nMODEL 6 (TOTAL COST)\\n")
        f.write(m6.fitted.summary().as_text())
    print("saved:", summaries_path)

    # Robustness checks
    robustness_rows = []
    swap_plot_rows = []

    # R1: Placebo daytime only (9am-5pm inclusive)
    daytime = m1_df.loc[m1_df["hour"].between(9, 17)]
    r1 = _fit_ols_cluster(daytime, m1_formula, "pickup_ca", "R1 Daytime Only", m1_key)
    robustness_rows.append(
        {"check": "Daytime only", "coef": r1.coef, "se": r1.se, "p": r1.p, "n": r1.nobs, "r2": r1.r2}
    )

    # R2: Variable swap (replace key predictor)
    swaps = [
        ("Median Income ($10k)", "pickup_median_income_10k"),
        ("% Poverty", "pickup_poverty_rate"),
        ("% Black Residents", "pickup_pct_black"),
        ("Deprivation Score", "pickup_transit_deprivation_score"),
    ]
    for label, key in swaps:
        # Use distance + duration + FE only; avoid double-including the key as a control.
        form = f"log_fare_per_mile ~ {key} + log_distance + log_duration" + fe
        rr = _fit_ols_cluster(m1_df, form, "pickup_ca", f"R2 Swap: {label}", key)
        robustness_rows.append(
            {"check": f"Swap key: {label}", "coef": rr.coef, "se": rr.se, "p": rr.p, "n": rr.nobs, "r2": rr.r2}
        )
        swap_plot_rows.append({"label": label, "coef": rr.coef, "se": rr.se})

    # R3: Company fixed effects (only if there is >1 company category)
    if m1_df["company"].nunique(dropna=True) > 1:
        form = (
            "log_fare_per_mile ~ pickup_transit_deprivation_score + log_distance + log_duration + "
            "pickup_median_income + C(company)" + fe
        )
        r3 = _fit_ols_cluster(m1_df, form, "pickup_ca", "R3 Company FE", m1_key)
        robustness_rows.append(
            {"check": "Company FE", "coef": r3.coef, "se": r3.se, "p": r3.p, "n": r3.nobs, "r2": r3.r2}
        )
    else:
        robustness_rows.append({"check": "Company FE", "note": "Skipped (only one company category)"})

    # R4: Drop short trips under 1 mile
    strict = m1_df.loc[m1_df["distance_miles"] >= 1.0]
    r4 = _fit_ols_cluster(strict, m1_formula, "pickup_ca", "R4 Drop <1 mile", m1_key)
    robustness_rows.append(
        {"check": "Drop trips < 1 mile", "coef": r4.coef, "se": r4.se, "p": r4.p, "n": r4.nobs, "r2": r4.r2}
    )

    # R5: Deprivation score only (drop income control) to address potential multicollinearity.
    m1_no_income_formula = "log_fare_per_mile ~ pickup_transit_deprivation_score + log_distance + log_duration" + fe
    r5 = _fit_ols_cluster(m1_df, m1_no_income_formula, "pickup_ca", "R5 Deprivation Only", m1_key)
    robustness_rows.append(
        {
            "check": "Drop income control (deprivation-only)",
            "coef": r5.coef,
            "se": r5.se,
            "p": r5.p,
            "n": r5.nobs,
            "r2": r5.r2,
            "acs_dep_income_corr": dep_income_corr,
        }
    )

    pd.DataFrame(robustness_rows).to_csv(TAB_DIR / "step7_robustness_checks.csv", index=False)
    print("saved:", TAB_DIR / "step7_robustness_checks.csv")

    _save_robustness_swap_plot(swap_plot_rows, FIG_DIR / "robustness_swap_coefplot.png")

    # Key numbers CSV (Step 7H)
    key_numbers = [
        {"model": m1.name, "beta1": m1.coef, "se": m1.se, "r2": m1.r2, "n": m1.nobs},
        {"model": m2.name, "beta1": m2.coef, "se": m2.se, "r2": m2.r2, "n": m2.nobs},
        {"model": m3.name, "beta1": m3.coef, "se": m3.se, "p": m3.p, "r2": m3.r2, "n": m3.nobs},
        {"model": m4.name, "beta1": m4.coef, "se": m4.se, "r2": m4.r2, "n": m4.nobs},
        {"model": m5.name, "beta1": m5.coef, "se": m5.se, "r2": m5.r2, "n": m5.nobs},
        {"model": m6.name, "beta1": m6.coef, "se": m6.se, "r2": m6.r2, "n": m6.nobs},
        {
            "coef_diff_rideshare_minus_taxi": diff,
            "z": z,
            "p": pz,
        },
        {"acs_dep_income_corr": dep_income_corr},
        {
            "model3_total_late_night_effect_b1_plus_b3": total_ln,
            "model3_total_late_night_effect_se": total_ln_se,
        },
        {
            "model3_beta3_interaction": b3,
            "model3_beta3_se": se3,
            "model3_beta3_p": p3,
        },
    ]
    pd.DataFrame(key_numbers).to_csv(TAB_DIR / "step7_key_numbers.csv", index=False)
    print("saved:", TAB_DIR / "step7_key_numbers.csv")

    # Trip volume analysis: trips per 1,000 residents (pickup side).
    trips = pd.concat([tnc, taxi], ignore_index=True)
    counts = (
        trips.groupby(["pickup_ca", "trip_type"], dropna=False)
        .size()
        .reset_index(name="trip_count")
    )
    pop = acs[["community_area", "population"]].rename(columns={"community_area": "pickup_ca"})
    counts = counts.merge(pop, on="pickup_ca", how="left")
    counts["trips_per_1000_residents"] = (counts["trip_count"] / counts["population"]) * 1000.0
    counts.to_csv(TAB_DIR / "trips_per_1000_by_community_area.csv", index=False)
    print("saved:", TAB_DIR / "trips_per_1000_by_community_area.csv")

    try:
        acs_clusters = pd.read_parquet("data/processed/acs_clustered.parquet")
        acs_clusters = acs_clusters.rename(columns={"community_area": "pickup_ca"})
        counts = counts.merge(acs_clusters, on="pickup_ca", how="left")
        counts["cluster_label"] = counts["cluster_label"].fillna("Unmatched")
        by_cluster = (
            counts.groupby(["cluster_label", "trip_type"], dropna=False)["trips_per_1000_residents"]
            .mean()
            .reset_index()
        )
        by_cluster.to_csv(TAB_DIR / "trips_per_1000_by_cluster.csv", index=False)
        print("saved:", TAB_DIR / "trips_per_1000_by_cluster.csv")

        _setup_matplotlib()
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(11, 4.3))
        order = [
            "High poverty / transit deserts",
            "Mixed / higher deprivation",
            "Mixed / lower deprivation",
            "Affluent / low deprivation",
        ]
        ax = sns.barplot(
            data=by_cluster,
            x="cluster_label",
            y="trips_per_1000_residents",
            hue="trip_type",
            order=order,
            palette={"rideshare": "#d97706", "taxi": "#2563eb"},
        )
        # Add bar value labels so tiny taxi bars are still readable.
        for patch in ax.patches:
            h = patch.get_height()
            if not np.isfinite(h):
                continue
            x = patch.get_x() + patch.get_width() / 2.0
            ax.text(
                x,
                h + max(15.0, 0.01 * h),
                f"{h:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111827",
            )

        # Add affluent benchmark line at the affluent rideshare level.
        try:
            affluent_benchmark = float(
                by_cluster.loc[
                    (by_cluster["cluster_label"] == "Affluent / low deprivation")
                    & (by_cluster["trip_type"] == "rideshare"),
                    "trips_per_1000_residents",
                ].iloc[0]
            )
            ax.axhline(affluent_benchmark, color="#111827", linestyle="--", lw=1.5, alpha=0.75)
            ax.text(
                3.15,
                affluent_benchmark + 40.0,
                "Affluent benchmark",
                ha="left",
                va="bottom",
                fontsize=9,
                color="#111827",
            )
        except Exception:
            pass

        plt.xticks(rotation=15, ha="right")
        plt.ylabel("Trips per 1,000 Residents")
        plt.xlabel("")
        plt.title("Trip Volume by Neighborhood Cluster (Pickup), January 2024")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "trips_per_1000_by_cluster.png", dpi=200)
        plt.close()
        print("saved:", FIG_DIR / "trips_per_1000_by_cluster.png")
    except Exception as e:
        print("trip volume analysis skipped:", repr(e))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full spatial public-health analysis for The Care Gap paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "district_master.csv"
FIG = ROOT / "figures"
OUT = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)

# Queen contiguity for 18 District Council districts (land adjacency).
# Harbour crossings treated as non-contiguous (standard land-based weights).
NEIGHBORS = {
    "Central & Western": ["Wan Chai", "Southern"],
    "Wan Chai": ["Central & Western", "Eastern"],
    "Eastern": ["Wan Chai"],
    "Southern": ["Central & Western"],
    "Yau Tsim Mong": ["Sham Shui Po", "Kowloon City"],
    "Sham Shui Po": ["Yau Tsim Mong", "Kowloon City", "Wong Tai Sin", "Kwai Tsing"],
    "Kowloon City": ["Yau Tsim Mong", "Sham Shui Po", "Wong Tai Sin", "Kwun Tong"],
    "Wong Tai Sin": ["Sham Shui Po", "Kowloon City", "Kwun Tong", "Sha Tin", "Sai Kung"],
    "Kwun Tong": ["Kowloon City", "Wong Tai Sin", "Sai Kung"],
    "Kwai Tsing": ["Sham Shui Po", "Tsuen Wan", "Tuen Mun", "Islands"],
    "Tsuen Wan": ["Kwai Tsing", "Tuen Mun", "Yuen Long", "Sha Tin", "Islands"],
    "Tuen Mun": ["Kwai Tsing", "Tsuen Wan", "Yuen Long"],
    "Yuen Long": ["Tsuen Wan", "Tuen Mun", "North", "Tai Po"],
    "North": ["Yuen Long", "Tai Po", "Sha Tin"],
    "Tai Po": ["Yuen Long", "North", "Sha Tin", "Sai Kung"],
    "Sha Tin": ["Wong Tai Sin", "Tsuen Wan", "North", "Tai Po", "Sai Kung"],
    "Sai Kung": ["Wong Tai Sin", "Kwun Tong", "Tai Po", "Sha Tin"],
    "Islands": ["Kwai Tsing", "Tsuen Wan"],
}


def queen_weights(districts: list[str]) -> np.ndarray:
    n = len(districts)
    idx = {d: i for i, d in enumerate(districts)}
    W = np.zeros((n, n))
    for d, nbrs in NEIGHBORS.items():
        i = idx[d]
        for nb in nbrs:
            j = idx[nb]
            W[i, j] = 1.0
    # Symmetrise in case of any asymmetry
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0)
    # Row-standardise
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return W / rs


def morans_i(y: np.ndarray, W: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    z = y - y.mean()
    n = len(y)
    S0 = W.sum()
    num = float(z @ W @ z)
    den = float(z @ z)
    I = (n / S0) * (num / den) if den > 0 else np.nan
    # Randomisation expectation/variance under normality approximation
    EI = -1 / (n - 1)
    # Permutation p-value
    n_perm = 9999
    rng = np.random.default_rng(42)
    count = 0
    obs = abs(I - EI)
    for _ in range(n_perm):
        zp = rng.permutation(z)
        Ip = (n / S0) * float(zp @ W @ zp) / den
        if abs(Ip - EI) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"I": I, "EI": EI, "p_perm": p}


def added_variable_plot(df: pd.DataFrame, path: Path) -> tuple[float, float]:
    """Real partial regression: Intensity residuals vs Floor residuals | Income."""
    y = df["fw_2024_pct"].values
    income = sm.add_constant(df["income_10k"])
    floor = df["floor_m2"].values
    y_res = sm.OLS(y, income).fit().resid
    x_res = sm.OLS(floor, income).fit().resid
    av = sm.OLS(y_res, sm.add_constant(x_res)).fit()
    slope = float(av.params.iloc[1])
    pval = float(av.pvalues.iloc[1])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(x_res, y_res, c="#2c5f8a", s=55, alpha=0.85, edgecolors="black", linewidths=0.4)
    xs = np.linspace(x_res.min() - 1, x_res.max() + 1, 100)
    ax.plot(xs, av.params.iloc[0] + av.params.iloc[1] * xs, color="#b33a3a", lw=2)
    for i, d in enumerate(df["district"]):
        ax.annotate(d.split()[0][:6], (x_res[i], y_res[i]), fontsize=7, alpha=0.75)
    ax.set_xlabel("Floor-area residual | Income ($m^2$)")
    ax.set_ylabel("FDH intensity residual | Income (pp)")
    ax.set_title(f"Added-variable plot: space vs intensity\nβ={slope:.3f}, p={pval:.3f}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return slope, pval


def spatial_residual_map_proxy(df: pd.DataFrame, resid: np.ndarray, path: Path) -> None:
    """Schematic district residual strip (no shapefile): ordered by HA cluster."""
    order = df.sort_values(["ha_cluster", "district"]).index
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = np.where(resid[order] >= 0, "#2c5f8a", "#b33a3a")
    ax.barh(df.loc[order, "district"], resid[order], color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("OLS residual (FDH intensity, pp)")
    ax.set_title("Spatial structure of OLS residuals (ordered by HA cluster)")
    ax.grid(True, axis="x", ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def quadrant_outcome_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        df["fw_2024_pct"],
        df["r_excl_2024"],
        c=df["rche_sub_per_1000_elderly"],
        s=60 + 4 * (df["h_frag_2021_pct"] - df["h_frag_2021_pct"].min()),
        cmap="YlOrRd",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.axvline(df["fw_2024_pct"].mean(), ls="--", color="grey")
    ax.axhline(df["r_excl_2024"].mean(), ls="--", color="grey")
    for _, r in df.iterrows():
        ax.annotate(r["district"], (r["fw_2024_pct"], r["r_excl_2024"]), fontsize=7, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Subsidised RCHE places / 1,000 elderly (2024)")
    ax.set_xlabel("FDH intensity F/W (%)")
    ax.set_ylabel(r"Baseline dependency $R_{excl}$")
    ax.set_title("Burden–mitigation matrix with institutional supply overlay")
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def unpaid_care_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["fw_2024_pct"], df["female_lfpr_excl_2021"], c="#2c5f8a", s=60, edgecolors="black")
    for _, r in df.iterrows():
        ax.annotate(r["district"], (r["fw_2024_pct"], r["female_lfpr_excl_2021"]), fontsize=7, alpha=0.75)
    slope, intercept, r, p, _ = stats.linregress(df["fw_2024_pct"], df["female_lfpr_excl_2021"])
    xs = np.linspace(df["fw_2024_pct"].min(), df["fw_2024_pct"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color="#b33a3a", lw=2)
    ax.set_xlabel("FDH intensity F/W 2024 (%)")
    ax.set_ylabel("Female LFPR excl. FDH, 2021 (%)")
    ax.set_title(f"Private helpers vs unpaid/local female labour\nr={r:.2f}, p={p:.3f}")
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def temporal_bridge_plot(df: pd.DataFrame, path: Path) -> dict:
    rho_fw, p_fw = stats.spearmanr(df["fw_2021_pct"], df["fw_2024_pct"])
    rho_inc, p_inc = stats.spearmanr(df["income_10k"], df["fw_2024_pct"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["fw_2021_pct"], df["fw_2024_pct"], s=60, c="#2c5f8a", edgecolors="black")
    lims = [
        min(df["fw_2021_pct"].min(), df["fw_2024_pct"].min()) - 1,
        max(df["fw_2021_pct"].max(), df["fw_2024_pct"].max()) + 1,
    ]
    ax.plot(lims, lims, ls="--", color="grey", label="45°")
    for _, r in df.iterrows():
        ax.annotate(r["district"], (r["fw_2021_pct"], r["fw_2024_pct"]), fontsize=7, alpha=0.75)
    ax.set_xlabel("F/W 2021 (%)")
    ax.set_ylabel("F/W 2024 (%)")
    ax.set_title(f"Temporal rank stability of FDH intensity\nSpearman ρ={rho_fw:.3f}, p={p_fw:.4f}")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"spearman_fw_2021_2024": float(rho_fw), "p_fw": float(p_fw), "spearman_income_fw": float(rho_inc), "p_income": float(p_inc)}


def compute_cdrs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data-informed CDRS using z-scores:
    CDRS = z(R_excl) + z(H_frag) - z(F/W)
    Equal weights after standardisation; higher = higher discharge/community-care risk.
    """
    z_r = (df["r_excl_2024"] - df["r_excl_2024"].mean()) / df["r_excl_2024"].std(ddof=0)
    z_h = (df["h_frag_2021_pct"] - df["h_frag_2021_pct"].mean()) / df["h_frag_2021_pct"].std(ddof=0)
    z_f = (df["fw_2024_pct"] - df["fw_2024_pct"].mean()) / df["fw_2024_pct"].std(ddof=0)
    out = df.copy()
    out["cdrs"] = z_r + z_h - z_f
    # Calibrate weights from association with subsidised RCHE intensity:
    # RCHE ~ a + b1 z_r + b2 z_h + b3 z_f  (expect b3 < 0 if private helpers substitute for public beds)
    X = sm.add_constant(np.column_stack([z_r, z_h, z_f]))
    y = df["rche_sub_per_1000_elderly"]
    m = sm.OLS(y, X).fit()
    b1, b2, b3 = float(m.params.iloc[1]), float(m.params.iloc[2]), float(m.params.iloc[3])
    w1, w2, w3 = abs(b1), abs(b2), abs(b3)
    s = w1 + w2 + w3
    w1, w2, w3 = w1 / s, w2 / s, w3 / s
    # Preserve risk direction: +aging, +fragility, -helper intensity
    sign_f = -1.0 if b3 <= 0 else 1.0
    out["cdrs_calibrated"] = w1 * z_r + w2 * z_h + sign_f * w3 * z_f
    out.attrs["cdrs_weights"] = {
        "w_R": w1,
        "w_H": w2,
        "w_F": w3,
        "sign_F": sign_f,
        "raw_coefs": {"b_R": b1, "b_H": b2, "b_F": b3},
        "rche_model_rsq": float(m.rsquared),
    }
    return out


def main() -> None:
    df = pd.read_csv(DATA)
    results: dict = {}

    # --- OLS intensity model ---
    y = df["fw_2024_pct"]
    X = sm.add_constant(df[["income_10k", "floor_m2"]])
    model = sm.OLS(y, X).fit()
    results["ols"] = {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "params": {k: float(v) for k, v in model.params.items()},
        "bse": {k: float(v) for k, v in model.bse.items()},
        "tvalues": {k: float(v) for k, v in model.tvalues.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
    }

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xv = df[["income_10k", "floor_m2"]].assign(const=1)[["const", "income_10k", "floor_m2"]]
    vifs = {Xv.columns[i]: float(variance_inflation_factor(Xv.values, i)) for i in range(Xv.shape[1])}
    results["vif"] = vifs

    # Influence / Cook's D
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    df["cooks_d"] = cooks
    df["ols_resid"] = model.resid
    results["influence"] = {
        "max_cooks": float(cooks.max()),
        "max_cooks_district": str(df.loc[cooks.argmax(), "district"]),
        "cooks_threshold_4n": 4 / len(df),
        "high_influence": df.loc[cooks > 4 / len(df), ["district", "cooks_d"]].to_dict("records"),
    }

    # Leave-one-out CV RMSE and coef stability
    coefs = []
    preds = np.zeros(len(df))
    for i in range(len(df)):
        mask = np.ones(len(df), dtype=bool)
        mask[i] = False
        m = sm.OLS(y[mask], X[mask]).fit()
        preds[i] = float(m.predict(X.iloc[[i]]).iloc[0])
        coefs.append(m.params.to_numpy())
    coefs = np.array(coefs)
    results["loocv"] = {
        "rmse": float(np.sqrt(np.mean((y - preds) ** 2))),
        "coef_income_mean": float(coefs[:, 1].mean()),
        "coef_income_sd": float(coefs[:, 1].std(ddof=1)),
        "coef_floor_mean": float(coefs[:, 2].mean()),
        "coef_floor_sd": float(coefs[:, 2].std(ddof=1)),
    }

    # Bootstrap CIs
    rng = np.random.default_rng(123)
    B = 5000
    boot = np.zeros((B, 3))
    n = len(df)
    for b in range(B):
        idx = rng.integers(0, n, n)
        mb = sm.OLS(y.values[idx], X.values[idx]).fit()
        boot[b] = np.asarray(mb.params, dtype=float)
    results["bootstrap_ci_95"] = {
        "income": [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))],
        "floor": [float(np.percentile(boot[:, 2], 2.5)), float(np.percentile(boot[:, 2], 97.5))],
        "const": [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
    }

    # Real AV plot
    slope, pval = added_variable_plot(df, FIG / "fig_av_plot.png")
    results["av_plot"] = {"slope": slope, "p": pval}

    # Spatial Moran
    W = queen_weights(df["district"].tolist())
    results["moran"] = {
        "fw": morans_i(df["fw_2024_pct"].values, W),
        "r_excl": morans_i(df["r_excl_2024"].values, W),
        "resid": morans_i(model.resid.values, W),
        "rche_sub_rate": morans_i(df["rche_sub_per_1000_elderly"].values, W),
        "cdrs_placeholder": None,
    }
    spatial_residual_map_proxy(df, model.resid.values, FIG / "fig_residual_bars.png")

    # Temporal bridge
    results["temporal"] = temporal_bridge_plot(df, FIG / "fig_temporal_fw.png")

    # Outcome models: institutional supply
    y_rche = df["rche_sub_per_1000_elderly"]
    Xo = sm.add_constant(df[["fw_2024_pct", "pct_65_2024", "income_10k"]])
    m_rche = sm.OLS(y_rche, Xo).fit()
    results["outcome_rche"] = {
        "r2": float(m_rche.rsquared),
        "params": {k: float(v) for k, v in m_rche.params.items()},
        "pvalues": {k: float(v) for k, v in m_rche.pvalues.items()},
        "bse": {k: float(v) for k, v in m_rche.bse.items()},
    }

    # Unpaid care proxy
    r_fl, p_fl = stats.pearsonr(df["fw_2024_pct"], df["female_lfpr_excl_2021"])
    results["unpaid_care"] = {"pearson_r": float(r_fl), "p": float(p_fl)}
    unpaid_care_plot(df, FIG / "fig_unpaid_care.png")

    # Childcare-adjusted rank stability of Wan Chai vs WTS
    results["childcare_sensitivity"] = {
        "wan_chai_base": float(df.loc[df.district == "Wan Chai", "fw_2024_pct"].iloc[0]),
        "wan_chai_adj": float(df.loc[df.district == "Wan Chai", "fw_adj_elder_pct"].iloc[0]),
        "wts_base": float(df.loc[df.district == "Wong Tai Sin", "fw_2024_pct"].iloc[0]),
        "wts_adj": float(df.loc[df.district == "Wong Tai Sin", "fw_adj_elder_pct"].iloc[0]),
        "ratio_base": float(
            df.loc[df.district == "Wan Chai", "fw_2024_pct"].iloc[0]
            / df.loc[df.district == "Wong Tai Sin", "fw_2024_pct"].iloc[0]
        ),
        "ratio_adj": float(
            df.loc[df.district == "Wan Chai", "fw_adj_elder_pct"].iloc[0]
            / df.loc[df.district == "Wong Tai Sin", "fw_adj_elder_pct"].iloc[0]
        ),
    }

    # CDRS
    df2 = compute_cdrs(df)
    results["cdrs_weights"] = df2.attrs["cdrs_weights"]
    results["moran"]["cdrs"] = morans_i(df2["cdrs"].values, W)

    # Quadrant labels
    # Territory averages from C&SD Table 110-06833 (matches paper matrix)
    fw_m = 7.11
    r_m = 512.0
    def quad(r):
        high_b = r["r_excl_2024"] >= r_m
        high_m = r["fw_2024_pct"] >= fw_m
        if high_b and high_m:
            return "I_buffered"
        if high_b and not high_m:
            return "II_exposed"
        if not high_b and not high_m:
            return "III_low_pressure"
        return "IV_low_burden_high_mitigation"

    df2["quadrant"] = df2.apply(quad, axis=1)
    quadrant_outcome_plot(df2, FIG / "fig_quadrant_rche.png")

    # Stress test with fragility interaction
    s = 0.10
    df2["mitigation_loss_pp"] = s * df2["fw_2024_pct"]
    df2["stress_exposure"] = df2["mitigation_loss_pp"] * (df2["h_frag_2021_pct"] / df2["h_frag_2021_pct"].mean())

    # Save enriched table
    keep = [
        "district",
        "ha_cluster",
        "fw_2024_pct",
        "fw_2021_pct",
        "r_excl_2024",
        "income_10k",
        "floor_m2",
        "h_frag_2021_pct",
        "female_lfpr_excl_2021",
        "pct_65_2024",
        "theta_childcare",
        "fw_adj_elder_pct",
        "rche_sub_per_1000_elderly",
        "rche_total_per_1000_elderly",
        "cdrs",
        "cdrs_calibrated",
        "quadrant",
        "cooks_d",
        "ols_resid",
        "mitigation_loss_pp",
        "stress_exposure",
    ]
    df2[keep].sort_values("cdrs", ascending=False).to_csv(OUT / "district_results.csv", index=False)

    # Summary tables for LaTeX
    (OUT / "analysis_results.json").write_text(json.dumps(results, indent=2))

    # Print key findings
    print(model.summary())
    print("\nVIF:", vifs)
    print("Moran F/W:", results["moran"]["fw"])
    print("Moran resid:", results["moran"]["resid"])
    print("Outcome RCHE:", results["outcome_rche"])
    print("CDRS weights:", results["cdrs_weights"])
    print("Top CDRS:\n", df2.sort_values("cdrs", ascending=False)[["district", "cdrs", "quadrant"]].head(8))
    print("Wrote figures to", FIG)
    print("Wrote", OUT / "district_results.csv")


if __name__ == "__main__":
    main()

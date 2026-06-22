"""Phase 3 - Exploratory data analysis on the BDG2 electricity subset.

Generates publication-quality figures (results/figures/eda_*) and a markdown
summary (reports/eda_summary.md).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from src.utils import get_logger, load_config
from src.viz import setup_style, savefig

log = get_logger("eda", logfile="logs/eda.log")


def _norm_profile(g: pd.Series) -> pd.Series:
    """Normalize a building's series by its own mean (for cross-building averaging)."""
    m = g.mean()
    return g / m if m and np.isfinite(m) and m > 0 else g


def run_eda(cfg) -> dict:
    setup_style()
    figs = cfg["paths"]["figures"]
    df = pd.read_parquet(cfg["paths"]["interim"] / "dataset.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]
    df["norm_load"] = df.groupby("building_id")["meter_reading"].transform(_norm_profile)

    stats = {}
    stats["n_buildings"] = int(df["building_id"].nunique())
    stats["n_sites"] = int(df["site_id"].nunique())
    stats["n_rows"] = int(len(df))

    # ---- 1. composition: buildings per site & per usage ---------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bs = df.groupby("site_id")["building_id"].nunique().sort_values(ascending=False)
    bs.plot.bar(ax=ax[0], color="#0072B2")
    ax[0].set(title="Buildings per site", xlabel="Site", ylabel="# buildings")
    bu = df.groupby("primaryspaceusage")["building_id"].nunique().sort_values()
    bu.plot.barh(ax=ax[1], color="#009E73")
    ax[1].set(title="Buildings per primary use", xlabel="# buildings", ylabel="")
    savefig(fig, figs / "eda_01_composition.png")

    # ---- 2. completeness distribution --------------------------------------
    comp = df.groupby("building_id")["meter_reading"].apply(lambda s: s.notna().mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(comp.values, bins=20, color="#56B4E9", edgecolor="white")
    ax.set(title="Per-building data completeness", xlabel="Fraction of non-missing hours",
           ylabel="# buildings")
    savefig(fig, figs / "eda_02_completeness.png")
    stats["min_completeness"] = float(comp.min())
    stats["mean_completeness"] = float(comp.mean())

    # ---- 3. mean daily profile by building type ----------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top_uses = df["primaryspaceusage"].value_counts().head(5).index
    for use in top_uses:
        prof = (df[df["primaryspaceusage"] == use]
                .groupby("hour")["norm_load"].mean())
        ax.plot(prof.index, prof.values, marker="o", ms=3, label=use)
    ax.set(title="Mean daily load profile by building type (own-mean normalized)",
           xlabel="Hour of day", ylabel="Normalized load", xticks=range(0, 24, 3))
    ax.legend(fontsize=8)
    savefig(fig, figs / "eda_03_daily_profile.png")

    # ---- 4. weekly profile (hour-of-week) ----------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    wk = df.groupby("hour_of_week")["norm_load"].mean()
    ax.plot(wk.index, wk.values, color="#D55E00")
    for d in range(1, 7):
        ax.axvline(d * 24, color="grey", ls=":", alpha=0.5)
    ax.set(title="Mean weekly load profile (Mon-Sun)", xlabel="Hour of week",
           ylabel="Normalized load", xticks=[12 + 24 * i for i in range(7)],
           xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    savefig(fig, figs / "eda_04_weekly_profile.png")

    # ---- 5. monthly seasonality --------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    mon = df.groupby("month")["norm_load"].mean()
    ax.bar(mon.index, mon.values, color="#CC79A7")
    ax.set(title="Monthly seasonality (normalized load)", xlabel="Month",
           ylabel="Normalized load", xticks=range(1, 13))
    savefig(fig, figs / "eda_05_monthly.png")

    # ---- 6. load vs temperature --------------------------------------------
    samp = df.dropna(subset=["meter_reading", "airTemperature"]).sample(
        min(40000, df["meter_reading"].notna().sum()), random_state=cfg["seed"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(samp["airTemperature"], samp["norm_load"], s=3, alpha=0.15,
                    c=samp["month"], cmap="twilight")
    ax.set(title="Normalized load vs air temperature", xlabel="Air temperature (°C)",
           ylabel="Normalized load", ylim=(0, samp["norm_load"].quantile(0.99)))
    plt.colorbar(sc, ax=ax, label="Month")
    savefig(fig, figs / "eda_06_load_vs_temp.png")

    # ---- 7. autocorrelation of a representative building -------------------
    bid = (df.groupby("building_id")["meter_reading"].apply(lambda s: s.notna().mean())
           .idxmax())
    series = (df[df["building_id"] == bid]
              .set_index("timestamp")["meter_reading"]
              .interpolate(limit=3).dropna())
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(series, lags=180, ax=ax, alpha=0.05)
    ax.set(title=f"Autocorrelation (building: {bid})", xlabel="Lag (hours)")
    savefig(fig, figs / "eda_07_acf.png")

    # ---- 8. distribution of readings (log) ---------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = df["meter_reading"].dropna()
    vals = vals[vals > 0]
    ax.hist(np.log1p(vals), bins=60, color="#0072B2", edgecolor="white")
    ax.set(title="Distribution of hourly electricity readings",
           xlabel="log(1 + kWh)", ylabel="Count")
    savefig(fig, figs / "eda_08_distribution.png")

    # ---- 9. example time series --------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=False)
    examples = (df.groupby("building_id")["meter_reading"].apply(lambda s: s.notna().mean())
                .sort_values(ascending=False).head(3).index)
    for ax, bid in zip(axes, examples):
        sub = df[df["building_id"] == bid].set_index("timestamp")["meter_reading"]
        sub = sub.loc["2017-03-01":"2017-03-21"]
        ax.plot(sub.index, sub.values, color="#009E73")
        ax.set(title=f"{bid}", ylabel="kWh")
    fig.suptitle("Example 3-week load traces (Mar 2017)", fontweight="bold")
    savefig(fig, figs / "eda_09_examples.png")

    log.info("EDA figures written to %s", figs)
    return stats


def write_summary(cfg, stats: dict) -> None:
    rep = cfg["paths"]["reports"] / "eda_summary.md"
    lines = [
        "# Exploratory Data Analysis - BDG2 Electricity Subset\n",
        f"- Buildings: **{stats['n_buildings']}** across **{stats['n_sites']}** sites",
        f"- Hourly observations: **{stats['n_rows']:,}**",
        f"- Completeness: min **{stats['min_completeness']:.2%}**, "
        f"mean **{stats['mean_completeness']:.2%}**\n",
        "## Figures",
        "1. `eda_01_composition` - buildings per site / per primary use",
        "2. `eda_02_completeness` - per-building data completeness",
        "3. `eda_03_daily_profile` - mean daily load by building type",
        "4. `eda_04_weekly_profile` - weekday/weekend structure",
        "5. `eda_05_monthly` - annual seasonality",
        "6. `eda_06_load_vs_temp` - weather dependence",
        "7. `eda_07_acf` - 24 h and 168 h autocorrelation cycles",
        "8. `eda_08_distribution` - reading distribution (log)",
        "9. `eda_09_examples` - example load traces",
        "\n## Key observations",
        "- Strong **daily (24 h)** and **weekly (168 h)** periodicity motivates the "
        "168 h lookback / 24 h horizon design and attention over these cycles.",
        "- Load profiles differ by **building type** (e.g. office vs lodging), motivating "
        "static building covariates and per-building scaling.",
        "- **Temperature dependence** is non-linear and seasonal, motivating weather "
        "covariates in the forecast models.",
    ]
    rep.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", rep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    stats = run_eda(cfg)
    write_summary(cfg, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

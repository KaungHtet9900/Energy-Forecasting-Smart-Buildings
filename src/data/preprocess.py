"""Phase 2 - clean, subset, and merge BDG2 into a tidy long table.

Steps:
  1. Read the wide electricity matrix, restrict to the analysis window.
  2. Score each building by completeness + mean load.
  3. Select a CPU-tractable, multi-site subset (round-robin across sites).
  4. Melt to long, merge static metadata, merge per-site weather.
  5. Clean weather (per-site time interpolation) and save tidy parquet.

Output: data/interim/dataset.parquet, selected_buildings.csv, subset_summary.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config

log = get_logger("preprocess", logfile="logs/preprocess.log")


# --------------------------------------------------------------------------
def load_metadata(cfg) -> pd.DataFrame:
    md = pd.read_csv(cfg["paths"]["raw_bdg2"] / "metadata.csv")
    keep = ["building_id", "site_id", "primaryspaceusage", "sub_primaryspaceusage",
            "sqm", "yearbuilt", "timezone", "lat", "lng",
            "building_id_kaggle", "site_id_kaggle"]
    keep = [c for c in keep if c in md.columns]
    return md[keep].copy()


def load_electricity_wide(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg["paths"]["raw_bdg2"] / "electricity.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    start = pd.Timestamp(cfg["data"]["start"])
    end = pd.Timestamp(cfg["data"]["end"])
    df = df.loc[start:end]
    return df


# --------------------------------------------------------------------------
def score_buildings(wide: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-building completeness + mean load, joined with metadata."""
    completeness = wide.notna().mean()
    mean_load = wide.mean()
    nonzero_frac = (wide > 0).mean()
    score = pd.DataFrame({
        "building_id": completeness.index,
        "completeness": completeness.values,
        "mean_kwh": mean_load.values,
        "nonzero_frac": nonzero_frac.values,
    })
    score = score.merge(meta, on="building_id", how="left")
    return score


def select_subset(cfg, score: pd.DataFrame) -> pd.DataFrame:
    """Filter by quality, then round-robin across sites up to max_buildings."""
    d = cfg["data"]
    cand = score[
        (score["completeness"] >= d["min_completeness"])
        & (score["mean_kwh"] >= d["min_mean_kwh"])
        & (score["nonzero_frac"] >= 0.5)
    ].copy()

    if d.get("sites"):
        cand = cand[cand["site_id"].isin(d["sites"])]

    log.info("candidates after quality filter: %d (across %d sites)",
             len(cand), cand["site_id"].nunique())

    # round-robin across sites, picking highest-completeness first within each site
    cand = cand.sort_values(["site_id", "completeness"], ascending=[True, False])
    groups = {s: list(g["building_id"]) for s, g in cand.groupby("site_id")}
    order = []
    while groups and len(order) < d["max_buildings"]:
        for s in list(groups.keys()):
            if not groups[s]:
                del groups[s]
                continue
            order.append(groups[s].pop(0))
            if len(order) >= d["max_buildings"]:
                break
    chosen = cand[cand["building_id"].isin(order)].copy()
    log.info("selected %d buildings across %d sites; usages: %s",
             len(chosen), chosen["site_id"].nunique(),
             dict(chosen["primaryspaceusage"].value_counts()))
    return chosen


# --------------------------------------------------------------------------
def clean_weather(cfg, sites) -> pd.DataFrame:
    w = pd.read_csv(cfg["paths"]["raw_bdg2"] / "weather.csv")
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    wcols = cfg["features"]["weather"]
    wcols = [c for c in wcols if c in w.columns]
    w = w[["timestamp", "site_id"] + wcols]
    w = w[w["site_id"].isin(sites)].copy()

    # per-site chronological interpolation + fill
    out = []
    for site, g in w.groupby("site_id"):
        g = g.set_index("timestamp").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_idx)
        g["site_id"] = site
        g[wcols] = (g[wcols].interpolate("time", limit=6)
                    .ffill().bfill())
        # any column still all-NaN -> fill with global median later
        out.append(g.rename_axis("timestamp").reset_index())
    w = pd.concat(out, ignore_index=True)
    for c in wcols:
        if w[c].isna().any():
            w[c] = w[c].fillna(w[c].median())
    return w


# --------------------------------------------------------------------------
def build_dataset(cfg) -> pd.DataFrame:
    meta = load_metadata(cfg)
    wide = load_electricity_wide(cfg)
    log.info("electricity window: %s .. %s  shape=%s",
             wide.index.min(), wide.index.max(), wide.shape)

    score = score_buildings(wide, meta)
    chosen = select_subset(cfg, score)
    chosen_ids = chosen["building_id"].tolist()

    # subset wide -> long
    sub = wide[chosen_ids].copy()
    long = sub.reset_index().melt(id_vars="timestamp",
                                  var_name="building_id",
                                  value_name="meter_reading")

    # merge static metadata
    static_cols = ["building_id", "site_id", "primaryspaceusage", "sqm",
                   "yearbuilt", "timezone"]
    static_cols = [c for c in static_cols if c in chosen.columns]
    long = long.merge(chosen[static_cols], on="building_id", how="left")

    # merge weather on (site_id, timestamp)
    sites = chosen["site_id"].unique().tolist()
    weather = clean_weather(cfg, sites)
    long = long.merge(weather, on=["site_id", "timestamp"], how="left")

    long = long.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    return long, chosen, score


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to config yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    long, chosen, score = build_dataset(cfg)
    interim = cfg["paths"]["interim"]

    out_path = interim / "dataset.parquet"
    long.to_parquet(out_path, index=False)
    chosen.to_csv(interim / "selected_buildings.csv", index=False)
    score.to_csv(interim / "building_scores.csv", index=False)

    summary = {
        "n_buildings": int(chosen.shape[0]),
        "n_sites": int(chosen["site_id"].nunique()),
        "sites": sorted(chosen["site_id"].unique().tolist()),
        "n_rows": int(long.shape[0]),
        "date_min": str(long["timestamp"].min()),
        "date_max": str(long["timestamp"].max()),
        "usages": {k: int(v) for k, v in chosen["primaryspaceusage"].value_counts().items()},
        "target_missing_frac": float(long["meter_reading"].isna().mean()),
        "columns": list(long.columns),
    }
    with open(interim / "subset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("saved %s  (%.1f MB, %d rows)", out_path,
             out_path.stat().st_size / 1e6, len(long))
    log.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

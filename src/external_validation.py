"""Phase 10 - external validation on ASHRAE GEPIII.

Transfers the BDG2-trained models to GEPIII electricity meters that are DISJOINT
from the BDG2 training buildings (using the building_id_kaggle linkage in the BDG2
metadata to drop overlaps), and reports zero-shot + optional fine-tuned metrics.

Requires the GEPIII files (see src/data/download.py --gepiii). If they are absent,
the script explains how to obtain them and exits cleanly.

GEPIII schema:
  building_metadata.csv: site_id, building_id, primary_use, square_feet, year_built, ...
  train.csv:             building_id, meter (0=electricity), timestamp, meter_reading
  weather_train.csv:     site_id, timestamp, air_temperature, dew_temperature,
                         wind_speed, cloud_coverage, sea_level_pressure, ...
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, get_logger, compute_metrics
from src.models import common
from src.data.features import add_calendar, add_holidays
from src.data.windowing import TIME_FEATS, WEATHER_FEATS
from src.models.rnn import build_lstm, build_gru
from src.models.cnn import build_tcn, build_cnn_lstm
from src.models.attention import build_transformer, build_attn_lstm, build_tft

log = get_logger("external", logfile="logs/external.log")

DL_BUILDERS = {"lstm": build_lstm, "gru": build_gru, "tcn": build_tcn,
               "cnn_lstm": build_cnn_lstm, "transformer": build_transformer,
               "attn_lstm": build_attn_lstm, "tft": build_tft}

# GEPIII primary_use -> BDG2 primaryspaceusage (category alignment)
USE_MAP = {
    "Education": "Education", "Office": "Office", "Lodging/residential": "Lodging/residential",
    "Entertainment/public assembly": "Entertainment/public assembly",
    "Public services": "Public services", "Healthcare": "Healthcare",
    "Manufacturing/industrial": "Manufacturing/industrial", "Retail": "Retail",
    "Services": "Services", "Other": "Other", "Parking": "Other",
    "Warehouse/storage": "Other", "Food sales and service": "Retail",
    "Religious worship": "Public services", "Technology/science": "Office",
    "Utility": "Other",
}
GEPIII_WEATHER = {"air_temperature": "airTemperature", "dew_temperature": "dewTemperature",
                  "wind_speed": "windSpeed", "cloud_coverage": "cloudCoverage",
                  "sea_level_pressure": "seaLvlPressure"}


def _check_data(cfg):
    g = cfg["paths"]["raw_gepiii"]
    need = ["building_metadata.csv", "train.csv", "weather_train.csv"]
    missing = [f for f in need if not (g / f).exists()]
    if missing:
        log.warning("GEPIII files missing: %s", missing)
        log.warning("Join the competition then run: python -m src.data.download --gepiii")
        log.warning("  https://www.kaggle.com/competitions/ashrae-energy-prediction/rules")
        return False
    return True


def _overlap_kaggle_ids(cfg):
    """GEPIII building_ids that correspond to BDG2 *training* buildings (to exclude)."""
    md = pd.read_csv(cfg["paths"]["raw_bdg2"] / "metadata.csv")
    sel = pd.read_csv(cfg["paths"]["interim"] / "selected_buildings.csv")
    train_md = md[md["building_id"].isin(sel["building_id"])]
    return set(train_md["building_id_kaggle"].dropna().astype(int).tolist())


def build_external(cfg, max_buildings=60):
    g = cfg["paths"]["raw_gepiii"]
    bmeta = pd.read_csv(g / "building_metadata.csv")
    overlap = _overlap_kaggle_ids(cfg) if cfg["external"]["exclude_overlap_with_bdg2"] else set()

    # electricity meter only
    train = pd.read_csv(g / "train.csv")
    train = train[train["meter"] == 0].copy()
    train["timestamp"] = pd.to_datetime(train["timestamp"])

    # candidate buildings: electricity, not overlapping, high completeness
    elec_ids = train["building_id"].unique()
    cand = bmeta[bmeta["building_id"].isin(elec_ids)]
    cand = cand[~cand["building_id"].isin(overlap)]
    log.info("GEPIII electricity buildings: %d | after overlap-exclusion: %d",
             len(elec_ids), len(cand))

    comp = (train.groupby("building_id")["meter_reading"]
            .apply(lambda s: s.notna().mean()))
    good = comp[comp >= 0.9].index
    cand = cand[cand["building_id"].isin(good)]
    # round-robin across sites for diversity
    cand = cand.sort_values("site_id")
    chosen = (cand.groupby("site_id", group_keys=False)
              .head(max(1, max_buildings // max(1, cand["site_id"].nunique()))))
    chosen = chosen.head(max_buildings)
    log.info("selected %d external GEPIII buildings across %d sites",
             len(chosen), chosen["site_id"].nunique())

    # weather
    w = pd.read_csv(g / "weather_train.csv")
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    w = w.rename(columns=GEPIII_WEATHER)[["site_id", "timestamp"] + WEATHER_FEATS]

    # assemble long table in BDG2 schema
    df = train[train["building_id"].isin(chosen["building_id"])].copy()
    df = df.merge(chosen[["building_id", "site_id", "primary_use", "square_feet",
                          "year_built"]], on="building_id", how="left")
    df["primaryspaceusage"] = df["primary_use"].map(USE_MAP).fillna("Other")
    df["sqm"] = df["square_feet"] * 0.092903
    df = df.rename(columns={"year_built": "yearbuilt"})
    df["timezone"] = "US/Eastern"  # GEPIII tz not provided per-site; holidays approx US
    df = df.merge(w, on=["site_id", "timestamp"], how="left")
    return df, chosen


def featurize_external(cfg, df):
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["meter_reading"] = df["meter_reading"].clip(lower=0)
    df["meter_reading"] = (df.groupby("building_id")["meter_reading"]
                           .transform(lambda s: s.interpolate(limit=3)))
    df["y_log"] = np.log1p(df["meter_reading"])
    df = add_calendar(df)
    df = add_holidays(df)
    for c in WEATHER_FEATS:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df["yearbuilt_missing"] = df["yearbuilt"].isna().astype("int8")
    df["yearbuilt"] = df["yearbuilt"].fillna(df["yearbuilt"].median())
    df["log_sqm"] = np.log1p(df["sqm"])
    return df


def window_external(cfg, df):
    """Build external windows; calibrate target scaling on each building's first 60%."""
    meta = common.load_meta(cfg)
    sc = np.load(cfg["paths"]["processed"] / "scalers.npz", allow_pickle=True)
    w_mean, w_std = sc["w_mean"], sc["w_std"]
    lb, h = meta["lookback"], meta["horizon"]

    # apply BDG2 weather scaler (domain shift is intentional)
    for i, c in enumerate(WEATHER_FEATS):
        df[c] = (df[c] - w_mean[i]) / w_std[i]

    # static scaling: reuse BDG2 stats by recomputing approx (z over external buildings)
    usages = [s.replace("use_", "") for s in meta["static_features"] if s.startswith("use_")]

    # exact BDG2 static scaler stats (match windowing.py: mean/std over training buildings)
    feat = pd.read_parquet(cfg["paths"]["interim"] / "features.parquet").drop_duplicates("building_id")
    s_mean = {"log_sqm": float(feat["log_sqm"].mean()), "yearbuilt": float(feat["yearbuilt"].mean())}
    s_std = {"log_sqm": float(feat["log_sqm"].std()) or 1.0, "yearbuilt": float(feat["yearbuilt"].std()) or 1.0}

    X_enc, X_dec, X_stat, Y, y_raw, ym, ys, bidx = ([] for _ in range(8))
    buildings = df["building_id"].unique().tolist()
    for j, b in enumerate(buildings):
        gdf = df[df["building_id"] == b].reset_index(drop=True)
        y_log = gdf["y_log"].to_numpy(np.float32)
        n = len(gdf)
        cal = int(n * 0.6)
        valid = np.isfinite(y_log[:cal])
        if valid.sum() < lb + h:
            continue
        mu = float(np.nanmean(y_log[:cal][valid]))
        sd = float(np.nanstd(y_log[:cal][valid])) or 1.0
        y_scaled = (y_log - mu) / sd
        tf = gdf[TIME_FEATS].to_numpy(np.float32)
        srow = gdf.iloc[0]
        stat = [(float(srow["log_sqm"]) - s_mean["log_sqm"]) / s_std["log_sqm"],
                (float(srow["yearbuilt"]) - s_mean["yearbuilt"]) / s_std["yearbuilt"],
                float(srow["yearbuilt_missing"])]
        stat += [1.0 if srow["primaryspaceusage"] == u else 0.0 for u in usages]
        stat = np.asarray(stat, np.float32)

        # evaluate on the last 40% (the calibration tail), non-overlapping days
        for o in range(max(lb, cal), n - h + 1, h):
            enc_y = y_scaled[o - lb:o]
            dec_y = y_scaled[o:o + h]
            if not (np.isfinite(enc_y).all() and np.isfinite(dec_y).all()):
                continue
            X_enc.append(np.column_stack([enc_y, tf[o - lb:o]]))
            X_dec.append(tf[o:o + h])
            X_stat.append(stat)
            Y.append(dec_y)
            y_raw.append(gdf["meter_reading"].to_numpy(np.float32)[o:o + h])
            ym.append(mu); ys.append(sd); bidx.append(j)

    arr = dict(X_enc=np.asarray(X_enc, np.float32), X_dec=np.asarray(X_dec, np.float32),
               X_stat=np.asarray(X_stat, np.float32), Y=np.asarray(Y, np.float32),
               y_raw=np.asarray(y_raw, np.float32), y_mean=np.asarray(ym, np.float32),
               y_std=np.asarray(ys, np.float32), b_idx=np.asarray(bidx, np.int32))
    log.info("external windows: %d", len(arr["Y"]))
    return arr


def evaluate_external(cfg, arr):
    meta = common.load_meta(cfg)
    rows = []
    # baselines that need no training
    from src.models.baselines import Persistence, SeasonalNaive
    for name, mdl in [("persistence", Persistence(meta["horizon"])),
                      ("seasonal_naive", SeasonalNaive(meta["horizon"]))]:
        pred = common.invert(mdl.predict_scaled(arr), arr["y_mean"], arr["y_std"])
        rows.append({"model": name, **compute_metrics(arr["y_raw"], np.clip(pred, 0, None))})

    for name, builder in DL_BUILDERS.items():
        path = cfg["paths"]["models"] / f"{name}.pt"
        if not path.exists():
            continue
        model = builder(cfg, meta)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        pred_scaled = common.predict_scaled(model, arr)
        pred = np.clip(common.invert(pred_scaled, arr["y_mean"], arr["y_std"]), 0, None)
        rows.append({"model": name, **compute_metrics(arr["y_raw"], pred)})
        np.savez_compressed(cfg["paths"]["models"] / f"{name}_external_pred.npz", pred=pred)

    res = pd.DataFrame(rows).sort_values("RMSE")
    res.to_csv(cfg["paths"]["tables"] / "external_metrics.csv", index=False)
    log.info("external (zero-shot transfer) metrics:\n%s", res.to_string(index=False))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-buildings", type=int, default=60)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not _check_data(cfg):
        return 0
    df, chosen = build_external(cfg, args.max_buildings)
    df = featurize_external(cfg, df)
    arr = window_external(cfg, df)
    if len(arr["Y"]) == 0:
        log.error("no external windows built")
        return 1
    evaluate_external(cfg, arr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

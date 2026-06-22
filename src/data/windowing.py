"""Phase 4b - build sliding-window tensors with leakage-free scaling and split.

For each (building, forecast origin) it creates:
  X_enc  (lookback, 1 + n_time)   scaled target history + known time covariates
  X_dec  (horizon,  n_time)       known-future time covariates
  X_stat (n_static,)              static building covariates
  Y      (horizon,)               scaled-log target (model label)
  y_raw  (horizon,)               true kWh (for metric evaluation)
  plus building index, origin time, and per-building (mean, std) to invert.

Scalers are fit on the TRAIN region only. Windows straddling a split boundary are
dropped so labels never leak across train/val/test.

Output: data/processed/{train,val,test}.npz, scalers.npz, feature_names.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config

log = get_logger("windowing", logfile="logs/windowing.log")

# known time-varying covariates (used in both encoder-past and decoder-future)
TIME_FEATS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
              "is_weekend", "is_holiday",
              "airTemperature", "dewTemperature", "windSpeed", "cloudCoverage",
              "seaLvlPressure"]
WEATHER_FEATS = ["airTemperature", "dewTemperature", "windSpeed", "cloudCoverage",
                 "seaLvlPressure"]
STATIC_NUM = ["log_sqm", "yearbuilt", "yearbuilt_missing"]


def build(cfg):
    df = pd.read_parquet(cfg["paths"]["interim"] / "features.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    lookback = cfg["task"]["lookback"]
    horizon = cfg["task"]["horizon"]
    stride = cfg["task"]["stride"]
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])

    # ---- fit scalers on TRAIN rows only -----------------------------------
    train_rows = df["timestamp"] <= train_end
    w_mean = df.loc[train_rows, WEATHER_FEATS].mean()
    w_std = df.loc[train_rows, WEATHER_FEATS].replace(0, np.nan).std().fillna(1.0)
    w_std[w_std == 0] = 1.0
    df[WEATHER_FEATS] = (df[WEATHER_FEATS] - w_mean) / w_std

    # static numeric scaling (over training buildings' static values)
    static_tbl = df.drop_duplicates("building_id").set_index("building_id")
    s_mean = static_tbl[["log_sqm", "yearbuilt"]].mean()
    s_std = static_tbl[["log_sqm", "yearbuilt"]].std().replace(0, 1.0)

    # usage one-hot (fixed order)
    usages = sorted(df["primaryspaceusage"].dropna().unique().tolist())
    static_feat_names = ["log_sqm_z", "yearbuilt_z", "yearbuilt_missing"] + \
                        [f"use_{u}" for u in usages]

    enc_feat_names = ["y_scaled"] + TIME_FEATS
    dec_feat_names = list(TIME_FEATS)

    buildings = df["building_id"].unique().tolist()
    b2idx = {b: i for i, b in enumerate(buildings)}

    out = {k: {"X_enc": [], "X_dec": [], "X_stat": [], "Y": [], "y_raw": [],
               "b_idx": [], "origin": [], "y_mean": [], "y_std": []}
           for k in ("train", "val", "test")}

    b_means = np.zeros(len(buildings), np.float32)
    b_stds = np.ones(len(buildings), np.float32)
    skipped_nan = 0

    for b in buildings:
        g = df[df["building_id"] == b]
        ts = g["timestamp"].values
        y_log = g["y_log"].to_numpy(np.float32)
        y_kwh = g["meter_reading"].to_numpy(np.float32)
        tf = g[TIME_FEATS].to_numpy(np.float32)

        # per-building target stats from train region
        tr_mask = (g["timestamp"] <= train_end).to_numpy() & np.isfinite(y_log)
        if tr_mask.sum() < lookback + horizon:
            continue
        mu = float(np.nanmean(y_log[tr_mask]))
        sd = float(np.nanstd(y_log[tr_mask])) or 1.0
        bi = b2idx[b]
        b_means[bi], b_stds[bi] = mu, sd
        y_scaled = (y_log - mu) / sd

        # static vector
        srow = static_tbl.loc[b]
        stat = [
            (srow["log_sqm"] - s_mean["log_sqm"]) / s_std["log_sqm"],
            (srow["yearbuilt"] - s_mean["yearbuilt"]) / s_std["yearbuilt"],
            float(srow["yearbuilt_missing"]),
        ] + [1.0 if srow["primaryspaceusage"] == u else 0.0 for u in usages]
        stat = np.asarray(stat, np.float32)

        T = len(g)
        for o in range(lookback, T - horizon + 1, stride):
            enc_y = y_scaled[o - lookback:o]
            dec_y = y_scaled[o:o + horizon]
            if not (np.isfinite(enc_y).all() and np.isfinite(dec_y).all()):
                skipped_nan += 1
                continue
            h_start = ts[o]
            h_end = ts[o + horizon - 1]
            if h_end <= np.datetime64(train_end):
                split = "train"
            elif h_start > np.datetime64(train_end) and h_end <= np.datetime64(val_end):
                split = "val"
            elif h_start > np.datetime64(val_end):
                split = "test"
            else:
                continue  # straddles a boundary

            x_enc = np.column_stack([enc_y, tf[o - lookback:o]])      # (lookback, 1+nt)
            x_dec = tf[o:o + horizon]                                 # (horizon, nt)
            d = out[split]
            d["X_enc"].append(x_enc)
            d["X_dec"].append(x_dec)
            d["X_stat"].append(stat)
            d["Y"].append(dec_y)
            d["y_raw"].append(y_kwh[o:o + horizon])
            d["b_idx"].append(bi)
            d["origin"].append(h_start)
            d["y_mean"].append(mu)
            d["y_std"].append(sd)

    # ---- stack & save ------------------------------------------------------
    proc = cfg["paths"]["processed"]
    counts = {}
    for split, d in out.items():
        if not d["X_enc"]:
            counts[split] = 0
            continue
        arr = dict(
            X_enc=np.asarray(d["X_enc"], np.float32),
            X_dec=np.asarray(d["X_dec"], np.float32),
            X_stat=np.asarray(d["X_stat"], np.float32),
            Y=np.asarray(d["Y"], np.float32),
            y_raw=np.asarray(d["y_raw"], np.float32),
            b_idx=np.asarray(d["b_idx"], np.int32),
            origin=np.asarray(d["origin"], dtype="datetime64[ns]").astype(np.int64),
            y_mean=np.asarray(d["y_mean"], np.float32),
            y_std=np.asarray(d["y_std"], np.float32),
        )
        np.savez_compressed(proc / f"{split}.npz", **arr)
        counts[split] = len(arr["Y"])

    np.savez(proc / "scalers.npz",
             b_means=b_means, b_stds=b_stds,
             buildings=np.array(buildings),
             w_mean=w_mean.values, w_std=w_std.values,
             weather_feats=np.array(WEATHER_FEATS))

    meta = {
        "lookback": lookback, "horizon": horizon, "stride": stride,
        "enc_features": enc_feat_names, "dec_features": dec_feat_names,
        "static_features": static_feat_names,
        "n_enc": len(enc_feat_names), "n_dec": len(dec_feat_names),
        "n_static": len(static_feat_names),
        "n_buildings": len(buildings),
        "counts": counts, "skipped_nan_windows": int(skipped_nan),
        "target_transform": "per-building z-score of log1p(kWh)",
    }
    with open(proc / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("counts: %s | skipped(NaN): %d", counts, skipped_nan)
    log.info("enc=%d dec=%d static=%d feats | saved to %s",
             meta["n_enc"], meta["n_dec"], meta["n_static"], proc)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    build(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

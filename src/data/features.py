"""Phase 4a - feature engineering.

Adds calendar (cyclical), holiday, cleaned-weather, and target-transform columns
to the tidy dataset and writes data/interim/features.parquet.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import holidays

from src.utils import get_logger, load_config

log = get_logger("features", logfile="logs/features.log")

# timezone -> holidays country code
TZ_TO_COUNTRY = {
    "US/Eastern": "US", "US/Central": "US", "US/Mountain": "US", "US/Pacific": "US",
    "Europe/London": "GB", "Europe/Dublin": "IE",
}


def _holiday_set(country: str, years) -> set:
    try:
        return set(holidays.country_holidays(country, years=years).keys())
    except Exception:  # noqa: BLE001
        return set()


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    # cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["timestamp"].dt.year.unique().tolist())
    df["date"] = df["timestamp"].dt.date
    df["is_holiday"] = 0
    for tz, country in TZ_TO_COUNTRY.items():
        mask = df["timezone"] == tz
        if not mask.any():
            continue
        hset = _holiday_set(country, years)
        df.loc[mask, "is_holiday"] = df.loc[mask, "date"].isin(hset).astype("int8")
    df["is_holiday"] = df["is_holiday"].astype("int8")
    df = df.drop(columns=["date"])
    return df


def clean_target_and_static(cfg, df: pd.DataFrame) -> pd.DataFrame:
    # clip non-physical negatives, interpolate short gaps per building
    df["meter_reading"] = df["meter_reading"].clip(lower=0)
    df["meter_reading"] = (df.groupby("building_id")["meter_reading"]
                           .transform(lambda s: s.interpolate(limit=3)))
    df["y_log"] = np.log1p(df["meter_reading"])

    # residual weather NaNs -> global median
    for c in cfg["features"]["weather"]:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # static: impute yearbuilt + missingness flag, log-area
    df["yearbuilt_missing"] = df["yearbuilt"].isna().astype("int8")
    df["yearbuilt"] = df["yearbuilt"].fillna(df["yearbuilt"].median())
    df["log_sqm"] = np.log1p(df["sqm"])
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = pd.read_parquet(cfg["paths"]["interim"] / "dataset.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    df = add_calendar(df)
    df = add_holidays(df)
    df = clean_target_and_static(cfg, df)

    out = cfg["paths"]["interim"] / "features.parquet"
    df.to_parquet(out, index=False)
    log.info("features.parquet: shape=%s cols=%s", df.shape, list(df.columns))
    log.info("holiday rows: %d (%.2f%%) | target still NaN: %d",
             int(df["is_holiday"].sum()), 100 * df["is_holiday"].mean(),
             int(df["y_log"].isna().sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

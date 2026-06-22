"""Phase 5-7 orchestrator: train/fit every model, save kWh predictions.

Predictions for val & test are written to results/models/<name>_<split>_pred.npz
(in kWh) so evaluate.py / interpret.py can analyse them uniformly. DL checkpoints
and run metadata (params, time, val loss) are saved alongside.

Usage::
    python -m src.train --all
    python -m src.train --models tft,lstm,lightgbm
    python -m src.train --all --quick          # fast smoke run (few epochs)
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

from src.utils import load_config, get_logger, set_seed, compute_metrics
from src.models import common
from src.models.baselines import build_baselines
from src.models.rnn import build_lstm, build_gru
from src.models.cnn import build_tcn, build_cnn_lstm
from src.models.attention import build_transformer, build_attn_lstm, build_tft

log = get_logger("train", logfile="logs/train.log")

DL_BUILDERS = {
    "lstm": build_lstm, "gru": build_gru,
    "tcn": build_tcn, "cnn_lstm": build_cnn_lstm,
    "transformer": build_transformer, "attn_lstm": build_attn_lstm, "tft": build_tft,
}
BASELINE_NAMES = ["persistence", "seasonal_naive", "linear", "lightgbm"]
ALL_ORDER = BASELINE_NAMES + list(DL_BUILDERS.keys())


def save_pred(cfg, name, split, pred_kwh):
    path = cfg["paths"]["models"] / f"{name}_{split}_pred.npz"
    np.savez_compressed(path, pred=pred_kwh.astype(np.float32))


def run(cfg, models, quick=False):
    meta = common.load_meta(cfg)
    splits = {s: common.load_split(cfg, s) for s in ("train", "val", "test")}
    if quick:
        cfg["train"]["epochs"] = 3
        cfg["train"]["patience"] = 2
        log.info("QUICK mode: epochs=3")

    baselines = build_baselines(cfg)
    run_meta = {}

    for name in models:
        log.info("============================== %s ==============================", name)
        set_seed(cfg["seed"])
        t0 = time.time()
        try:
            if name in BASELINE_NAMES:
                mdl = baselines[name]
                mdl.fit(splits["train"], splits["val"])
                params = 0
                # feature importance (lightgbm) saved for interpretability comparison
                if name == "lightgbm":
                    fi = mdl.feature_importance()
                    with open(cfg["paths"]["interpret"] / "lightgbm_importance.json",
                              "w", encoding="utf-8") as f:
                        json.dump(fi, f, indent=2)
                predict = mdl.predict_scaled
            elif name in DL_BUILDERS:
                model = DL_BUILDERS[name](cfg, meta)
                model, history, best_val = common.train_model(
                    model, cfg, splits["train"], splits["val"], name)
                common.save_checkpoint(model, cfg, name)
                params = common.count_params(model)
                with open(cfg["paths"]["models"] / f"{name}_history.json",
                          "w", encoding="utf-8") as f:
                    json.dump(history, f)
                predict = lambda arr, _m=model: common.predict_scaled(_m, arr)
            else:
                log.warning("unknown model '%s' - skipping", name)
                continue

            row = {"params": int(params), "train_time_s": None}
            for split in ("val", "test"):
                arr = splits[split]
                pred_scaled = predict(arr)
                pred_kwh = common.invert(pred_scaled, arr["y_mean"], arr["y_std"])
                pred_kwh = np.clip(pred_kwh, 0, None)
                save_pred(cfg, name, split, pred_kwh)
                if split == "test":
                    m = compute_metrics(arr["y_raw"], pred_kwh)
                    row.update({f"test_{k}": v for k, v in m.items()})
            row["train_time_s"] = round(time.time() - t0, 1)
            run_meta[name] = row
            log.info("[%s] DONE in %.1fs | test MAE=%.3f RMSE=%.3f CV_RMSE=%.2f%% R2=%.3f",
                     name, row["train_time_s"], row["test_MAE"], row["test_RMSE"],
                     row["test_CV_RMSE"], row["test_R2"])
        except Exception as e:  # noqa: BLE001
            log.error("[%s] FAILED: %s\n%s", name, e, traceback.format_exc())
        # persist run metadata incrementally so a crash never loses completed work
        with open(cfg["paths"]["tables"] / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

    # persist run metadata
    with open(cfg["paths"]["tables"] / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    log.info("run_meta saved for %d models", len(run_meta))
    return run_meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models", default=None, help="comma-separated model names")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.all:
        models = ALL_ORDER
    else:
        models = ALL_ORDER
    run(cfg, models, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 9 - interpretability analysis of the attention models.

Loads trained TFT / Transformer / Attn-LSTM checkpoints and extracts:
  - temporal attention vs lag (do models learn the 24/168 h cycles?)
  - attention heatmaps (horizon step x past position)
  - TFT variable-selection importances (static / encoder / decoder)
  - comparison of TFT decoder importances with LightGBM gain importances
  - attention-vs-lag broken down by building type

Figures -> results/interpret/, narrative -> reports/interpretability.md
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.utils import load_config, get_logger
from src.models import common
from src.models.attention import build_tft, build_transformer, build_attn_lstm
from src.viz import setup_style, savefig

log = get_logger("interpret", logfile="logs/interpret.log")

BUILDERS = {"tft": build_tft, "transformer": build_transformer, "attn_lstm": build_attn_lstm}


def load_model(cfg, meta, name):
    path = cfg["paths"]["models"] / f"{name}.pt"
    if not path.exists():
        return None
    model = BUILDERS[name](cfg, meta)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def collect_attn(model, arr, max_n=1500):
    n = min(max_n, arr["X_enc"].shape[0])
    xe = torch.from_numpy(arr["X_enc"][:n])
    xd = torch.from_numpy(arr["X_dec"][:n])
    xs = torch.from_numpy(arr["X_stat"][:n])
    _, att = model(xe, xd, xs, return_attn=True)
    return {k: v.cpu().numpy() for k, v in att.items()}, np.arange(n)


def interpret(cfg):
    setup_style()
    out = cfg["paths"]["interpret"]
    meta = common.load_meta(cfg)
    L, H = meta["lookback"], meta["horizon"]
    test = common.load_split(cfg, "test")

    sc = np.load(cfg["paths"]["processed"] / "scalers.npz", allow_pickle=True)
    buildings = list(sc["buildings"])
    sel = pd.read_csv(cfg["paths"]["interim"] / "selected_buildings.csv")
    b2type = dict(zip(sel["building_id"], sel["primaryspaceusage"]))

    summary = {}

    # ===================================================================
    # TFT
    # ===================================================================
    tft = load_model(cfg, meta, "tft")
    if tft is not None:
        att, idx = collect_attn(tft, test)
        temporal = att["temporal_attn"]              # (n, H, L+H)
        enc_w = att["encoder_weights"].mean(axis=(0, 1))   # (n_enc,)
        dec_w = att["decoder_weights"].mean(axis=(0, 1))   # (n_dec,)
        stat_w = att["static_weights"].mean(axis=0)        # (n_static,)

        # attention to past positions, averaged over windows & horizon
        past_attn = temporal[:, :, :L].mean(axis=(0, 1))   # (L,)
        lag = np.arange(L, 0, -1)                           # position 0 -> lag L

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(lag, past_attn, color="#D55E00")
        for k in (24, 48, 72, 96, 120, 144, 168):
            ax.axvline(k, color="grey", ls=":", alpha=0.5)
        ax.set(title="TFT temporal attention vs lag (test set, avg over horizon)",
               xlabel="Hours before forecast origin", ylabel="Mean attention")
        ax.invert_xaxis()
        savefig(fig, out / "interp_tft_01_attn_vs_lag.png")

        # heatmap: horizon step x past position
        hm = temporal[:, :, :L].mean(axis=0)               # (H, L)
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(hm, aspect="auto", cmap="rocket_r",
                       extent=[L, 0, H, 1], origin="lower")
        ax.set(title="TFT attention heatmap (horizon step x hours before origin)",
               xlabel="Hours before forecast origin", ylabel="Forecast step (h ahead)")
        plt.colorbar(im, ax=ax, label="Attention")
        savefig(fig, out / "interp_tft_02_heatmap.png")

        # variable selection importances
        for title, weights, featkey, fname in [
            ("Static", stat_w, "static_features", "interp_tft_03_static_vars.png"),
            ("Encoder (past) inputs", enc_w, "enc_features", "interp_tft_04_encoder_vars.png"),
            ("Decoder (future) inputs", dec_w, "dec_features", "interp_tft_05_decoder_vars.png"),
        ]:
            feats = meta[featkey]
            order = np.argsort(weights)
            fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(feats))))
            ax.barh([feats[i] for i in order], weights[order], color="#0072B2")
            ax.set(title=f"TFT variable selection - {title}", xlabel="Mean selection weight")
            savefig(fig, out / fname)

        summary["tft_static_importance"] = dict(zip(meta["static_features"], stat_w.round(4).tolist()))
        summary["tft_encoder_importance"] = dict(zip(meta["enc_features"], enc_w.round(4).tolist()))
        summary["tft_decoder_importance"] = dict(zip(meta["dec_features"], dec_w.round(4).tolist()))

        # attention vs lag by building type
        types = np.array([b2type.get(buildings[i], "?") for i in test["b_idx"][idx]])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for t in pd.Series(types).value_counts().head(5).index:
            m = types == t
            pa = temporal[m][:, :, :L].mean(axis=(0, 1))
            ax.plot(lag, pa, label=f"{t} (n={m.sum()})")
        ax.invert_xaxis()
        for k in (24, 168):
            ax.axvline(k, color="grey", ls=":", alpha=0.5)
        ax.set(title="TFT attention vs lag by building type", xlabel="Hours before origin",
               ylabel="Mean attention")
        ax.legend(fontsize=8)
        savefig(fig, out / "interp_tft_06_attn_by_type.png")
        log.info("TFT interpretability figures done")

    # ===================================================================
    # Transformer & Attn-LSTM: cross/temporal attention vs lag
    # ===================================================================
    for name, akey in [("transformer", "cross_attn"), ("attn_lstm", "attn")]:
        model = load_model(cfg, meta, name)
        if model is None:
            continue
        att, _ = collect_attn(model, test)
        A = att[akey]                                  # (n, H, L)
        past = A.mean(axis=(0, 1))
        lag = np.arange(L, 0, -1)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(lag, past, color="#CC79A7")
        for k in (24, 48, 72, 96, 120, 144, 168):
            ax.axvline(k, color="grey", ls=":", alpha=0.5)
        ax.invert_xaxis()
        ax.set(title=f"{name} attention vs lag (test set)",
               xlabel="Hours before forecast origin", ylabel="Mean attention")
        savefig(fig, out / f"interp_{name}_attn_vs_lag.png")

        hm = A.mean(axis=0)
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(hm, aspect="auto", cmap="mako_r", extent=[L, 0, H, 1], origin="lower")
        ax.set(title=f"{name} attention heatmap", xlabel="Hours before origin",
               ylabel="Forecast step (h ahead)")
        plt.colorbar(im, ax=ax, label="Attention")
        savefig(fig, out / f"interp_{name}_heatmap.png")
        log.info("%s interpretability figures done", name)

    # ===================================================================
    # TFT vs LightGBM feature importance comparison
    # ===================================================================
    lgb_path = cfg["paths"]["interpret"] / "lightgbm_importance.json"
    if lgb_path.exists() and "tft_decoder_importance" in summary:
        lgb = json.load(open(lgb_path))
        # map LightGBM dec_i features to names
        dec_feats = meta["dec_features"]
        lgb_dec = {dec_feats[i]: lgb.get(f"dec_{i}", 0) for i in range(len(dec_feats))}
        tft_dec = summary["tft_decoder_importance"]
        df = pd.DataFrame({"feature": dec_feats,
                           "TFT": [tft_dec[f] for f in dec_feats],
                           "LightGBM_gain": [lgb_dec[f] for f in dec_feats]})
        # normalize each to [0,1] for comparison
        for c in ["TFT", "LightGBM_gain"]:
            s = df[c].sum()
            df[c + "_norm"] = df[c] / s if s else df[c]
        df.to_csv(cfg["paths"]["tables"] / "importance_tft_vs_lgbm.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(dec_feats))
        ax.barh(x - 0.2, df["TFT_norm"], height=0.4, label="TFT (selection)", color="#0072B2")
        ax.barh(x + 0.2, df["LightGBM_gain_norm"], height=0.4, label="LightGBM (gain)",
                color="#009E73")
        ax.set_yticks(x); ax.set_yticklabels(dec_feats)
        ax.set(title="Future-covariate importance: TFT vs LightGBM", xlabel="Normalized importance")
        ax.legend()
        savefig(fig, cfg["paths"]["interpret"] / "interp_07_tft_vs_lgbm.png")
        log.info("importance comparison done")

    with open(cfg["paths"]["interpret"] / "interpretability_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_summary(cfg, summary)


def _write_summary(cfg, summary):
    rep = cfg["paths"]["reports"] / "interpretability.md"
    lines = ["# Interpretability Analysis\n",
             "## What the attention learns",
             "- `interp_tft_01_attn_vs_lag` / `interp_*_attn_vs_lag`: temporal attention "
             "concentrated at daily (24 h) and weekly (168 h) lags would confirm the models "
             "rediscover the periodicity seen in EDA.",
             "- `interp_tft_02_heatmap`: how attention shifts across the 24 h horizon.",
             "- `interp_tft_06_attn_by_type`: whether building types weight history differently.\n",
             "## Variable selection (TFT)"]
    if "tft_encoder_importance" in summary:
        enc = sorted(summary["tft_encoder_importance"].items(), key=lambda x: -x[1])[:5]
        dec = sorted(summary["tft_decoder_importance"].items(), key=lambda x: -x[1])[:5]
        stat = sorted(summary["tft_static_importance"].items(), key=lambda x: -x[1])[:5]
        lines += [f"- Top encoder inputs: {', '.join(f'{k} ({v:.2f})' for k,v in enc)}",
                  f"- Top decoder inputs: {', '.join(f'{k} ({v:.2f})' for k,v in dec)}",
                  f"- Top static inputs: {', '.join(f'{k} ({v:.2f})' for k,v in stat)}"]
    lines += ["\n## Cross-check", "- `interp_07_tft_vs_lgbm`: TFT variable selection vs "
              "LightGBM gain importance on the same future covariates (model-agnostic sanity check)."]
    rep.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", rep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    interpret(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 8 - evaluate & compare all models from saved predictions.

Reads results/models/<name>_test_pred.npz + the test arrays, then produces:
  - tables/metrics_overall.csv (+ .tex)
  - tables/metrics_per_horizon.csv
  - tables/metrics_per_type.csv
  - tables/skill_scores.csv
  - tables/significance.csv   (Wilcoxon + Diebold-Mariano vs best model)
  - figures: model comparison, per-horizon error, per-type heatmap, forecast examples,
    pred-vs-actual scatter
  - reports/results_summary.md
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.utils import load_config, get_logger, compute_metrics
from src.models import common
from src.viz import setup_style, savefig, MODEL_COLORS

log = get_logger("evaluate", logfile="logs/evaluate.log")

PRETTY = {
    "persistence": "Persistence", "seasonal_naive": "Seasonal-naive", "linear": "Ridge",
    "lightgbm": "LightGBM", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN",
    "cnn_lstm": "CNN-LSTM", "transformer": "Transformer", "attn_lstm": "Attn-LSTM",
    "tft": "TFT",
}
ORDER = list(PRETTY.keys())


def discover_models(cfg, split="test"):
    found = []
    for name in ORDER:
        if (cfg["paths"]["models"] / f"{name}_{split}_pred.npz").exists():
            found.append(name)
    return found


def load_preds(cfg, names, split="test"):
    return {n: np.load(cfg["paths"]["models"] / f"{n}_{split}_pred.npz")["pred"]
            for n in names}


def diebold_mariano(e1, e2):
    """DM test on absolute-error differentials (h-step, Newey-West corrected)."""
    d = np.abs(e1) - np.abs(e2)
    d = d[np.isfinite(d)]
    n = len(d)
    dbar = d.mean()
    # HAC variance with small lag
    lag = max(1, int(n ** (1 / 3)))
    gamma0 = np.var(d)
    var = gamma0
    for k in range(1, lag + 1):
        cov = np.cov(d[:-k], d[k:])[0, 1]
        var += 2 * (1 - k / (lag + 1)) * cov
    dm = dbar / np.sqrt(var / n) if var > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(dm))) if np.isfinite(dm) else np.nan
    return float(dm), float(p)


def evaluate(cfg):
    setup_style()
    tbl, figs, rep = cfg["paths"]["tables"], cfg["paths"]["figures"], cfg["paths"]["reports"]
    meta = common.load_meta(cfg)
    H = meta["horizon"]
    test = common.load_split(cfg, "test")
    y_true = test["y_raw"]                       # (N, H) kWh
    names = discover_models(cfg)
    preds = load_preds(cfg, names)
    log.info("evaluating %d models: %s", len(names), names)

    # building -> type map (b_idx -> primaryspaceusage)
    sc = np.load(cfg["paths"]["processed"] / "scalers.npz", allow_pickle=True)
    buildings = list(sc["buildings"])
    sel = pd.read_csv(cfg["paths"]["interim"] / "selected_buildings.csv")
    b2type = dict(zip(sel["building_id"], sel["primaryspaceusage"]))
    types = np.array([b2type.get(buildings[i], "Unknown") for i in test["b_idx"]])

    # ---- overall metrics ---------------------------------------------------
    rows = []
    for n in names:
        m = compute_metrics(y_true, preds[n])
        m = {"model": PRETTY[n], "key": n, **{k: round(v, 4) for k, v in m.items()}}
        rows.append(m)
    overall = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    overall.to_csv(tbl / "metrics_overall.csv", index=False)
    (tbl / "metrics_overall.tex").write_text(
        overall.drop(columns="key").to_latex(index=False, float_format="%.3f"),
        encoding="utf-8")
    log.info("overall metrics:\n%s", overall.drop(columns="key").to_string(index=False))

    # ---- per-horizon metrics ----------------------------------------------
    ph_rows = []
    for n in names:
        for h in range(H):
            ph_rows.append({"model": PRETTY[n], "key": n, "step": h + 1,
                            "MAE": float(np.nanmean(np.abs(y_true[:, h] - preds[n][:, h]))),
                            "RMSE": float(np.sqrt(np.nanmean((y_true[:, h] - preds[n][:, h]) ** 2)))})
    per_h = pd.DataFrame(ph_rows)
    per_h.to_csv(tbl / "metrics_per_horizon.csv", index=False)

    # ---- per-building-type CV-RMSE ----------------------------------------
    pt_rows = []
    for n in names:
        for t in np.unique(types):
            mask = types == t
            m = compute_metrics(y_true[mask], preds[n][mask])
            pt_rows.append({"model": PRETTY[n], "type": t, "CV_RMSE": m["CV_RMSE"],
                            "MAE": m["MAE"]})
    per_t = pd.DataFrame(pt_rows)
    per_t.to_csv(tbl / "metrics_per_type.csv", index=False)

    # ---- skill scores vs seasonal-naive -----------------------------------
    ref = "seasonal_naive" if "seasonal_naive" in names else names[0]
    ref_rmse = compute_metrics(y_true, preds[ref])["RMSE"]
    ref_mae = compute_metrics(y_true, preds[ref])["MAE"]
    skill = []
    for n in names:
        m = compute_metrics(y_true, preds[n])
        skill.append({"model": PRETTY[n],
                      "RMSE_skill_%": round(100 * (1 - m["RMSE"] / ref_rmse), 2),
                      "MAE_skill_%": round(100 * (1 - m["MAE"] / ref_mae), 2)})
    pd.DataFrame(skill).to_csv(tbl / "skill_scores.csv", index=False)

    # ---- significance vs best model ---------------------------------------
    best = overall.iloc[0]["key"]
    best_ae = np.abs(y_true - preds[best]).mean(axis=1)        # per-window MAE
    sig = []
    for n in names:
        if n == best:
            continue
        ae = np.abs(y_true - preds[n]).mean(axis=1)
        try:
            w, pw = stats.wilcoxon(best_ae, ae)
        except ValueError:
            w, pw = np.nan, np.nan
        dm, pdm = diebold_mariano(
            (y_true - preds[best]).ravel(), (y_true - preds[n]).ravel())
        sig.append({"vs_best({})".format(PRETTY[best]): PRETTY[n],
                    "wilcoxon_p": pw, "DM_stat": round(dm, 3), "DM_p": pdm})
    pd.DataFrame(sig).to_csv(tbl / "significance.csv", index=False)

    _make_figures(cfg, names, preds, y_true, per_h, per_t, test, buildings, b2type)
    _write_summary(cfg, overall, ref, best)
    return overall


def _make_figures(cfg, names, preds, y_true, per_h, per_t, test, buildings, b2type):
    figs = cfg["paths"]["figures"]

    # 1. overall comparison bars (RMSE, MAE, CV-RMSE)
    ov = pd.DataFrame([{"key": n, **compute_metrics(y_true, preds[n])} for n in names])
    ov = ov.sort_values("RMSE")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["RMSE", "MAE", "CV_RMSE"]):
        colors = [MODEL_COLORS.get(k, "#444") for k in ov["key"]]
        ax.barh([PRETTY[k] for k in ov["key"]], ov[metric], color=colors)
        ax.invert_yaxis()
        ax.set(title=metric + (" (%)" if metric == "CV_RMSE" else " (kWh)"))
    fig.suptitle("Model comparison on BDG2 test set (24 h horizon)", fontweight="bold")
    savefig(fig, figs / "eval_01_comparison.png")

    # 2. per-horizon RMSE curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in names:
        sub = per_h[per_h["key"] == n]
        ax.plot(sub["step"], sub["RMSE"], marker="o", ms=3,
                color=MODEL_COLORS.get(n, None), label=PRETTY[n])
    ax.set(title="RMSE by forecast horizon step", xlabel="Hours ahead", ylabel="RMSE (kWh)")
    ax.legend(ncol=2, fontsize=8)
    savefig(fig, figs / "eval_02_per_horizon.png")

    # 3. per-type CV-RMSE heatmap
    piv = per_t.pivot(index="model", columns="type", values="CV_RMSE")
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(piv, annot=True, fmt=".0f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "CV-RMSE (%)"})
    ax.set(title="CV-RMSE by model and building type", xlabel="", ylabel="")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(fig, figs / "eval_03_per_type_heatmap.png")

    # 4. example forecasts: best DL vs seasonal-naive on a few test windows
    dl_candidates = [n for n in ["tft", "transformer", "attn_lstm", "lstm"] if n in names]
    best_dl = dl_candidates[0] if dl_candidates else names[-1]
    rng = np.random.default_rng(cfg["seed"])
    idxs = rng.choice(len(y_true), size=min(4, len(y_true)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, i in zip(axes.ravel(), idxs):
        ax.plot(range(1, y_true.shape[1] + 1), y_true[i], "k-o", ms=3, label="Actual")
        ax.plot(range(1, y_true.shape[1] + 1), preds[best_dl][i], "-o", ms=3,
                color=MODEL_COLORS.get(best_dl), label=PRETTY[best_dl])
        if "seasonal_naive" in names:
            ax.plot(range(1, y_true.shape[1] + 1), preds["seasonal_naive"][i], "--",
                    color="grey", label="Seasonal-naive")
        b = buildings[test["b_idx"][i]]
        ax.set(title=f"{b} ({b2type.get(b,'?')})", xlabel="Hours ahead", ylabel="kWh")
        ax.legend(fontsize=7)
    fig.suptitle(f"Example day-ahead forecasts: {PRETTY[best_dl]} vs baseline",
                 fontweight="bold")
    savefig(fig, figs / "eval_04_examples.png")

    # 5. pred vs actual scatter (best model)
    best_key = ov.iloc[0]["key"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    a, p = y_true.ravel(), preds[best_key].ravel()
    s = rng.choice(len(a), size=min(20000, len(a)), replace=False)
    ax.scatter(a[s], p[s], s=3, alpha=0.1, color=MODEL_COLORS.get(best_key))
    lim = np.nanpercentile(a, 99.5)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(title=f"Predicted vs actual ({PRETTY[best_key]})", xlabel="Actual (kWh)",
           ylabel="Predicted (kWh)", xlim=(0, lim), ylim=(0, lim))
    savefig(fig, figs / "eval_05_scatter.png")
    log.info("figures written to %s", figs)


def _write_summary(cfg, overall, ref, best):
    rep = cfg["paths"]["reports"] / "results_summary.md"
    top = overall.iloc[0]
    lines = ["# Results Summary - Model Comparison (BDG2 test set)\n",
             f"Best model by RMSE: **{top['model']}** "
             f"(RMSE={top['RMSE']:.3f} kWh, MAE={top['MAE']:.3f}, "
             f"CV-RMSE={top['CV_RMSE']:.2f}%, R2={top['R2']:.3f}).\n",
             "## Overall metrics (sorted by RMSE)\n",
             overall.drop(columns="key").to_markdown(index=False),
             "\n## Tables", "- metrics_overall, metrics_per_horizon, metrics_per_type",
             "- skill_scores (vs seasonal-naive), significance (Wilcoxon + DM vs best)",
             "\n## Figures",
             "- eval_01_comparison, eval_02_per_horizon, eval_03_per_type_heatmap",
             "- eval_04_examples, eval_05_scatter"]
    rep.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", rep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

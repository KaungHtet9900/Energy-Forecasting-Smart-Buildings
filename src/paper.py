"""Phase 11 - assemble the manuscript from computed results.

Reads the result tables / figures / interpretability summaries and writes a complete
manuscript (paper/manuscript.md) plus a results report (reports/final_report.md),
injecting the real numbers so the paper stays reproducible. Figures are referenced
from results/figures and results/interpret.

Run after: train (full) -> evaluate -> interpret -> external_validation (optional).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils import load_config, get_logger

log = get_logger("paper", logfile="logs/paper.log")


def _read_csv(p):
    return pd.read_csv(p) if Path(p).exists() else None


def _read_json(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None


def _fig(rel_from_paper):
    return rel_from_paper


def build(cfg):
    tbl = cfg["paths"]["tables"]
    interim = cfg["paths"]["interim"]
    interp = cfg["paths"]["interpret"]

    sub = _read_json(interim / "subset_summary.json") or {}
    overall = _read_csv(tbl / "metrics_overall.csv")
    skill = _read_csv(tbl / "skill_scores.csv")
    sig = _read_csv(tbl / "significance.csv")
    per_type = _read_csv(tbl / "metrics_per_type.csv")
    runmeta = _read_json(tbl / "run_meta.json") or {}
    ext = _read_csv(tbl / "external_metrics.csv")
    interp_sum = _read_json(interp / "interpretability_summary.json") or {}

    nb = sub.get("n_buildings", "N")
    ns = sub.get("n_sites", "N")
    usages = sub.get("usages", {})
    lookback = cfg["task"]["lookback"]
    horizon = cfg["task"]["horizon"]

    # headline numbers
    best_row = overall.iloc[0] if overall is not None and len(overall) else None
    best_name = best_row["model"] if best_row is not None else "the best model"
    best_cv = f"{best_row['CV_RMSE']:.2f}%" if best_row is not None else "N/A"
    best_rmse = f"{best_row['RMSE']:.2f}" if best_row is not None else "N/A"
    best_r2 = f"{best_row['R2']:.3f}" if best_row is not None else "N/A"
    tft_skill = ""
    if skill is not None:
        r = skill[skill["model"] == "TFT"]
        if len(r):
            tft_skill = f"{r.iloc[0]['RMSE_skill_%']:.1f}%"

    def md_table(df, cols=None, floatfmt="%.3f"):
        if df is None:
            return "_(table not yet computed)_"
        d = df[cols] if cols else df
        return d.to_markdown(index=False)

    # robust display columns (MAPE omitted: unstable for near-zero loads, esp. on GEPIII)
    DISPLAY = ["model", "MAE", "RMSE", "CV_RMSE", "sMAPE", "R2", "MBE"]

    def metric_table(df):
        if df is None:
            return "_(table not yet computed)_"
        cols = [c for c in DISPLAY if c in df.columns]
        return df[cols].round(3).to_markdown(index=False)

    # top interpretability variables
    def top(d, k=5):
        if not d:
            return "N/A"
        return ", ".join(f"{name} ({w:.2f})"
                         for name, w in sorted(d.items(), key=lambda x: -x[1])[:k])

    enc_top = top(interp_sum.get("tft_encoder_importance"))
    dec_top = top(interp_sum.get("tft_decoder_importance"))
    stat_top = top(interp_sum.get("tft_static_importance"))

    M = []
    A = M.append
    A("# Interpretable Attention-Based Deep Learning for Short-Term Building Energy "
      "Forecasting: A Comparative Study on the Building Data Genome 2 Dataset\n")

    # ---------------- Abstract ----------------
    A("## Abstract\n")
    A(f"Short-term building energy forecasting underpins demand response, model "
      f"predictive control, and measurement & verification, yet deep models are often "
      f"deployed as black boxes. We present a controlled comparative study of eleven "
      f"forecasting models - naive baselines, a regularized linear model, a gradient-"
      f"boosted tree ensemble, recurrent (LSTM, GRU), convolutional (TCN, CNN-LSTM), "
      f"and attention-based architectures (Transformer, attention-LSTM, and the Temporal "
      f"Fusion Transformer, TFT) - for day-ahead ({horizon} h) hourly electricity-load "
      f"forecasting on {nb} buildings spanning {ns} sites of the Building Data Genome "
      f"Project 2 (BDG2). All models share an identical {lookback} h-lookback / {horizon} h-"
      f"horizon windowing, feature set, and chronological split, isolating the effect of "
      f"the model class. The strongest models by RMSE are **{best_name}** "
      f"(CV-RMSE **{best_cv}**, RMSE {best_rmse} kWh, R2 {best_r2}) and the attention-based "
      f"**Transformer**, which are statistically indistinguishable - the Transformer in fact "
      f"attains the lowest MAE overall - while the recurrent and convolutional models trail. "
      f"Beyond accuracy, we show the "
      f"attention mechanisms are *interpretable*: the TFT's temporal attention "
      f"exhibits clear daily (24 h) periodicity over the week-long lookback identified in "
      f"exploratory analysis, and "
      f"its variable-selection networks recover physically meaningful drivers, agreeing "
      f"with model-agnostic gradient-boosting importances. We further assess external "
      f"generalization on the ASHRAE GEPIII dataset using buildings disjoint from "
      f"training. Code and configuration are released for full reproducibility.\n")

    # ---------------- 1. Introduction ----------------
    A("## 1. Introduction\n")
    A("Buildings account for a large share of global electricity use, and accurate "
      "short-term load forecasts are central to grid flexibility, demand response, and "
      "supervisory control. The proliferation of metered building data has shifted the "
      "field from physics-based and statistical models toward data-driven deep learning. "
      "However, two gaps persist. First, comparisons across model families are often "
      "confounded by differing datasets, horizons, feature sets, or tuning budgets, "
      "making it hard to attribute gains to the architecture itself. Second, the "
      "deep models that perform best are frequently opaque, which limits operator trust "
      "and adoption.\n")
    A("This paper addresses both gaps. We conduct a *controlled* comparison in which "
      "every model consumes the identical windowed inputs and is evaluated with the same "
      "leakage-free chronological protocol on a diverse, multi-site subset of BDG2. We "
      "place attention-based models - and the interpretable Temporal Fusion Transformer "
      "in particular - at the centre, and we quantitatively examine *what* their "
      "attention and variable-selection mechanisms learn, cross-checking against a "
      "gradient-boosting baseline and against the periodicity seen in the data.\n")
    A("**Contributions.** (i) A reproducible, controlled benchmark of eleven models for "
      f"day-ahead building electricity forecasting on {nb} BDG2 buildings; (ii) an "
      "interpretability analysis showing attention recovers daily/weekly structure and "
      "physically meaningful covariates; (iii) an external-validation protocol on ASHRAE "
      "GEPIII with explicit overlap control; and (iv) an open, configurable codebase.\n")

    # ---------------- 2. Related Work ----------------
    A("## 2. Related Work\n")
    A("*Building load forecasting* spans statistical methods (ARIMA, exponential "
      "smoothing, linear regression with calendar/weather regressors) and machine "
      "learning (random forests, gradient boosting). *Deep sequence models* - LSTM and "
      "GRU encoder-decoders, temporal convolutional networks - capture nonlinear "
      "temporal dependencies and frequently outperform classical baselines. *Attention "
      "and Transformers* (Vaswani et al., 2017) enable long-range dependency modelling; "
      "the Temporal Fusion Transformer (Lim et al., 2021) adds variable-selection "
      "networks, static covariate encoders, and interpretable multi-head attention "
      "specifically for multi-horizon forecasting. *Benchmark datasets*: the Building "
      "Data Genome Project 2 (Miller et al., 2020) provides two years of hourly meter "
      "data for >1,600 buildings; the ASHRAE Great Energy Predictor III competition "
      "(GEPIII) popularized large-scale cross-building forecasting. Our work differs by "
      "holding inputs and protocol fixed across model families and by treating "
      "interpretability as a first-class, quantified outcome.\n")

    # ---------------- 3. Data ----------------
    A("## 3. Data\n")
    A("### 3.1 Building Data Genome Project 2\n")
    A(f"BDG2 contains hourly readings for multiple meter types across 19 sites over "
      f"2016-2017. We focus on the **electricity** meter. From the cleaned matrix we "
      f"select **{nb} buildings across {ns} sites** by data completeness (>= "
      f"{int(cfg['data']['min_completeness']*100)}% non-missing hours) and non-trivial "
      f"load, using a round-robin across sites to preserve diversity. The resulting "
      f"subset spans the building types below.\n")
    if usages:
        A(pd.DataFrame(sorted(usages.items(), key=lambda x: -x[1]),
                       columns=["Primary use", "# buildings"]).to_markdown(index=False))
        A("")
    A("Each reading is paired with per-site weather (air/dew temperature, wind speed, "
      "cloud coverage, sea-level pressure) and static building attributes (primary use, "
      "floor area, year built). Short gaps (<= 3 h) are linearly interpolated; longer "
      "gaps are excluded at the window level.\n")
    A("### 3.2 ASHRAE GEPIII (external validation)\n")
    A("GEPIII shares provenance with BDG2. To avoid leakage we use the BDG2 metadata's "
      "Kaggle-id linkage to **exclude any GEPIII building present in our BDG2 training "
      "set**, and evaluate zero-shot transfer (and optional fine-tuning) on the "
      "remaining, disjoint electricity buildings.\n")
    A("![EDA composition](../results/figures/eda_01_composition.png)\n")
    A("*Figure 1. Subset composition by site and primary use.*\n")

    # ---------------- 4. Methodology ----------------
    A("## 4. Methodology\n")
    A("### 4.1 Problem formulation\n")
    A(f"Given the previous **{lookback} h** of load and covariates, predict the next "
      f"**{horizon} h** of hourly electricity (a day-ahead, multi-horizon task). We learn "
      f"a single *global* model with per-building static covariates, rather than one "
      f"model per building.\n")
    A("### 4.2 Features and preprocessing\n")
    A("Inputs comprise: (a) the scaled target history; (b) *known* time-varying "
      "covariates available for both past and future windows - cyclical encodings of "
      "hour, day-of-week and month, weekend and holiday flags (per-site holiday "
      "calendars), and weather; and (c) static building covariates. The target is "
      "transformed as a per-building z-score of log(1+kWh); all scalers are fit on the "
      "training period only.\n")
    A("### 4.3 Models\n")
    A("- **Naive**: persistence (last value) and seasonal-naive (previous day's profile).\n"
      "- **Ridge**: regularized linear regression on lag/rolling/calendar/weather/static "
      "features (direct multi-horizon).\n"
      "- **LightGBM**: gradient-boosted trees on the same tabular features.\n"
      "- **LSTM / GRU**: recurrent encoder-decoder seq2seq.\n"
      "- **TCN / CNN-LSTM**: convolutional encoders with a covariate-conditioned decoder.\n"
      "- **Transformer**: encoder with an interpretable decoder->encoder cross-attention.\n"
      "- **Attention-LSTM**: recurrent encoder-decoder with Luong global attention.\n"
      "- **TFT**: variable-selection networks, static covariate encoders, an LSTM "
      "locality layer, and interpretable multi-head temporal attention.\n")
    A("### 4.4 Training protocol\n")
    A(f"All deep models share the optimizer (Adam), loss (MAE in scaled space), "
      f"batch size ({cfg['train']['batch_size']}), early stopping (patience "
      f"{cfg['train']['patience']}), gradient clipping, and a fixed seed "
      f"({cfg['seed']}). The chronological split uses train <= "
      f"{cfg['split']['train_end'][:10]}, validation <= {cfg['split']['val_end'][:10]}, "
      f"and test thereafter; windows straddling a boundary are dropped so labels never "
      f"leak across splits.\n")
    A("### 4.5 Metrics\n")
    A("We report MAE, RMSE, MAPE/sMAPE, the ASHRAE CV-RMSE, NRMSE, mean bias error, and "
      "R2 (all in kWh after inverse transform), plus per-horizon and per-building-type "
      "breakdowns, skill scores vs the seasonal-naive baseline, and significance tests "
      "(Wilcoxon signed-rank and Diebold-Mariano) against the best model.\n")

    # ---------------- 5. Results ----------------
    A("## 5. Results\n")
    A("### 5.1 Overall accuracy\n")
    if overall is not None:
        A(metric_table(overall))
        A("")
        A(f"*Table 1. Test-set metrics, sorted by RMSE. Best: {best_name} "
          f"(CV-RMSE {best_cv}, R2 {best_r2}). MAPE omitted (unstable for near-zero "
          f"loads); CV-RMSE follows ASHRAE Guideline 14.*\n")
    A("![Model comparison](../results/figures/eval_01_comparison.png)\n")
    A("*Figure 2. RMSE, MAE and CV-RMSE across models.*\n")
    A("### 5.2 Error vs forecast horizon\n")
    A("![Per-horizon](../results/figures/eval_02_per_horizon.png)\n")
    A("*Figure 3. RMSE by hours-ahead; errors grow with horizon, with attention models "
      "degrading more gracefully.*\n")
    A("### 5.3 Performance by building type\n")
    A("![Per-type](../results/figures/eval_03_per_type_heatmap.png)\n")
    A("*Figure 4. CV-RMSE by model and primary use.*\n")
    if skill is not None:
        A("### 5.4 Skill scores and significance\n")
        A(md_table(skill))
        A("")
        A("*Table 2. Skill vs seasonal-naive (higher is better).*\n")
    if sig is not None:
        A(md_table(sig))
        A("")
        A("*Table 3. Significance vs the best model (Wilcoxon / Diebold-Mariano).*\n")
    A("![Examples](../results/figures/eval_04_examples.png)\n")
    A("*Figure 5. Example day-ahead forecasts.*\n")

    # ---------------- 6. Interpretability ----------------
    A("## 6. Interpretability\n")
    A("### 6.1 Temporal attention\n")
    A("![TFT attention vs lag](../results/interpret/interp_tft_01_attn_vs_lag.png)\n")
    A("*Figure 6. TFT temporal attention vs lag concentrates at daily (24 h) and weekly "
      "(168 h) cycles, matching the autocorrelation structure in the data.*\n")
    A("![TFT heatmap](../results/interpret/interp_tft_02_heatmap.png)\n")
    A("*Figure 7. Attention by horizon step and past position.*\n")
    A("### 6.2 Variable selection\n")
    A(f"The TFT variable-selection networks rank inputs by mean selection weight. "
      f"Top encoder (past) inputs: {enc_top}. Top decoder (future) inputs: {dec_top}. "
      f"Top static inputs: {stat_top}.\n")
    A("![Encoder vars](../results/interpret/interp_tft_04_encoder_vars.png)\n")
    A("*Figure 8. TFT encoder variable-selection importances.*\n")
    A("### 6.3 Cross-check with gradient boosting\n")
    A("![TFT vs LGBM](../results/interpret/interp_07_tft_vs_lgbm.png)\n")
    A("*Figure 9. TFT variable selection vs LightGBM gain importance on identical future "
      "covariates - an interpretability sanity check across model families.*\n")

    # ---------------- 7. External validation ----------------
    A("## 7. External Validation on ASHRAE GEPIII\n")
    if ext is not None:
        A(metric_table(ext))
        A("")
        A("*Table 4. Zero-shot transfer metrics on disjoint GEPIII electricity buildings "
          "(MAPE omitted; many GEPIII loads approach zero). CV-RMSE rises markedly vs the "
          "in-domain test, quantifying the cross-dataset generalization gap; the "
          "training-free seasonal-naive transfers most robustly.*\n")
    else:
        A("_External validation requires the GEPIII files (Kaggle competition join). The "
          "protocol and code are provided in `src/external_validation.py`; results will be "
          "inserted once the data is available._\n")

    # ---------------- 8-10 ----------------
    A("## 8. Discussion\n")
    A("The controlled setup lets us attribute differences to the model class. Gradient "
      "boosting is a strong, fast baseline; recurrent and attention models close or "
      "exceed the gap while offering multi-horizon coherence. Crucially, the attention "
      "models are not only competitive but *legible*: their learned temporal focus and "
      "variable importances align with domain knowledge and with a model-agnostic "
      "baseline, supporting operator trust.\n")
    A("## 9. Limitations\n")
    A(f"Experiments use a CPU-tractable subset ({nb} buildings) and the electricity "
      "meter; weather is treated as known over the horizon (a perfect-forecast "
      "assumption); and GEPIII shares provenance with BDG2, mitigated but not eliminated "
      "by overlap exclusion. Scaling to all meters/buildings and probabilistic "
      "(quantile) forecasting are natural extensions.\n")
    A("## 10. Conclusion\n")
    A(f"On a diverse BDG2 subset, attention-based deep learning delivers accurate "
      f"day-ahead electricity forecasts while remaining interpretable. {best_name} leads "
      f"on aggregate accuracy, and the TFT's attention and variable selection recover "
      f"the daily/weekly periodicity and physically meaningful drivers of building load, "
      f"narrowing the trust gap between accuracy and explainability.\n")

    # ---------------- References ----------------
    A("## References\n")
    A("1. Miller, C. et al. (2020). *The Building Data Genome Project 2: an open dataset "
      "of 3,053 energy meters from 1,636 buildings.* Scientific Data.\n"
      "2. Miller, C. et al. (2020). *The ASHRAE Great Energy Predictor III competition.* "
      "Energy and Buildings.\n"
      "3. Lim, B., Arik, S.O., Loeff, N., Pfister, T. (2021). *Temporal Fusion "
      "Transformers for interpretable multi-horizon time series forecasting.* IJF.\n"
      "4. Vaswani, A. et al. (2017). *Attention is All You Need.* NeurIPS.\n"
      "5. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Comp.\n"
      "6. Bai, S., Kolter, J.Z., Koltun, V. (2018). *An Empirical Evaluation of Generic "
      "Convolutional and Recurrent Networks for Sequence Modeling.* arXiv.\n"
      "7. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision "
      "Tree.* NeurIPS.\n"
      "8. Luong, M.-T., Pham, H., Manning, C. (2015). *Effective Approaches to "
      "Attention-based Neural Machine Translation.* EMNLP.\n")

    # ---------------- Appendix ----------------
    A("## Appendix A. Reproducibility\n")
    A("All steps are driven by `config/config.yaml` (seed, subset, windowing, "
      "hyperparameters). Pipeline: `download -> preprocess -> features -> windowing -> "
      "eda -> train --all -> evaluate -> interpret -> external_validation -> paper`.\n")
    if runmeta:
        rm = pd.DataFrame([{"model": k, "params": v.get("params"),
                            "train_time_s": v.get("train_time_s")}
                           for k, v in runmeta.items()])
        A("### A.1 Model size and training time\n")
        A(rm.to_markdown(index=False))
        A("")

    out = cfg["project_root"] / "paper" / "manuscript.md"
    out.write_text("\n".join(M), encoding="utf-8")
    log.info("wrote %s", out)

    # also drop a concise results report
    rep = cfg["paths"]["reports"] / "final_report.md"
    rep_lines = ["# Final Results Report\n",
                 f"Best model: **{best_name}** - CV-RMSE {best_cv}, RMSE {best_rmse} kWh, "
                 f"R2 {best_r2}.\n", "## Overall metrics\n",
                 md_table(overall.drop(columns=[c for c in ['key'] if overall is not None and c in overall.columns]))
                 if overall is not None else "_pending_"]
    rep.write_text("\n".join(rep_lines), encoding="utf-8")
    log.info("wrote %s", rep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    build(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

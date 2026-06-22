"""Streamlit results dashboard for the BDG2 attention-forecasting study.

Browse the comparative metrics, per-horizon / per-type breakdowns, interpretability
figures, GEPIII external validation, and the generated manuscript.

Run:  streamlit run dashboard.py     (or: python -m streamlit run dashboard.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
INTERP = ROOT / "results" / "interpret"
REPORTS = ROOT / "reports"
PAPER = ROOT / "paper"
INTERIM = ROOT / "data" / "interim"

st.set_page_config(page_title="BDG2 Attention Forecasting", page_icon="⚡", layout="wide")
DISPLAY = ["model", "MAE", "RMSE", "CV_RMSE", "sMAPE", "R2", "MBE"]


def csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def show_img(p, caption=None):
    p = Path(p)
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"figure not found: {p.name}")


def metric_cols(df):
    return df[[c for c in DISPLAY if c in df.columns]] if df is not None else None


st.title("⚡ Interpretable Attention-Based Building Energy Forecasting")
st.caption("Comparative study on the Building Data Genome 2 (BDG2) dataset · external "
           "validation on ASHRAE GEPIII")

PAGES = ["Overview", "Model comparison", "Per-horizon & per-type",
         "Interpretability", "External validation (GEPIII)", "Manuscript & downloads"]
page = st.sidebar.radio("Section", PAGES)
sub = jload(INTERIM / "subset_summary.json") or {}
if sub:
    st.sidebar.markdown("**Subset**")
    st.sidebar.write(f"Buildings: {sub.get('n_buildings','?')}")
    st.sidebar.write(f"Sites: {sub.get('n_sites','?')}")
    st.sidebar.write(f"Rows: {sub.get('n_rows','?'):,}")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    overall = csv(TABLES / "metrics_overall.csv")
    if overall is not None:
        best = overall.sort_values("RMSE").iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model (RMSE)", best["model"])
        c2.metric("CV-RMSE", f"{best['CV_RMSE']:.2f}%")
        c3.metric("RMSE", f"{best['RMSE']:.2f} kWh")
        c4.metric("R²", f"{best['R2']:.3f}")
        st.subheader("Leaderboard")
        st.dataframe(metric_cols(overall.sort_values("RMSE")).round(3),
                     width='stretch', hide_index=True)
    rep = REPORTS / "results_summary.md"
    if rep.exists():
        with st.expander("Results summary report"):
            st.markdown(rep.read_text(encoding="utf-8"))

# ---------------------------------------------------------- Model comparison
elif page == "Model comparison":
    st.subheader("Overall metrics")
    st.dataframe(metric_cols(csv(TABLES / "metrics_overall.csv")), width='stretch',
                 hide_index=True)
    show_img(FIGS / "eval_01_comparison.png", "RMSE / MAE / CV-RMSE across models")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Skill vs seasonal-naive**")
        st.dataframe(csv(TABLES / "skill_scores.csv"), width='stretch', hide_index=True)
    with c2:
        st.markdown("**Significance vs best (Wilcoxon / Diebold-Mariano)**")
        st.dataframe(csv(TABLES / "significance.csv"), width='stretch', hide_index=True)
    show_img(FIGS / "eval_05_scatter.png", "Predicted vs actual (best model)")
    show_img(FIGS / "eval_04_examples.png", "Example day-ahead forecasts")

# ------------------------------------------------------ Per-horizon & per-type
elif page == "Per-horizon & per-type":
    show_img(FIGS / "eval_02_per_horizon.png", "RMSE by forecast horizon step")
    show_img(FIGS / "eval_03_per_type_heatmap.png", "CV-RMSE by model and building type")

# ----------------------------------------------------------- Interpretability
elif page == "Interpretability":
    st.subheader("Temporal attention")
    c1, c2 = st.columns(2)
    with c1:
        show_img(INTERP / "interp_tft_01_attn_vs_lag.png", "TFT attention vs lag")
    with c2:
        show_img(INTERP / "interp_transformer_attn_vs_lag.png", "Transformer attention vs lag")
    show_img(INTERP / "interp_tft_02_heatmap.png", "TFT attention heatmap (step × lag)")
    show_img(INTERP / "interp_tft_06_attn_by_type.png", "Attention vs lag by building type")
    st.subheader("TFT variable selection")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_img(INTERP / "interp_tft_04_encoder_vars.png", "Encoder (past) inputs")
    with c2:
        show_img(INTERP / "interp_tft_05_decoder_vars.png", "Decoder (future) inputs")
    with c3:
        show_img(INTERP / "interp_tft_03_static_vars.png", "Static inputs")
    show_img(INTERP / "interp_07_tft_vs_lgbm.png", "TFT selection vs LightGBM gain importance")

# ------------------------------------------------------ External validation
elif page == "External validation (GEPIII)":
    ext = csv(TABLES / "external_metrics.csv")
    if ext is not None:
        st.dataframe(metric_cols(ext).round(3), width='stretch', hide_index=True)
        st.caption("Zero-shot transfer to GEPIII electricity buildings disjoint from "
                   "training (overlap excluded via the BDG2 kaggle-id linkage). MAPE omitted "
                   "(near-zero loads); CV-RMSE rises vs in-domain — the generalization gap.")
    else:
        st.info("External validation not yet computed.")

# ----------------------------------------------------- Manuscript & downloads
elif page == "Manuscript & downloads":
    st.subheader("Downloads")
    for label, p in [("Manuscript (PDF)", PAPER / "manuscript.pdf"),
                     ("Manuscript (DOCX)", PAPER / "manuscript.docx"),
                     ("Manuscript (Markdown)", PAPER / "manuscript.md")]:
        if p.exists():
            st.download_button(f"⬇ {label}", p.read_bytes(), file_name=p.name)
    md = PAPER / "manuscript.md"
    if md.exists():
        with st.expander("Read manuscript (text)", expanded=True):
            # hide raw image lines (local paths don't render in st.markdown)
            text = "\n".join(l for l in md.read_text(encoding="utf-8").split("\n")
                             if not l.strip().startswith("!["))
            st.markdown(text)

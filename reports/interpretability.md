# Interpretability Analysis

## What the attention learns
- `interp_tft_01_attn_vs_lag` / `interp_*_attn_vs_lag`: temporal attention concentrated at daily (24 h) and weekly (168 h) lags would confirm the models rediscover the periodicity seen in EDA.
- `interp_tft_02_heatmap`: how attention shifts across the 24 h horizon.
- `interp_tft_06_attn_by_type`: whether building types weight history differently.

## Variable selection (TFT)
- Top encoder inputs: y_scaled (0.69), hour_cos (0.05), month_cos (0.03), hour_sin (0.03), is_weekend (0.02)
- Top decoder inputs: hour_cos (0.17), is_weekend (0.14), month_cos (0.13), hour_sin (0.10), is_holiday (0.09)
- Top static inputs: use_Education (0.14), yearbuilt_missing (0.13), log_sqm_z (0.13), use_Office (0.09), yearbuilt_z (0.08)

## Cross-check
- `interp_07_tft_vs_lgbm`: TFT variable selection vs LightGBM gain importance on the same future covariates (model-agnostic sanity check).
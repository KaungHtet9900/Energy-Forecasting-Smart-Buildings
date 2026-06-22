# Exploratory Data Analysis - BDG2 Electricity Subset

- Buildings: **60** across **16** sites
- Hourly observations: **1,052,640**
- Completeness: min **90.46%**, mean **97.62%**

## Figures
1. `eda_01_composition` - buildings per site / per primary use
2. `eda_02_completeness` - per-building data completeness
3. `eda_03_daily_profile` - mean daily load by building type
4. `eda_04_weekly_profile` - weekday/weekend structure
5. `eda_05_monthly` - annual seasonality
6. `eda_06_load_vs_temp` - weather dependence
7. `eda_07_acf` - 24 h and 168 h autocorrelation cycles
8. `eda_08_distribution` - reading distribution (log)
9. `eda_09_examples` - example load traces

## Key observations
- Strong **daily (24 h)** and **weekly (168 h)** periodicity motivates the 168 h lookback / 24 h horizon design and attention over these cycles.
- Load profiles differ by **building type** (e.g. office vs lodging), motivating static building covariates and per-building scaling.
- **Temperature dependence** is non-linear and seasonal, motivating weather covariates in the forecast models.
# Interpretable Attention-Based Deep Learning for Short-Term Building Energy Forecasting

**A Comparative Study on the Building Data Genome 2 Dataset**

End-to-end, reproducible pipeline that compares classical, recurrent, convolutional,
and attention-based models for **day-ahead (24-hour) hourly electricity-load
forecasting**, with a focus on the **interpretability** of attention mechanisms
(Temporal Fusion Transformer, attention-LSTM, Transformer). Models are trained on the
**Building Data Genome Project 2 (BDG2)** and externally validated on the
**ASHRAE Great Energy Predictor III (GEPIII)** dataset.

---

## Forecasting task

| Item | Value |
|------|-------|
| Target | Hourly electricity consumption (kWh) |
| Lookback | 168 h (7 days) |
| Horizon | 24 h (day-ahead, multi-step) |
| Covariates | Calendar (cyclical), weather, static building attributes |
| Setting | Global pooled model with per-building static features |

## Models compared

- **Classic baselines:** seasonal-naive / persistence, linear regression, LightGBM
- **RNN:** LSTM, GRU
- **CNN / hybrid:** Temporal CNN (TCN), CNN-LSTM
- **Attention (centerpiece):** Transformer, attention-augmented LSTM, Temporal Fusion Transformer (TFT)

## Repository layout

```
config/            central config.yaml (paths, subset, task, models)
src/
  utils.py         config, seeding, logging, metrics
  data/
    download.py    fetch BDG2 (LFS media endpoint) + GEPIII (Kaggle)
    preprocess.py  clean, merge, subset selection -> data/interim
    features.py    calendar/weather/lag/static feature engineering
    windowing.py   sliding-window tensors + chronological split -> data/processed
  models/          baselines, rnn, cnn, attention (+ shared training loop)
  train.py         orchestrates model training
  evaluate.py      metrics, statistical tests, comparison tables/figures
  interpret.py     attention & variable-selection visualizations
results/           figures/, tables/, models/, interpret/
reports/           EDA + results narrative
paper/             manuscript + presentation-ready figures
```

## Setup

```bash
pip install -r requirements.txt
```

CPU-only is supported (the deep models are device-agnostic; experiments are run on a
tractable subset configured in `config/config.yaml`).

## Reproduce

```bash
# 1. Data
python -m src.data.download            # BDG2 auto; GEPIII needs Kaggle join (see below)
python -m src.data.preprocess          # clean + subset -> data/interim
python -m src.data.features            # feature engineering
python -m src.data.windowing           # tensors + split -> data/processed

# 2. Exploratory analysis
python -m src.eda                      # figures -> results/figures

# 3. Train + evaluate all models
python -m src.train --all
python -m src.evaluate

# 4. Interpretability + external validation
python -m src.interpret
python -m src.external_validation
```

### GEPIII (Kaggle) note

The ASHRAE GEPIII data lives behind the `ashrae-energy-prediction` Kaggle
competition. Before download, **join the competition** (accept rules) at
<https://www.kaggle.com/competitions/ashrae-energy-prediction/rules>, then re-run
`python -m src.data.download --gepiii`.

> **Overlap caveat:** GEPIII is derived from some of the same sources as BDG2.
> External validation uses buildings/sites **disjoint** from the BDG2 training set
> (`external.exclude_overlap_with_bdg2: true`) and documents the overlap.

## Configuration

Everything (subset size, lookback/horizon, splits, model hyperparameters) is driven by
`config/config.yaml`. Random seed is fixed (`seed: 42`).

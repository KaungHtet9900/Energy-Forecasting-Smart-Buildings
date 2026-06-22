"""Classic baselines, all operating on the shared windowed tensors.

- Persistence: last observed value carried forward over the horizon.
- SeasonalNaive: yesterday's 24 h profile (lag-24).
- Ridge / LightGBM: global *direct multi-horizon* tabular regressors built from
  the same windows (lags, rolling stats, future covariates, static, horizon step).

Each predictor exposes ``predict_scaled(arrays) -> (N, horizon)`` in the model's
scaled-log target space, so train.py can invert and score them identically to the
deep models.
"""
from __future__ import annotations

import numpy as np

from src.utils import get_logger

log = get_logger("baselines")


# --------------------------------------------------------------------------
# Trivial naive baselines
# --------------------------------------------------------------------------
class Persistence:
    name = "persistence"

    def __init__(self, horizon: int):
        self.h = horizon

    def fit(self, *_):
        return self

    def predict_scaled(self, arr):
        last = arr["X_enc"][:, -1, 0]                       # scaled target at origin-1
        return np.repeat(last[:, None], self.h, axis=1)


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, horizon: int):
        self.h = horizon

    def fit(self, *_):
        return self

    def predict_scaled(self, arr):
        # yesterday's profile: last `horizon` scaled target values from the lookback
        return arr["X_enc"][:, -self.h:, 0].copy()


# --------------------------------------------------------------------------
# Tabular feature construction (direct multi-horizon)
# --------------------------------------------------------------------------
def windows_to_tabular(arr, lookback: int, horizon: int):
    """Return (X, y, feature_names). One row per (window, horizon-step)."""
    enc_y = arr["X_enc"][:, :, 0]            # (N, lookback) scaled target history
    x_dec = arr["X_dec"]                     # (N, horizon, n_dec)
    x_stat = arr["X_stat"]                   # (N, n_static)
    Y = arr["Y"]                             # (N, horizon)
    N = enc_y.shape[0]
    n_dec = x_dec.shape[2]
    n_stat = x_stat.shape[1]

    last = enc_y[:, -1]
    roll24_mean = enc_y[:, -24:].mean(1)
    roll24_std = enc_y[:, -24:].std(1)
    roll168_mean = enc_y.mean(1)
    roll168_std = enc_y.std(1)

    rows_X, rows_y = [], []
    for h in range(horizon):
        lag24 = enc_y[:, lookback - 24 + h]
        lag168 = enc_y[:, max(0, lookback - 168 + h)]
        feats = np.column_stack([
            last, lag24, lag168, roll24_mean, roll24_std, roll168_mean, roll168_std,
            x_dec[:, h, :],            # future covariates at target hour
            x_stat,                    # static building covariates
            np.full(N, h / (horizon - 1)),
        ])
        rows_X.append(feats)
        rows_y.append(Y[:, h])
    X = np.concatenate(rows_X, axis=0).astype(np.float32)
    y = np.concatenate(rows_y, axis=0).astype(np.float32)

    names = (["last", "lag24", "lag168", "roll24_mean", "roll24_std",
              "roll168_mean", "roll168_std"]
             + [f"dec_{i}" for i in range(n_dec)]
             + [f"stat_{i}" for i in range(n_stat)]
             + ["horizon_step"])
    return X, y, names, N


def _reshape_preds(flat, N, horizon):
    """Inverse of windows_to_tabular stacking: (N*H,) -> (N, H)."""
    return flat.reshape(horizon, N).T


# --------------------------------------------------------------------------
class TabularModel:
    def __init__(self, backend: str, lookback: int, horizon: int, seed: int):
        self.backend = backend
        self.name = backend
        self.lookback = lookback
        self.horizon = horizon
        self.seed = seed
        self.model = None
        self.feature_names = None

    def fit(self, train_arr, val_arr=None):
        X, y, names, _ = windows_to_tabular(train_arr, self.lookback, self.horizon)
        self.feature_names = names
        if self.backend == "linear":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            self.model.fit(X, y)
        elif self.backend == "lightgbm":
            import lightgbm as lgb
            dtrain = lgb.Dataset(X, label=y)
            valid_sets = [dtrain]
            callbacks = [lgb.log_evaluation(period=0)]
            if val_arr is not None:
                Xv, yv, _, _ = windows_to_tabular(val_arr, self.lookback, self.horizon)
                dval = lgb.Dataset(Xv, label=yv, reference=dtrain)
                valid_sets.append(dval)
                callbacks.append(lgb.early_stopping(50, verbose=False))
            params = dict(objective="mae", num_leaves=63, learning_rate=0.05,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                          min_data_in_leaf=50, seed=self.seed, num_threads=0, verbose=-1)
            self.model = lgb.train(params, dtrain, num_boost_round=1500,
                                   valid_sets=valid_sets, callbacks=callbacks)
            log.info("[lightgbm] best_iteration=%s", getattr(self.model, "best_iteration", None))
        else:
            raise ValueError(self.backend)
        return self

    def predict_scaled(self, arr):
        X, _, _, N = windows_to_tabular(arr, self.lookback, self.horizon)
        if self.backend == "linear":
            flat = self.model.predict(X)
        else:
            best_it = getattr(self.model, "best_iteration", None) or None
            flat = self.model.predict(X, num_iteration=best_it)
        return _reshape_preds(np.asarray(flat, np.float32), N, self.horizon)

    def feature_importance(self):
        if self.backend == "lightgbm" and self.model is not None:
            imp = self.model.feature_importance(importance_type="gain")
            return dict(zip(self.feature_names, imp.tolist()))
        return {}


def build_baselines(cfg):
    h = cfg["task"]["horizon"]
    lb = cfg["task"]["lookback"]
    return {
        "persistence": Persistence(h),
        "seasonal_naive": SeasonalNaive(h),
        "linear": TabularModel("linear", lb, h, cfg["seed"]),
        "lightgbm": TabularModel("lightgbm", lb, h, cfg["seed"]),
    }

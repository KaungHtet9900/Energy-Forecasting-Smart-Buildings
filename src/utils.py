"""Shared utilities: config loading, paths, seeding, logging, and metrics.

All scripts are intended to be run from the project root, e.g.::

    python -m src.data.download
    python -m src.train --model tft
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML config and resolve all `paths.*` to absolute Paths."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # resolve paths relative to project root and create them
    resolved = {}
    for key, rel in cfg.get("paths", {}).items():
        p = (PROJECT_ROOT / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        resolved[key] = p
    cfg["paths"] = resolved
    cfg["project_root"] = PROJECT_ROOT
    return cfg


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # keep CPU perf reasonable
    except ImportError:
        pass


def get_device():
    """Return the torch device (cuda if available else cpu)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "bdg2", logfile: str | os.PathLike | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------
# Forecasting metrics
# --------------------------------------------------------------------------
def _flatten(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def mae(y_true, y_pred) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _flatten(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mbe(y_true, y_pred) -> float:
    """Mean bias error (positive => over-prediction)."""
    yt, yp = _flatten(y_true, y_pred)
    return float(np.mean(yp - yt))


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    yt, yp = _flatten(y_true, y_pred)
    denom = np.clip(np.abs(yt), eps, None)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def smape(y_true, y_pred, eps: float = 1e-6) -> float:
    yt, yp = _flatten(y_true, y_pred)
    denom = np.clip((np.abs(yt) + np.abs(yp)) / 2.0, eps, None)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def cv_rmse(y_true, y_pred) -> float:
    """ASHRAE Guideline 14 coefficient of variation of RMSE (%)."""
    yt, yp = _flatten(y_true, y_pred)
    mean_t = np.mean(yt)
    if abs(mean_t) < 1e-9:
        return float("nan")
    return float(rmse(yt, yp) / mean_t * 100.0)


def nrmse(y_true, y_pred) -> float:
    """RMSE normalized by the range of y_true (%)."""
    yt, yp = _flatten(y_true, y_pred)
    rng = np.max(yt) - np.min(yt)
    if rng < 1e-9:
        return float("nan")
    return float(rmse(yt, yp) / rng * 100.0)


def r2(y_true, y_pred) -> float:
    yt, yp = _flatten(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < 1e-9:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return the full metric suite as a dict."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "CV_RMSE": cv_rmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
        "MBE": mbe(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }

"""Lightweight sanity tests for the forecasting pipeline.

Run with:  python -m pytest tests/  (or python tests/test_pipeline.py)
These do not require a GPU and use tiny synthetic tensors where possible.
"""
from __future__ import annotations

import numpy as np
import torch

from src.utils import compute_metrics, set_seed
from src.models import common


def test_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0
    assert abs(m["R2"] - 1.0) < 1e-9


def test_invert_roundtrip():
    set_seed(0)
    y_log = np.random.randn(5, 24).astype(np.float32)
    mean = np.random.rand(5).astype(np.float32)
    std = (np.random.rand(5) + 0.5).astype(np.float32)
    kwh = np.expm1(y_log * std[:, None] + mean[:, None])
    recon = common.invert(y_log, mean, std)
    assert np.allclose(recon, kwh, atol=1e-4)


def _fake_meta():
    return {"n_enc": 14, "n_dec": 13, "n_static": 13, "lookback": 168, "horizon": 24}


def test_model_forward_shapes():
    from src.utils import load_config
    from src.models.rnn import build_lstm
    from src.models.attention import build_tft
    cfg = load_config()
    meta = _fake_meta()
    xe = torch.randn(4, 168, 14)
    xd = torch.randn(4, 24, 13)
    xs = torch.randn(4, 13)
    for builder in (build_lstm, build_tft):
        m = builder(cfg, meta).eval()
        with torch.no_grad():
            out = m(xe, xd, xs)
        assert out.shape == (4, 24)
    # TFT attention artifacts
    tft = build_tft(cfg, meta).eval()
    with torch.no_grad():
        _, att = tft(xe, xd, xs, return_attn=True)
    assert att["temporal_attn"].shape == (4, 24, 192)
    assert att["encoder_weights"].shape == (4, 168, 14)


if __name__ == "__main__":
    test_metrics_perfect()
    test_invert_roundtrip()
    test_model_forward_shapes()
    print("all sanity tests passed")

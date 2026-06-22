"""Convolutional forecasters: Temporal CNN (TCN) and CNN-LSTM hybrid.

Both encode the lookback into a context vector, then a shared covariate-conditioned
MLP decoder produces the 24 h forecast from known-future covariates + static.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm

from src.models.layers import CovariateDecoder


class _Chomp(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous() if self.chomp else x


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, channels, kernel, dropout):
        super().__init__()
        layers, c_in = [], n_enc
        for i, c_out in enumerate(channels):
            layers.append(_TCNBlock(c_in, c_out, kernel, dilation=2 ** i, dropout=dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.decoder = CovariateDecoder(channels[-1], n_dec, n_static,
                                        hidden=channels[-1], dropout=dropout)

    def forward(self, x_enc, x_dec, x_stat):
        h = self.tcn(x_enc.transpose(1, 2))      # (B, C, L)
        ctx = h[:, :, -1]                         # last timestep representation
        return self.decoder(ctx, x_dec, x_stat)


class CNNLSTM(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, cnn_channels, kernel, hidden,
                 layers, dropout):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_enc, cnn_channels, kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel, padding=pad), nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.decoder = CovariateDecoder(hidden, n_dec, n_static, hidden=hidden,
                                        dropout=dropout)

    def forward(self, x_enc, x_dec, x_stat):
        h = self.conv(x_enc.transpose(1, 2)).transpose(1, 2)   # (B, L, C)
        out, _ = self.lstm(h)
        ctx = out[:, -1, :]
        return self.decoder(ctx, x_dec, x_stat)


def build_tcn(cfg, meta):
    m = cfg["models"]["tcn"]
    return TCN(meta["n_enc"], meta["n_dec"], meta["n_static"],
               m["channels"], m["kernel"], m["dropout"])


def build_cnn_lstm(cfg, meta):
    m = cfg["models"]["cnn_lstm"]
    return CNNLSTM(meta["n_enc"], meta["n_dec"], meta["n_static"],
                   m["cnn_channels"], m["kernel"], m["hidden"], m["layers"], m["dropout"])

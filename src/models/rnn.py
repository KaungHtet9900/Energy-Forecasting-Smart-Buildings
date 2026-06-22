"""Recurrent seq2seq forecasters: LSTM and GRU encoder-decoder."""
from __future__ import annotations

import torch
from torch import nn


class RNNSeq2Seq(nn.Module):
    """Encoder RNN over the lookback -> decoder RNN over known future covariates.

    Static covariates are projected and concatenated to every decoder input step.
    Non-autoregressive (decoder consumes covariates, not past predictions) to avoid
    error accumulation over the 24 h horizon.
    """

    def __init__(self, cell: str, n_enc: int, n_dec: int, n_static: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cell = cell
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.static_proj = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU())
        self.encoder = rnn(n_enc, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.decoder = rnn(n_dec + hidden, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x_enc, x_dec, x_stat):
        _, state = self.encoder(x_enc)
        s = self.static_proj(x_stat).unsqueeze(1).expand(-1, x_dec.size(1), -1)
        dec_in = torch.cat([x_dec, s], dim=-1)
        out, _ = self.decoder(dec_in, state)
        return self.head(out).squeeze(-1)


def build_lstm(cfg, meta):
    m = cfg["models"]["lstm"]
    return RNNSeq2Seq("lstm", meta["n_enc"], meta["n_dec"], meta["n_static"],
                      m["hidden"], m["layers"], m["dropout"])


def build_gru(cfg, meta):
    m = cfg["models"]["gru"]
    return RNNSeq2Seq("gru", meta["n_enc"], meta["n_dec"], meta["n_static"],
                      m["hidden"], m["layers"], m["dropout"])

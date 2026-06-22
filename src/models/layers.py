"""Reusable neural building blocks."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):  # x: (B, L, d)
        return x + self.pe[:, : x.size(1)]


class CovariateDecoder(nn.Module):
    """Context-conditioned MLP head shared by TCN / CNN-LSTM.

    Maps (context vector, per-step future covariates, static) -> one value per step.
    """

    def __init__(self, ctx_dim: int, n_dec: int, n_stat: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + n_dec + n_stat, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, ctx, x_dec, x_stat):
        h = x_dec.size(1)
        ctx_e = ctx.unsqueeze(1).expand(-1, h, -1)
        stat_e = x_stat.unsqueeze(1).expand(-1, h, -1)
        z = torch.cat([ctx_e, x_dec, stat_e], dim=-1)
        return self.net(z).squeeze(-1)


# --------------------------------------------------------------------------
# Temporal Fusion Transformer building blocks
# --------------------------------------------------------------------------
class GLU(nn.Module):
    """Gated Linear Unit."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out * 2)
        self.d_out = d_out

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network (Lim et al., 2021)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int = None,
                 dropout: float = 0.1, context_dim: int = None):
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(context_dim, d_hidden, bias=False) if context_dim else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else None

    def forward(self, x, context=None):
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            if context.dim() == h.dim() - 1:
                context = context.unsqueeze(1).expand(*h.shape[:-1], -1)
            h = h + self.ctx(context)
        h = torch.relu(h)
        h = self.drop(self.fc2(h))
        h = self.glu(h)
        res = x if self.skip is None else self.skip(x)
        return self.norm(h + res)


class VariableSelectionNetwork(nn.Module):
    """Variable-selection network with softmax selection weights (interpretable).

    Vectorized: a per-variable scalar->d_model embedding (one weight per variable)
    followed by a *shared* GRN applied across the variable axis, replacing the
    Python loop over per-variable GRNs (much faster on CPU). The selection weights
    are unchanged, so interpretability is preserved.
    """

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1,
                 context_dim: int = None):
        super().__init__()
        self.n_vars = n_vars
        # per-variable scalar embedding: x[...,v] * W[v,:] + b[v,:]
        self.embed_w = nn.Parameter(torch.randn(n_vars, d_model) * 0.1)
        self.embed_b = nn.Parameter(torch.zeros(n_vars, d_model))
        # shared GRN applied to each variable embedding (broadcasts over var axis)
        self.var_grn = GRN(d_model, d_model, d_model, dropout)
        # selection weights from the flattened inputs (+ optional static context)
        self.flat_grn = GRN(n_vars, d_model, n_vars, dropout, context_dim=context_dim)

    def forward(self, x, context=None):
        # x: (..., n_vars). Returns (..., d_model) and weights (..., n_vars)
        weights = torch.softmax(self.flat_grn(x, context), dim=-1)        # (..., n_vars)
        emb = x.unsqueeze(-1) * self.embed_w + self.embed_b              # (..., n_vars, d)
        emb = self.var_grn(emb)                                          # shared GRN
        out = (weights.unsqueeze(-1) * emb).sum(dim=-2)                  # (..., d_model)
        return out, weights

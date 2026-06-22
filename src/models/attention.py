"""Attention-based forecasters (the interpretable centerpiece).

All three expose interpretability artifacts through ``return_attn=True``:
  * Transformer   -> decoder->encoder cross-attention map  (B, H, L)
  * AttnLSTM      -> Luong attention map                   (B, H, L)
  * TFT           -> temporal attention (B, H, L+H) and variable-selection weights
                     for static / encoder / decoder inputs.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.layers import PositionalEncoding, GRN, GLU, VariableSelectionNetwork


# ==========================================================================
# 1. Transformer (encoder + interpretable cross-attention decoder)
# ==========================================================================
class TransformerForecaster(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, d_model=128, nhead=4, layers=3,
                 ff=256, dropout=0.1):
        super().__init__()
        self.enc_proj = nn.Linear(n_enc, d_model)
        self.dec_proj = nn.Linear(n_dec, d_model)
        self.static_proj = nn.Linear(n_static, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_enc, x_dec, x_stat, return_attn=False):
        mem = self.encoder(self.pos(self.enc_proj(x_enc)))             # (B,L,d)
        q = self.pos(self.dec_proj(x_dec) + self.static_proj(x_stat).unsqueeze(1))
        attn_out, attn_w = self.cross_attn(q, mem, mem, need_weights=True,
                                           average_attn_weights=True)   # (B,H,L)
        h = self.norm1(q + attn_out)
        h = self.norm2(h + self.ffn(h))
        y = self.head(h).squeeze(-1)
        if return_attn:
            return y, {"cross_attn": attn_w}
        return y


# ==========================================================================
# 2. Attention-augmented LSTM (Luong global attention)
# ==========================================================================
class AttnLSTM(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=128, layers=1, attn_dim=64,
                 dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.static_proj = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU())
        self.encoder = nn.LSTM(n_enc, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.decoder = nn.LSTM(n_dec + hidden, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(hidden, hidden, bias=False)             # Luong "general"
        self.combine = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(),
                                     nn.Dropout(dropout))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_enc, x_dec, x_stat, return_attn=False):
        enc_out, state = self.encoder(x_enc)                          # (B,L,hid)
        s = self.static_proj(x_stat)                                  # (B,hid)
        H = x_dec.size(1)
        s_seq = s.unsqueeze(1).expand(-1, H, -1)
        dec_in = torch.cat([x_dec, s_seq], dim=-1)
        dec_out, _ = self.decoder(dec_in, state)                      # (B,H,hid)

        scores = torch.bmm(self.attn(dec_out), enc_out.transpose(1, 2))  # (B,H,L)
        attn_w = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_w, enc_out)                          # (B,H,hid)
        out = self.combine(torch.cat([dec_out, context], dim=-1))
        y = self.head(out).squeeze(-1)
        if return_attn:
            return y, {"attn": attn_w}
        return y


# ==========================================================================
# 3. Temporal Fusion Transformer (compact, faithful)
# ==========================================================================
class TFT(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, d_model=64, nhead=4, lstm_layers=1,
                 dropout=0.1, hidden_continuous=32):
        super().__init__()
        self.d = d_model
        self.n_enc, self.n_dec, self.n_static = n_enc, n_dec, n_static

        # static variable selection + context vectors
        self.static_vsn = VariableSelectionNetwork(n_static, d_model, dropout)
        self.c_select = GRN(d_model, d_model, d_model, dropout)
        self.c_enrich = GRN(d_model, d_model, d_model, dropout)
        self.c_h = GRN(d_model, d_model, d_model, dropout)
        self.c_c = GRN(d_model, d_model, d_model, dropout)

        # temporal variable selection (context-conditioned on c_select)
        self.enc_vsn = VariableSelectionNetwork(n_enc, d_model, dropout, context_dim=d_model)
        self.dec_vsn = VariableSelectionNetwork(n_dec, d_model, dropout, context_dim=d_model)

        # locality-enhancement seq2seq
        self.enc_lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)
        self.dec_lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)
        self.glu1 = GLU(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # static enrichment
        self.enrich = GRN(d_model, d_model, d_model, dropout, context_dim=d_model)

        # interpretable multi-head attention
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.glu2 = GLU(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # position-wise feed-forward
        self.ffn = GRN(d_model, d_model, d_model, dropout)
        self.glu3 = GLU(d_model, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_enc, x_dec, x_stat, return_attn=False):
        B, L, _ = x_enc.shape
        H = x_dec.size(1)

        # static contexts
        static_emb, static_w = self.static_vsn(x_stat)               # (B,d), (B,n_static)
        c_sel = self.c_select(static_emb)
        c_enr = self.c_enrich(static_emb)
        h0 = self.c_h(static_emb).unsqueeze(0).contiguous()
        c0 = self.c_c(static_emb).unsqueeze(0).contiguous()

        # temporal variable selection
        enc_emb, enc_w = self.enc_vsn(x_enc, c_sel)                   # (B,L,d), (B,L,n_enc)
        dec_emb, dec_w = self.dec_vsn(x_dec, c_sel)                   # (B,H,d), (B,H,n_dec)

        # seq2seq with gated skip
        enc_lstm, state = self.enc_lstm(enc_emb, (h0, c0))
        dec_lstm, _ = self.dec_lstm(dec_emb, state)
        lstm_out = torch.cat([enc_lstm, dec_lstm], dim=1)
        emb_cat = torch.cat([enc_emb, dec_emb], dim=1)
        temporal = self.norm1(self.glu1(lstm_out) + emb_cat)         # (B,L+H,d)

        # static enrichment
        enriched = self.enrich(temporal, c_enr)                      # (B,L+H,d)

        # interpretable attention (decoder queries attend over masked full sequence)
        q = enriched[:, L:, :]                                       # (B,H,d)
        idx = torch.arange(L + H, device=x_enc.device)
        # query i (global pos L+i) may attend to keys j <= L+i
        mask = idx[None, :] > (L + torch.arange(H, device=x_enc.device))[:, None]  # (H,L+H)
        attn_out, attn_w = self.attn(q, enriched, enriched, attn_mask=mask,
                                     need_weights=True, average_attn_weights=True)
        x = self.norm2(self.glu2(attn_out) + q)

        # feed-forward + final gated skip to decoder lstm features
        x = self.ffn(x)
        x = self.norm3(self.glu3(x) + dec_lstm)
        y = self.head(x).squeeze(-1)

        if return_attn:
            return y, {"temporal_attn": attn_w, "static_weights": static_w,
                       "encoder_weights": enc_w, "decoder_weights": dec_w}
        return y


# --------------------------------------------------------------------------
def build_transformer(cfg, meta):
    m = cfg["models"]["transformer"]
    return TransformerForecaster(meta["n_enc"], meta["n_dec"], meta["n_static"],
                                 m["d_model"], m["nhead"], m["layers"], m["ff"],
                                 m["dropout"])


def build_attn_lstm(cfg, meta):
    m = cfg["models"]["attn_lstm"]
    return AttnLSTM(meta["n_enc"], meta["n_dec"], meta["n_static"],
                    m["hidden"], m["layers"], m["attn_dim"], m["dropout"])


def build_tft(cfg, meta):
    m = cfg["models"]["tft"]
    return TFT(meta["n_enc"], meta["n_dec"], meta["n_static"], m["d_model"],
               m["nhead"], m["lstm_layers"], m["dropout"], m["hidden_continuous"])

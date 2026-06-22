"""Shared infrastructure for the deep-learning forecasters.

- WindowDataset / loaders over the .npz tensors
- a generic CPU-friendly training loop with early stopping
- scaled <-> kWh inverse transform helpers
- prediction + checkpoint utilities
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils import get_logger, get_device

log = get_logger("models")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_split(cfg, split: str) -> dict:
    d = np.load(cfg["paths"]["processed"] / f"{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_meta(cfg) -> dict:
    with open(cfg["paths"]["processed"] / "feature_names.json", encoding="utf-8") as f:
        return json.load(f)


class WindowDataset(Dataset):
    def __init__(self, arrays: dict):
        self.x_enc = torch.from_numpy(arrays["X_enc"])
        self.x_dec = torch.from_numpy(arrays["X_dec"])
        self.x_stat = torch.from_numpy(arrays["X_stat"])
        self.y = torch.from_numpy(arrays["Y"])

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.x_enc[i], self.x_dec[i], self.x_stat[i], self.y[i]


def make_loader(arrays: dict, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(WindowDataset(arrays), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


# --------------------------------------------------------------------------
# Inverse transform / metrics helpers
# --------------------------------------------------------------------------
def invert(pred_scaled: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """scaled-log space -> kWh."""
    y_log = pred_scaled * y_std[:, None] + y_mean[:, None]
    return np.expm1(y_log)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, cfg, train_arr, val_arr, name: str):
    device = get_device()
    model = model.to(device)
    tcfg = cfg["train"]
    torch.manual_seed(cfg["seed"])

    train_loader = make_loader(train_arr, tcfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_arr, tcfg["batch_size"], shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"],
                           weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=3)
    loss_fn = nn.L1Loss() if tcfg["loss"] == "mae" else nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience, bad = tcfg["patience"], 0
    history = {"train": [], "val": []}
    log.info("[%s] params=%s | train=%d val=%d | device=%s",
             name, f"{count_params(model):,}", len(train_arr["Y"]), len(val_arr["Y"]), device)

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for xe, xd, xs, y in train_loader:
            xe, xd, xs, y = xe.to(device), xd.to(device), xs.to(device), y.to(device)
            opt.zero_grad()
            pred = model(xe, xd, xs)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xe, xd, xs, y in val_loader:
                xe, xd, xs, y = xe.to(device), xd.to(device), xs.to(device), y.to(device)
                va_loss += loss_fn(model(xe, xd, xs), y).item() * len(y)
        va_loss /= len(val_loader.dataset)
        sched.step(va_loss)
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val - 1e-5:
            best_val, best_state, bad = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        log.info("[%s] epoch %2d | train %.4f | val %.4f | %.1fs%s",
                 name, epoch, tr_loss, va_loss, time.time() - t0,
                 "  *best" if bad == 0 else f"  (bad {bad})")
        if bad >= patience:
            log.info("[%s] early stop at epoch %d (best val %.4f)", name, epoch, best_val)
            break

    model.load_state_dict(best_state)
    return model, history, best_val


@torch.no_grad()
def predict_scaled(model, arrays, batch_size=256) -> np.ndarray:
    device = get_device()
    model = model.to(device).eval()
    loader = make_loader(arrays, batch_size, shuffle=False)
    preds = []
    for xe, xd, xs, _ in loader:
        xe, xd, xs = xe.to(device), xd.to(device), xs.to(device)
        preds.append(model(xe, xd, xs).cpu().numpy())
    return np.concatenate(preds, axis=0)


def save_checkpoint(model, cfg, name: str):
    path = cfg["paths"]["models"] / f"{name}.pt"
    torch.save(model.state_dict(), path)
    return path

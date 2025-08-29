#!/usr/bin/env python3
"""
train_batch_binary.py — revision 2 (July 2025)
===========================================
Trains GRU‑based agents on the **binary** helicopter task produced by
``TaskConfig_Generator_binary.py``.

Key targets
-----------
* **truth models** – labels:
    * report → ``trueReport``   (ground truth latent state, −1/+1)
    * hazard → ``truePredict``  (ground truth predict,    −1/+1)
* **norm  models** – labels:
    * report → ``rep_norm``     (Bayesian report,          −1/+1)
    * hazard → ``resp_pred``    (Bayesian predict,         −1/+1)

All four columns are converted from −1/+1 to 0/1 inside the dataset so that
``BCEWithLogitsLoss`` receives *binary* targets.  Baseline probes remain
anchored to the ground‑truth columns (``trueReport`` & ``truePredict``).

Outputs are written under::

    models/<truth|norm>/seed_<n>/
        checkpoint_best.pt   final.pt
        loss_history.json    baseline_loss_history.json
        hp.json
"""
from __future__ import annotations
import ast
import glob
import json
import os
import random
import time
from typing import List, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rnn_models import GRUModel

# ───────────────────────────── configuration ──────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VAR_TRAIN  = os.path.join(BASE_DIR, "variants", "train")
VAR_TEST   = os.path.join(BASE_DIR, "variants", "test")


def get_default_hp() -> dict:
    return {
        "n_input": 1,
        "n_rnn": 128,
        "batch_size": 25,
        "learning_rate": 3e-4,
        "target_loss": 2e-3,
    }

# ───────────────────────────── dataset helper ─────────────────────────────
class HelicopterDataset(Dataset):
    """Convert one CSV of trials into (evidence, report, hazard) tensors."""

    def __init__(self, df: pd.DataFrame, rep_col: str, haz_col: str):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            self.x.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            # map −1/+1 → 0/1 for BCEWithLogitsLoss
            self.y_rep.append(torch.tensor([(row[rep_col] + 1) * 0.5], dtype=torch.float32))
            self.y_haz.append(torch.tensor([(row[haz_col] + 1) * 0.5], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ───────────────────────────── CSV utilities ──────────────────────────────

def _list_train_variants() -> List[str]:
    paths = sorted(glob.glob(os.path.join(VAR_TRAIN, "train_*.csv")))
    if not paths:
        raise FileNotFoundError("No training CSVs found under variants/train/")
    return paths


def _pick_test_csv() -> pd.DataFrame:
    paths = glob.glob(os.path.join(VAR_TEST, "test_*.csv"))
    if not paths:
        raise FileNotFoundError("No test CSVs found under variants/test/")
    return pd.read_csv(random.choice(paths))

# ───────────────────────────── training loop ──────────────────────────────

def train_model(model_cls: Type[nn.Module], type_key: str, seed: int, *, use_norm: bool) -> None:
    """Train a single model (*truth* vs *norm*) for a given seed."""
    variants = _list_train_variants()
    n_epochs = len(variants)
    hp = get_default_hp(); hp["max_epochs"] = n_epochs

    # reproducibility ---------------------------------------------------
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    order = np.random.default_rng(seed).permutation(n_epochs)

    # column mapping ----------------------------------------------------
    if use_norm:
        rep_col, haz_col = "rep_norm", "resp_pred"    # Bayesian labels
    else:
        rep_col, haz_col = "trueReport", "truePredict" # ground‑truth labels
    rep_bl, haz_bl = "trueReport", "truePredict"       # baseline always GT

    # folders -----------------------------------------------------------
    model_dir = os.path.join(BASE_DIR, "models", type_key, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    baseline_df = _pick_test_csv()

    # model & optimisation ---------------------------------------------
    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    best = float("inf"); loss_hist = []; base_hist = []
    t0 = time.time()

    for epoch, idx in enumerate(order):
        df = pd.read_csv(variants[idx])
        dl = DataLoader(
            HelicopterDataset(df, rep_col, haz_col),
            batch_size=hp["batch_size"], shuffle=True, drop_last=True)

        model.train(); running = 0.0
        for x, y_r, y_h in dl:
            x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
            opt.zero_grad()
            out_r, out_h = model(x)
            loss = loss_fn(out_r[:, -1], y_r) + 0.5*loss_fn(out_h, y_h)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); running += loss.item()
        epoch_loss = running / len(dl); loss_hist.append(epoch_loss)

        # baseline probe every 5 epochs --------------------------------
        if epoch % 5 == 0:
            model.eval(); b_tot = 0.0
            with torch.no_grad():
                bl_dl = DataLoader(
                    HelicopterDataset(baseline_df, rep_bl, haz_bl),
                    batch_size=hp["batch_size"])
                for x, y_r, y_h in bl_dl:
                    x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
                    o_r, o_h = model(x)
                    b_tot += loss_fn(o_r[:, -1], y_r).item() + 0.5*loss_fn(o_h, y_h).item()
                base_hist.append(b_tot / len(bl_dl))

        # checkpoints & early stopping ---------------------------------
        if epoch_loss < best:
            best = epoch_loss; torch.save(model.state_dict(), ckpt_best)
        if best < hp["target_loss"]:
            print(f"{type_key}|seed{seed} early‑stop @ ep {epoch} (best {best:.4e})"); break
        if epoch % 50 == 0 or epoch == n_epochs-1:
            print(f"{type_key}|seed{seed} ep {epoch:03}/{n_epochs} loss {epoch_loss:.4e}")

    # save artefacts ----------------------------------------------------
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(base_hist, open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp, open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    print(f"{type_key}|seed{seed} finished in {time.time() - t0:.1f}s | best {best:.4e}")

# ───────────────────────────── entry‑point ───────────────────────────────

def main() -> None:
    seeds = range(1)  # adjust as needed
    SPECS: List[Tuple[Type[nn.Module], bool, str]] = [
        (GRUModel, False, "truth"),
        (GRUModel, True,  "norm"),
    ]
    for seed in seeds:
        for cls, use_norm, key in SPECS:
            train_model(cls, key, seed, use_norm=use_norm)
    print("All trainings complete.")

if __name__ == "__main__":
    main()

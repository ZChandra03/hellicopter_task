#!/usr/bin/env python3
"""
train_batch_binary_onehot.py — Beta-sweep edition (binary hazard, one-hot reward loss)
======================================================================================
Compares how different symmetric Beta priors over hazard rates affect GRU training
on the helicopter task.

Reward-style, one-hot correctness objective:
- Full reward (1.0) if BOTH heads choose correctly
- Half reward (0.5) if exactly ONE head is correct
- Zero reward (0.0) if BOTH are wrong

We optimize expected reward under the model's probabilities:
  p_correct = p(y=1) if label=1 else p(y=0)
  head_reward = p_correct
  total reward R = 0.5 * (report_head_reward + hazard_head_reward)
  loss = mean(1 - R)

This preserves the two-head architecture and uses the final timestep for report.
"""

from __future__ import annotations
import ast
import glob
import os
import random
import time
import json
from typing import List, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rnn_models import GRUModel

# ───────────────────────────── configuration ──────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(BASE_DIR, "variants")

BETA_KEYS = [
    "beta_0p1",
    "beta_0p5",
    "beta_1p0",
    "beta_2p0",
    "beta_10p0",
]

def get_default_hp() -> dict:
    return {
        "n_input": 1,
        "n_rnn": 128,
        "batch_size": 25,
        "learning_rate": 3e-4,
        "target_loss": 1e-2,   # reward loss ∈ [0,1]; adjust target accordingly
    }

# ───────────────────────────── dataset helper ─────────────────────────────
class HelicopterDataset(Dataset):
    """Convert one CSV of trials into tensors consumable by PyTorch.

    Targets:
      - report → trueReport  (−1/+1 → 0/1)
      - hazard → truePredict (−1/+1 → 0/1)  **binary hazard**
    """

    def __init__(self, df: pd.DataFrame):
        xs, yr, yh = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))

            # X: (T, 1)
            x = torch.tensor(evid, dtype=torch.float32).unsqueeze(-1)
            xs.append(x)

            # map −1/+1 → 0/1 and store as (1,) float
            yr.append(torch.tensor([(row["trueReport"] + 1) * 0.5], dtype=torch.float32))
            yh.append(torch.tensor([(row["truePredict"] + 1) * 0.5], dtype=torch.float32))

        self.x, self.y_rep, self.y_haz = xs, yr, yh

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y_rep[i], self.y_haz[i]

# ───────────────────────────── CSV utilities ─────────────────────────────
def _list_train_variants(beta_key: str) -> List[str]:
    """All training CSVs for one Beta prior."""
    pat = os.path.join(VARIANTS_DIR, beta_key, "trainConfig_*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No files match {pat}")
    return paths

def _load_baseline_df() -> pd.DataFrame:
    """Always probe on Beta 1.0 testConfig_0."""
    path = os.path.join(VARIANTS_DIR, "beta_1p0", "testConfig_0.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline CSV not found: {path}")
    return pd.read_csv(path)

# ───────────────────────────── reward-style loss helpers ──────────────────
def _ensure_col(v: torch.Tensor) -> torch.Tensor:
    """Make sure tensor is (B,1) float."""
    v = v.float()
    if v.ndim == 1:
        v = v.unsqueeze(1)
    return v

def _p_correct_from_logits(logits: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    """Probability of being correct for a binary logit vs label in {0,1}.

    logits: (B,1) or (B,) raw logits
    y01   : (B,1) or (B,) labels in {0,1}
    returns: (B,1) p_correct (differentiable w.r.t. logits)
    """
    logits = _ensure_col(logits)
    y01    = _ensure_col(y01)

    p1 = torch.sigmoid(logits)  # P(class=1)
    p0 = 1.0 - p1               # P(class=0)
    p_corr = torch.where(y01 > 0.5, p1, p0)
    return p_corr  # (B,1)

def reward_loss(o_rep_last: torch.Tensor, o_haz: torch.Tensor,
                y_rep: torch.Tensor, y_haz: torch.Tensor) -> torch.Tensor:
    """Compute mean(1 - reward) with reward = 0.5*(p_corr_rep + p_corr_haz)."""
    p_corr_rep = _p_correct_from_logits(o_rep_last, y_rep)  # (B,1)
    p_corr_haz = _p_correct_from_logits(o_haz,       y_haz) # (B,1)
    reward = 0.5 * (p_corr_rep + p_corr_haz)               # (B,1) in [0,1]
    loss = (1.0 - reward).mean()
    return loss

# ───────────────────────────── training loop ──────────────────────────────
def train_beta(
    model_cls: Type[nn.Module],
    beta_key: str,
    seed: int,
) -> None:
    """Train one model on one Beta prior folder."""
    csvs = _list_train_variants(beta_key)
    n_epochs = len(csvs)  # one epoch per training CSV

    hp = get_default_hp()
    hp["max_epochs"] = n_epochs

    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(n_epochs)

    # directories ----------------------------------------------------------
    model_dir = os.path.join(BASE_DIR, "models", beta_key, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    # fixed baseline CSV ---------------------------------------------------
    baseline_df = _load_baseline_df()

    # model / optimiser ----------------------------------------------------
    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])

    best, loss_hist, bl_hist = float("inf"), [], []
    t0 = time.time()

    for epoch, idx in enumerate(perm):
        df = pd.read_csv(csvs[idx])

        # ------------------- training ----------------------------------
        dl = DataLoader(
            HelicopterDataset(df),
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=True
        )
        model.train()
        running = 0.0

        for x, y_r, y_h in dl:
            x   = x.to(DEVICE)      # (B, T, 1)
            y_r = y_r.to(DEVICE)    # (B, 1)
            y_h = y_h.to(DEVICE)    # (B, 1)

            opt.zero_grad()
            o_r, o_h = model(x)     # o_r: (B, T, 1) logits; o_h: (B, 1) logits

            # final step for report head
            o_r_last = o_r[:, -1]   # (B, 1) or (B,)
            o_r_last = _ensure_col(o_r_last)

            # reward-style loss
            loss = reward_loss(o_r_last, o_h, y_r, y_h)

            # safety check during development
            assert loss.requires_grad, "Loss is not connected to the graph"

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        epoch_loss = running / len(dl)
        loss_hist.append(epoch_loss)

        # ------------------- periodic checkpoints ----------------------
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f"checkpoint_ep{epoch+1:03}.pt"))

        # ------------------- baseline probe (every 5 epochs) -----------
        if epoch % 5 == 0:
            model.eval()
            btot = 0.0
            with torch.no_grad():
                bl_dl = DataLoader(HelicopterDataset(baseline_df),
                                   batch_size=hp["batch_size"])
                for x, y_r, y_h in bl_dl:
                    x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
                    o_r, o_h = model(x)
                    btot += reward_loss(_ensure_col(o_r[:, -1]), o_h, y_r, y_h).item()
            bl_hist.append(btot / len(bl_dl))

        # ------------------- logging / best model ----------------------
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"{beta_key}|seed{seed}  ep {epoch:03}/{n_epochs}  loss {epoch_loss:.4e}")

        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), ckpt_best)

        if best < hp["target_loss"]:
            print(f"{beta_key}|seed{seed} early-stop @ ep {epoch} (best {best:.4e})")
            break

    # ------------------- artefacts ---------------------------------------
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(bl_hist,  open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp,        open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    print(f"{beta_key}|seed{seed} finished in {time.time() - t0:.1f}s | best {best:.4e}")

# ───────────────────────────── entry-point ────────────────────────────────
def main() -> None:
    seeds = range(10)  # adjust as needed
    for seed in seeds:
        for beta_key in BETA_KEYS:
            train_beta(GRUModel, beta_key, seed)
    print("All trainings complete.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_batch_binary_onehot.py — uniform-groups edition (binary hazard, one-hot reward loss)
=========================================================================================
Compares how different *uniform* hazard groups affect GRU training on the helicopter task.

Groups expected under variants/:
- hz_0_0p4
- hz_0p6_1
- hz_edges_0_0p2_0p8_1
- hz_0p3_0p7
- hz_flat_0_1

Reward-style, one-hot correctness objective (unchanged):
  R = 0.5 * (report_head_reward + hazard_head_reward), rewards in [0,1]
Training uses REINFORCE with hard Bernoulli samples; we also log the proxy reward loss.
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

# New hazard groups (match the generator’s folder slugs)
GROUP_KEYS = [
    "hz_0_0p4",
    "hz_0p6_1",
    "hz_edges_0_0p2_0p8_1",
    "hz_0p3_0p7",
    "hz_flat_0_1",
]

def get_default_hp() -> dict:
    return {
        "n_input": 1,
        "n_rnn": 128,
        "batch_size": 25,
        "learning_rate": 3e-4,
        "target_loss": 1e-2,         # reward loss ∈ [0,1]
        # old-style schedule knobs
        "epochs_per_csv": 100,         # several passes per CSV
        "max_csv": 20,              # take first max_csv from each folder
    }

# ───────────────────────────── dataset helper ─────────────────────────────
class HelicopterDataset(Dataset):
    """Convert one CSV of trials into tensors for PyTorch.

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
def _list_train_variants(group_key: str) -> List[str]:
    """All training CSVs for one hazard group."""
    pat = os.path.join(VARIANTS_DIR, group_key, "trainConfig_*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No files match {pat}")
    return paths

def _load_baseline_df() -> pd.DataFrame:
    """Probe on the flat group testConfig_0 for consistency across runs."""
    path = os.path.join(VARIANTS_DIR, "hz_flat_0_1", "testConfig_0.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline CSV not found: {path}")
    return pd.read_csv(path)

# ───────────────────────────── reward-style loss helpers ──────────────────
def _ensure_col(v: torch.Tensor) -> torch.Tensor:
    v = v.float()
    if v.ndim == 1:
        v = v.unsqueeze(1)
    return v

def _p_correct_from_logits(logits: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    logits = _ensure_col(logits)
    y01    = _ensure_col(y01)
    p1 = torch.sigmoid(logits)  # P(class=1)
    p0 = 1.0 - p1               # P(class=0)
    return torch.where(y01 > 0.5, p1, p0)

def reward_loss(o_rep_last: torch.Tensor, o_haz: torch.Tensor,
                y_rep: torch.Tensor, y_haz: torch.Tensor) -> torch.Tensor:
    p_corr_rep = _p_correct_from_logits(o_rep_last, y_rep)
    p_corr_haz = _p_correct_from_logits(o_haz,       y_haz)
    reward = 0.5 * (p_corr_rep + p_corr_haz)
    return (1.0 - reward).mean()

# --------------------- non-differentiable reward via REINFORCE ----------
class _PGState:
    def __init__(self, baseline: float = 0.0):
        self.baseline = baseline  # kept for API compatibility

def reinforce_loss(o_rep_last: torch.Tensor, o_haz: torch.Tensor,
                   y_rep: torch.Tensor, y_haz: torch.Tensor,
                   state: _PGState,
                   entropy_coef: float = 0.0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PURE Option A: discrete reward, no baseline/entropy.

      R = 0.5 * ( 1[a_rep == y_rep] + 1[a_haz == y_haz] ), so R ∈ {0, 0.5, 1}
    """
    z_r = _ensure_col(o_rep_last).squeeze(1)
    z_h = _ensure_col(o_haz).squeeze(1)
    y_r = _ensure_col(y_rep).squeeze(1)
    y_h = _ensure_col(y_haz).squeeze(1)

    m_r = torch.distributions.Bernoulli(logits=z_r)
    m_h = torch.distributions.Bernoulli(logits=z_h)
    a_r = m_r.sample()
    a_h = m_h.sample()

    corr_r = (a_r == y_r).float()
    corr_h = (a_h == y_h).float()
    R = 0.5 * (corr_r + corr_h)

    logp = m_r.log_prob(a_r) + m_h.log_prob(a_h)
    loss = -(R.detach() * logp).mean()

    return loss, R.mean().detach()

# ───────────────────────────── training loop ──────────────────────────────
def train_group(
    model_cls: Type[nn.Module],
    group_key: str,
    seed: int,
) -> None:
    """Train one model on one hazard-group folder with old-style schedule."""
    csvs_all = _list_train_variants(group_key)
    hp = get_default_hp()

    # take only the first max_csv files
    max_csv = int(hp.get("max_csv", len(csvs_all)))
    csvs = csvs_all[:max_csv]
    if not csvs:
        raise RuntimeError(f"No CSVs selected for {group_key}")

    n_csv = len(csvs)
    epochs_per_csv = int(hp.get("epochs_per_csv", 1))
    n_total_epochs = n_csv * epochs_per_csv
    hp["max_epochs"] = n_total_epochs

    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(n_csv)

    # directories ----------------------------------------------------------
    model_dir = os.path.join(BASE_DIR, "models_OTS", group_key, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    # fixed baseline CSV ---------------------------------------------------
    baseline_df = _load_baseline_df()

    # model / optimiser ----------------------------------------------------
    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    pg_state = _PGState(baseline=0.0)

    best, loss_hist, bl_hist = float("inf"), [], []
    t0 = time.time()
    global_epoch = 0
    early_stop = False

    for idx in perm:
        df = pd.read_csv(csvs[idx])
        dl = DataLoader(
            HelicopterDataset(df),
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=True
        )

        # several passes over THIS CSV
        for _local_ep in range(epochs_per_csv):
            if early_stop:
                break

            model.train()
            running = 0.0

            for x, y_r, y_h in dl:
                x   = x.to(DEVICE)
                y_r = y_r.to(DEVICE)
                y_h = y_h.to(DEVICE)

                opt.zero_grad()
                o_r, o_h = model(x)     # o_r: (B, T, 1) logits; o_h: (B, 1) logits
                o_r_last = _ensure_col(o_r[:, -1])

                loss, batch_R = reinforce_loss(o_r_last, o_h, y_r, y_h, pg_state)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running += float(1.0 - batch_R)

            epoch_loss = running / len(dl)
            loss_hist.append(epoch_loss)

            # periodic checkpoints
            if (global_epoch + 1) % 50 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, f"checkpoint_ep{global_epoch+1:03}.pt")
                )

            # baseline probe (every 10 epochs)
            if global_epoch % 10 == 0:
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

            # logging / best model
            if global_epoch % 50 == 0 or global_epoch == n_total_epochs - 1:
                print(f"{group_key}|seed{seed}  ep {global_epoch:03}/{n_total_epochs}  loss {epoch_loss:.4e}")

            if epoch_loss < best:
                best = epoch_loss
                torch.save(model.state_dict(), ckpt_best)

            if best < hp["target_loss"]:
                print(f"{group_key}|seed{seed} early-stop @ ep {global_epoch} (best {best:.4e})")
                early_stop = True
                break

            global_epoch += 1

        if early_stop:
            break

    # artefacts ------------------------------------------------------------
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(bl_hist,  open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp,        open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    print(f"{group_key}|seed{seed} finished in {time.time() - t0:.1f}s | best {best:.4e}")

# ───────────────────────────── entry-point ────────────────────────────────
def main() -> None:
    seeds = range(10)  # adjust as needed
    for seed in seeds:
        for group_key in GROUP_KEYS:
            train_group(GRUModel, group_key, seed)
    print("All trainings complete.")

if __name__ == "__main__":
    main()

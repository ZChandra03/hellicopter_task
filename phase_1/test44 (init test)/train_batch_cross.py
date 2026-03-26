#!/usr/bin/env python3
"""
train_batch_comprehensive.py
============================
Trains 80 models across all combinations of:
- 2 noise levels (sigma_1, sigma_2)
- 2 loss functions (REINFORCE, BCE)
- 2 initializations (kaiming_default, orthogonal)
- 10 seeds (0-9)

Total: 2 × 2 × 2 × 10 = 80 models
"""

from __future__ import annotations
import ast
import glob
import os
import random
import time
import json
from typing import List, Type, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rnn_models import GRUModel

# ─────────────────────────── configuration ───────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(BASE_DIR, "variants")

# Experimental factors
SIGMA_LEVELS = ["sigma_1", "sigma_2"]
LOSS_TYPES = ["reinforce", "bce"]
INIT_TYPES = ["kaiming", "orthogonal"]
SEEDS = list(range(10))

def get_default_hp() -> dict:
    return {
        "n_input": 1,
        "n_rnn": 128,
        "batch_size": 25,
        "learning_rate": 3e-4,
        "target_loss": 1e-2,
        "epochs_per_csv": 4,
        "max_csv": 500,
    }

# ─────────────────────────── dataset helper ───────────────────────────
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

# ─────────────────────────── CSV utilities ───────────────────────────
def _list_train_variants(sigma_key: str) -> List[str]:
    """All training CSVs for one sigma level."""
    pat = os.path.join(VARIANTS_DIR, sigma_key, "trainConfig_*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No files match {pat}")
    return paths

def _load_baseline_df(sigma_key: str) -> pd.DataFrame:
    """Load test CSV for the given sigma level."""
    path = os.path.join(VARIANTS_DIR, sigma_key, "testConfig_0.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline CSV not found: {path}")
    return pd.read_csv(path)

# ─────────────────────────── initialization helpers ───────────────────────────
def apply_orthogonal_init(model: nn.Module) -> None:
    """Apply orthogonal initialization to all weights in the model."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim >= 2:
                nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

def apply_kaiming_init(model: nn.Module) -> None:
    """Apply kaiming (default PyTorch) initialization."""
    # PyTorch already uses kaiming by default for most layers
    # This function is here for explicit clarity, but does nothing special
    pass

# ─────────────────────────── loss function helpers ───────────────────────────
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

class _PGState:
    def __init__(self, baseline: float = 0.0):
        self.baseline = baseline

def reinforce_loss(o_rep_last: torch.Tensor, o_haz: torch.Tensor,
                   y_rep: torch.Tensor, y_haz: torch.Tensor,
                   state: _PGState,
                   entropy_coef: float = 0.0
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PURE Option A: discrete reward with no baseline and no entropy bonus.

      R = 0.5 * ( 1[a_rep == y_rep] + 1[a_haz == y_haz] ), so R ∈ {0, 0.5, 1}

    Returns
    -------
    loss : scalar tensor (minimize)
    reward_mean : detached scalar tensor (mean(R))
    """
    # shapes -> (B,)
    z_r = _ensure_col(o_rep_last).squeeze(1)
    z_h = _ensure_col(o_haz).squeeze(1)
    y_r = _ensure_col(y_rep).squeeze(1)
    y_h = _ensure_col(y_haz).squeeze(1)

    # Bernoulli policies and hard samples
    m_r = torch.distributions.Bernoulli(logits=z_r)
    m_h = torch.distributions.Bernoulli(logits=z_h)
    a_r = m_r.sample()  # (B,)
    a_h = m_h.sample()  # (B,)

    # Discrete reward in {0, 0.5, 1}
    corr_r = (a_r == y_r).float()
    corr_h = (a_h == y_h).float()
    R = 0.5 * (corr_r + corr_h)  # (B,)

    # Pure REINFORCE: maximize E[R] => minimize -(R * log p(a))
    logp = m_r.log_prob(a_r) + m_h.log_prob(a_h)  # (B,)
    loss = -(R.detach() * logp).mean()            # no baseline, no entropy

    batch_mean = R.mean().detach()
    return loss, batch_mean

def bce_loss(o_rep_last: torch.Tensor, o_haz: torch.Tensor,
             y_rep: torch.Tensor, y_haz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BCE with logits loss for both heads.
    
    Returns
    -------
    loss : scalar tensor (minimize)
    reward_mean : detached scalar tensor (mean accuracy as pseudo-reward)
    """
    bce_fn = nn.BCEWithLogitsLoss()
    
    o_rep_last = _ensure_col(o_rep_last).squeeze(1)
    o_haz = _ensure_col(o_haz).squeeze(1)
    y_rep = _ensure_col(y_rep).squeeze(1)
    y_haz = _ensure_col(y_haz).squeeze(1)
    
    # Compute BCE loss for each head
    loss_rep = bce_fn(o_rep_last, y_rep)
    loss_haz = bce_fn(o_haz, y_haz)
    
    # Combined loss
    loss = 0.5 * (loss_rep + loss_haz)
    
    # Compute accuracy as pseudo-reward for logging
    with torch.no_grad():
        pred_rep = (torch.sigmoid(o_rep_last) > 0.5).float()
        pred_haz = (torch.sigmoid(o_haz) > 0.5).float()
        acc_rep = (pred_rep == y_rep).float().mean()
        acc_haz = (pred_haz == y_haz).float().mean()
        pseudo_reward = 0.5 * (acc_rep + acc_haz)
    
    return loss, pseudo_reward

# ─────────────────────────── training loop ───────────────────────────
def train_model(
    model_cls: Type[nn.Module],
    sigma_key: str,
    loss_type: str,
    init_type: str,
    seed: int,
) -> None:
    """Train one model with specified configuration."""
    
    # Setup experiment name
    exp_name = f"{sigma_key}_{loss_type}_{init_type}_seed{seed}"
    print(f"\n{'='*80}")
    print(f"Starting: {exp_name}")
    print(f"{'='*80}")
    
    csvs_all = _list_train_variants(sigma_key)
    hp = get_default_hp()

    # Take only the first max_csv files
    max_csv = int(hp.get("max_csv", len(csvs_all)))
    csvs = csvs_all[:max_csv]
    if not csvs:
        raise RuntimeError(f"No CSVs selected for {sigma_key}")

    n_csv = len(csvs)
    epochs_per_csv = int(hp.get("epochs_per_csv", 1))
    n_total_epochs = n_csv * epochs_per_csv
    hp["max_epochs"] = n_total_epochs

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Shuffle CSV order deterministically
    perm = np.random.default_rng(seed).permutation(n_csv)

    # Directories
    model_dir = os.path.join(BASE_DIR, "models", sigma_key, loss_type, init_type, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    # Load baseline CSV
    baseline_df = _load_baseline_df(sigma_key)

    # Initialize model
    model = model_cls(hp).to(DEVICE)
    
    # Apply initialization
    if init_type == "orthogonal":
        apply_orthogonal_init(model)
    elif init_type == "kaiming":
        apply_kaiming_init(model)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    
    # Loss function state
    pg_state = _PGState(baseline=0.0)
    
    # Select loss function
    if loss_type == "reinforce":
        loss_fn = lambda o_r, o_h, y_r, y_h: reinforce_loss(o_r, o_h, y_r, y_h, pg_state)
    elif loss_type == "bce":
        loss_fn = bce_loss
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    best, loss_hist, bl_hist = float("inf"), [], []
    t0 = time.time()
    global_epoch = 0
    early_stop = False

    for idx in perm:
        # Load one CSV and create dataloader
        df = pd.read_csv(csvs[idx])
        dl = DataLoader(
            HelicopterDataset(df),
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=True
        )

        # Run several passes over this CSV
        for _local_ep in range(epochs_per_csv):
            if early_stop:
                break

            model.train()
            running = 0.0

            for x, y_r, y_h in dl:
                x   = x.to(DEVICE)      # (B, T, 1)
                y_r = y_r.to(DEVICE)    # (B, 1)
                y_h = y_h.to(DEVICE)    # (B, 1)

                opt.zero_grad()
                o_r, o_h = model(x)     # o_r: (B, T, 1) logits; o_h: (B, 1) logits

                # Final step for report head
                o_r_last = _ensure_col(o_r[:, -1])

                # Compute loss
                loss, batch_metric = loss_fn(o_r_last, o_h, y_r, y_h)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                # For BCE, batch_metric is accuracy; for REINFORCE, it's reward
                # Convert to loss-like metric (higher is worse)
                if loss_type == "bce":
                    running += loss.item()
                else:  # reinforce
                    running += float(1.0 - batch_metric)

            epoch_loss = running / len(dl)
            loss_hist.append(epoch_loss)

            # Periodic checkpoints
            if (global_epoch + 1) % 50 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, f"checkpoint_ep{global_epoch+1:03}.pt")
                )

            # Baseline probe (every 10 epochs)
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

            # Logging
            if global_epoch % 50 == 0 or global_epoch == n_total_epochs - 1:
                print(f"{exp_name}  ep {global_epoch:03}/{n_total_epochs}  loss {epoch_loss:.4e}")

            if epoch_loss < best:
                best = epoch_loss
                torch.save(model.state_dict(), ckpt_best)

            if best < hp["target_loss"]:
                print(f"{exp_name} early-stop @ ep {global_epoch} (best {best:.4e})")
                early_stop = True
                break

            global_epoch += 1

        if early_stop:
            break

    # Save final artifacts
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(bl_hist,  open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    
    # Save hyperparameters with experiment config
    hp_full = hp.copy()
    hp_full.update({
        "sigma_key": sigma_key,
        "loss_type": loss_type,
        "init_type": init_type,
        "seed": seed,
    })
    json.dump(hp_full, open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    
    print(f"{exp_name} finished in {time.time() - t0:.1f}s | best {best:.4e}")

# ─────────────────────────── entry-point ───────────────────────────
def main() -> None:
    """Train all 80 models: 2 sigmas × 2 losses × 2 inits × 10 seeds."""
    
    total_models = len(SIGMA_LEVELS) * len(LOSS_TYPES) * len(INIT_TYPES) * len(SEEDS)
    print(f"\n{'='*80}")
    print(f"Training {total_models} models total")
    print(f"  Sigma levels: {SIGMA_LEVELS}")
    print(f"  Loss types: {LOSS_TYPES}")
    print(f"  Init types: {INIT_TYPES}")
    print(f"  Seeds: {SEEDS}")
    print(f"{'='*80}\n")
    
    model_count = 0
    for sigma in SIGMA_LEVELS:
        for loss in LOSS_TYPES:
            for init in INIT_TYPES:
                for seed in SEEDS:
                    model_count += 1
                    print(f"\n[{model_count}/{total_models}] Training model...")
                    train_model(GRUModel, sigma, loss, init, seed)
    
    print(f"\n{'='*80}")
    print(f"All {total_models} trainings complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
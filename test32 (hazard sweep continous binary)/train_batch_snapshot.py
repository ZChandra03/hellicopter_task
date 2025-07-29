#!/usr/bin/env python3
"""
train_batch_snapshot.py — Beta‑sweep edition (July 2025)
=======================================================
Compares how different symmetric Beta priors over hazard rates affect
GRU training on the helicopter task.

Directory layout expected
-------------------------
variants/
    beta_0p1/
        trainConfig_0.csv   [≥1 per beta; name pattern *trainConfig_*.csv*]
        testConfig_0.csv
    beta_0p5/
        …
    beta_1p0/              ← *baseline probe* uses testConfig_0.csv here
        …
    beta_2p0/
    beta_10p0/

Outputs per run::

    models/<beta_key>/seed_<n>/
        ├── checkpoint_best.pt
        ├── checkpoint_ep020.pt
        ├── …
        ├── final.pt
        ├── loss_history.json
        ├── baseline_loss_history.json
        └── hp.json
"""

from __future__ import annotations
import glob, os, random, time, json
from typing import List, Type, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rnn_models import GRUModel

# ───────────────────────────── configuration ──────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
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
        "target_loss": 2e-3,
    }

# ───────────────────────────── dataset helper ─────────────────────────────
class HelicopterDataset(Dataset):
    """Convert one CSV of trials into tensors consumable by PyTorch."""

    def __init__(self, df: pd.DataFrame):
        xs, yr, yh = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = eval(str(evid))
            xs.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            yr.append(torch.tensor([(row["trueReport"] + 1) * 0.5], dtype=torch.float32))
            yh.append(torch.tensor([row["trueHazard"]], dtype=torch.float32))
        self.x, self.y_rep, self.y_haz = xs, yr, yh

    def __len__(self):  return len(self.x)
    def __getitem__(self, i):  return self.x[i], self.y_rep[i], self.y_haz[i]

# ───────────────────────────── CSV utilities ──────────────────────────────
def _list_train_variants(beta_key: str) -> List[str]:
    """All training CSVs for one Beta prior."""
    pat = os.path.join(VARIANTS_DIR, beta_key, "trainConfig_*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No files match {pat}")
    return paths

def _load_baseline_df() -> pd.DataFrame:
    """Always probe on Beta 1.0 testConfig_0."""
    path = os.path.join(VARIANTS_DIR, "beta_1p0", "testConfig_0.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline CSV not found: {path}")
    return pd.read_csv(path)

# ───────────────────────────── training loop ──────────────────────────────
def train_beta(
    model_cls: Type[nn.Module],
    beta_key: str,
    seed: int,
) -> None:
    """Train one model on one Beta prior folder."""
    csvs = _list_train_variants(beta_key)
    n_epochs = len(csvs)                 # one epoch per training CSV

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

    # model / optimiser / loss --------------------------------------------
    model   = model_cls(hp).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce     = nn.BCEWithLogitsLoss()

    best, loss_hist, bl_hist = float("inf"), [], []
    t0 = time.time()

    for epoch, idx in enumerate(perm):
        df = pd.read_csv(csvs[idx])

        # ------------------- training ----------------------------------
        dl = DataLoader(HelicopterDataset(df),
                        batch_size=hp["batch_size"],
                        shuffle=True, drop_last=True)
        model.train(); running = 0.0
        for x, y_r, y_h in dl:
            x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
            opt.zero_grad()
            o_r, o_h = model(x)
            loss = bce(o_r[:, -1], y_r) + 0.5 * bce(o_h, y_h)
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

        # ------------------- baseline probe (every 5 epochs) -----------
        if epoch % 5 == 0:
            model.eval(); btot = 0.0
            with torch.no_grad():
                bl_dl = DataLoader(HelicopterDataset(baseline_df),
                                   batch_size=hp["batch_size"])
                for x, y_r, y_h in bl_dl:
                    x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
                    o_r, o_h = model(x)
                    btot += (bce(o_r[:, -1], y_r) + 0.5 * bce(o_h, y_h)).item()
            bl_hist.append(btot / len(bl_dl))

        # ------------------- logging / best model ----------------------
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"{beta_key}|seed{seed}  ep {epoch:03}/{n_epochs}"
                  f"  loss {epoch_loss:.4e}")
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), ckpt_best)
        if best < hp["target_loss"]:
            print(f"{beta_key}|seed{seed} early‑stop @ ep {epoch}"
                  f" (best {best:.4e})"); break

    # ------------------- artefacts ---------------------------------------
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(bl_hist,  open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp,        open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    print(f"{beta_key}|seed{seed} finished in {time.time() - t0:.1f}s | best {best:.4e}")

# ───────────────────────────── entry‑point ────────────────────────────────
def main() -> None:
    seeds = range(1)          # adjust as needed
    for seed in seeds:
        for beta_key in BETA_KEYS:
            train_beta(GRUModel, beta_key, seed)
    print("All trainings complete.")

if __name__ == "__main__":
    main()

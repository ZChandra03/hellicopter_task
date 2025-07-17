#!/usr/bin/env python3
"""
train_batch_finished_27.py — revision​11 (17 Jul 2025)
============================================================

This update *aligns the baseline‑loss logging with* **train_batch_finished_26.py**:

* **Baseline probe** is *always* evaluated against the *ground‑truth* column
  ``truePredict`` – regardless of which family (``truth`` | ``norm``) is
  being trained.
* Baseline losses are stored as a flat list of floats (one value per probe)
  and written to **``baseline_loss_history.json``** (identical filename &
  format to *train 26*).
* ``loss_history.json`` now contains **only** the per‑epoch training losses,
  matching *train 26*.
* A lightweight ``hp.json`` with the main hyper‑parameters is also saved so all
  artefacts match the older pipeline.

Run the script from *any* directory; it will discover CSVs under the
``variants/`` folder that lives next to the script.
"""
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json, time, ast, random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------- #
# Hyper‑parameters & paths
# ---------------------------------------------------------------------------- #
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR        = Path(__file__).resolve().parent
TRAIN_DIR       = BASE_DIR / "variants" / "train"          # train_00.csv …
TEST_CSV        = BASE_DIR / "variants" / "test" / "test_unsorted_00.csv"
SEEDS           = range(3)         # 0–9
BATCH_SIZE      = 25
HIDDEN_SIZE     = 128
LEARNING_RATE   = 3e-4
TARGET_LOSS     = 2e-3
PROBE_EVERY     = 5                 # baseline probe cadence (epochs)

# Filled *after* discovering CSVs so it matches their count
MAX_EPOCHS: int | None = None

# ---------------------------------------------------------------------------- #
# Dataset helpers
# ---------------------------------------------------------------------------- #
class HelicopterDataset(Dataset):
    """Binary hazard dataset: evidence sequence → target (stay=0 / switch=1)."""

    def __init__(self, df: pd.DataFrame, label_col: str):
        xs, ys = [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            xs.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            ys.append((row[label_col] + 1) * 0.5)  # −1/+1 → 0/1
        self.x = xs
        self.y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------------------------------------------------------------------------- #
# Minimal GRU (single logit output per trial)
# ---------------------------------------------------------------------------- #
class SimpleGRU(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (B, T, 1)
        h, _ = self.gru(x)
        return self.out(h[:, -1])  # (B, 1) logit

# ---------------------------------------------------------------------------- #
# Discover CSVs & prepare baseline DataFrame
# ---------------------------------------------------------------------------- #
TRAIN_CSVS: List[Path] = sorted(TRAIN_DIR.glob("train_*.csv"))
if not TRAIN_CSVS:
    raise RuntimeError(f"No training CSVs found under {TRAIN_DIR}. Run Task_Config_Generator_binary.py.")

MAX_EPOCHS = len(TRAIN_CSVS)  # one epoch per CSV

if TEST_CSV.exists():
    BASELINE_DF = pd.read_csv(TEST_CSV)
else:  # fall back to the first training variant
    BASELINE_DF = pd.read_csv(TRAIN_CSVS[0])

# ---------------------------------------------------------------------------- #
# Training routine (one family, one seed)
# ---------------------------------------------------------------------------- #
LOSS_FN = nn.BCEWithLogitsLoss()

def train_family(seed: int, fam_key: str, label_col: str) -> None:
    """Train *one*  GRU  on hazard labels defined by *label_col*."""
    # deterministic shuffling of CSV order per seed
    rng = np.random.default_rng(seed)
    perm = rng.permutation(MAX_EPOCHS)

    model_dir = BASE_DIR / "models" / fam_key / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # model/optimiser
    model = SimpleGRU().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # fixed baseline loader (GROUND‑TRUTH labels **always**)
    bl_loader = DataLoader(HelicopterDataset(BASELINE_DF, "truePredict"),
                           batch_size=BATCH_SIZE)

    best_loss: float = float("inf")
    train_hist: list[float] = []           # per‑epoch training loss
    base_hist:  list[float] = []           # every PROBE_EVERY epochs
    t0 = time.time()

    for epoch, vidx in enumerate(perm):
        # ---- load epoch CSV --------------------------------------------------
        df = pd.read_csv(TRAIN_CSVS[vidx])
        train_loader = DataLoader(HelicopterDataset(df, label_col),
                                  batch_size=BATCH_SIZE, shuffle=True)

        # ---- SGD step --------------------------------------------------------
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss   = LOSS_FN(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
        epoch_loss = running / len(train_loader.dataset)
        train_hist.append(epoch_loss)

        # ---- baseline probe --------------------------------------------------
        if epoch % PROBE_EVERY == 0:
            model.eval(); tot = 0.0
            with torch.no_grad():
                for x, y in bl_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    tot += LOSS_FN(model(x), y).item() * x.size(0)
            bl_loss = tot / len(bl_loader.dataset)
            base_hist.append(bl_loss)
            print(f"{fam_key}|seed{seed} ep{epoch:03d} train {epoch_loss:.4f} base {bl_loss:.4f}")

        # early stop & checkpointing
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_dir / "checkpoint_best.pt")
        if best_loss < TARGET_LOSS:
            print(f"{fam_key}|seed{seed} early‑stop @ ep {epoch}")
            break

    # ---- save artefacts -----------------------------------------------------
    torch.save(model.state_dict(), model_dir / "final.pt")
    json.dump(train_hist, open(model_dir / "loss_history.json", "w"), indent=2)
    json.dump(base_hist,  open(model_dir / "baseline_loss_history.json", "w"), indent=2)
    json.dump({
        "hidden_size"   : HIDDEN_SIZE,
        "batch_size"    : BATCH_SIZE,
        "learning_rate" : LEARNING_RATE,
        "target_loss"   : TARGET_LOSS,
        "probe_every"   : PROBE_EVERY,
        "max_epochs"    : MAX_EPOCHS,
    }, open(model_dir / "hp.json", "w"), indent=2)

    print(f"{fam_key}|seed{seed} done in {time.time()-t0:.1f}s | best {best_loss:.4f}")

# ---------------------------------------------------------------------------- #
# Entry‑point
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    for s in SEEDS:
        train_family(s, "truth", "truePredict")
        train_family(s, "norm",  "resp_pred")

    print("All trainings complete.")

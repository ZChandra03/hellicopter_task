#!/usr/bin/env python3
"""
train_batch_finished.py — revision 7 (July 2025)
================================================
This script trains GRU‑based agents on the helicopter hazard‑inference task.

Key changes in *r7* (compared with r6):
* **Periodic checkpoints:** model weights are now saved every 20 epochs
  as ``checkpoint_epXXX.pt`` (e.g., ``checkpoint_ep020.pt``), in addition to
  ``checkpoint_best.pt`` (lowest training loss) and ``final.pt``.
* Baseline probe logic from r6 is unchanged.

Outputs per run::

    models/<type_key>/seed_<n>/
        ├── checkpoint_best.pt
        ├── checkpoint_ep020.pt
        ├── checkpoint_ep040.pt
        ├── …                               # one every 20 epochs
        ├── final.pt
        ├── loss_history.json
        ├── baseline_loss_history.json
        └── hp.json
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

from rnn_models import GRUModel, LSTMModel, RNNModel

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
    """Convert one CSV of trials into tensors consumable by PyTorch."""

    def __init__(self, df: pd.DataFrame, rep_col: str, haz_col: str):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            self.x.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            self.y_rep.append(torch.tensor([(row[rep_col] + 1) * 0.5],
                                           dtype=torch.float32))
            self.y_haz.append(torch.tensor([row[haz_col]], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]


# ───────────────────────────── CSV utilities ──────────────────────────────
def _list_train_variants(category: str) -> List[str]:
    """Return *sorted* list of training CSVs for a category."""
    pattern = os.path.join(VAR_TRAIN, category, f"train_{category}_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs match {pattern}")
    return paths


def _pick_test_unsorted() -> pd.DataFrame:
    """Load **one** random unsorted‑test CSV as a DataFrame."""
    pattern = os.path.join(VAR_TEST, "unsorted", "test_unsorted_*.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError("[baseline‑probe] No test_unsorted CSVs found")
    return pd.read_csv(random.choice(paths))


# ───────────────────────────── training loop ──────────────────────────────
def train_model(
    model_cls: Type[nn.Module],
    type_key: str,
    seed: int,
    *,
    category: str,
    use_norm: bool,
) -> None:
    """Train a single model (*type_key*, *seed*) on one category of data."""

    variants = _list_train_variants(category)
    n_epochs = len(variants)  # one epoch per CSV

    # hyperparameters ---------------------------------------------------
    hp = get_default_hp()
    hp["max_epochs"] = n_epochs

    # reproducibility ---------------------------------------------------
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(n_epochs)

    # column names ------------------------------------------------------
    rep_col, haz_col = ("rep_norm", "haz_norm") if use_norm else (
        "trueReport", "trueHazard"
    )
    rep_col_bl, haz_col_bl = "trueReport", "trueHazard"  # always ground truth

    # folder layout -----------------------------------------------------
    model_dir = os.path.join(BASE_DIR, "models", type_key, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    # fixed *test* CSV for baseline probe ------------------------------
    baseline_df = _pick_test_unsorted()

    # model / optimiser / loss -----------------------------------------
    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    best = float("inf")
    loss_hist, bl_hist = [], []
    t0 = time.time()

    for epoch, v_idx in enumerate(perm):
        df = pd.read_csv(variants[v_idx])
        if category == "unsorted":        # keep epoch size constant
            df = df.iloc[:300]

        # ---- training step -----------------------------------------
        dl = DataLoader(
            HelicopterDataset(df, rep_col, haz_col),
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        model.train()
        running = 0.0
        for x, y_r, y_h in dl:
            x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
            opt.zero_grad()
            out_r, out_h = model(x)
            loss = loss_fn(out_r[:, -1], y_r) + 0.5 * loss_fn(out_h, y_h)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        epoch_loss = running / len(dl)
        loss_hist.append(epoch_loss)

        # ---- periodic checkpoint every 20 epochs -------------------
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f"checkpoint_ep{epoch+1:03d}.pt"),
            )

        # ---- baseline probe every 5 epochs -------------------------
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                bl_dl = DataLoader(
                    HelicopterDataset(baseline_df, rep_col_bl, haz_col_bl),
                    batch_size=hp["batch_size"],
                )
                btot = 0.0
                for x, y_r, y_h in bl_dl:
                    x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
                    o_r, o_h = model(x)
                    btot += (
                        loss_fn(o_r[:, -1], y_r) + 0.5 * loss_fn(o_h, y_h)
                    ).item()
                bl_hist.append(btot / len(bl_dl))

        # ---- logging / best‑checkpoint / early‑stop -----------------
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"{type_key}|seed{seed} ep {epoch:03d}/{n_epochs}"
                  f" loss {epoch_loss:.4e}")

        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), ckpt_best)

        if best < hp["target_loss"]:
            print(f"{type_key}|seed{seed} early‑stop @ ep {epoch}"
                  f" (best {best:.4e})")
            break

    # ---- save artefacts ---------------------------------------------
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"),
              indent=2)
    json.dump(bl_hist,
              open(os.path.join(model_dir, "baseline_loss_history.json"), "w"),
              indent=2)
    json.dump(hp, open(os.path.join(model_dir, "hp.json"), "w"), indent=2)

    print(f"{type_key}|seed{seed} finished in {time.time() - t0:.1f}s"
          f" | best {best:.4e}")


# ───────────────────────────── entry‑point ───────────────────────────────
def main() -> None:
    seeds = range(5) # change to range(10) for all seeds

    SPECS: List[Tuple[Type[nn.Module], str, bool, str]] = [
        # (ModelClass, variant_category, use_norm_labels, type_key)
        (GRUModel, "informative",  False, "inf_truth"),
        (GRUModel, "informative",  True,  "inf_norm"),
        (GRUModel, "uninformative",False, "unin_truth"),
        (GRUModel, "uninformative",True,  "unin_norm"),
        (GRUModel, "misleading",   False, "mis_truth"),
        (GRUModel, "misleading",   True,  "mis_norm"),
        (GRUModel, "unsorted", False, "uns_truth"),
        (GRUModel, "unsorted", True,  "uns_norm"),
    ]

    for seed in seeds:
        for cls, cat, use_norm, key in SPECS:
            train_model(cls, key, seed, category=cat, use_norm=use_norm)

    print("All trainings complete.")


if __name__ == "__main__":
    main()

# train.py – variant‑aware trainer for the GRU helicopter task (updated to save checkpoints
# in the same style as train_old.py)
# -----------------------------------------------------------------------------
import os, json, time, ast
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gru_model import GRUModel

# -----------------------------------------------------------------------------
# Paths & device
# -----------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR = os.path.join(BASE_DIR, "variants")  # same as TaskConfig_Generator

# Model‑output directory (mirrors train_old.py layout:  <BASE>/models/<tag>/)
MODEL_TAG   = "gru_trained"         # analogous to the seed_name used previously
MODEL_DIR   = os.path.join(BASE_DIR, "models", MODEL_TAG)
os.makedirs(MODEL_DIR, exist_ok=True)
CKPT_PATH   = os.path.join(MODEL_DIR, "checkpoint.pt")

# -----------------------------------------------------------------------------
# Hyper‑parameter utilities
# -----------------------------------------------------------------------------

def get_default_hp():
    """Return reasonable defaults for this task."""
    return {
        "n_input"      : 1,
        "n_rnn"        : 128,
        "batch_size"   : 32,
        "epochs"       : 5000,
        "learning_rate": 3e-4,
        "target_loss"  : 2e-3,
    }

# -----------------------------------------------------------------------------
# Dataset for a single CSV variant
# -----------------------------------------------------------------------------
class HelicopterDataset(Dataset):
    """Given a <trainConfig_varX.csv>, returns   x_t  , y_report , y_haz   tensors."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.x:  List[torch.Tensor] = []
        self.y_rep: List[torch.Tensor] = []
        self.y_haz: List[torch.Tensor] = []

        for _, row in df.iterrows():
            evid = torch.tensor(ast.literal_eval(row["evidence"]), dtype=torch.float32).unsqueeze(-1)  # (T,1)
            self.x.append(evid)

            # Map labels from {‑1,1} → {0,1} for BCEWithLogitsLoss
            rep = 0.5 * (float(row["trueReport"])  + 1)
            haz = 0.5 * (float(row["truePredict"]) + 1)
            self.y_rep.append(torch.tensor([rep], dtype=torch.float32))
            self.y_haz.append(torch.tensor([haz], dtype=torch.float32))

    # ‑‑ PyTorch Dataset API ‑‑------------------------------------------------
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# -----------------------------------------------------------------------------
# Helper builders
# -----------------------------------------------------------------------------

def build_model(hp):
    return GRUModel(hp).to(DEVICE)

# -----------------------------------------------------------------------------
# Training for one variant CSV
# -----------------------------------------------------------------------------

def train_variant(model: nn.Module, hp: dict, csv_path: str):
    """Train *in‑place* on one variant, return final loss (early‑stop aware)."""
    ds   = HelicopterDataset(csv_path)
    dl   = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True)
    opt  = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce  = nn.BCEWithLogitsLoss()

    for epoch in range(hp["epochs"]):
        epoch_loss = 0.0
        model.train()

        for x, y_rep, y_haz in dl:
            x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)

            opt.zero_grad()
            logits_rep, logits_haz = model(x)

            # Broadcast y_rep to sequence length
            # NEW – use only the last GRU time-step for the report head
            logits_rep_last = logits_rep[:, -1]        # shape (B, 1)
            loss_rep        = bce(logits_rep_last, y_rep)
            loss_haz = bce(logits_haz, y_haz)
            loss     = loss_rep + 0.5 * loss_haz
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dl)
        if epoch % 100 == 0:
            print(f"[{os.path.basename(csv_path)}]  Epoch {epoch:4d} | loss {epoch_loss:.4e}")

        if epoch_loss < hp["target_loss"]:
            print(f"‑ reached target {hp['target_loss']:.2e} after {epoch} epochs.")
            return True, epoch_loss   # early‑stop and report loss

    return False, epoch_loss

# -----------------------------------------------------------------------------
# Main experiment loop (multiple variants) — with best‑checkpoint saving
# -----------------------------------------------------------------------------

def run(max_variants: int = 40):
    print(f"Training on up to {max_variants} variants…")
    hp = get_default_hp()

    torch.manual_seed(0)
    model = build_model(hp)

    losses, times = [], []
    best_loss = float("inf")
    start_global = time.time()

    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"trainConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            print(f"Variant file {csv_path} not found — stopping.")
            break

        print(f"\n=== Training on variant {k} ===")
        t0 = time.time()
        early, v_loss = train_variant(model, hp, csv_path)
        times.append(time.time() - t0)
        losses.append(float(v_loss))

        # --- checkpoint if improved ---------------------------
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  >> new best loss {best_loss:.4e} — checkpoint saved")

        if early:
            print("Early‑stopping criterion met; ending training across variants.")
            break

    # ---------------------------------------------------------------------
    # Persist run metadata (loss curve, times, hp)
    # ---------------------------------------------------------------------
    with open(os.path.join(MODEL_DIR, "loss_history.json"), "w") as f:
        json.dump(losses, f)
    with open(os.path.join(MODEL_DIR, "times.json"), "w") as f:
        json.dump(times, f)
    with open(os.path.join(MODEL_DIR, "hp.json"), "w") as f:
        json.dump(hp, f)

    total_time = time.time() - start_global
    print(f"\nTotal time: {total_time:.1f}s | variants used: {len(losses)} | best loss: {best_loss:.4e}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run(max_variants=40)

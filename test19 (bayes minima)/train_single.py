#!/usr/bin/env python3
"""
train_single.py — Train RNNs on the helicopter task.

**July 2025 revision**
The supervision source (ground truth vs. Bayesian normative labels) is chosen
per‑model via a Boolean in `MODEL_SPECS`; no environment variables required.
"""
# ---------------------------------------------------------------------------
# std libs
import os, json, time, copy, random, ast
from typing import List, Type

# third‑party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# project modules
from rnn_models import GRUModel, LSTMModel, RNNModel
import TaskConfig_Generator as TCG
from NormativeModel import BayesianObserver

# ---------------------------------------------------------------------------
# Constants for the Bayesian observer (match TaskConfig settings)
HS_GRID  = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # robust on Windows & POSIX

# ---------------------------------------------------------------------------
# Helper: normative prediction

def normative_predict(evidence: List[float], sigma: float) -> tuple[int, float]:
    """Return (state_report ∈{−1,1}, posterior_mean_hazard ∈[0,1])."""
    L_haz, _, rep, _ = BayesianObserver(evidence, MU1, MU2, sigma, HS_GRID.copy())
    haz_mean = float(np.dot(HS_GRID, L_haz[:, -1]))
    return int(rep), haz_mean

# ---------------------------------------------------------------------------
# Hyper‑parameters

def get_default_hp():
    return {
        "n_input"       : 1,
        "n_rnn"         : 128,
        "batch_size"    : 25,
        "max_epochs"    : 500,
        "learning_rate" : 3e-4,
        "target_loss"   : 2e-3,   # early‑stop once any epoch beats this
    }

# ---------------------------------------------------------------------------
# Dataset wrapper
class HelicopterDataset(Dataset):
    """Turn a row of the trial DataFrame into model‑ready tensors."""
    def __init__(self, df: pd.DataFrame):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            evid = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
            self.x.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            self.y_rep.append(torch.tensor([0.5 * (row["trueReport"] + 1)], dtype=torch.float32))
            self.y_haz.append(torch.tensor([row["trueHazard"]], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ---------------------------------------------------------------------------
# Trial generation — called once per *epoch*

def generate_trials(use_norm: bool) -> pd.DataFrame:
    """Generate a fresh trial set; replace targets with normative labels if requested."""
    params = copy.deepcopy(TCG.params)
    df = pd.DataFrame(TCG.makeBlockTrials(params))

    if use_norm:
        for idx, row in df.iterrows():
            evidence = row["evidence"]
            if not isinstance(evidence, list):
                evidence = ast.literal_eval(str(evidence))
            sigma = float(row["sigma"])
            rep_norm, haz_norm = normative_predict(evidence, sigma)
            df.at[idx, "trueReport"] = rep_norm
            df.at[idx, "trueHazard"]  = haz_norm

    return df

# ---------------------------------------------------------------------------
# Core trainer

def train_model(model_cls: Type[nn.Module], tag: str, use_norm: bool, seed: int = 0):
    hp = get_default_hp()

    # Reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Paths
    suffix    = "_norm" if use_norm else ""
    model_dir = os.path.join(BASE_DIR, "models", f"{tag}{suffix}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    # Model & optimiser
    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce   = nn.BCEWithLogitsLoss()

    best_loss, loss_hist = float("inf"), []
    t_start = time.time()

    # Training loop
    for epoch in range(hp["max_epochs"]):
        df_trials = generate_trials(use_norm)
        dl = DataLoader(HelicopterDataset(df_trials),
                        batch_size=hp["batch_size"], shuffle=True, drop_last=True)

        model.train()
        running = 0.0
        for x, y_rep, y_haz in dl:
            x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)

            opt.zero_grad()
            logits_rep, logits_haz = model(x)
            loss = bce(logits_rep[:, -1], y_rep) + 0.5 * bce(logits_haz, y_haz)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        epoch_loss = running / len(dl)
        loss_hist.append(epoch_loss)
        if epoch % 10 == 0 or epoch == hp["max_epochs"] - 1:
            print(f"{tag}{suffix} | epoch {epoch:3d} | loss {epoch_loss:.4e}")

        # checkpoints
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ckpt_best)
        if best_loss < hp["target_loss"]:
            print(f"{tag}{suffix} | early stop at epoch {epoch} (loss {best_loss:.4e})")
            break

    # final save & metadata
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(hp,         open(os.path.join(model_dir, "hp.json"),          "w"), indent=2)

    print(f"{tag}{suffix} finished in {time.time() - t_start:.1f}s | best {best_loss:.4e}")

# ---------------------------------------------------------------------------

def main():
    MODEL_SPECS = [
        (GRUModel,  "gru_truth", False),  # ground truth
        (GRUModel,  "gru_norm",  True),   # normative labels
    ]
    for cls, tag, use_norm in MODEL_SPECS:
        train_model(cls, tag, use_norm, seed=0)
    print("All models done — each epoch uses fresh, non‑repeated trials.")


if __name__ == "__main__":
    main()

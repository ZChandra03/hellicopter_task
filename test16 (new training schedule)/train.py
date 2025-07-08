# train.py – trainer that generates *fresh* trials every epoch  ────────────────────────────────
# --------------------------------------------------------------------------------------
# This version **never** re‑uses a trial.  For each training epoch we call
# `TaskConfig_Generator.makeBlockTrials` to build a brand‑new set of trials, train on it
# once, then discard it.  Training stops when either `max_epochs` is reached or the
# running loss drops below `target_loss`.
# --------------------------------------------------------------------------------------
import os, json, time, copy, random
from typing import List, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Project‑local imports -------------------------------------------------------------
from rnn_models import GRUModel, LSTMModel, RNNModel
import TaskConfig_Generator as TCG

# Global constants -----------------------------------------------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MAX_VARIANTS = 40  # number of independent seed runs per model class

# Hyper‑parameters ------------------------------------------------------------------

def get_default_hp():
    """Return a dict of default hyper‑parameters."""
    return {
        "n_input"       : 1,
        "n_rnn"         : 128,
        "batch_size"    : 32,
        "max_epochs"    : 100,
        "learning_rate" : 3e-4,
        "target_loss"   : 2e-3,
    }

# Dataset wrapper -------------------------------------------------------------------

class HelicopterDataset(Dataset):
    """Wrap a *pandas.DataFrame* into tensors for the helicopter task."""

    def __init__(self, df: pd.DataFrame):
        self.x:      List[torch.Tensor] = []
        self.y_rep:  List[torch.Tensor] = []
        self.y_haz:  List[torch.Tensor] = []

        for _, row in df.iterrows():
            evid_raw = row["evidence"]
            evid     = evid_raw if isinstance(evid_raw, list) else eval(evid_raw)
            evid_t   = torch.tensor(evid, dtype=torch.float32).unsqueeze(-1)  # (T, 1)
            self.x.append(evid_t)

            rep_bin = 0.5 * (float(row["trueReport"]) + 1)  # map −1/+1 → 0/1
            haz_val = float(row["trueHazard"])

            self.y_rep.append(torch.tensor([rep_bin], dtype=torch.float32))
            self.y_haz.append(torch.tensor([haz_val], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# Training helpers ------------------------------------------------------------------

def generate_trials() -> pd.DataFrame:
    """Return a new DataFrame of trials from TaskConfig_Generator every call."""
    params = copy.deepcopy(TCG.params)
    trials = TCG.makeBlockTrials(params)  # returns list[dict] or DataFrame‑like object
    return pd.DataFrame(trials)

# ----------------------------------------------------------------------------------
# Core training routine – one seed / "variant" --------------------------------------
# ----------------------------------------------------------------------------------

def train_variant(model: nn.Module, hp: dict, seed: int):
    """Train `model` on freshly generated trials for up to `max_epochs`."""
    # RNG reproducibility -----------------------------------------------------------
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce = nn.BCEWithLogitsLoss()

    best_loss = float("inf")

    for epoch in range(hp["max_epochs"]):
        # 1) new trials every epoch --------------------------------------------------
        df_trials = generate_trials()
        dl = DataLoader(HelicopterDataset(df_trials),
                        batch_size=hp["batch_size"], shuffle=True, drop_last=True)

        # 2) one pass over this epoch's data ----------------------------------------
        model.train()
        running, batches = 0.0, 0

        for x, y_rep, y_haz in dl:
            x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)

            opt.zero_grad()
            logits_rep, logits_haz = model(x)

            loss_rep = bce(logits_rep[:, -1], y_rep)
            loss_haz = bce(logits_haz,        y_haz)
            loss     = loss_rep + 0.5 * loss_haz
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            batches += 1

        epoch_loss = running / batches
        print(f"    epoch {epoch:3d} | loss {epoch_loss:.4e}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        # Early‑stop if we meet target ------------------------------------------------
        if best_loss < hp["target_loss"]:
            print("    target loss reached – stopping early")
            break

    return best_loss

# Wrapper that trains one model class across multiple seeds -------------------------

def train_model_type(model_cls: Type[nn.Module], tag: str, max_variants: int = MAX_VARIANTS):
    print(f"\n===== Training {tag} =====")
    hp = get_default_hp()

    best_global = float("inf")
    losses, times = [], []

    model_dir = os.path.join(BASE_DIR, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")

    t_global = time.time()

    for v in range(max_variants):
        print(f"\n=== Variant / seed {v} ===")
        model = model_cls(hp).to(DEVICE)  # fresh weights per seed
        t0 = time.time()
        v_loss = train_variant(model, hp, seed=v)
        elapsed = time.time() - t0

        losses.append(float(v_loss))
        times.append(elapsed)

        if v_loss < best_global:
            best_global = v_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> new best loss {best_global:.4e} – checkpoint saved")

        if best_global < hp["target_loss"]:
            print("Early‑stopping across seeds – global target met.")
            break

    # Save logs --------------------------------------------------------------------
    with open(os.path.join(model_dir, "loss_history.json"), "w") as f:
        json.dump(losses, f)
    with open(os.path.join(model_dir, "times.json"), "w") as f:
        json.dump(times, f)
    with open(os.path.join(model_dir, "hp.json"), "w") as f:
        json.dump(hp, f)

    print(f"Finished {tag}: {time.time() - t_global:.1f}s | variants {len(losses)} | best loss {best_global:.4e}")

# Main entry ------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_SPECS = [
        (GRUModel,  "gru_trained"),
        (LSTMModel, "lstm_trained"),
        (RNNModel,  "rnn_trained"),
    ]

    for cls, tag in MODEL_SPECS:
        train_model_type(cls, tag)

    print("\nAll model classes finished – training ran with unique trials every epoch.")

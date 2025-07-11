# train.py – unified trainer that sequentially trains one RNN, one GRU and one LSTM model
# --------------------------------------------------------------------------------------
# This merges the three task‑specific scripts (train_vanilla.py, train.py, train_LSTM.py)
# into ONE file.  It loops over the three model classes exposed by rnn_models.py and
# repeats exactly the same variant‑aware training procedure for each, saving separate
# checkpoints / logs under   models/<tag>/ .  No argparse or extra dependencies added.
# --------------------------------------------------------------------------------------
import os, json, time, ast
from typing import List, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------------------
# Import the three encoder classes from *one* place
# --------------------------------------------------------------------------------------
from rnn_models import GRUModel, LSTMModel, RNNModel

# --------------------------------------------------------------------------------------
# Global constants (shared for all model types)
# --------------------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR = os.path.join(BASE_DIR, "variants")  # generated by TaskConfig_Generator
MAX_VARIANTS = 40                                   # hard‑coded like before

# --------------------------------------------------------------------------------------
# Hyper‑parameter utilities (identical defaults for all models)
# --------------------------------------------------------------------------------------

def get_default_hp():
    """Return reasonable defaults for this task."""
    return {
        "n_input"      : 1,
        "n_rnn"        : 128,
        "batch_size"   : 32,
        "epochs"       : 100,
        "learning_rate": 3e-4,
        "target_loss"  : 2e-3,
    }

# --------------------------------------------------------------------------------------
# Dataset for a single CSV variant (unchanged)
# --------------------------------------------------------------------------------------

class HelicopterDataset(Dataset):
    """Given a <trainConfig_varX_bayes.csv>, returns  x_t , y_report , y_haz  tensors."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.x:      List[torch.Tensor] = []
        self.y_rep:  List[torch.Tensor] = []
        self.y_haz:  List[torch.Tensor] = []

        for _, row in df.iterrows():
            evid = torch.tensor(ast.literal_eval(row["evidence"]),
                                dtype=torch.float32).unsqueeze(-1)      # (T, 1)
            self.x.append(evid)

            rep = 0.5 * (float(row["rep_norm"]) + 1)   # −1/+1 → 0/1
            haz = float(row["haz_norm"])               # 0.0 … 1.0

            self.y_rep.append(torch.tensor([rep], dtype=torch.float32))
            self.y_haz.append(torch.tensor([haz], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# --------------------------------------------------------------------------------------
# Core training routine (shared) --------------------------------------------------------
# --------------------------------------------------------------------------------------

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

            loss_rep = bce(logits_rep[:, -1], y_rep)  # last‑step report head
            loss_haz = bce(logits_haz, y_haz)         # predict hazard rate
            loss     = loss_rep + 0.5 * loss_haz
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dl)
        if epoch % 100 == 0:
            print(f"[{os.path.basename(csv_path)}]  Epoch {epoch:4d} | loss {epoch_loss:.4e}")

        if epoch_loss < hp["target_loss"]:
            print(f"  ‑ reached target {hp['target_loss']:.2e} after {epoch} epochs.")
            return True, epoch_loss  # early stop

    return False, epoch_loss

# --------------------------------------------------------------------------------------
# Wrapper that trains *one* model type across all variants and handles bookkeeping
# --------------------------------------------------------------------------------------

def train_model_type(model_cls: Type[nn.Module], tag: str, max_variants: int = MAX_VARIANTS):
    """Train the given model class across variants; save best checkpoint & metadata."""
    print(f"\n===== Training {tag} =====")
    hp = get_default_hp()

    torch.manual_seed(0)
    model = model_cls(hp).to(DEVICE)

    # prepare output dirs unique to this tag
    model_dir = os.path.join(BASE_DIR, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")

    losses, times = [], []
    best_loss     = float("inf")
    start_global  = time.time()

    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"trainConfig_var{k}_bayes.csv")
        if not os.path.isfile(csv_path):
            print(f"Variant file {csv_path} not found — stopping.")
            break

        print(f"\n=== Variant {k} ===")
        t0 = time.time()
        early, v_loss = train_variant(model, hp, csv_path)
        times.append(time.time() - t0)
        losses.append(float(v_loss))

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> new best loss {best_loss:.4e} — checkpoint saved")

        if early:
            print("Early‑stopping criterion met; ending training across variants.")
            break

    # save run metadata
    with open(os.path.join(model_dir, "loss_history.json"), "w") as f:
        json.dump(losses, f)
    with open(os.path.join(model_dir, "times.json"), "w") as f:
        json.dump(times, f)
    with open(os.path.join(model_dir, "hp.json"), "w") as f:
        json.dump(hp, f)

    total_time = time.time() - start_global
    print(f"Finished {tag}: {total_time:.1f}s | variants: {len(losses)} | best loss: {best_loss:.4e}")

# --------------------------------------------------------------------------------------
# Main: iterate over the three encoder types ------------------------------------------------
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_SPECS = [  # (class, directory‑friendly tag)
        (GRUModel,   "gru_trained"),
        (LSTMModel,  "lstm_trained"),
        (RNNModel,   "rnn_trained"),
    ]

    for cls, tag in MODEL_SPECS:
        train_model_type(cls, tag, max_variants=MAX_VARIANTS)

    print("\nAll model types finished.")

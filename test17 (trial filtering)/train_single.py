#!/usr/bin/env python3
"""
train_single.py — Train one initialization per RNN model on the helicopter task.

Key differences from the previous `train_seeded.py`:
• No outer “variants / seeds” loop — each model class is trained exactly once.
• Fresh trials are still generated every epoch (the model never sees the same
  trial twice).
• The best‑performing epoch (lowest loss) is checkpointed to
  `models/<tag>/checkpoint_best.pt`
• The final model after training/early‑stop is saved to
  `models/<tag>/final.pt`
• Epoch‑wise loss history and hyper‑parameters are stored alongside the
  weights for later inspection.
"""
# std libs
import os, json, time, copy, random
from typing import List, Type

# third‑party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# project
from rnn_models import GRUModel, LSTMModel, RNNModel
import TaskConfig_Generator_filtered as TCG

# ---------------------------------------------------------------------------
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    """Wrap a DataFrame row into tensors understood by the network."""
    def __init__(self, df: pd.DataFrame):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            evid = evid if isinstance(evid, list) else eval(evid)
            self.x     .append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            self.y_rep .append(torch.tensor([0.5*(row["trueReport"]+1)], dtype=torch.float32))
            self.y_haz .append(torch.tensor([row["trueHazard"]], dtype=torch.float32))
    def __len__(self):                return len(self.x)
    def __getitem__(self, idx):       return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ---------------------------------------------------------------------------
# Trial generation — called once per *epoch*
def generate_trials() -> pd.DataFrame:
    params = copy.deepcopy(TCG.params)
    return pd.DataFrame(TCG.makeBlockTrials(params))

# ---------------------------------------------------------------------------
# Core trainer (single seed / single model instance)
def train_model(model_cls: Type[nn.Module], tag: str, seed: int = 0):
    hp = get_default_hp()

    # Reproducibility
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    # Set‑up ----------------------------------------------------------------
    model_dir  = os.path.join(BASE_DIR, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce   = nn.BCEWithLogitsLoss()

    best_loss, loss_hist = float("inf"), []
    t_start = time.time()

    # Training loop ---------------------------------------------------------
    for epoch in range(hp["max_epochs"]):
        df_trials = generate_trials()
        dl = DataLoader(
            HelicopterDataset(df_trials),
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=True,
        )

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
        loss_hist.append(epoch_loss)
        print(f"{tag} | epoch {epoch:3d} | loss {epoch_loss:.4e}")

        # checkpoint best model so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ckpt_best)

        # early stopping
        if best_loss < hp["target_loss"]:
            print(f"{tag} | target loss reached ({best_loss:.4e}) – early stop at epoch {epoch}")
            break

    # Save *final* model after training loop (may or may not be best)
    torch.save(model.state_dict(), ckpt_final)

    # Persist training metadata --------------------------------------------
    with open(os.path.join(model_dir, "loss_history.json"), "w") as f:
        json.dump(loss_hist, f, indent=2)
    with open(os.path.join(model_dir, "hp.json"), "w") as f:
        json.dump(hp, f, indent=2)

    duration = time.time() - t_start
    print(f"{tag} finished in {duration:.1f}s | best loss {best_loss:.4e} | checkpoints saved to {model_dir}")

# ---------------------------------------------------------------------------
def main():
    MODEL_SPECS = [
        (GRUModel,  "gru_trained"),
    ]
    for cls, tag in MODEL_SPECS:
        train_model(cls, tag, seed=0)
    print("All model classes finished — training ran with unique trials each epoch.")

if __name__ == "__main__":
    main()

# train.py  – variant-aware trainer for the GRU helicopter task
# ------------------------------------------------------------
import os, json, time, math, ast
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gru_model import GRUModel                   # ← new!
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR = os.path.join(BASE_DIR, "variants")  # same as TaskConfig_Generator
# ------------------------------------------------------------------
# Hyper-parameter utilities
# ------------------------------------------------------------------
def get_default_hp():
    return {
        "n_input": 1,
        "n_rnn": 128,
        "batch_size": 32,
        "epochs": 5000,
        "learning_rate": 3e-4,
        "target_loss": 2e-3,
    }
# ------------------------------------------------------------------
# Dataset for one CSV variant
# ------------------------------------------------------------------
class HelicopterDataset(Dataset):
    """Loads one <trainConfig_varX.csv> and returns tensors:
         x_t     : (T, 1)   evidence sequence
         y_report: (1,)     latest state  (0 / 1)
         y_haz   : (1,)     hazard class  (0 / 1)
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.x      : List[torch.Tensor] = []
        self.y_rep  : List[torch.Tensor] = []
        self.y_haz  : List[torch.Tensor] = []

        for _, row in df.iterrows():
            evid = torch.tensor(
                ast.literal_eval(row["evidence"]), dtype=torch.float32
            ).unsqueeze(-1)                    # (T,1)
            self.x.append(evid)

            # map {-1,1} → {0,1} for BCE
            rep = 0.5 * (float(row["trueReport" ]) + 1)
            haz = 0.5 * (float(row["truePredict"]) + 1)
            self.y_rep.append(torch.tensor([rep], dtype=torch.float32))
            self.y_haz.append(torch.tensor([haz], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]
# ------------------------------------------------------------------
def build_model(hp):
    return GRUModel(hp).to(DEVICE)
# ------------------------------------------------------------------
def train_variant(model, hp, csv_path):
    ds  = HelicopterDataset(csv_path)
    dl  = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(hp["epochs"]):
        epoch_loss = 0.0
        model.train()

        for x, y_rep, y_haz in dl:
            x, y_rep, y_haz = x.to(DEVICE), y_rep.to(DEVICE), y_haz.to(DEVICE)

            opt.zero_grad()
            logits_rep, logits_haz = model(x)               # (B,T,1)  &  (B,1)

            # expand y_rep → (B,T,1) to match logits_rep
            y_rep_seq = y_rep.unsqueeze(1).repeat(1, logits_rep.size(1), 1)

            loss_rep = bce(logits_rep, y_rep_seq)
            loss_haz = bce(logits_haz, y_haz)
            loss = loss_rep + 0.5 * loss_haz          # simple weighting
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dl)
        if epoch % 10 == 0:
            print(f"[{os.path.basename(csv_path)}]  "
                  f"Epoch {epoch:4d} | loss {epoch_loss:.4e}")

        if epoch_loss < hp["target_loss"]:
            print(f"- reached target {hp['target_loss']:.2e} after {epoch} epochs.")
            return True, epoch_loss    # early-stop for whole training loop

    return False, epoch_loss
# ------------------------------------------------------------------
def run(max_variants=40):
    print(f"Training on up to {max_variants} variants...")
    hp = get_default_hp()
    torch.manual_seed(0)
    model = build_model(hp)

    losses, start = [], time.time()
    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"trainConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            print(f"Variant file {csv_path} not found — stopping.")
            break

        print(f"\n=== Training on variant {k} ===")
        early, v_loss = train_variant(model, hp, csv_path)
        losses.append(float(v_loss))

        if early:
            break

    # save weights + log
    out_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "gru_checkpoint.pt"))
    json.dump(losses, open(os.path.join(out_dir, "loss_history.json"), "w"))

    print(f"\nTotal time: {time.time()-start:.1f}s | "
          f"variants used: {len(losses)} | "
          f"best loss: {min(losses):.4e}")
# ------------------------------------------------------------------
if __name__ == "__main__":
    run(max_variants=40)          # change here or pass via argparse

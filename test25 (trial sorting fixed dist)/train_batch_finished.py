#!/usr/bin/env python3
"""
train_batch_pilot.py — *revision 4*  (July 2025)
================================================

### New directory layout
```
models/
├── inf_truth/
│   ├── seed_0/
│   │   ├── checkpoint_best.pt
│   │   ├── final.pt
│   │   └── …
│   └── seed_1/
│       └── …
├── inf_norm/
│   └── seed_0/ …
├── unin_truth/
│   └── seed_0/ …
├── unin_norm/
│   └── seed_0/ …
├── mis_truth/
│   └── seed_0/ …
├── mis_norm/
│   └── seed_0/ …
├── uns_truth/
│   └── seed_0/ …
└── uns_norm/
    └── seed_0/ …
```
Each **of the eight training types** now has its own folder; inside it, one
sub‑folder per seed keeps that run’s checkpoints and JSON logs.

Other behaviour (one CSV per epoch, seed‑shuffled, baseline probes, early‑stop,
`rep_norm`/`haz_norm` columns for normative models) is unchanged.
"""
# ---------------------------------------------------------------------------
import os, json, time, random, ast, glob
from typing import List, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rnn_models import GRUModel, LSTMModel, RNNModel

# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANT_ROOT = os.path.join(BASE_DIR, "variants", "train")

# ---------------------------------------------------------------------------

def get_default_hp():
    return {
        "n_input"      : 1,
        "n_rnn"        : 128,
        "batch_size"   : 25,
        "learning_rate": 3e-4,
        "target_loss"  : 2e-3,
    }

# ---------------------------------------------------------------------------
class HelicopterDataset(Dataset):
    """Tensor view of one CSV. Column names select targets."""

    def __init__(self, df: pd.DataFrame, rep_col: str, haz_col: str):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            evid = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
            self.x.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            self.y_rep.append(torch.tensor([(row[rep_col] + 1) * 0.5], dtype=torch.float32))
            self.y_haz.append(torch.tensor([row[haz_col]], dtype=torch.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ---------------------------------------------------------------------------

def list_variant_paths(category: str) -> List[str]:
    pat = os.path.join(VARIANT_ROOT, category, f"train_{category}_*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No CSVs match {pat}")
    return paths

# ---------------------------------------------------------------------------

def train_model(model_cls: Type[nn.Module], type_key: str, seed: int, category: str, use_norm: bool):
    """Train a single model (defined by *type_key* & *seed*)."""

    variants = list_variant_paths(category)
    n_epochs = len(variants)

    hp = get_default_hp(); hp["max_epochs"] = n_epochs

    # Seed RNGs + permutation
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_epochs)

    rep_col, haz_col = ("rep_norm", "haz_norm") if use_norm else ("trueReport", "trueHazard")

    # -------------------------------------------------------------------
    # Directory layout: models/<type_key>/seed_<n>/…
    model_dir = os.path.join(BASE_DIR, "models", type_key, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    # Baseline unsorted CSV (fixed per run)
    unsorted_paths = list_variant_paths("unsorted")
    baseline_df = pd.read_csv(unsorted_paths[rng.integers(len(unsorted_paths))])

    model = model_cls(hp).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    best, loss_hist, bl_hist = float("inf"), [], []
    t0 = time.time()

    for epoch, v_idx in enumerate(perm):
        df = pd.read_csv(variants[v_idx])
        if category == "unsorted":
            df = df.iloc[:300]

        dl = DataLoader(HelicopterDataset(df, rep_col, haz_col), batch_size=hp["batch_size"], shuffle=True, drop_last=True)
        model.train(); running = 0.0
        for x, y_r, y_h in dl:
            x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
            opt.zero_grad(); out_r, out_h = model(x)
            loss = loss_fn(out_r[:, -1], y_r) + 0.5 * loss_fn(out_h, y_h)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            running += loss.item()
        epoch_loss = running / len(dl); loss_hist.append(epoch_loss)

        # Baseline probe
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval(); btot = 0.0
                bl_dl = DataLoader(HelicopterDataset(baseline_df, rep_col, haz_col), batch_size=hp["batch_size"])
                for x, y_r, y_h in bl_dl:
                    x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
                    o_r, o_h = model(x)
                    btot += (loss_fn(o_r[:, -1], y_r) + 0.5 * loss_fn(o_h, y_h)).item()
                bl_hist.append(btot / len(bl_dl))

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"{type_key}|seed{seed} ep {epoch:03d}/{n_epochs} loss {epoch_loss:.4e}")

        if epoch_loss < best:
            best = epoch_loss; torch.save(model.state_dict(), ckpt_best)
        if best < hp["target_loss"]:
            print(f"{type_key}|seed{seed} early‑stop @ ep {epoch} (best {best:.4e})"); break

    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(bl_hist,   open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp,        open(os.path.join(model_dir, "hp.json"),             "w"), indent=2)

    print(f"{type_key}|seed{seed} done in {time.time() - t0:.1f}s | best {best:.4e}")

# ---------------------------------------------------------------------------

def main():
    seeds = range(10)  # adjust as needed

    # (Model‑class, category, use_norm flag, folder key)
    SPECS: list[Tuple[Type[nn.Module], str, bool, str]] = [
        (GRUModel, "informative",  False, "inf_truth"),
        (GRUModel, "informative",  True,  "inf_norm"),
        (GRUModel, "uninformative",False, "unin_truth"),
        (GRUModel, "uninformative",True,  "unin_norm"),
        (GRUModel, "misleading",   False, "mis_truth"),
        (GRUModel, "misleading",   True,  "mis_norm"),
        (GRUModel, "unsorted",     False, "uns_truth"),
        (GRUModel, "unsorted",     True,  "uns_norm"),
    ]

    for seed in seeds:
        for cls, cat, use_norm, key in SPECS:
            train_model(cls, key, seed, category=cat, use_norm=use_norm)

    print("All trainings complete.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

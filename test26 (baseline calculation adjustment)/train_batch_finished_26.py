#!/usr/bin/env python3
"""
train_batch_finished.py — revision 5 (July 2025)

Eight training types × 10 seeds → 80 GRU runs, each epoch fed by one 300‑trial CSV.
Every 5 epochs we probe the model on a fixed 300‑trial *unsorted* file to log a
“baseline” loss curve that is saved to
    models/<type_key>/seed_<n>/baseline_loss_history.json

**New in r5** – the baseline probe is *always* evaluated against the ground‑truth
columns (`trueReport`, `trueHazard`), even for models trained on normative
labels, so the resulting curves are on the same footing.
"""
# ──────────────────────────────────────────────────────────────────────────────
import os, json, time, random, ast, glob
from typing import List, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rnn_models import GRUModel, LSTMModel, RNNModel

# ──────────────────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
VAR_ROOT  = os.path.join(BASE_DIR, "variants", "train")

# ──────────────────────────────────────────────────────────────────────────────
def get_default_hp():
    return {
        "n_input"      : 1,
        "n_rnn"        : 128,
        "batch_size"   : 25,
        "learning_rate": 3e-4,
        "target_loss"  : 2e-3,
    }

# ──────────────────────────────────────────────────────────────────────────────
class HelicopterDataset(Dataset):
    """Turn one CSV of trials into (evidence, report‑target, hazard‑target) tensors."""

    def __init__(self, df: pd.DataFrame, rep_col: str, haz_col: str):
        self.x, self.y_rep, self.y_haz = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            evid = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
            self.x.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            self.y_rep.append(torch.tensor([(row[rep_col] + 1) * 0.5], dtype=torch.float32))
            self.y_haz.append(torch.tensor([row[haz_col]], dtype=torch.float32))

    def __len__(self):  return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y_rep[idx], self.y_haz[idx]

# ──────────────────────────────────────────────────────────────────────────────
def list_variant_paths(category: str) -> List[str]:
    pat = os.path.join(VAR_ROOT, category, f"train_{category}_*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No CSVs match {pat}")
    return paths

# ──────────────────────────────────────────────────────────────────────────────
def train_model(model_cls: Type[nn.Module], type_key: str,
                seed: int, *, category: str, use_norm: bool) -> None:
    """Train one model run (defined by *type_key* and *seed*)."""

    variants = list_variant_paths(category)
    n_epochs = len(variants)

    hp = get_default_hp(); hp["max_epochs"] = n_epochs

    # deterministic shuffling per seed
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(n_epochs)

    # training targets
    rep_col, haz_col = ("rep_norm", "haz_norm") if use_norm else ("trueReport", "trueHazard")
    # baseline probe targets – always ground truth
    rep_col_bl, haz_col_bl = "trueReport", "trueHazard"       # NEW  :contentReference[oaicite:1]{index=1}

    # ── folder layout ────────────────────────────────────────────────────────
    model_dir  = os.path.join(BASE_DIR, "models", type_key, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_best  = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    # fixed unsorted CSV for the 5‑epoch probe
    baseline_df = pd.read_csv(list_variant_paths("unsorted")[np.random.randint(0, 999999) % len(list_variant_paths("unsorted"))])

    model   = model_cls(hp).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    best, loss_hist, bl_hist = float("inf"), [], []
    t0 = time.time()

    for epoch, v_idx in enumerate(perm):
        df = pd.read_csv(variants[v_idx])
        if category == "unsorted":
            df = df.iloc[:300]                         #   300‑trial subset

        dl = DataLoader(HelicopterDataset(df, rep_col, haz_col),
                        batch_size=hp["batch_size"], shuffle=True, drop_last=True)

        # ── training step ───────────────────────────────────────────────────
        model.train(); running = 0.0
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

        # ── baseline probe every 5 epochs ───────────────────────────────────
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval(); btot = 0.0
                bl_dl = DataLoader(HelicopterDataset(baseline_df, rep_col_bl, haz_col_bl),   # CHANGED
                                   batch_size=hp["batch_size"])
                for x, y_r, y_h in bl_dl:
                    x, y_r, y_h = x.to(DEVICE), y_r.to(DEVICE), y_h.to(DEVICE)
                    o_r, o_h = model(x)
                    btot += (loss_fn(o_r[:, -1], y_r) + 0.5 * loss_fn(o_h, y_h)).item()
                bl_hist.append(btot / len(bl_dl))

        # progress + early‑stop
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"{type_key}|seed{seed} ep {epoch:03d}/{n_epochs} loss {epoch_loss:.4e}")

        if epoch_loss < best:
            best = epoch_loss; torch.save(model.state_dict(), ckpt_best)
        if best < hp["target_loss"]:
            print(f"{type_key}|seed{seed} early‑stop @ ep {epoch} (best {best:.4e})")
            break

    # ── write artefacts ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(bl_hist,   open(os.path.join(model_dir, "baseline_loss_history.json"), "w"), indent=2)
    json.dump(hp,        open(os.path.join(model_dir, "hp.json"),             "w"), indent=2)

    print(f"{type_key}|seed{seed} done in {time.time() - t0:.1f}s | best {best:.4e}")

# ──────────────────────────────────────────────────────────────────────────────
def main():
    seeds = range(10)   # 0‑9

    SPECS: List[Tuple[Type[nn.Module], str, bool, str]] = [
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

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

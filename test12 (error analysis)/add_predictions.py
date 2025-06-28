#!/usr/bin/env python3
# add_predictions.py -----------------------------------------------------------
# Append report + hazard predictions from the Bayesian normative model and
# six trained RNNs (models/…  +  models_1/…) to every test CSV.
# ------------------------------------------------------------------------------
import os, ast, json
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch

from NormativeModel import BayesianObserver
from rnn_models    import GRUModel, LSTMModel, RNNModel

# ───────────────────────── constants
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VAR_DIR    = os.path.join(BASE_DIR, "variants")      # where testConfig_var*.csv live
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HS_GRID    = np.arange(0, 1.05, 0.05)
MU1, MU2   = -1, 1                                   # location means for the observer

# folders → suffix used in column names
MODEL_FOLDERS: Tuple[Tuple[str, str], ...] = (
    ("models",   ""),    # → rep_gru, haz_gru, …
    ("models_1", "1"),   # → rep_gru1, haz_gru1, …
)

MODEL_TAGS: Tuple[Tuple[str, type, str], ...] = (
    ("gru_trained",  GRUModel,  "gru"),
    ("lstm_trained", LSTMModel, "lstm"),
    ("rnn_trained",  RNNModel,  "rnn"),
)

# ───────────────────────── helpers
def default_hp() -> Dict[str, int]:
    return {"n_input": 1, "n_rnn": 128}

def load_hp(model_dir: str) -> Dict:
    path = os.path.join(model_dir, "hp.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return default_hp()

def load_checkpoint(cls, model_dir: str) -> torch.nn.Module:
    hp   = load_hp(model_dir)
    mdl  = cls(hp).to(DEVICE)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"),
                      map_location=DEVICE)
    mdl.load_state_dict(ckpt)
    mdl.eval()
    return mdl

def build_rnn_bank() -> Dict[str, torch.nn.Module]:
    """Return dict  key → model  where key is e.g. 'gru', 'gru1', …"""
    bank = {}
    for folder, suffix in MODEL_FOLDERS:
        root = os.path.join(BASE_DIR, folder)
        for tag, cls, short in MODEL_TAGS:
            mdir = os.path.join(root, tag)
            ckpt = os.path.join(mdir, "checkpoint.pt")
            if os.path.isfile(ckpt):
                key = f"{short}{suffix}"
                bank[key] = load_checkpoint(cls, mdir)
            else:
                print(f"[warning] checkpoint missing: {ckpt} – skipping")
    return bank

@torch.no_grad()
def rnn_predict(model: torch.nn.Module, evidence) -> Tuple[int, float]:
    """Return (report ∈{-1,1}, hazard ∈[0,1])."""
    x = torch.tensor(evidence, dtype=torch.float32)\
             .unsqueeze(0).unsqueeze(-1).to(DEVICE)
    loc_logits, haz_logits = model(x)
    rep = 1 if loc_logits[0, -1, 0].item() > 0 else -1
    haz = torch.sigmoid(haz_logits)[0, 0].item()
    return rep, haz

def normative_predict(evidence, sigma) -> Tuple[int, float]:
    L_haz, L_state, rep, _ = BayesianObserver(
        evidence, MU1, MU2, sigma, HS_GRID.copy()
    )
    haz = float(np.dot(HS_GRID, L_haz[:, -1]))       # posterior mean
    return int(rep), haz

# ───────────────────────── core
def process_csv(path: str, bank: Dict[str, torch.nn.Module]) -> None:
    df = pd.read_csv(path)

    # prepare all new column names
    cols = ["rep_norm", "haz_norm"]
    for key in bank:
        cols += [f"rep_{key}", f"haz_{key}"]
    for c in cols:
        if c in df.columns:           # overwrite if already present
            df.drop(columns=c, inplace=True)
        df[c] = np.nan

    # iterate rows
    for idx, row in df.iterrows():
        ev     = ast.literal_eval(row["evidence"])
        sigma  = float(row["sigma"])

        rep_n, haz_n = normative_predict(ev, sigma)
        df.at[idx, "rep_norm"] = rep_n
        df.at[idx, "haz_norm"] = haz_n

        for key, mdl in bank.items():
            rep_r, haz_r = rnn_predict(mdl, ev)
            df.at[idx, f"rep_{key}"] = rep_r
            df.at[idx, f"haz_{key}"] = haz_r

    out = path.replace(".csv", "_preds.csv")
    df.to_csv(out, index=False)
    print(f"✓ {os.path.basename(out)} saved")

def main(max_variants: int = 40):
    bank = build_rnn_bank()
    if not bank:
        raise RuntimeError("No RNN checkpoints found!")

    for k in range(max_variants):
        csv = os.path.join(VAR_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv):
            break
        process_csv(csv, bank)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TaskConfig_Generator_binary.py
==============================

Simplified 300‑trial variant generator **(+ Bayesian `rep_norm` + `resp_pred`)**.
"""
from __future__ import annotations
import sys, os, copy, ast
from typing import Sequence, Dict

import numpy as np
import pandas as pd

import TaskConfig_Generator_Trial as TCG           # trial‑level generator
from NormativeModel import BayesianObserver        # Bayesian ideal observer

# ---------------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------------- #
HS_GRID = np.arange(0, 1.05, 0.05)     # hazard‑rate grid for observer
MU1, MU2 = -1, 1                       # latent means (must match trial gen)
N_TRIALS_PER_VARIANT = 300             # keep in sync with TCG

# ---------------------------------------------------------------------------- #
# Helpers – Bayesian responses for one trial
# ---------------------------------------------------------------------------- #
def _bayes_responses(evidence: Sequence[float], sigma: float) -> tuple[int, int]:
    """
    Return (rep_norm, resp_pred) for a single trial.

    • `rep_norm`  = Bayesian *report* response  (state) → −1 stay / +1 switch  
    • `resp_pred` = Bayesian *predict* response (hazard) → −1 stay / +1 switch
    """
    _, _, rep_norm, resp_pred = BayesianObserver(
        evidence, MU1, MU2, sigma, HS_GRID.copy()
    )
    return int(rep_norm), int(resp_pred)

# ---------------------------------------------------------------------------- #
# Augment DataFrame with Bayesian columns
# ---------------------------------------------------------------------------- #
def append_norm_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Run observer on every trial and add `rep_norm`, `resp_pred` columns."""
    rep_vals, pred_vals = [], []
    for _, row in df.iterrows():
        evid = row["evidence"]
        if not isinstance(evid, list):                       # CSV↔obj round‑trip
            evid = ast.literal_eval(str(evid))
        sigma = float(row.get("sigma", row.get("noise", 0)))
        rep, pred = _bayes_responses(evid, sigma)
        rep_vals.append(rep)
        pred_vals.append(pred)
    df["rep_norm"]  = rep_vals
    df["resp_pred"] = pred_vals
    return df

# ---------------------------------------------------------------------------- #
# Variant factory – one 300‑trial unsorted block
# ---------------------------------------------------------------------------- #
def generate_variant(_: int) -> pd.DataFrame:
    params: Dict = copy.deepcopy(TCG.params)
    params["nTrials"] = N_TRIALS_PER_VARIANT      # enforce 300
    trials_df = pd.DataFrame(TCG.makeBlockTrials(params))
    trials_df = append_norm_responses(trials_df)  # ★ new – adds both columns
    return trials_df

# ---------------------------------------------------------------------------- #
# Saver – write CSV under train/ or test/ > unsorted/
# ---------------------------------------------------------------------------- #
def save_variant(df: pd.DataFrame, split: str, vidx: int, root: str) -> None:
    out_dir = os.path.join(root, split)
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{split}_{vidx:02d}.csv")
    df.to_csv(fname, index=False)

# ---------------------------------------------------------------------------- #
# CLI driver
# ---------------------------------------------------------------------------- #
def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_test  = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants")
    os.makedirs(root, exist_ok=True)

    for vidx in range(n_train):
        df = generate_variant(vidx)
        save_variant(df, "train", vidx, root)
        print(f"[train {vidx:02d}] variant written – {len(df)} trials")

    for vidx in range(n_test):
        df = generate_variant(vidx)
        save_variant(df, "test", vidx, root)
        print(f"[test  {vidx:02d}] variant written – {len(df)} trials")

if __name__ == "__main__":
    main()

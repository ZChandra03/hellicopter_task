#!/usr/bin/env python3
"""
TaskConfig_Generator_stratified.py
=================================

**Simplified variant generator – no sorting, 300‑trial blocks, appends `resp_pred`.**

This script replaces the previous “stratified” version that split a 900‑trial
block into *Informative*, *Uninformative* and *Misleading* subsets.  For the new
pipeline each *variant* **is generated directly with 300 trials** using
`TaskConfig_Generator_Trial.makeBlockTrials`, and the normative Bayesian
observer’s *predict* response (`resp_pred`) is written as an additional CSV
column.

Directory layout created by default:

```
variants/
    train/unsorted/train_unsorted_00.csv  (… up to n_train‑1)
    test/unsorted/test_unsorted_00.csv   (… up to n_test‑1)
```

Each file contains exactly 300 rows (one per trial) in the same order they were
sampled – **no further sorting or category labels are applied**.

Run from the command line:

```bash
python TaskConfig_Generator_stratified.py [n_train] [n_test]
```

where the optional positional arguments default to `n_train=200` and
`n_test=20`.
"""

from __future__ import annotations

import sys, os, copy, ast
from typing import Sequence, Dict

import numpy as np
import pandas as pd

import TaskConfig_Generator_Trial as TCG  # trial‑level generator (300 trials)
from NormativeModel import BayesianObserver  # Bayesian ideal observer

# ---------------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------------- #
HS_GRID = np.arange(0, 1.05, 0.05)   # hazard‑rate grid passed to observer
MU1, MU2 = -1, 1                     # latent means (must match trial gen)
N_TRIALS_PER_VARIANT = 300           # hard‑coded size – keep in sync with TCG

# ---------------------------------------------------------------------------- #
# Helper: compute Bayesian *predict* response for one trial
# ---------------------------------------------------------------------------- #

def _predict_response(evidence: Sequence[float], sigma: float) -> int:
    """Return `resp_pred` (−1 stay / +1 switch) from BayesianObserver."""
    _, _, _, resp_pred = BayesianObserver(
        evidence, MU1, MU2, sigma, HS_GRID.copy()
    )
    return int(resp_pred)

# ---------------------------------------------------------------------------- #
# Augment DataFrame with `resp_pred`
# ---------------------------------------------------------------------------- #

def append_resp_pred(df: pd.DataFrame) -> pd.DataFrame:
    """Run observer on every row and add a `resp_pred` column."""
    preds = []
    for _, row in df.iterrows():
        evid = row["evidence"]
        if not isinstance(evid, list):
            evid = ast.literal_eval(str(evid))  # CSV‑to‑list round‑trip safety
        sigma = float(row.get("sigma", row.get("noise", 0)))
        preds.append(_predict_response(evid, sigma))
    df["resp_pred"] = preds
    return df

# ---------------------------------------------------------------------------- #
# Variant factory – one 300‑trial unsorted block
# ---------------------------------------------------------------------------- #

def generate_variant(_: int) -> pd.DataFrame:
    params: Dict = copy.deepcopy(TCG.params)
    params["nTrials"] = N_TRIALS_PER_VARIANT  # enforce 300
    trials_df = pd.DataFrame(TCG.makeBlockTrials(params))
    trials_df = append_resp_pred(trials_df)
    return trials_df

# ---------------------------------------------------------------------------- #
# Saver – write a single CSV under train/ or test/ > unsorted/
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
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_test  = int(sys.argv[2]) if len(sys.argv) > 2 else 50

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

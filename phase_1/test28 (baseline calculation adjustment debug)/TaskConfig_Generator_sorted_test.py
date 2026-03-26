#!/usr/bin/env python3
"""
TaskConfig_Generator_sorted.py
=============================

Generate **Bayesian‑tagged trial variants** for both *training* and *testing*
phases and save them in a four‑way folder structure **with normative model
outputs already embedded in every CSV.**

```
variants/
├── train/
│   ├── informative/
│   ├── uninformative/
│   ├── misleading/
│   └── unsorted/
└── test/
    ├── informative/
    ├── uninformative/
    ├── misleading/
    └── unsorted/
```

Each CSV now contains two extra columns:

* **`rep_norm`** – Bayesian state report (−1 or 1)
* **`haz_norm`** – Bayesian posterior mean hazard‑rate estimate ∈ [0,1]

Default counts are **n_train_variants = 30** and **n_test_variants = 10**, but
both can be overridden from the command line:

```bash
python TaskConfig_Generator_sorted.py            # 30 train, 10 test
python TaskConfig_Generator_sorted.py 40 12      # 40 train, 12 test
```
"""

# --------------------------------------------------------------------------- #
from __future__ import annotations

import sys, os, copy, ast
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy                     # Shannon H

import TaskConfig_Generator_Trial as TCG            # legacy trial generator
from NormativeModel import BayesianObserver         # Bayesian ideal observer

# --------------------------------------------------------------------------- #
# Shared Bayesian constants (must match the rest of the project)
HS_GRID  = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1

# --------------------------------------------------------------------------- #
# Normative evaluation helpers
# --------------------------------------------------------------------------- #

def normative_eval(evidence: List[float], sigma: float, true_hazard: float) -> Tuple[int, float, float, float]:
    """Return `(rep_norm, haz_norm, dist, H)` for a single trial."""
    L_haz, _, rep_norm, _ = BayesianObserver(
        evidence, MU1, MU2, sigma, HS_GRID.copy()
    )
    post_final = L_haz[:, -1]
    haz_norm   = float(np.dot(HS_GRID, post_final))   # posterior mean
    dist       = abs(haz_norm - true_hazard)          # distance to ground truth
    H          = float(entropy(post_final))           # posterior width
    return int(rep_norm), haz_norm, dist, H


# --------------------------------------------------------------------------- #
# Category splitter WITH normative columns attached
# --------------------------------------------------------------------------- #

def label_and_augment(df: pd.DataFrame) -> pd.DataFrame:
    """Add rep_norm / haz_norm and assign each trial to one of three categories."""
    dists, entrs = [], []
    rep_list, haz_list = [], []

    for idx, row in df.iterrows():
        evid  = row["evidence"]
        evid  = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
        sigma = float(row.get("sigma", row.get("noise")))
        h_true = float(row.get("hazard", row.get("trueHazard", row.get("hazardRate"))))

        rep_norm, haz_norm, dist, H = normative_eval(evid, sigma, h_true)

        rep_list.append(rep_norm)
        haz_list.append(haz_norm)
        dists.append(dist)
        entrs.append(H)

    # attach normative columns
    df["rep_norm"] = rep_list
    df["haz_norm"] = haz_list

    dists = np.array(dists)
    entrs = np.array(entrs)

    # select 300 most misleading by |E[h] - h_true|
    mis_idx = np.argsort(dists)[::-1][:300]

    remaining = np.setdiff1d(np.arange(len(df)), mis_idx)
    sorted_by_H = remaining[np.argsort(entrs[remaining])]   # narrow → wide
    inf_idx  = sorted_by_H[:300]
    unin_idx = sorted_by_H[300:]

    labels = np.full(len(df), "Uninformative", dtype=object)
    labels[inf_idx] = "Informative"
    labels[mis_idx] = "Misleading"
    df["category"] = labels

    return df

# --------------------------------------------------------------------------- #
# File write helpers
# --------------------------------------------------------------------------- #

def save_variant(df: pd.DataFrame, split: str, vidx: int, root: str) -> None:
    """Write one 900‑trial variant into its four CSV destinations."""
    prefix = split  # "train" or "test"

    # ensure sub‑folders exist
    for sub in ("informative", "uninformative", "misleading", "unsorted"):
        os.makedirs(os.path.join(root, split, sub), exist_ok=True)

    # 1) unsorted (complete) file
    df.to_csv(os.path.join(root, split, "unsorted", f"{prefix}_unsorted_{vidx:02d}.csv"), index=False)

    # 2) category‑specific files (300 rows each)
    for cat in ("Informative", "Uninformative", "Misleading"):
        cat_df = df[df["category"] == cat]
        path   = os.path.join(root, split, cat.lower(), f"{prefix}_{cat.lower()}_{vidx:02d}.csv")
        cat_df.to_csv(path, index=False)

# --------------------------------------------------------------------------- #
# Variant generator
# --------------------------------------------------------------------------- #

def generate_variant(vidx: int) -> pd.DataFrame:
    params: Dict = copy.deepcopy(TCG.params)
    params["nTrials"] = 900
    trials = pd.DataFrame(TCG.makeBlockTrials(params))
    trials = label_and_augment(trials)
    return trials

# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

def main():
    # defaults: 30 train, 10 test (override via CLI)
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_test  = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants")
    os.makedirs(root, exist_ok=True)

    # TRAIN variants
    for vidx in range(n_train):
        df = generate_variant(vidx)
        save_variant(df, "train", vidx, root)
        print(f"[train {vidx:02d}] variants written … {len(df)} trials")

    # TEST variants
    for vidx in range(n_test):
        df = generate_variant(vidx)
        save_variant(df, "test", vidx, root)
        print(f"[test  {vidx:02d}] variants written … {len(df)} trials")


if __name__ == "__main__":
    main()

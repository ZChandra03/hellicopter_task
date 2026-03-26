#!/usr/bin/env python3
"""
TaskConfig_Generator_Informativeness.py
======================================

Generate `n_variants` × 900 trials, tag each as Informative / Uninformative /
Misleading, and save them into THREE SEPARATE FOLDERS.

Usage
-----
    python TaskConfig_Generator_Informativeness.py        # 40 variants
    python TaskConfig_Generator_Informativeness.py 12     # 12 variants
"""
# --------------------------------------------------------------------------- #
from __future__ import annotations
import sys, os, copy, ast
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.stats import entropy                           # Shannon H

import TaskConfig_Generator_Trial as TCG                  # generator
from NormativeModel import BayesianObserver               # ideal observer

# --------------------------------------------------------------------------- #
# Constants that must match your project setup
HS_GRID  = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1
# --------------------------------------------------------------------------- #
def analyse_trial(evidence: List[float], sigma: float, true_hazard: float):
    """Return Bayesian E[h], |error|, and posterior entropy."""
    L_haz, _, _, _ = BayesianObserver(evidence, MU1, MU2, sigma, HS_GRID.copy())
    post   = L_haz[:, -1]
    h_bayes = float((HS_GRID * post).sum())
    H       = float(entropy(post))
    dist    = abs(h_bayes - true_hazard)
    return dist, H

def split_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Tag 900 trials → Informative / Uninformative / Misleading."""
    dists, entrs = [], []
    for _, row in df.iterrows():
        evid = row["evidence"]
        evid = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
        sigma = float(row.get("sigma", row.get("noise")))
        h_true = float(row.get("hazard", row.get("trueHazard",
                       row.get("hazardRate"))))
        dist, H = analyse_trial(evid, sigma, h_true)
        dists.append(dist); entrs.append(H)

    dists, entrs = np.array(dists), np.array(entrs)

    mis_idx = np.argsort(dists)[::-1][:300]                       # 300 worst
    remaining = np.setdiff1d(np.arange(len(df)), mis_idx)
    sorted_by_H = remaining[np.argsort(entrs[remaining])]         # narrow→wide
    inf_idx, unin_idx = sorted_by_H[:300], sorted_by_H[300:]

    labels = np.full(len(df), "Uninformative", dtype=object)
    labels[inf_idx] = "Informative"
    labels[mis_idx] = "Misleading"
    df["category"] = labels
    return df

def generate_variant(v: int, root: str):
    params: Dict = copy.deepcopy(TCG.params)
    params["nTrials"] = 900
    trials = pd.DataFrame(TCG.makeBlockTrials(params))
    trials = split_categories(trials)

    for cat in ["Informative", "Uninformative", "Misleading"]:
        cat_dir = os.path.join(root, cat.lower())
        os.makedirs(cat_dir, exist_ok=True)
        fn = os.path.join(cat_dir, f"train_{cat.lower()}_{v:02d}.csv")
        trials[trials["category"] == cat].to_csv(fn, index=False)
        print(f"[v{v:02d}] {fn} written")

def main():
    n_variants = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root       = os.path.join(script_dir, "variants")
    for v in range(n_variants):
        generate_variant(v, root)

if __name__ == "__main__":
    main()

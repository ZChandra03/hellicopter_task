#!/usr/bin/env python3
"""
TaskConfig_Generator_sorted_test.py
==================================

Balanced three‑way sorter **with fixed hazard distribution**
------------------------------------------------------------
This script partitions each 900‑trial variant into three 300‑trial CSVs
(*Informative*, *Uninformative*, *Misleading*) **without perturbing the a‑priori
hazard‑rate distribution** defined in **`TaskConfig_Generator_Trial.py`**.

Key idea – instead of measuring the empirical hazard counts of the freshly
sampled 900‑trial block, we import the *target* probabilities from the upstream
generator and allocate *exact integer quotas* per hazard value inside every
300‑trial subset.

The remainder of the public interface – CLI, folder tree, normative‑model
columns – is unchanged so nothing downstream needs to be edited.
"""

# --------------------------------------------------------------------------- #
from __future__ import annotations

import sys, os, copy, ast
from typing import List, Dict, Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy                     # Shannon H

import TaskConfig_Generator_Trial as TCG            # ← gives us hazard_probs
from NormativeModel import BayesianObserver         # Bayesian ideal observer

# --------------------------------------------------------------------------- #
# Shared constants
HS_GRID  = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1
N_PER_CAT = 300                                     # trials per subset

# --------------------------------------------------------------------------- #
# Pre‑compute **exact** target counts per hazard value (for a 300‑trial file)
# --------------------------------------------------------------------------- #
HAZARD_PROBS: Dict[float, float] | None = TCG.params.get("hazard_probs")
TARGET_COUNTS_PER_HAZARD: Dict[float, int] | None = None

if HAZARD_PROBS and isinstance(HAZARD_PROBS, dict):
    # keep deterministic ordering for reproducibility
    _hz, _pr = zip(*sorted(HAZARD_PROBS.items()))
    _float   = np.array(_pr) * N_PER_CAT
    _base    = np.floor(_float).astype(int)
    # distribute leftover trials to largest fractional parts
    leftover = N_PER_CAT - _base.sum()
    if leftover > 0:
        order = np.argsort(_float - _base)[::-1]
        _base[order[:leftover]] += 1
    TARGET_COUNTS_PER_HAZARD = dict(zip(_hz, _base))

# --------------------------------------------------------------------------- #
# Normative evaluation helper
# --------------------------------------------------------------------------- #

def normative_eval(evidence: Sequence[float], sigma: float, true_hazard: float) -> Tuple[int, float, float, float]:
    """Return `(rep_norm, haz_norm, dist, H)` for a single trial."""
    L_haz, _, rep_norm, _ = BayesianObserver(
        evidence, MU1, MU2, sigma, HS_GRID.copy()
    )
    post_final = L_haz[:, -1]
    haz_norm   = float(np.dot(HS_GRID, post_final))   # posterior mean
    dist       = abs(haz_norm - true_hazard)          # |E[h] − h_true|
    H          = float(entropy(post_final))           # posterior entropy
    return int(rep_norm), haz_norm, dist, H

# --------------------------------------------------------------------------- #
# Main labelling + balanced split
# --------------------------------------------------------------------------- #

def label_and_augment(df: pd.DataFrame) -> pd.DataFrame:
    """Attach normative columns **and** balanced category labels."""
    # --- 1. run Bayesian observer on every trial ------------------------------------
    rep_norm, haz_norm, dist_arr, entr_arr, haz_true = [], [], [], [], []

    for _, row in df.iterrows():
        evid    = row["evidence"]
        evid    = evid if isinstance(evid, list) else ast.literal_eval(str(evid))
        sigma   = float(row.get("sigma", row.get("noise")))
        h_true  = float(row.get("trueHazard", row.get("hazard", row.get("hazardRate"))))

        r, h_est, d, H = normative_eval(evid, sigma, h_true)
        rep_norm.append(r)
        haz_norm.append(h_est)
        dist_arr.append(d)
        entr_arr.append(H)
        haz_true.append(h_true)

    df["rep_norm"] = rep_norm
    df["haz_norm"] = haz_norm

    dist_arr = np.asarray(dist_arr)
    entr_arr = np.asarray(entr_arr)
    haz_true = np.asarray(haz_true)

    # --- 2. decide per‑hazard quotas -------------------------------------------------
    if TARGET_COUNTS_PER_HAZARD is not None:
        target_counts = TARGET_COUNTS_PER_HAZARD
    else:
        # fall back: preserve empirical distribution of this 900‑trial block
        unique_haz, haz_counts = np.unique(haz_true, return_counts=True)
        target_counts = {h: int(round(c * N_PER_CAT / len(df))) for h, c in zip(unique_haz, haz_counts)}
        # adjust rounding
        delta = N_PER_CAT - sum(target_counts.values())
        if delta != 0:
            target_counts[unique_haz[0]] += delta

    # label array (default uninformative)
    labels = np.full(len(df), "Uninformative", dtype=object)

    mis_pool, inf_pool = set(), set()

    # --- 3. per‑hazard allocation ---------------------------------------------------
    for h_val, q in target_counts.items():
        idx_h = np.where(haz_true == h_val)[0]
        if not len(idx_h):
            continue  # this hazard absent in block (unlikely)

        # Misleading ⇒ largest |E[haz] − h_true|
        order_dist = idx_h[np.argsort(dist_arr[idx_h])[::-1]]
        mis_take   = min(q, len(order_dist))
        mis_idx    = order_dist[:mis_take]
        mis_pool.update(mis_idx)

        # Informative ⇒ smallest posterior entropy among remaining trials
        remaining  = np.setdiff1d(idx_h, mis_idx, assume_unique=True)
        order_H    = remaining[np.argsort(entr_arr[remaining])]
        inf_take   = min(q, len(order_H))
        inf_idx    = order_H[:inf_take]
        inf_pool.update(inf_idx)
        # leftover of this hazard automatically stays Uninformative

    # --- 4. safety top‑ups to hit exactly 300/300 ------------------------------------
    def _top_up(pool: set, goal: int, key_scores: np.ndarray, reverse: bool, banned: set) -> set:
        if len(pool) >= goal:
            return pool
        need      = goal - len(pool)
        leftovers = np.setdiff1d(np.arange(len(df)), np.concatenate([list(pool), list(banned)]))
        order     = leftovers[np.argsort(key_scores[leftovers])[::-1 if reverse else 1]]
        pool.update(order[:need])
        return pool

    mis_pool = _top_up(mis_pool, N_PER_CAT, dist_arr,  True,  inf_pool)
    inf_pool = _top_up(inf_pool, N_PER_CAT, entr_arr, False, mis_pool)

    # --- 5. write labels -------------------------------------------------------------
    labels[list(mis_pool)] = "Misleading"
    labels[list(inf_pool)] = "Informative"
    df["category"] = labels

    # Sanity checks
    assert (df["category"] == "Misleading").sum() == N_PER_CAT
    assert (df["category"] == "Informative").sum() == N_PER_CAT

    return df

# --------------------------------------------------------------------------- #
# File write helpers (unchanged)
# --------------------------------------------------------------------------- #

def save_variant(df: pd.DataFrame, split: str, vidx: int, root: str) -> None:
    prefix = split  # "train" or "test"
    for sub in ("informative", "uninformative", "misleading", "unsorted"):
        os.makedirs(os.path.join(root, split, sub), exist_ok=True)

    # unsorted full 900‑trial CSV
    df.to_csv(os.path.join(root, split, "unsorted", f"{prefix}_unsorted_{vidx:02d}.csv"), index=False)

    for cat in ("Informative", "Uninformative", "Misleading"):
        cat_df = df[df["category"] == cat]
        path   = os.path.join(root, split, cat.lower(), f"{prefix}_{cat.lower()}_{vidx:02d}.csv")
        cat_df.to_csv(path, index=False)

# --------------------------------------------------------------------------- #
# Variant factory (unchanged generation – still 900 trials per variant)
# --------------------------------------------------------------------------- #

def generate_variant(vidx: int) -> pd.DataFrame:
    params: Dict = copy.deepcopy(TCG.params)
    params["nTrials"] = 900
    trials = pd.DataFrame(TCG.makeBlockTrials(params))
    trials = label_and_augment(trials)
    return trials

# --------------------------------------------------------------------------- #
# CLI driver
# --------------------------------------------------------------------------- #

def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_test  = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants")
    os.makedirs(root, exist_ok=True)

    for vidx in range(n_train):
        df = generate_variant(vidx)
        save_variant(df, "train", vidx, root)
        print(f"[train {vidx:02d}] variants written … {len(df)} trials")

    for vidx in range(n_test):
        df = generate_variant(vidx)
        save_variant(df, "test", vidx, root)
        print(f"[test  {vidx:02d}] variants written … {len(df)} trials")


if __name__ == "__main__":
    main()

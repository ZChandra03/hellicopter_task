#!/usr/bin/env python3
# add_predictions.py -----------------------------------------------------------
# Append report + hazard predictions from the Bayesian normative model
# to every test CSV produced by TaskConfig_Generator.py
# ------------------------------------------------------------------------------

import os, ast
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from NormativeModel import BayesianObserver   # local module

# ───────────────────────── constants ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VAR_DIR    = os.path.join(BASE_DIR, "variants")      # where testConfig_var*.csv live
HS_GRID    = np.arange(0, 1.05, 0.05)              # 0 … 1, step 0.005
MU1, MU2   = -1, 1                                   # location means for the observer

# ───────────────────────── helpers ────────────────────────────────────────────
def normative_predict(evidence, sigma) -> Tuple[int, float]:
    """
    Run the Bayesian observer and return:
        rep  ∈ {−1, 1}  – site report (state)
        haz  ∈ [0, 1]   – posterior mean hazard estimate
    """
    L_haz, _, rep, _ = BayesianObserver(
        evidence, MU1, MU2, sigma, HS_GRID.copy()
    )
    haz = float(np.dot(HS_GRID, L_haz[:, -1]))   # posterior mean over hazard grid
    return int(rep), haz

# ───────────────────────── core ───────────────────────────────────────────────
def process_csv(path: str) -> None:
    df = pd.read_csv(path)

    # ensure columns exist (overwrite if already present)
    for col in ("rep_norm", "haz_norm"):
        if col in df.columns:
            df.drop(columns=col, inplace=True)
        df[col] = np.nan

    # iterate rows
    for idx, row in df.iterrows():
        evidence = ast.literal_eval(row["evidence"])
        sigma    = float(row["sigma"])

        rep, haz = normative_predict(evidence, sigma)
        df.at[idx, "rep_norm"] = rep
        df.at[idx, "haz_norm"] = haz

    out_path = path.replace(".csv", "_bayes.csv")
    df.to_csv(out_path, index=False)
    print(f"✓ {os.path.basename(out_path)} saved")

def main(max_variants: int = 40):
    for k in range(max_variants):
        csv_path = os.path.join(VAR_DIR, f"trainConfig_var{k}.csv")
        if not os.path.isfile(csv_path):   # stop when variants run out
            break
        process_csv(csv_path)

if __name__ == "__main__":
    main()

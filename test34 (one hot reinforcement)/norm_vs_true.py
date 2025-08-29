#!/usr/bin/env python3
"""
plot_haz_bias.py
----------------
Visualise the bias  haz_norm – trueHazard  on the first 10 *unsorted*
test‑set CSVs.

* Assumes the repo layout used by `evaluate_3_max.py`.
* Produces a histogram and prints mean / std / quartiles to the console.
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── config ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VARIANT_ROOT  = os.path.join(BASE_DIR, "variants", "test", "unsorted")
MAX_CSVS      = 10                          # first N CSVs to inspect
BINS          = 40                          # histogram resolution

# ── find the CSVs (same rule as evaluate_3_max) ─────────────────────────
csv_paths = sorted(glob.glob(os.path.join(VARIANT_ROOT,
                                          "test_unsorted_*.csv")))[:MAX_CSVS]
if not csv_paths:
    raise FileNotFoundError(f"No test CSVs found under {VARIANT_ROOT}")

# ── load & concatenate ──────────────────────────────────────────────────
df_list = [pd.read_csv(p) for p in csv_paths]
df      = pd.concat(df_list, ignore_index=True)

# guard against column‑name drift
if not {"haz_norm", "trueHazard"} <= set(df.columns):
    raise KeyError("Expected columns 'haz_norm' and 'trueHazard' not found.")

# ── compute bias ────────────────────────────────────────────────────────
bias = df["haz_norm"] - df["trueHazard"]

# ── plot ────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.hist(bias, bins=BINS, edgecolor='black', alpha=0.8)
plt.axvline(bias.mean(), ls="--", lw=2, label=f"mean = {bias.mean():+.3f}")
plt.title("Distribution of haz_norm – trueHazard\n(first 10 unsorted test CSVs)")
plt.xlabel("haz_norm – trueHazard")
plt.ylabel("Count")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# ── quick stats ─────────────────────────────────────────────────────────
print(f"N = {len(bias):,}")
print(f"Mean  : {bias.mean():+.4f}")
print(f"Std   : {bias.std():.4f}")
print("Quartiles:")
for q in (0.25, 0.50, 0.75):
    print(f"  {int(q*100):>2}% : {bias.quantile(q):+.4f}")

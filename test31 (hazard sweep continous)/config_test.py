#!/usr/bin/env python3
"""
hazard_dist_ALL_exact.py
────────────────────────
Counts how many trials landed on each *exact* hazard value (0.00, 0.05, …, 1.00)
and plots the empirical probability **mass** against the analytic Beta mass for
the corresponding 0.05‑wide cells.
"""

import glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

# ─────────────── settings ───────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR    = os.path.join(SCRIPT_DIR, "variants")

FILE_PATTERN = "trainConfig_*.csv"
ALPHA, BETA  = 0.1, 0.1            # Beta parameters
RES          = 0.05                # hazard grid step
GRID         = np.round(np.arange(0, 1 + RES/2, RES), 2)  # 0.00 … 1.00
OUT_PNG      = os.path.join(CFG_DIR, "hazard_mass_ALL.png")
# ─────────────────────────────────────────

# 1) gather hazards
csv_paths = sorted(glob.glob(os.path.join(CFG_DIR, FILE_PATTERN)))
if not csv_paths:
    raise FileNotFoundError(f"No {FILE_PATTERN} in {CFG_DIR}")

hazards = pd.concat(
    (pd.read_csv(p, usecols=["trueHazard"]) for p in csv_paths),
    ignore_index=True)["trueHazard"].to_numpy()
N = len(hazards)

# 2) empirical probability MASS at each exact grid value
counts = np.array([np.sum(hazards == g) for g in GRID])
emp_mass = counts / N                 # P(h = g)

# 3) analytic Beta MASS for the same cells
#    cell edges: left = g-Δ/2 (clamped to 0), right = g+Δ/2 (clamped to 1)
left_edges  = np.maximum(GRID - RES/2, 0)
right_edges = np.minimum(GRID + RES/2, 1)
beta_mass = beta.cdf(right_edges, ALPHA, BETA) - beta.cdf(left_edges, ALPHA, BETA)

# 4) plot MASS comparison
plt.figure(figsize=(7, 4.5))
plt.bar(GRID, emp_mass, width=RES*0.9, align="center",
        edgecolor="black", alpha=0.7, label="Empirical mass")
plt.plot(GRID, beta_mass, "k--", lw=2, label=f"Beta({ALPHA:.0f},{BETA:.0f}) mass")

plt.title("Probability mass at each exact hazard value (all train configs)")
plt.xlabel("Hazard h (grid values)")
plt.ylabel("Probability mass")
plt.xlim(-0.025, 1.025)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[✓] Saved {OUT_PNG}   ({N} hazards from {len(csv_paths)} configs)")

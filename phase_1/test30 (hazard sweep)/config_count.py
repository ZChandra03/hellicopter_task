#!/usr/bin/env python3
"""
hazard_rate_counts.py
─────────────────────
Print **per‑Beta counts** of exact hazard values found in the training CSVs
under `variants/`.   For every folder that matches `beta_*` we report how many
trials fall on each 0.05‑spaced grid point (0.00 → 1.00) plus a *None* bucket
for off‑grid / missing entries.

### Quick sampling
Set `MAX_CSVS` (integer) to limit how many CSV files **per Beta folder** are
loaded.  `None` ⇒ load everything.  A deterministic shuffle keeps runs stable.

Example (truncated):
```text
=== beta_0p1 (250 CSVs) ===
h = 0.00 :   3,000
h = 0.05 :   7,502
  …
h = None :       0

=== beta_0p5 (250 CSVs) ===
  …
[✓] Processed 5 Beta folders (total CSVs = 1,250, trials = 375,000)
```
"""
from __future__ import annotations

import os, glob, random
from typing import List, Dict

import numpy as np
import pandas as pd

# ─────────────── settings ───────────────
HERE          = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR  = os.path.join(HERE, "variants")
BETA_PATTERN  = os.path.join(VARIANTS_DIR, "beta_*")
GRID_STEP     = 0.05
GRID_VALS     = np.round(np.arange(0, 1 + GRID_STEP/2, GRID_STEP), 2)  # 0.00‑1.00

# Limit number of CSVs **per‑Beta** (None → no cap)
MAX_CSVS: int = 10     # e.g. 250 for a quick check
RNG_SEED: int         = 42

# ─────────────── locate Beta dirs ───────
beta_dirs: List[str] = sorted(d for d in glob.glob(BETA_PATTERN) if os.path.isdir(d))
if not beta_dirs:
    raise FileNotFoundError(f"No beta_* directories under {VARIANTS_DIR}")

grand_total_csvs = 0
grand_total_trials = 0

for beta_dir in beta_dirs:
    beta_key = os.path.basename(beta_dir)  # e.g. beta_0p1
    csv_paths: List[str] = sorted(glob.glob(os.path.join(beta_dir, "trainConfig_*.csv")))
    if not csv_paths:
        print(f"[!] {beta_key}: no trainConfig_*.csv files found – skipping")
        continue

    # optional sampling
    if MAX_CSVS is not None and MAX_CSVS < len(csv_paths):
        random.Random(RNG_SEED).shuffle(csv_paths)
        csv_paths = csv_paths[:MAX_CSVS]
        sample_msg = f" (sampled {MAX_CSVS} of {len(csv_paths)} CSVs)"
    else:
        sample_msg = f" ({len(csv_paths)} CSVs)"

    # load hazard column
    hazards = pd.concat(
        (pd.read_csv(p, usecols=["trueHazard"]) for p in csv_paths),
        ignore_index=True
    )["trueHazard"].to_numpy()

    # count occurrences on the 0.05 grid
    counts: Dict[float, int] = {float(g): int(np.sum(hazards == g)) for g in GRID_VALS}
    mask_matched = np.isin(hazards, GRID_VALS)
    counts_none = int(np.sum(~mask_matched))

    width = max(len(f"{int(v):,}") for v in (*counts.values(), counts_none))

    print(f"\n=== {beta_key}{sample_msg} ===")
    for g in GRID_VALS:
        print(f"h = {g:4.2f} : {counts[g]:{width},d}")
    print(f"h = None : {counts_none:{width},d}")

    grand_total_csvs   += len(csv_paths)
    grand_total_trials += hazards.size

print(f"\n[✓] Processed {len(beta_dirs)} Beta folders (total CSVs = {grand_total_csvs:,}, "
      f"trials = {grand_total_trials:,})")

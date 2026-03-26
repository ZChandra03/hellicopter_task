#!/usr/bin/env python3
"""
baseline_plot.py – **Revision 2**
Visualise baseline‑loss curves for the eight model types, averaged over the
first 10 seeds (0‑9), given the *new* folder layout:
```
models/<type_key>/seed_<n>/baseline_loss_history.json
```
* Thin translucent lines  → individual seeds.
* Thick solid line        → mean across seeds.
"""

from __future__ import annotations
import os, re, json
from collections import defaultdict
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

# ───── configuration ────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(HERE, "models")
OUTFIG      = os.path.join(HERE, "baseline_loss_over_time.png")

SEED_RE     = re.compile(r"seed_(\d+)")
MAX_SEEDS   = 3  # plot seeds 0‑9
COLOR_CYCLE = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

MODEL_TYPES = ["truth", "norm"]

# ───── gather baseline histories ────────────────────────────────────────
base_hist: dict[str, dict[int, list[float]]] = defaultdict(dict)

for type_key in MODEL_TYPES:
    type_dir = os.path.join(MODELS_DIR, type_key)
    if not os.path.isdir(type_dir):
        continue

    for seed_dir in os.listdir(type_dir):
        m = SEED_RE.fullmatch(seed_dir)
        if not m:
            continue
        seed = int(m.group(1))
        if seed >= MAX_SEEDS:
            continue

        bl_path = os.path.join(type_dir, seed_dir, "baseline_loss_history.json")
        if not os.path.exists(bl_path):
            continue

        with open(bl_path) as f:
            losses = json.load(f)
        base_hist[type_key][seed] = losses

# Sanity check ----------------------------------------------------------
if not any(base_hist.values()):
    raise RuntimeError(f"No baseline histories found under {MODELS_DIR} – check paths!")

# ───── plotting ─────────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))

for type_key in MODEL_TYPES:
    if type_key not in base_hist or not base_hist[type_key]:
        continue

    color = next(COLOR_CYCLE)

    # individual seeds
    for losses in base_hist[type_key].values():
        plt.plot(range(0, len(losses) * 5, 5),  # x‑axis in epochs (every 5)
                 losses,
                 color=color, linewidth=0.8, alpha=0.3)

    # mean curve
    max_len = max(len(lst) for lst in base_hist[type_key].values())
    stacked = np.full((len(base_hist[type_key]), max_len), np.nan)
    for i, losses in enumerate(base_hist[type_key].values()):
        stacked[i, :len(losses)] = losses
    mean_curve = np.nanmean(stacked, axis=0)
    plt.plot(range(0, max_len * 5, 5), mean_curve,
             color=color, linewidth=2.5, label=type_key)

plt.xlabel("Epoch (5‑epoch resolution)")
plt.ylabel("Baseline Loss")
plt.title("Baseline Loss vs Epoch – mean of first 10 seeds")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(OUTFIG, dpi=300)
print(f"Figure saved to {OUTFIG}")

plt.show()

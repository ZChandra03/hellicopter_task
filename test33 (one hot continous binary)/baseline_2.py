#!/usr/bin/env python3
"""
loss_plot.py — Beta‑sweep edition (July 2025)
===========================================
Plots **baseline‑probe loss** (solid) and **training loss** sampled every
5 epochs (dashed) for each Beta‑prior setting.

Directory layout expected ::

    models/<beta_key>/seed_<n>/
        ├── baseline_loss_history.json   ← probe on held‑out Beta 1.0 set
        └── loss_history.json            ← per‑epoch training loss

where ``<beta_key>`` starts with ``"beta_"`` (e.g. *beta_0p1*, *beta_1p0*, …)
produced by *train_batch_snapshot.py*.

Run this script from the repo root **after** training; it will auto‑discover
all Beta folders under *models/* and aggregate across seeds.
"""
from __future__ import annotations

import os, re, json
from collections import defaultdict
from itertools import cycle
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(HERE, "models")
OUTFIG      = os.path.join(HERE, "loss_over_time_beta.png")

SEED_RE     = re.compile(r"seed_(\d+)")
MAX_SEEDS   = 6                                # seeds 0‑9
COLOR_CYCLE = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# auto‑discover every sub‑folder that looks like a Beta‑prior run
MODEL_TYPES: List[str] = sorted(
    d for d in os.listdir(MODELS_DIR)
    if d.startswith("beta_") and os.path.isdir(os.path.join(MODELS_DIR, d))
)
if not MODEL_TYPES:
    raise RuntimeError(f"No beta_* folders found under {MODELS_DIR}")

# ── gather loss histories ────────────────────────────────────────────────
# base_hist[type][seed]  → List[float]
base_hist:  Dict[str, Dict[int, List[float]]] = defaultdict(dict)
train_hist: Dict[str, Dict[int, List[float]]] = defaultdict(dict)

for t_key in MODEL_TYPES:
    t_dir = os.path.join(MODELS_DIR, t_key)

    for s_dir in os.listdir(t_dir):
        m = SEED_RE.fullmatch(s_dir)
        if not m:
            continue
        seed = int(m.group(1))
        if seed >= MAX_SEEDS:
            continue

        bl_path   = os.path.join(t_dir, s_dir, "baseline_loss_history.json")
        train_path= os.path.join(t_dir, s_dir, "loss_history.json")

        if os.path.exists(bl_path):
            base_hist[t_key][seed] = json.load(open(bl_path))
        if os.path.exists(train_path):
            full = json.load(open(train_path))
            train_hist[t_key][seed] = full[::5]                 # sample every 5 epochs

# sanity guard
if not any(base_hist.values()):
    raise RuntimeError(f"No loss histories found under {MODELS_DIR}")

# ── plotting ─────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))

for t_key in MODEL_TYPES:
    if t_key not in base_hist:
        continue
    color = next(COLOR_CYCLE)

    # ─ baseline probes (solid) ────────────────────────────────────────
    for losses in base_hist[t_key].values():                     # thin lines per seed
        plt.plot(range(0, len(losses)*5, 5), losses,
                 color=color, alpha=.30, linewidth=.8)
    # mean over seeds
    max_len = max(map(len, base_hist[t_key].values()))
    stack   = np.full((len(base_hist[t_key]), max_len), np.nan)
    for i, l in enumerate(base_hist[t_key].values()):
        stack[i, :len(l)] = l
    plt.plot(range(0, max_len*5, 5), np.nanmean(stack, 0),
             color=color, linewidth=2.5, label=f"{t_key} baseline")

    # ─ training loss (dashed) ─────────────────────────────────────────
    if t_key not in train_hist:
        continue
    for losses in train_hist[t_key].values():                    # thin dashed per seed
        plt.plot(range(0, len(losses)*5, 5), losses,
                 color=color, alpha=.30, linewidth=.8, linestyle="--")
    max_len_tr = max(map(len, train_hist[t_key].values()))
    stack_tr   = np.full((len(train_hist[t_key]), max_len_tr), np.nan)
    for i, l in enumerate(train_hist[t_key].values()):
        stack_tr[i, :len(l)] = l
    plt.plot(range(0, max_len_tr*5, 5), np.nanmean(stack_tr, 0),
             color=color, linewidth=2.5, linestyle="--", label=f"{t_key} train")

plt.xlabel("Epoch (5‑epoch resolution)")
plt.ylabel("Loss (BCE + 0.5·BCE)")
plt.title("Training vs Baseline‑probe Loss across Beta priors")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(OUTFIG, dpi=300)
print(f"Figure saved to {OUTFIG}")
plt.show()

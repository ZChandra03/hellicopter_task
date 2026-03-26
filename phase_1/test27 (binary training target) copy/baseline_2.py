#!/usr/bin/env python3
"""
loss_plot.py – Revision 3
Baseline loss   (solid)  +  Training loss every 5 epochs (dashed)
-----------------------------------------------------------------
Directory layout expected:
    models/<type_key>/seed_<n>/{baseline_loss_history.json, loss_history.json}
"""
from __future__ import annotations
import os, re, json
from collections import defaultdict
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

# ── config ───────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(HERE, "models")
OUTFIG      = os.path.join(HERE, "loss_over_time.png")

SEED_RE     = re.compile(r"seed_(\d+)")
MAX_SEEDS   = 10                                # seeds 0‑9
COLOR_CYCLE = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

MODEL_TYPES = [
    "truth", "norm",
]

# ── gather loss histories ────────────────────────────────────────────────────
base_hist:  dict[str, dict[int, list[float]]] = defaultdict(dict)
train_hist: dict[str, dict[int, list[float]]] = defaultdict(dict)

for t_key in MODEL_TYPES:
    t_dir = os.path.join(MODELS_DIR, t_key)
    if not os.path.isdir(t_dir):
        continue

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
            train_hist[t_key][seed] = full[::5]                 # every 5 epochs

# sanity guard
if not any(base_hist.values()):
    raise RuntimeError(f"No histories found under {MODELS_DIR}")

# ── plotting ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))

for t_key in MODEL_TYPES:
    if t_key not in base_hist:
        continue
    color = next(COLOR_CYCLE)

    # ─ baseline ────────────────────────────────────────────────────────────
    for losses in base_hist[t_key].values():                     # thin lines
        plt.plot(range(0, len(losses)*5, 5), losses,
                 color=color, alpha=.30, linewidth=.8)
    # mean
    max_len = max(map(len, base_hist[t_key].values()))
    stack   = np.full((len(base_hist[t_key]), max_len), np.nan)
    for i,l in enumerate(base_hist[t_key].values()):
        stack[i,:len(l)] = l
    plt.plot(range(0, max_len*5, 5), np.nanmean(stack,0),
             color=color, linewidth=2.5, label=f"{t_key} baseline")

    # ─ training ────────────────────────────────────────────────────────────
    if t_key not in train_hist:
        continue
    for losses in train_hist[t_key].values():                    # thin dashed
        plt.plot(range(0, len(losses)*5, 5), losses,
                 color=color, alpha=.30, linewidth=.8, linestyle="--")
    max_len_tr = max(map(len, train_hist[t_key].values()))
    stack_tr   = np.full((len(train_hist[t_key]), max_len_tr), np.nan)
    for i,l in enumerate(train_hist[t_key].values()):
        stack_tr[i,:len(l)] = l
    plt.plot(range(0, max_len_tr*5, 5), np.nanmean(stack_tr,0),
             color=color, linewidth=2.5, linestyle="--",
             label=f"{t_key} train")

plt.xlabel("Epoch (5‑epoch resolution)")
plt.ylabel("Loss")
plt.title("Baseline vs Training Loss (every 5 epochs)")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(OUTFIG, dpi=300)
print(f"Figure saved to {OUTFIG}")
plt.show()

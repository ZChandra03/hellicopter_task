#!/usr/bin/env python3
"""
plot_losses.py – visualise training loss for the first 16 seeds of every model type,
                 colouring by supervision family (“norm” vs “truth”).
"""

from __future__ import annotations
import os, re, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ───── config ────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(HERE, "models")      # adjust if your models live elsewhere
OUTFIG      = os.path.join(HERE, "loss_over_time.png")

FOLDER_RE   = re.compile(r"^(?P<tag>.+?)_s(?P<seed>\d+)$")   # <base>_s#
MAX_SEEDS   = 1                                             # first 16 seeds
COLORS      = {"norm": "tab:blue",
               "truth": "tab:orange",
               "other": "tab:green"}       # fallback colour



# ───── helpers ───────────────────────────────────────────────────────────
def family_of(tag: str) -> str:
    """Return 'norm', 'truth', or 'other' based on the tag name."""
    if "_norm" in tag:
        return "norm"
    if "_truth" in tag:
        return "truth"
    return "other"



# ───── gather loss histories ────────────────────────────────────────────
loss_hist: dict[str, dict[int, list[float]]] = defaultdict(dict)

for dirpath, _, filenames in os.walk(MODELS_DIR):
    if "loss_history.json" not in filenames:
        continue

    folder = os.path.basename(dirpath)
    m = FOLDER_RE.match(folder)
    if not m:
        continue

    tag  = m.group("tag")
    seed = int(m.group("seed"))
    if seed >= MAX_SEEDS:            # keep only seeds 0-15
        continue

    with open(os.path.join(dirpath, "loss_history.json")) as f:
        losses = json.load(f)
    loss_hist[tag][seed] = losses



# ───── sanity check ─────────────────────────────────────────────────────
if not loss_hist:
    raise RuntimeError(f"No loss_history.json files found under {MODELS_DIR}")



# ───── plotting ─────────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))

for tag, seed_dict in sorted(loss_hist.items()):
    fam   = family_of(tag)
    color = COLORS[fam]

    # thin individual curves
    for losses in seed_dict.values():
        plt.plot(range(len(losses)), losses,
                 color=color, linewidth=0.8, alpha=0.3)

    # thick mean curve
    max_len  = max(len(L) for L in seed_dict.values())
    stacked  = np.full((len(seed_dict), max_len), np.nan)
    for i, losses in enumerate(seed_dict.values()):
        stacked[i, :len(losses)] = losses
    mean_curve = np.nanmean(stacked, axis=0)

    plt.plot(range(len(mean_curve)), mean_curve,
             color=color, linewidth=2.5,
             label=f"{tag} (mean)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss – first 16 seeds\nnorm vs truth colour-coded")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUTFIG, dpi=300)
print(f"Figure saved to {OUTFIG}")

plt.show()

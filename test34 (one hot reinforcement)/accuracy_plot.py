#!/usr/bin/env python3
"""
accuracy_vs_epoch.py – v2 (July 2025)
=====================================
Plot **classification accuracy** of each trained network’s two heads as a
function of training epoch:

* **Report / state head**  → figure ``accuracy_rep_vs_epoch.png``
* **Hazard / predict head**→ figure ``accuracy_haz_vs_epoch.png``

The script keeps exactly the same folder conventions, epoch filter, colour
mapping, and seed‑averaging logic found in *pearson_plot.py* so the figures
remain directly comparable.

Usage
-----
    python accuracy_vs_epoch.py            # writes two png files next to the script

Hard requirements
-----------------
* ``evaluate_3_max.py`` must be importable from the working directory because
  its helper functions (`_list_test_csvs`, `MAX_TEST_CSVS`, `DEVICE`, and the
  ``GRUModel`` definition) are reused verbatim.
* PyTorch ≥ 2.1, pandas, NumPy, and matplotlib must be installed.
"""
from __future__ import annotations

import ast
import glob
import os
import re
from collections import defaultdict
from itertools import cycle
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ───────────────────────── internal imports ──────────────────────────────
import evaluate_3_max as ev                     # helper funcs + GRUModel

# ─────────────────────────── configuration ───────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(HERE, "models")
OUT_REP     = os.path.join(HERE, "accuracy_rep_vs_epoch.png")
OUT_HAZ     = os.path.join(HERE, "accuracy_haz_vs_epoch.png")

MODEL_TYPES = [
    "inf_truth", "inf_norm",
    "unin_truth", "unin_norm",
    "mis_truth",  "mis_norm",
    "uns_truth",  "uns_norm",
]

MAX_SEEDS   = 1                               # quick‑pass; raise for full sweep
CKPT_RE     = re.compile(r"checkpoint_ep(\d{3})\.pt")
COLOR_CYCLE = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# filter to the exact epochs requested by the user (matches pearson_plot v2)
# EPOCH_FILTER: Set[int] = {
#     20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 500
# }

EPOCH_FILTER: Set[int] = {
    20, 100, 200
}

# ───────────────────────── helper: compute accuracies ────────────────────

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid for float32 tensors (scalar return)."""
    return 0.5 * torch.tanh(0.5 * x) + 0.5

@torch.inference_mode()
def accuracy_by_head(ckpt_path: str) -> Tuple[float, float]:
    """Return ``(acc_rep, acc_haz)`` for *one* checkpoint on unsorted‑test CSVs."""
    # quick‑load model (no gradient tracking)
    hp = {"n_input": 1, "n_rnn": 128}
    model = ev.GRUModel(hp).to(ev.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=ev.DEVICE))
    model.eval()

    # evaluation set: unsorted test variants (same as baseline probes)
    csvs = ev._list_test_csvs(ev.MAX_TEST_CSVS)

    rep_hit = haz_hit = total = 0
    for csv in csvs:
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            x = torch.tensor(evid, dtype=torch.float32, device=ev.DEVICE).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

            loc_logits, haz_logits = model(x)                # (1, T, 1), (1, 1)
            rep_pred = -1 if _sigmoid(loc_logits[0, -1, 0]).item() < 0.5 else 1
            haz_pred = -1 if _sigmoid(haz_logits).item()      < 0.5 else 1

            if rep_pred == int(row["trueReport"]):   rep_hit += 1
            if haz_pred == int(row["truePredict"]):  haz_hit += 1
            total += 1

    if total == 0:
        return float("nan"), float("nan")
    return rep_hit / total, haz_hit / total

# ───────────────────────── gather accuracies ─────────────────────────────
print("[+] Scanning checkpoints …")

data_rep: Dict[str, Dict[int, List[Tuple[int, float]]]] = defaultdict(dict)
data_haz: Dict[str, Dict[int, List[Tuple[int, float]]]] = defaultdict(dict)

for t_key in MODEL_TYPES:
    t_dir = os.path.join(MODELS_DIR, t_key)
    if not os.path.isdir(t_dir):
        print(f"[skip] {t_dir} not found")
        continue

    for s_dir in os.listdir(t_dir):
        m_seed = re.fullmatch(r"seed_(\d+)", s_dir)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))
        if seed >= MAX_SEEDS:
            continue

        ckpts = sorted(glob.glob(os.path.join(t_dir, s_dir, "checkpoint_ep*.pt")))
        if not ckpts:
            continue

        hist_rep: List[Tuple[int, float]] = []
        hist_haz: List[Tuple[int, float]] = []
        for ck in ckpts:
            m_ck = CKPT_RE.search(os.path.basename(ck))
            if not m_ck:
                continue
            epoch = int(m_ck.group(1))
            if epoch not in EPOCH_FILTER:
                continue

            acc_rep, acc_haz = accuracy_by_head(ck)
            hist_rep.append((epoch, acc_rep))
            hist_haz.append((epoch, acc_haz))
            print(f"  {t_key}|seed{seed} ep{epoch:03d} rep={acc_rep:.2%} haz={acc_haz:.2%}")

        if hist_rep:
            data_rep[t_key][seed] = sorted(hist_rep, key=lambda x: x[0])
        if hist_haz:
            data_haz[t_key][seed] = sorted(hist_haz, key=lambda x: x[0])

if not any(data_rep.values()) and not any(data_haz.values()):
    raise RuntimeError("No accuracy histories found – ensure checkpoints + CSVs exist.")

# ───────────────────────── plotting helper ───────────────────────────────

def _plot_head(data: Dict[str, Dict[int, List[Tuple[int, float]]]], ylabel: str, out_path: str) -> None:
    plt.figure(figsize=(12, 7))
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for t_key in MODEL_TYPES:
        if t_key not in data:
            continue
        color = next(color_cycle)

        # thin dashed – individual seeds
        for seed, pairs in data[t_key].items():
            if not pairs:
                continue
            ep, acc = zip(*pairs)
            plt.plot(ep, acc, color=color, alpha=0.3, linewidth=.8, linestyle="--")

        # bold solid – mean curve
        all_ep = sorted(EPOCH_FILTER)
        stack = np.full((len(data[t_key]), len(all_ep)), np.nan)
        for i, pairs in enumerate(data[t_key].values()):
            idx = {e: j for j, e in enumerate(all_ep)}
            for e, a in pairs:
                stack[i, idx[e]] = a
        mean_acc = np.nanmean(stack, axis=0)
        plt.plot(all_ep, mean_acc, color=color, linewidth=2.5, label=t_key)

    plt.axhline(0.5, color="k", linewidth=.6, alpha=.5, linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epoch (filtered)")
    plt.ylim(0, 1)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[+] Figure saved to {out_path}")
    plt.close()

# ───────────────────────── render both heads ─────────────────────────────
print("[+] Plotting report‑head accuracy …")
_plot_head(data_rep, "Report‑head accuracy", OUT_REP)

print("[+] Plotting hazard‑head accuracy …")
_plot_head(data_haz, "Hazard‑head accuracy", OUT_HAZ)

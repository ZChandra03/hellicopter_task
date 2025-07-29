#!/usr/bin/env python3
"""
correlation_vs_epoch.py – v2 (July 2025)
=======================================
Plot **Pearson correlation** between each trained network’s hazard‑prediction
errors and the Bayesian normative model **as a function of training epoch**.

🔄 **Change log v2** – Only the following **specific checkpoints** are sampled
and plotted (others are ignored):

```text
20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 500
```

Everything else (styling, CLI‑less execution, folder layout) remains identical
to *v1*.

Usage
-----
    python correlation_vs_epoch.py              # saves correlation_vs_epoch.png

Requirements
------------
* `evaluate_3_max.py` **must** be importable from the same directory because we
  reuse its helper functions (`_evaluate_csv`, `_list_test_csvs`, etc.).
* PyTorch and matplotlib must be installed.
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from itertools import cycle
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import torch

# ───────────────────────── internal imports ──────────────────────────────
import evaluate_3_max as ev  # relies on helper funcs + GRUModel definition

# ─────────────────────────── configuration ───────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(HERE, "models")
OUTFIG      = os.path.join(HERE, "correlation_vs_epoch.png")

MODEL_TYPES = [
    "inf_truth", "inf_norm",
    "unin_truth", "unin_norm",
    "mis_truth",  "mis_norm",
    "uns_truth",  "uns_norm",
]

MAX_SEEDS       = 1                      # scan seeds 0‑9
CKPT_RE         = re.compile(r"checkpoint_ep(\d{3})\.pt")
COLOR_CYCLE     = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# ▶▶ NEW: only these epochs are kept ◀◀
EPOCH_FILTER: Set[int] = {
    20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 500
}

# ───────────────────────── normative reference ───────────────────────────
print("[+] Computing normative error vector …")
_, _, _, _, ERR_NORM = ev.evaluate_norm()       # shape ≈ (#trials,)
ERR_NORM = ERR_NORM.astype(np.float32)

# guard against degenerate variance
if ERR_NORM.std() < 1e-12:
    raise RuntimeError(
        "Normative error vector has ~zero variance — Pearson r undefined.")

# ───────────────────────── helper: correlation ───────────────────────────

def pearson_vs_norm(ckpt_path: str, err_norm: np.ndarray) -> float:
    """Return Pearson *r* between a checkpoint’s hazard errors and *err_norm*."""
    # quick‑load GRU model (hp matches training code)
    hp = {"n_input": 1, "n_rnn": 128}
    model = ev.GRUModel(hp).to(ev.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=ev.DEVICE))
    model.eval()

    # evaluate on the test‑unsorted CSV block(s)
    csvs = ev._list_test_csvs(ev.MAX_TEST_CSVS)
    recs: List[Dict] = []
    for p in csvs:
        recs.extend(ev._evaluate_csv(p, model))

    err_net = np.asarray([r["haz_err_net"] for r in recs], dtype=np.float32)

    # guard against pathological zero‑variance cases
    if err_net.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(err_net, err_norm)[0, 1])

# ───────────────────────── gather correlations ───────────────────────────
print("[+] Scanning checkpoints …")

data: Dict[str, Dict[int, List[Tuple[int, float]]]] = defaultdict(dict)

for t_key in MODEL_TYPES:
    t_dir = os.path.join(MODELS_DIR, t_key)
    if not os.path.isdir(t_dir):
        print(f"[skip] missing {t_dir}")
        continue

    for s_dir in os.listdir(t_dir):
        m_seed = re.fullmatch(r"seed_(\d+)", s_dir)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))
        if seed >= MAX_SEEDS:
            continue

        ckpt_glob = os.path.join(t_dir, s_dir, "checkpoint_ep*.pt")
        ckpts = sorted(glob.glob(ckpt_glob))
        if not ckpts:
            continue

        corr_hist: List[Tuple[int, float]] = []
        for ck in ckpts:
            m_ck = CKPT_RE.search(os.path.basename(ck))
            if not m_ck:
                continue
            epoch = int(m_ck.group(1))
            if epoch not in EPOCH_FILTER:    # <<< NEW filter
                continue
            r_val = pearson_vs_norm(ck, ERR_NORM)
            corr_hist.append((epoch, r_val))
            print(f"  {t_key}|seed{seed} ep{epoch:03d} r={r_val:+.4f}")

        if corr_hist:
            # ensure chronological order
            data[t_key][seed] = sorted(corr_hist, key=lambda x: x[0])

# sanity check
if not any(data.values()):
    raise RuntimeError("No correlation histories found for the specified epochs – are the corresponding checkpoints present?")

# ───────────────────────── plotting ───────────────────────────────────────
print("[+] Plotting …")
plt.figure(figsize=(12, 7))
for t_key in MODEL_TYPES:
    if t_key not in data:
        continue
    color = next(COLOR_CYCLE)

    # thin dashed lines – individual seeds
    for seed, pairs in data[t_key].items():
        if not pairs:
            continue
        epochs, corr = zip(*pairs)
        plt.plot(epochs, corr, color=color, alpha=0.30, linewidth=.8,
                 linestyle="--")

    # bold solid – mean across seeds (interpolated NaNs)
    all_epochs = sorted(EPOCH_FILTER)  # fixed x‑axis
    stack = np.full((len(data[t_key]), len(all_epochs)), np.nan)
    for i, pairs in enumerate(data[t_key].values()):
        ep_idx = {ep: idx for idx, ep in enumerate(all_epochs)}
        for ep, r_val in pairs:
            stack[i, ep_idx[ep]] = r_val
    mean_corr = np.nanmean(stack, axis=0)
    plt.plot(all_epochs, mean_corr, color=color, linewidth=2.5, label=t_key)

plt.axhline(0, color="k", linewidth=.6, alpha=.5)
plt.xlabel("Epoch")
plt.ylabel("Pearson r (net vs normative)")
plt.title("Error‑Correlation with Normative Model vs Epoch (filtered)")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(OUTFIG, dpi=300)
print(f"[+] Figure saved to {OUTFIG}")
plt.show()

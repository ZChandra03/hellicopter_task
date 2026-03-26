#!/usr/bin/env python3
"""
evaluate_diff.py  – concise evaluation for the first six seeds of every GRU model
────────────────────────────────────────────────────────────────────────────────
Console output example
    GRU-NORM  (6 seeds, 12 000 trials)
        report 99.633 % | hazard ±0.10 79.142 % | hazard Hi/Lo 98.317 %
        losses: 1.17e-03 1.21e-03 1.19e-03 1.16e-03 1.22e-03 1.18e-03
    GRU-TRUTH (6 seeds, 12 000 trials)
        report 97.842 % | hazard ±0.10 72.404 % | hazard Hi/Lo 94.755 %
        losses: 2.07e-03 …

Only the headline numbers and the list of per-seed losses are printed.
"""
# ───────────────────────── imports & config
import os, ast, itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch
from rnn_models import GRUModel
from NormativeModel import BayesianObserver      # unchanged helper

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR  = os.path.join(BASE_DIR, "variants")
HS_GRID      = np.arange(0, 1.05, 0.05)
MU1, MU2     = -1, 1
EPS          = 1e-10

BASE_TAGS = ["gru_norm", "gru_truth"]           # evaluate both training regimes
SEEDS      = range(6)                           # s0 … s5
MODEL_SPECS: List[Tuple[str, Type[torch.nn.Module]]] = [
    (f"{tag}_s{seed}", GRUModel)                # e.g. ("gru_norm_s3", GRUModel)
    for tag, seed in itertools.product(BASE_TAGS, SEEDS)
]

# ───────────────────────── helpers
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_model(tag: str, cls: Type[torch.nn.Module]):
    hp = {"n_input": 1, "n_rnn": 128}
    m  = cls(hp).to(DEVICE)
    ck = os.path.join(BASE_DIR, "models", tag, "checkpoint_best.pt")
    if not os.path.isfile(ck):
        raise FileNotFoundError(f"checkpoint missing: {ck}")
    m.load_state_dict(torch.load(ck, map_location=DEVICE))
    m.eval()
    return m

def evaluate_single(tag: str, cls: Type[torch.nn.Module]) -> Dict[str, float]:
    """Return overall accuracies and mean BCE loss for one checkpoint."""
    mdl = load_model(tag, cls)
    recs: List[Dict] = []

    for k in range(40):                         # 40 test variants
        csv = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv):
            break
        df = pd.read_csv(csv)

        with torch.no_grad():
            for _, row in df.iterrows():
                ev   = ast.literal_eval(row["evidence"])
                x    = torch.tensor(ev, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                loc_logits, haz_logits = mdl(x)
                loc_L  = loc_logits[0, -1, 0].item()
                haz_L  = haz_logits[0, 0].item()

                # predictions & targets
                p_loc  = sigmoid(loc_L)
                p_haz  = sigmoid(haz_L)
                rep_t  = 0.5 * (row["trueReport"] + 1)        # −1/1 → 0/1
                haz_t  = row["trueHazard"]

                # accuracies
                recs.append({
                    "rep_corr"   : int((loc_L > 0) == bool(rep_t)),
                    "haz_acc"    : int(abs(p_haz - haz_t) <= 0.10),
                    "haz_hilo"   : int((p_haz > 0.5) == (haz_t > 0.5)),
                    # BCE loss mirroring training objective
                    "loss"       : -(rep_t * np.log(p_loc + EPS) + (1 - rep_t) * np.log(1 - p_loc + EPS))
                                   - 0.5*(haz_t * np.log(p_haz + EPS) + (1 - haz_t) * np.log(1 - p_haz + EPS)),
                })

    arr = pd.DataFrame(recs)
    return {
        "report" : arr["rep_corr"].mean(),
        "haz10"  : arr["haz_acc"].mean(),
        "hilo"   : arr["haz_hilo"].mean(),
        "loss"   : arr["loss"].mean(),
    }

# ───────────────────────── main
def main():
    stats = defaultdict(list)                   # keyed by base tag, e.g. "gru_norm"
    for tag, cls in MODEL_SPECS:
        res = evaluate_single(tag, cls)
        base = "_".join(tag.split("_")[:2])     # strip the seed suffix
        stats[base].append(res)

    for base, lst in stats.items():
        A = pd.DataFrame(lst)
        # headline mean across seeds
        rep, hz10, hilo = A[["report", "haz10", "hilo"]].mean()
        losses          = " ".join(f"{x:.2e}" for x in A["loss"])
        print(f"{base.upper():<10} ({len(lst)} seeds, {len(lst)*300*40:6,d} trials)")
        print(f"    report {rep:6.3%} | hazard ±0.10 {hz10:6.3%} | hazard Hi/Lo {hilo:6.3%}")
        print(f"    losses: {losses}\n")

if __name__ == "__main__":
    main()

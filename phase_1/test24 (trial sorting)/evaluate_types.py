#!/usr/bin/env python3
"""
evaluate_diff.py
================

Evaluate trained RNN models on **three trial categories** that were created by
`TaskConfig_Generator_Informativeness.py`:
    • Informative
    • Uninformative
    • Misleading

For each category we aggregate *all* trials across the first `MAX_VARIANTS`
variant files and print a **single‑line summary**:

    MODEL CATEGORY: report XX.XXX% | hazard ±0.10 XX.XXX% | hazard Hi/Lo XX.XXX%

No tables, no plots – just the headline numbers you asked for.

Run
    python evaluate_diff.py            # default MAX_VARIANTS = 40
    python evaluate_diff.py 20         # evaluate the first 20 variants
"""

import os
import sys
import ast
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch

from rnn_models import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver

# ───────────────────────── configuration ────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR  = os.path.join(BASE_DIR, "variants")
HS_GRID      = np.arange(0.0, 1.05, 0.05)
MU1, MU2     = -1, 1
EPS          = 1e-10  # numerical safety margin

# (label, class, checkpoint‑subfolder)
MODEL_SPECS: List[Tuple[str, Type[torch.nn.Module], str]] = [
    ("gru",  GRUModel,  "gru_truth_s0"),
    # Uncomment below when checkpoints are available
    # ("lstm", LSTMModel, "lstm_norm_s4"),
    # ("rnn",  RNNModel,  "rnn_norm_s4"),
]

CATEGORIES = {
    "informative"  : "Informative",
    "uninformative": "Uninformative",
    "misleading"   : "Misleading",
}

# ───────────────────────── helpers ──────────────────────────────────────

def get_default_hp() -> Dict[str, int]:
    """Return the hyper‑parameters that were used during training."""
    return {"n_input": 1, "n_rnn": 128}


def load_model(model_cls: Type[torch.nn.Module], tag: str):
    """Instantiate the network and load its checkpoint."""
    hp = get_default_hp()
    model = model_cls(hp).to(DEVICE)

    ckpt_path = os.path.join(BASE_DIR, "models", tag, "final.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def evaluate_csv(model: torch.nn.Module, csv_path: str, label: str) -> List[Dict]:
    """Run one CSV (900 trials) through the network and Bayesian observer."""
    df = pd.read_csv(csv_path)
    recs: List[Dict] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            ev_list = row["evidence"]
            # Stored as list‑string → convert if necessary
            if not isinstance(ev_list, list):
                ev_list = ast.literal_eval(str(ev_list))

            sigma    = float(row["sigma"])
            haz_true = float(row["trueHazard"])
            rep_true = int(row["trueReport"])

            # ---- Network prediction -----------------------------------
            x = torch.tensor(ev_list, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            loc_logits, haz_logits = model(x)

            loc_logit_last = loc_logits[0, -1, 0]
            prob_state2    = torch.sigmoid(loc_logit_last).item()
            haz_pred       = torch.sigmoid(haz_logits)[0, 0].item()

            loc_pred        = 1 if loc_logit_last.item() > 0 else -1
            rep_corr_net    = int(loc_pred == rep_true)
            haz_acc_net     = int(abs(haz_pred - haz_true) <= 0.10)
            haz_hilo_net    = int((haz_pred > 0.5) == (haz_true > 0.5))

            # ---- Bayesian normative observer --------------------------
            L_haz, L_state, rep_norm, _ = BayesianObserver(ev_list, MU1, MU2, sigma, HS_GRID.copy())
            rep_corr_norm = int(rep_norm == rep_true)
            haz_est_norm  = float(np.dot(HS_GRID, L_haz[:, -1]))
            haz_acc_norm  = int(abs(haz_est_norm - haz_true) <= 0.10)
            haz_hilo_norm = int((haz_est_norm > 0.5) == (haz_true > 0.5))

            recs.append({
                "rep_corr_net" : rep_corr_net,
                "haz_acc_net"  : haz_acc_net,
                "haz_hilo_net" : haz_hilo_net,
                "rep_corr_norm": rep_corr_norm,
                "haz_acc_norm" : haz_acc_norm,
                "haz_hilo_norm": haz_hilo_norm,
            })
    return recs


def compute_metrics(recs: List[Dict], key_suffix: str):
    """Return mean accuracies for either 'net' or 'norm'."""
    rep  = np.mean([r[f"rep_corr_{key_suffix}"]  for r in recs])
    haz  = np.mean([r[f"haz_acc_{key_suffix}"]   for r in recs])
    hilo = np.mean([r[f"haz_hilo_{key_suffix}"]  for r in recs])
    return rep, haz, hilo

# ───────────────────────── per‑model evaluation ─────────────────────────

def evaluate_model(label: str, model_cls: Type[torch.nn.Module], tag: str, max_variants: int):
    print(f"\n===== {label.upper()} =====")
    model = load_model(model_cls, tag)

    for cat_short, cat_name in CATEGORIES.items():
        all_recs: List[Dict] = []
        for k in range(max_variants):
            csv_path = os.path.join(VARIANT_DIR, cat_short, f"train_{cat_short}_{k:02d}.csv")
            if not os.path.isfile(csv_path):
                if k == 0:
                    print(f"[!] No variants found for category '{cat_name}'. Skipping.")
                break
            all_recs.extend(evaluate_csv(model, csv_path, label))

        if not all_recs:
            continue

        rep_norm, haz_norm, hilo_norm = compute_metrics(all_recs, "norm")
        rep_net,  haz_net,  hilo_net  = compute_metrics(all_recs, "net")

        print(f"NORM {cat_name}: report {rep_norm:.3%} | hazard ±0.10 {haz_norm:.3%} | hazard Hi/Lo {hilo_norm:.3%}")
        print(f"{label.upper()} {cat_name}: report {rep_net:.3%} | hazard ±0.10 {haz_net:.3%} | hazard Hi/Lo {hilo_net:.3%}")

# ───────────────────────── main ─────────────────────────────────────────

def main():
    max_variants = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    for label, cls, tag in MODEL_SPECS:
        evaluate_model(label, cls, tag, max_variants)

    print("\nAll evaluations finished.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
evaluate_types.py
=================
Evaluate trained RNN models on the *stratified* variants produced by
TaskConfig_Generator_stratified.py.

The directory tree we now expect is

    project_root/
    ├─ models/
    │   ├─ seed_0/
    │   │   └─ final.pt
    │   └─ …
    └─ variants/
        ├─ train/
        └─ test/          ← we evaluate this split
            ├─ informative/
            │   └─ test_informative_00.csv …
            ├─ uninformative/
            ├─ misleading/
            └─ unsorted/

If you later want to evaluate the *train* split instead, just launch the
script with the flag “--split train”.
"""

import os, sys, ast, argparse
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch

from rnn_models    import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver

# ────────────────────────── CLI ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=("train", "test"), default="test",
                    help="Which split in variants/ to load (default: %(default)s)")
parser.add_argument("--max_variants", type=int, default=40,
                    help="How many CSV variants per category to aggregate (default: %(default)s)")
args = parser.parse_args()

# ───────────────────────── configuration ────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR = os.path.join(BASE_DIR, "variants", args.split)   # <── changed
HS_GRID     = np.arange(0.0, 1.05, 0.05)
MU1, MU2    = -1,  1
EPS         = 1e-10

# (label, class, checkpoint‑subfolder)
MODEL_SPECS: List[Tuple[str, Type[torch.nn.Module], str]] = [
    ("gru", GRUModel, "seed_0"),
    # ("lstm", LSTMModel, "seed_0"),   # enable when checkpoints exist
    # ("rnn",  RNNModel,  "seed_0"),
]

CATEGORIES = {
    "informative"  : "Informative",
    "uninformative": "Uninformative",
    "misleading"   : "Misleading",
    "unsorted"     : "Unsorted",
}

# ───────────────────────── helpers ──────────────────────────────────────
def get_default_hp() -> Dict[str, int]:
    return {"n_input": 1, "n_rnn": 128}

def load_model(model_cls: Type[torch.nn.Module], tag: str):
    hp    = get_default_hp()
    model = model_cls(hp).to(DEVICE)

    ckpt_path = os.path.join(BASE_DIR, "models", tag, "final.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model

def evaluate_csv(model: torch.nn.Module, csv_path: str) -> List[Dict]:
    df   = pd.read_csv(csv_path)
    recs: List[Dict] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            ev_list = row["evidence"]
            if not isinstance(ev_list, list):
                ev_list = ast.literal_eval(str(ev_list))

            sigma    = float(row["sigma"])
            haz_true = float(row["trueHazard"])
            rep_true = int(row["trueReport"])

            # network prediction
            x = torch.tensor(ev_list, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            loc_logits, haz_logits = model(x)

            loc_last   = loc_logits[0, -1, 0]
            haz_pred   = torch.sigmoid(haz_logits)[0, 0].item()

            loc_pred       = 1 if loc_last.item() > 0 else -1
            recs.append({
                "rep_corr_net":  int(loc_pred == rep_true),
                "haz_acc_net":   int(abs(haz_pred - haz_true) <= 0.10),
                "haz_hilo_net":  int((haz_pred > 0.5) == (haz_true > 0.5)),
            })

            # Bayesian normative observer
            L_haz, _, rep_norm, _ = BayesianObserver(ev_list, MU1, MU2, sigma, HS_GRID.copy())
            haz_est_norm = float(np.dot(HS_GRID, L_haz[:, -1]))
            recs[-1].update({
                "rep_corr_norm": int(rep_norm == rep_true),
                "haz_acc_norm":  int(abs(haz_est_norm - haz_true) <= 0.10),
                "haz_hilo_norm": int((haz_est_norm > 0.5) == (haz_true > 0.5)),
            })
    return recs

def compute_metrics(recs: List[Dict], suffix: str):
    rep  = np.mean([r[f"rep_corr_{suffix}"]  for r in recs])
    haz  = np.mean([r[f"haz_acc_{suffix}"]   for r in recs])
    hilo = np.mean([r[f"haz_hilo_{suffix}"]  for r in recs])
    return rep, haz, hilo

# ───────────────────────── evaluation loop ──────────────────────────────
def evaluate_model(label: str, model_cls: Type[torch.nn.Module], tag: str):
    print(f"\n===== {label.upper()} =====")
    model = load_model(model_cls, tag)

    for cat_short, cat_name in CATEGORIES.items():
        all_recs: List[Dict] = []

        for k in range(args.max_variants):
            fname   = f"{args.split}_{cat_short}_{k:02d}.csv"      # <── changed
            csv_path = os.path.join(VARIANT_DIR, cat_short, fname)
            if not os.path.isfile(csv_path):
                if k == 0:
                    print(f"[!] No variants found for category '{cat_name}'. Skipping.")
                break
            all_recs.extend(evaluate_csv(model, csv_path))

        if not all_recs:
            continue

        rep_n, haz_n, hilo_n = compute_metrics(all_recs, "norm")
        rep_m, haz_m, hilo_m = compute_metrics(all_recs, "net")

        print(f"NORM {cat_name:13}: report {rep_n:.3%} | hazard ±0.10 {haz_n:.3%} | hazard Hi/Lo {hilo_n:.3%}")
        print(f"{label.upper()} {cat_name:9}: report {rep_m:.3%} | hazard ±0.10 {haz_m:.3%} | hazard Hi/Lo {hilo_m:.3%}")

# ────────────────────────── main ────────────────────────────────────────
def main() -> None:
    for lbl, cls, tag in MODEL_SPECS:
        evaluate_model(lbl, cls, tag)

    print("\nAll evaluations finished.")

if __name__ == "__main__":
    main()

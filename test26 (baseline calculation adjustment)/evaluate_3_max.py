#!/usr/bin/env python3
"""
UPDATED evaluate_3.py (2025‑07‑17)
==================================
Adds a *hard* cap on the number of test CSVs to load.  Set the constant
MAX_TEST_CSVS below.  If it is None or larger than the number of files
available, every CSV is used (original behaviour).

Everything else – output format, model paths, evaluation logic – is
unchanged.
"""

from __future__ import annotations
import ast, os, glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from rnn_models import GRUModel
from NormativeModel import BayesianObserver

# ───────────────────────── config ───────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
VARIANT_ROOT  = os.path.join(BASE_DIR, "variants", "test", "unsorted")

MODEL_TYPES   = [
    "inf_norm", "inf_truth",
    "unin_norm", "unin_truth",
    "mis_norm", "mis_truth",
    "uns_norm", "uns_truth",
]
SEEDS         = [0, 1, 2]          # evaluate first three seeds → 24 nets

MAX_TEST_CSVS = 10               # ← set to an int (e.g. 10) to limit CSVs

HS_GRID, MU1, MU2 = np.arange(0, 1.05, 0.05), -1, 1
EPS = 1e-10

# ───────────────────────── helpers ──────────────────────────────────────
def _clip(p: float) -> float:
    """Avoid log(0) in loss calc."""
    return float(np.clip(p, EPS, 1 - EPS))


def _list_test_csvs(max_csvs: int | None = None) -> List[str]:
    """Return up to *max_csvs* unsorted‑test CSV paths (sorted)."""
    paths = sorted(glob.glob(os.path.join(VARIANT_ROOT, "test_unsorted_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No test CSVs found under {VARIANT_ROOT}")
    return paths[:max_csvs] if max_csvs is not None else paths


def _load_model(type_key: str, seed: int) -> torch.nn.Module:
    ckpt = os.path.join(MODELS_DIR, type_key, f"seed_{seed}", "checkpoint_best.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"[missing ckpt] {ckpt}")
    hp = {"n_input": 1, "n_rnn": 128}
    model = GRUModel(hp).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model

# ───────────────────────── per‑CSV evaluation ───────────────────────────
def _evaluate_csv(csv_path: str, model: torch.nn.Module | None) -> List[Dict]:
    """Return list of per‑trial dicts. If *model* is None → normative only."""
    df = pd.read_csv(csv_path)
    recs: List[Dict] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            # unpack trial -----------------------------------------------------
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            sigma = float(row.get("sigma", row.get("noise", 0.0)))
            haz_true = float(row.get("trueHazard", row.get("hazard", row.get("hazardRate"))))
            rep_true = int(row.get("trueReport", row.get("report", 0)))
            y_rep = 0.5 * (rep_true + 1)

            # normative observer -----------------------------------------------
            L_haz, L_state, rep_norm, _ = BayesianObserver(
                evid, MU1, MU2, sigma, HS_GRID.copy())
            p_state2_norm = _clip(L_state[1, -1])
            haz_est_norm  = float(np.dot(HS_GRID, L_haz[:, -1]))
            p_haz_norm    = _clip(haz_est_norm)

            rec = {
                "rep_corr_norm": int(rep_norm == rep_true),
                "haz_acc_norm" : int(abs(haz_est_norm - haz_true) <= 0.10),
                "haz_hilo_norm": int((haz_est_norm > .5) == (haz_true > .5)),
                "loss_norm"    : (
                    -(y_rep * np.log(p_state2_norm) + (1 - y_rep) * np.log(1 - p_state2_norm))
                    + 0.5 * (-(haz_true * np.log(p_haz_norm) + (1 - haz_true) * np.log(1 - p_haz_norm)))
                ),
                "haz_err_norm" : haz_est_norm - haz_true,
            }

            # network prediction -----------------------------------------------
            if model is not None:
                x = torch.tensor(evid, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                loc_logits, haz_logits = model(x)
                loc_last = loc_logits[0, -1, 0]
                p_state2_net = _clip(torch.sigmoid(loc_last).item())
                p_haz_net = _clip(torch.sigmoid(haz_logits)[0, 0].item())
                rep_pred = 1 if loc_last.item() > 0 else -1

                rec.update({
                    "rep_corr_net": int(rep_pred == rep_true),
                    "haz_acc_net" : int(abs(p_haz_net - haz_true) <= 0.10),
                    "haz_hilo_net": int((p_haz_net > .5) == (haz_true > .5)),
                    "loss_net"    : (
                        -(y_rep * np.log(p_state2_net) + (1 - y_rep) * np.log(1 - p_state2_net))
                        + 0.5 * (-(haz_true * np.log(p_haz_net) + (1 - haz_true) * np.log(1 - p_haz_net)))
                    ),
                    "haz_err_net" : p_haz_net - haz_true,
                })

            recs.append(rec)
    return recs


def _aggregate(recs: List[Dict], use_net: bool) -> Tuple[float, float, float, float]:
    """Return (rep_acc, haz_acc_±0.10, haz_hi/lo, loss)."""
    df = pd.DataFrame(recs)
    cols = ("rep_corr_net", "haz_acc_net", "haz_hilo_net", "loss_net") if use_net else (
            "rep_corr_norm", "haz_acc_norm", "haz_hilo_norm", "loss_norm")
    return tuple(df[c].mean() for c in cols)  # type: ignore[return-value]

# ───────────────────────── top‑level evals ──────────────────────────────
def evaluate_norm() -> Tuple[float, float, float, float, np.ndarray]:
    csvs = _list_test_csvs(MAX_TEST_CSVS)
    recs: List[Dict] = []
    for p in csvs:
        recs.extend(_evaluate_csv(p, model=None))
    rep, haz, hilo, loss = _aggregate(recs, use_net=False)
    errs = np.array([r["haz_err_norm"] for r in recs], dtype=float)
    return rep, haz, hilo, loss, errs


def evaluate_network(type_key: str, seed: int, err_norm: np.ndarray) -> None:
    model = _load_model(type_key, seed)
    csvs = _list_test_csvs(MAX_TEST_CSVS)
    recs: List[Dict] = []
    for p in csvs:
        recs.extend(_evaluate_csv(p, model))
    rep, haz, hilo, loss = _aggregate(recs, use_net=True)
    errs = np.array([r["haz_err_net"] for r in recs], dtype=float)

    # Pearson r with normative errors
    r = float('nan') if errs.std() < 1e-12 or err_norm.std() < 1e-12 else float(
            np.corrcoef(errs, err_norm)[0, 1])

    tag_disp = f"{type_key}_s{seed}".upper()
    print(f"{tag_disp:<20}: report {rep:6.3%} | hazard ±0.10 {haz:6.3%} | "
          f"hazard Hi/Lo {hilo:6.3%} | loss {loss:.4f} | r(NORM) = {r:+.4f}")

# ───────────────────────── entry‑point ──────────────────────────────────
def main():
    print("========= Bayesian Normative Baseline =========")
    rep_n, haz_n, hilo_n, loss_n, err_norm = evaluate_norm()
    print(f"NORM: report {rep_n:6.3%} | hazard ±0.10 {haz_n:6.3%} | "
          f"hazard Hi/Lo {hilo_n:6.3%} | loss {loss_n:.4f}\n")

    print("========= Trained Networks (first 3 seeds each) =========")
    for type_key in MODEL_TYPES:
        for seed in SEEDS:
            try:
                evaluate_network(type_key, seed, err_norm)
            except FileNotFoundError as e:
                print(f"[skip] {e}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

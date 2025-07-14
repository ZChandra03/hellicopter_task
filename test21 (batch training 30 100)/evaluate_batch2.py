"""
Evaluate a *batch* of trained networks (first six seeds of every model type)
==========================================================================
Print **one concise summary line** per network **plus** a normative‐model
baseline.  No per‑condition tables, no histograms – just the high‑level
numbers you asked for:

    GRU_TRUTH_s0: report 99.633% | hazard ±0.10 79.142% | hazard Hi/Lo 98.317% | loss 0.0231

The script discovers available model directories automatically.  A model
*type* is the directory prefix *before* the trailing “_s<seed>” — for
instance “gru_truth” or “lstm_norm”.  We evaluate **seeds 0–5** for every
such type if the checkpoint `<models>/<tag>/final.pt` exists.

Usage
-----
Run the script directly (it expects the usual project structure)::

    python evaluate_diff.py
"""

from __future__ import annotations

import ast
import os
import re
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch

from rnn_models import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver

# ───────────────────────── configuration ────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VARIANT_DIR = os.path.join(BASE_DIR, "variants")  # trial CSVs

HS_GRID = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1
EPS      = 1e-10  # numerical safety for log / div
SEEDS    = range(6)  # evaluate seeds 0–5

# Mapping from directory prefix → network class
MODEL_CLASS_FROM_PREFIX = {
    "gru":  GRUModel,
    "lstm": LSTMModel,
    "rnn":  RNNModel,
}

# ───────────────────────── helpers ──────────────────────────────────────

def _clip(p: float) -> float:
    """Safe‑clamp a probability to (0, 1)."""
    return float(np.clip(p, EPS, 1 - EPS))


def load_model(model_cls: Type[torch.nn.Module], tag: str):
    """Instantiate *and* load checkpoint for one trained network."""
    ckpt = os.path.join(MODELS_DIR, tag, "checkpoint_best.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Missing checkpoint {ckpt!s}")

    hp = {"n_input": 1, "n_rnn": 128}
    model = model_cls(hp).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


# ───────────────────────── core per‑trial evaluation ────────────────────

def _evaluate_csv(csv_path: str,
                  model: torch.nn.Module | None = None) -> List[Dict]:
    """Return a list of per‑trial dictionaries for one CSV variant."""
    df = pd.read_csv(csv_path)
    recs: List[Dict] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            evid      = ast.literal_eval(row["evidence"])
            sigma     = float(row["sigma"])
            haz_true  = float(row["trueHazard"])
            rep_true  = int(row["trueReport"])
            y_rep     = 0.5 * (rep_true + 1)  # −1/+1 → 0/1

            # ─── Bayesian normative observer ────────────────────────────
            L_haz, L_state, rep_norm, _ = BayesianObserver(evid, MU1, MU2,
                                                           sigma, HS_GRID.copy())
            p_state2_norm = _clip(L_state[1, -1])
            haz_est_norm  = float(np.dot(HS_GRID, L_haz[:, -1]))
            p_haz_norm    = _clip(haz_est_norm)

            loss_norm = (
                -(y_rep * np.log(p_state2_norm) + (1 - y_rep) * np.log(1 - p_state2_norm)) +
                0.5 * (-(haz_true * np.log(p_haz_norm) + (1 - haz_true) * np.log(1 - p_haz_norm)))
            )

            rec: Dict[str, float | int] = {
                "rep_corr_norm"  : int(rep_norm == rep_true),
                "haz_acc_norm"  : int(abs(haz_est_norm - haz_true) <= 0.10),
                "haz_hilo_norm" : int((haz_est_norm > 0.5) == (haz_true > 0.5)),
                "loss_norm"      : loss_norm,
            }

            # ─── network predictions (optional) ────────────────────────
            if model is not None:
                x = torch.tensor(evid, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                loc_logits, haz_logits = model(x)
                loc_logit_last = loc_logits[0, -1, 0]
                p_state2_net   = _clip(torch.sigmoid(loc_logit_last).item())
                p_haz_net      = _clip(torch.sigmoid(haz_logits)[0, 0].item())

                rep_pred     = 1 if loc_logit_last.item() > 0 else -1
                rec.update({
                    "rep_corr_net"  : int(rep_pred == rep_true),
                    "haz_acc_net"  : int(abs(p_haz_net - haz_true) <= 0.10),
                    "haz_hilo_net" : int((p_haz_net > 0.5) == (haz_true > 0.5)),
                    "loss_net"      : (
                        -(y_rep * np.log(p_state2_net) + (1 - y_rep) * np.log(1 - p_state2_net)) +
                        0.5 * (-(haz_true * np.log(p_haz_net) + (1 - haz_true) * np.log(1 - p_haz_net)))
                    ),
                })

            recs.append(rec)
    return recs


def _aggregate(recs: List[Dict], use_net: bool) -> Tuple[float, float, float, float]:
    """Return weighted averages (report_acc, hazard_acc, hazard_hilo, loss)."""
    df = pd.DataFrame(recs)
    if use_net:
        cols = ("rep_corr_net", "haz_acc_net", "haz_hilo_net", "loss_net")
    else:
        cols = ("rep_corr_norm", "haz_acc_norm", "haz_hilo_norm", "loss_norm")

    weights = np.ones(len(df))  # every trial counts equally
    means   = [np.average(df[c], weights=weights) for c in cols]
    return tuple(means)  # type: ignore[return-value]


# ───────────────────────── evaluation façade ────────────────────────────

def evaluate_network(tag: str, model_cls: Type[torch.nn.Module]) -> Tuple[float, float, float, float]:
    """Evaluate one trained network across *all* CSV variants."""
    model = load_model(model_cls, tag)

    all_recs: List[Dict] = []
    k = 0
    while True:
        csv_path = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            break
        all_recs.extend(_evaluate_csv(csv_path, model))
        k += 1
    if k == 0:
        raise RuntimeError("No testConfig_var*.csv found in ./variants – run TaskConfig_Generator first?")

    return _aggregate(all_recs, use_net=True)


def evaluate_normative() -> Tuple[float, float, float, float]:
    """Evaluate Bayesian normative observer (no network)."""
    all_recs: List[Dict] = []
    k = 0
    while True:
        csv_path = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            break
        all_recs.extend(_evaluate_csv(csv_path, model=None))
        k += 1
    if k == 0:
        raise RuntimeError("No testConfig_var*.csv found in ./variants – run TaskConfig_Generator first?")

    return _aggregate(all_recs, use_net=False)


# ───────────────────────── model discovery ──────────────────────────────

def discover_model_tags() -> List[Tuple[str, Type[torch.nn.Module]]]:
    """Return (tag, model_cls) pairs for every *base* prefix and seed 0–5."""
    pattern = re.compile(r"^(?P<base>.+?)_s(?P<seed>\d+)$")
    groups: Dict[str, List[int]] = {}

    for dirname in os.listdir(MODELS_DIR):
        m = pattern.match(dirname)
        if not m:
            continue
        base  = m.group("base")
        seed  = int(m.group("seed"))
        if seed in SEEDS:
            groups.setdefault(base, []).append(seed)

    tags: List[Tuple[str, Type[torch.nn.Module]]] = []
    for base, seeds in sorted(groups.items()):
        # infer model class from prefix (gru / lstm / rnn)
        prefix = base.split("_")[0].lower()
        model_cls = None
        for key, cls in MODEL_CLASS_FROM_PREFIX.items():
            if prefix.startswith(key):
                model_cls = cls
                break
        if model_cls is None:
            print(f"[skip] Could not infer model class for '{base}'.")
            continue

        for s in sorted(seeds)[: len(SEEDS)]:
            tags.append((f"{base}_s{s}", model_cls))
    return tags


# ───────────────────────── main ─────────────────────────────────────────

def main():
    print("====== Bayesian Normative Baseline ======")
    rep_acc, haz_acc, haz_hilo, loss = evaluate_normative()
    print(f"NORM: report {rep_acc:6.3%} | hazard ±0.10 {haz_acc:6.3%} | hazard Hi/Lo {haz_hilo:6.3%} | loss {loss:.4f}")

    print("\n====== Trained Networks (seeds 0–5) ======")
    for tag, cls in discover_model_tags():
        try:
            rep_acc, haz_acc, haz_hilo, loss = evaluate_network(tag, cls)
            print(f"{tag.upper():<20}: report {rep_acc:6.3%} | hazard ±0.10 {haz_acc:6.3%} | hazard Hi/Lo {haz_hilo:6.3%} | loss {loss:.4f}")
        except FileNotFoundError as err:
            print(f"[skip] {err}")

    print("\nEvaluation complete (normative + first 6 seeds for each model type).")


if __name__ == "__main__":
    main()

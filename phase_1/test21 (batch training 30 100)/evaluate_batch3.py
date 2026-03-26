#!/usr/bin/env python3
"""
evaluate_batch_2.py ─ Batch evaluation **with pair-wise Pearson error correlations**

Prints:
  • One concise summary line per network **plus** the Bayesian baseline
  • A list of Pearson r values for the hazard-error vectors of every pair
"""

from __future__ import annotations
import ast, os, re
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch

from rnn_models import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver


# ───────────────────────── configuration ────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VARIANT_DIR = os.path.join(BASE_DIR, "variants")

HS_GRID = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1
EPS      = 1e-10
SEEDS    = range(15)

MODEL_CLASS_FROM_PREFIX = {
    "gru":  GRUModel,
    "lstm": LSTMModel,
    "rnn":  RNNModel,
}

# ───────────────────────── helpers ──────────────────────────────────────
def _clip(p: float) -> float:
    return float(np.clip(p, EPS, 1 - EPS))


def load_model(model_cls: Type[torch.nn.Module], tag: str):
    ckpt = os.path.join(MODELS_DIR, tag, "checkpoint_best.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Missing checkpoint {ckpt!s}")
    hp = {"n_input": 1, "n_rnn": 128}
    m = model_cls(hp).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    return m

# ───────────────────────── per-trial evaluation ─────────────────────────
def _evaluate_csv(csv_path: str,
                  model: torch.nn.Module | None = None) -> List[Dict]:
    df  = pd.read_csv(csv_path)
    rec = []

    with torch.no_grad():
        for _, row in df.iterrows():
            evid     = ast.literal_eval(row["evidence"])
            sigma    = float(row["sigma"])
            haz_true = float(row["trueHazard"])
            rep_true = int(row["trueReport"])
            y_rep    = 0.5 * (rep_true + 1)

            # ─ Bayesian observer (normative) ─
            L_haz, L_state, rep_norm, _ = BayesianObserver(
                evid, MU1, MU2, sigma, HS_GRID.copy())
            p_state2_norm = _clip(L_state[1, -1])
            haz_est_norm  = float(np.dot(HS_GRID, L_haz[:, -1]))
            p_haz_norm    = _clip(haz_est_norm)

            rec_d = {
                "rep_corr_norm" : int(rep_norm == rep_true),
                "haz_acc_norm"  : int(abs(haz_est_norm - haz_true) <= 0.10),
                "haz_hilo_norm" : int((haz_est_norm > .5) == (haz_true > .5)),
                "loss_norm"     : (
                    -(y_rep * np.log(p_state2_norm) + (1 - y_rep) * np.log(1 - p_state2_norm))
                    + 0.5 * (-(haz_true * np.log(p_haz_norm) + (1 - haz_true) * np.log(1 - p_haz_norm)))
                ),
                "haz_err_norm"  : haz_est_norm - haz_true,   # NEW
            }

            # ─ Network predictions (if supplied) ─
            if model is not None:
                x = (torch.tensor(evid, dtype=torch.float32)
                           .unsqueeze(0).unsqueeze(-1).to(DEVICE))
                loc_logits, haz_logits = model(x)
                loc_last = loc_logits[0, -1, 0]
                p_state2_net = _clip(torch.sigmoid(loc_last).item())
                p_haz_net    = _clip(torch.sigmoid(haz_logits)[0, 0].item())

                rep_pred = 1 if loc_last.item() > 0 else -1
                rec_d.update({
                    "rep_corr_net" : int(rep_pred == rep_true),
                    "haz_acc_net"  : int(abs(p_haz_net - haz_true) <= 0.10),
                    "haz_hilo_net" : int((p_haz_net > .5) == (haz_true > .5)),
                    "loss_net"     : (
                        -(y_rep * np.log(p_state2_net) + (1 - y_rep) * np.log(1 - p_state2_net))
                        + 0.5 * (-(haz_true * np.log(p_haz_net) + (1 - haz_true) * np.log(1 - p_haz_net)))
                    ),
                    "haz_err_net"  : p_haz_net - haz_true,   # NEW
                })

            rec.append(rec_d)
    return rec


def _aggregate(recs: List[Dict], use_net: bool) -> Tuple[float, float, float, float]:
    df = pd.DataFrame(recs)
    cols = ("rep_corr_net", "haz_acc_net", "haz_hilo_net", "loss_net") if use_net \
        else ("rep_corr_norm", "haz_acc_norm", "haz_hilo_norm", "loss_norm")
    means = [df[c].mean() for c in cols]
    return tuple(means)  # type: ignore[return-value]

# ───────────────────────── evaluation façade ────────────────────────────
def evaluate_network(tag: str, model_cls: Type[torch.nn.Module]):
    model = load_model(model_cls, tag)
    all_recs: List[Dict] = []
    k = 0
    while True:
        p = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(p): break
        all_recs.extend(_evaluate_csv(p, model))
        k += 1
    if k == 0:
        raise RuntimeError("No testConfig_var*.csv found in ./variants")

    rep_acc, haz_acc, haz_hilo, loss = _aggregate(all_recs, use_net=True)
    errors = np.array([r["haz_err_net"] for r in all_recs], dtype=float)
    return rep_acc, haz_acc, haz_hilo, loss, errors


def evaluate_normative():
    all_recs: List[Dict] = []
    k = 0
    while True:
        p = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(p): break
        all_recs.extend(_evaluate_csv(p, model=None))
        k += 1
    if k == 0:
        raise RuntimeError("No testConfig_var*.csv found in ./variants")

    rep_acc, haz_acc, haz_hilo, loss = _aggregate(all_recs, use_net=False)
    errors = np.array([r["haz_err_norm"] for r in all_recs], dtype=float)
    return rep_acc, haz_acc, haz_hilo, loss, errors

# ───────────────────────── model discovery ──────────────────────────────
def discover_model_tags():
    pattern = re.compile(r"^(?P<base>.+?)_s(?P<seed>\d+)$")
    groups: Dict[str, List[int]] = {}

    for d in os.listdir(MODELS_DIR):
        m = pattern.match(d)
        if not m: continue
        base, seed = m.group("base"), int(m.group("seed"))
        if seed in SEEDS: groups.setdefault(base, []).append(seed)

    tags = []
    for base, seeds in sorted(groups.items()):
        prefix = base.split("_")[0].lower()
        model_cls = next((cls for key, cls in MODEL_CLASS_FROM_PREFIX.items()
                          if prefix.startswith(key)), None)
        if model_cls is None:
            print(f"[skip] could not infer model class for '{base}'"); continue
        for s in sorted(seeds)[: len(SEEDS)]:
            tags.append((f"{base}_s{s}", model_cls))
    return tags

# ───────────────────────── main ─────────────────────────────────────────
def main():
    errors_dict: Dict[str, np.ndarray] = {}

    print("====== Bayesian Normative Baseline ======")
    r, h, hh, L, e = evaluate_normative()
    errors_dict["NORM"] = e
    print(f"NORM: report {r:6.3%} | hazard ±0.10 {h:6.3%} | hazard Hi/Lo {hh:6.3%} | loss {L:.4f}")

    print("\n====== Trained Networks (seeds 0–5) ======")
    for tag, cls in discover_model_tags():
        try:
            r, h, hh, L, e = evaluate_network(tag, cls)
            errors_dict[tag.upper()] = e
            print(f"{tag.upper():<20}: report {r:6.3%} | hazard ±0.10 {h:6.3%} | hazard Hi/Lo {hh:6.3%} | loss {L:.4f}")
        except FileNotFoundError as err:
            print(f"[skip] {err}")

    # ─ Pearson error correlations ─
    print("\n====== Hazard-Error Pearson Correlations ======")
    tags = list(errors_dict.keys())
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            e1, e2 = errors_dict[tags[i]], errors_dict[tags[j]]
            if e1.std() < 1e-12 or e2.std() < 1e-12:
                corr = float('nan')
            else:
                corr = float(np.corrcoef(e1, e2)[0, 1])
            print(f"{tags[i]:<20} ↔ {tags[j]:<20}: r = {corr:+.4f}")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()

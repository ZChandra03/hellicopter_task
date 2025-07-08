"""evaluate.py – compare GRU, LSTM and vanilla RNN performance against the
Bayesian normative model
--------------------------------------------------------------------------
This script now loops over **three** trained network architectures that share
an identical read‑out structure (see `rnn_models.py`):

    • GRUModel   – checkpoint under  models/gru_trained/
    • LSTMModel  – checkpoint under  models/lstm_trained/
    • RNNModel   – checkpoint under  models/rnn_trained/

For each network we compute the same per‑trial statistics as before and then
print a concise summary next to the Bayesian ideal observer.
"""
import os, ast
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from rnn_models import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver

# ───────────────────────── configuration
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR  = os.path.join(BASE_DIR, "variants")
HS_GRID      = np.arange(0, 1.05, 0.05)
MU1, MU2     = -1, 1
EPS          = 1e-10  # numerical safety for log / div

# Each tuple: (short label, model class, checkpoint sub‑directory)
MODEL_SPECS: List[Tuple[str, Type[torch.nn.Module], str]] = [
    ("gru",  GRUModel,  "gru_trained"),
]

# ───────────────────────── helpers ----------------------------------------------------

def get_default_hp() -> Dict[str, int]:
    """Return the common hyper‑parameter set used during training."""
    return {"n_input": 1, "n_rnn": 128}


def load_model(model_cls: Type[torch.nn.Module], tag: str):
    """Instantiate *and* load checkpoint for the requested network."""
    hp = get_default_hp()
    model = model_cls(hp).to(DEVICE)

    ckpt_path = os.path.join(BASE_DIR, "models", tag, "checkpoint_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found – expected {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model

# ───────────────────────── per‑variant evaluation -------------------------------------

def evaluate_variant(model: torch.nn.Module, csv_path: str, label: str) -> List[Dict]:
    """Compute trial‑level metrics for one *trained* network and Bayesian observer."""
    df = pd.read_csv(csv_path)
    recs: List[Dict] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            ev_list  = ast.literal_eval(row["evidence"])
            sigma    = float(row["sigma"])
            haz_true = float(row["trueHazard"])
            rep_true = int(row["trueReport"])

            # ---- Network prediction ---------------------------------------
            x = torch.tensor(ev_list, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            loc_logits, haz_logits = model(x)

            loc_logit_last = loc_logits[0, -1, 0]
            prob_state2    = torch.sigmoid(loc_logit_last).item()  # P(state = 2)
            haz_pred_prob  = torch.sigmoid(haz_logits)[0, 0].item()

            loc_pred      =  1 if loc_logit_last.item() > 0 else -1
            rep_corr_net  = int(loc_pred == rep_true)

            haz_err_net   = haz_pred_prob - haz_true
            haz_acc_net   = int(abs(haz_err_net) <= 0.10)
            haz_hilo_net  = int((haz_pred_prob > 0.5) == (haz_true > 0.5))

            # ---- Bayesian normative observer ------------------------------
            L_haz, L_state, rep_norm, _ = BayesianObserver(ev_list, MU1, MU2, sigma, HS_GRID.copy())

            #   • probabilities for BCE
            p_state2_norm = float(L_state[1, -1])
            p_state2_norm = np.clip(p_state2_norm, EPS, 1 - EPS)

            posterior      = L_haz[:, -1]
            haz_est_norm   = float(np.dot(HS_GRID, posterior))
            p_haz_norm     = np.clip(haz_est_norm, EPS, 1 - EPS)

            rep_corr_norm  = int(rep_norm == rep_true)
            haz_err_norm   = haz_est_norm - haz_true
            haz_acc_norm   = int(abs(haz_err_norm) <= 0.10)
            haz_hilo_norm  = int((haz_est_norm > 0.5) == (haz_true > 0.5))

            #   • Ideal‑observer BCE loss (unchanged)
            y_rep = 0.5 * (rep_true + 1)            # −1/+1 → 0/1
            loss_rep = -(y_rep * np.log(p_state2_norm) + (1 - y_rep) * np.log(1 - p_state2_norm))
            loss_haz = -(haz_true * np.log(p_haz_norm) + (1 - haz_true) * np.log(1 - p_haz_norm))
            loss_total_norm = loss_rep + 0.5 * loss_haz

            recs.append({
                "sigma"                 : sigma,
                "hazard"                : haz_true,

                # network‑specific keys (dynamically named)
                f"rep_corr_{label}"     : rep_corr_net,
                f"haz_acc_{label}"     : haz_acc_net,
                f"haz_hilo_{label}"    : haz_hilo_net,
                f"haz_err_{label}"     : haz_err_net,

                # normative keys (identical for all networks)
                "rep_corr_norm"         : rep_corr_norm,
                "haz_acc_norm"         : haz_acc_norm,
                "haz_hilo_norm"        : haz_hilo_norm,
                "haz_err_norm"         : haz_err_norm,
                "loss_norm"            : loss_total_norm,
            })
    return recs

# ───────────────────────── aggregation helpers ----------------------------------------

def aggregate(records: List[Dict], label: str):
    """Return condition‑wise summary table and underlying DataFrame."""
    df = pd.DataFrame(records)

    col_rep   = f"rep_corr_{label}"
    col_hacc  = f"haz_acc_{label}"
    col_hilo  = f"haz_hilo_{label}"

    grouped = (df.groupby(["sigma", "hazard"]).agg(
        N                 = (col_rep,  "size"),
        report_acc_net    = (col_rep,  "mean"),
        hazard_acc_net    = (col_hacc, "mean"),
        hazard_hilo_net   = (col_hilo, "mean"),
        report_acc_norm   = ("rep_corr_norm", "mean"),
        hazard_acc_norm   = ("haz_acc_norm", "mean"),
        hazard_hilo_norm  = ("haz_hilo_norm", "mean"),
        loss_norm         = ("loss_norm", "mean"),
    ).reset_index().sort_values(["sigma", "hazard"]))

    return grouped, df

# ───────────────────────── per‑model driver -------------------------------------------

def evaluate_model(label: str, model_cls: Type[torch.nn.Module], tag: str, max_variants: int = 40):
    print(f"\n===== Evaluating {label.upper()} =====")
    model = load_model(model_cls, tag)

    all_recs: List[Dict] = []
    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            break
        all_recs.extend(evaluate_variant(model, csv_path, label))

    table, raw_df = aggregate(all_recs, label)

    # ---- print tables ------------------------------------------------------
    print("Per‑condition performance (network vs Normative)")
    print("=" * 110)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    weights = table["N"]
    overall = {
        label.upper(): {
            "report_acc": np.average(table["report_acc_net"],  weights=weights),
            "hazard_acc": np.average(table["hazard_acc_net"],  weights=weights),
            "hazard_hilo":np.average(table["hazard_hilo_net"], weights=weights),
        },
        "Norm": {
            "report_acc": np.average(table["report_acc_norm"], weights=weights),
            "hazard_acc": np.average(table["hazard_acc_norm"], weights=weights),
            "hazard_hilo":np.average(table["hazard_hilo_norm"],weights=weights),
        },
    }
    print("\nOverall performance")
    for m, s in overall.items():
        print(f"{m}: report {s['report_acc']:.3%} | hazard ±0.10 {s['hazard_acc']:.3%} | hazard Hi/Lo {s['hazard_hilo']:.3%}")

    # ---- plot histogram ----------------------------------------------------
    bins = np.arange(-1.0, 1.0001, 0.05)
    plt.figure(figsize=(8, 4.5))
    plt.hist(raw_df["haz_err_norm"], bins=bins, alpha=0.6, label="Bayesian", edgecolor="black")
    plt.hist(raw_df[f"haz_err_{label}"], bins=bins, alpha=0.6, label=label.upper(), edgecolor="black")
    plt.axvline(0, color="k", linewidth=1)
    plt.xlabel("Prediction error: (predicted − true hazard)")
    plt.ylabel("Count")
    plt.title(f"Hazard prediction error distribution – {label.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return raw_df  # useful if caller wants the numeric loss etc.

# ───────────────────────────────────────── main ---------------------------------------
if __name__ == "__main__":
    max_variants = 40  # keep default

    # Evaluate each network architecture sequentially -----------------------
    for label, cls, tag in MODEL_SPECS:
        evaluate_model(label, cls, tag, max_variants=max_variants)

    print("\nEvaluation finished for all network types.")

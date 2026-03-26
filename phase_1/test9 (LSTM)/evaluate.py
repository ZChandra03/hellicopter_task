"""evaluate.py – compare LSTM vs Normative model, plot error histograms,
   and compute the Bayesian ideal observer’s BCE loss
-----------------------------------------------------------------------"""
import os
import ast
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt  # already present in the env

# ── MODEL IMPORT -------------------------------------------------------------
from rnn_models import LSTMModel
from NormativeModel import BayesianObserver

# ───────────────────────── configuration
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH: str = os.path.join(BASE_DIR, "models", "lstm_trained", "checkpoint.pt")
VARIANT_DIR: str = os.path.join(BASE_DIR, "variants")
HS_GRID: np.ndarray = np.arange(0.0, 1.0, 0.05)
MU1, MU2 = -1, 1
EPS: float = 1e-10  # numerical safety for log / div

# ───────────────────────── helpers

def get_default_hp() -> Dict[str, int]:
    """Return the default hyper‑parameter dictionary expected by LSTMModel."""
    return {"n_input": 1, "n_rnn": 128}


def load_model(hp: Dict[str, int]) -> LSTMModel:
    """Instantiate an LSTMModel, load weights, and set to eval‑mode."""
    model = LSTMModel(hp).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ───────────────────────── evaluation per CSV variant

def evaluate_variant(model: LSTMModel, csv_path: str) -> List[Dict]:
    """Run the trained LSTM on one task variant CSV and collect metrics."""
    df = pd.read_csv(csv_path)
    recs: List[Dict] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            ev_list = ast.literal_eval(row["evidence"])
            sigma = float(row["sigma"])
            haz_true = float(row["trueHazard"])
            rep_true = int(row["trueReport"])

            # ------------- LSTM forward pass --------------------------------
            x = (
                torch.tensor(ev_list, dtype=torch.float32)
                .unsqueeze(0)  # batch dim
                .unsqueeze(-1)  # feature dim (n_input = 1)
                .to(DEVICE)
            )
            loc_logits, haz_logits = model(x)  # (B,T,1), (B,1)

            loc_logit_last = loc_logits[0, -1, 0]
            prob_state2_net = torch.sigmoid(loc_logit_last).item()
            haz_pred_prob = torch.sigmoid(haz_logits)[0, 0].item()

            loc_pred = 1 if loc_logit_last.item() > 0 else -1
            rep_corr_net = int(loc_pred == rep_true)

            haz_err_net = haz_pred_prob - haz_true
            haz_acc_net = int(abs(haz_err_net) <= 0.10)
            haz_hilo_net = int((haz_pred_prob > 0.5) == (haz_true > 0.5))

            # ------------- Normative (ideal Bayesian) model -----------------
            L_haz, L_state, rep_norm, _ = BayesianObserver(
                ev_list, MU1, MU2, sigma, HS_GRID.copy()
            )

            #  • BCE probabilities
            p_state2_norm = float(L_state[1, -1])  # P(state = 2)
            p_state2_norm = np.clip(p_state2_norm, EPS, 1 - EPS)

            posterior = L_haz[:, -1]  # posterior over hazard grid
            haz_est_norm = float(np.dot(HS_GRID, posterior))  # E[h]
            p_haz_norm = np.clip(haz_est_norm, EPS, 1 - EPS)

            rep_corr_norm = int(rep_norm == rep_true)
            haz_err_norm = haz_est_norm - haz_true
            haz_acc_norm = int(abs(haz_err_norm) <= 0.10)
            haz_hilo_norm = int((haz_est_norm > 0.5) == (haz_true > 0.5))

            # ------------- Ideal‑observer BCE loss --------------------------
            y_rep = 0.5 * (rep_true + 1)  # −1/+1 → 0/1
            loss_rep = -(
                y_rep * np.log(p_state2_norm) + (1 - y_rep) * np.log(1 - p_state2_norm)
            )
            loss_haz = -(
                haz_true * np.log(p_haz_norm)
                + (1 - haz_true) * np.log(1 - p_haz_norm)
            )
            loss_total_norm = loss_rep + 0.5 * loss_haz

            recs.append(
                {
                    "sigma": sigma,
                    "hazard": haz_true,
                    # LSTM
                    "rep_corr_net": rep_corr_net,
                    "haz_acc_net": haz_acc_net,
                    "haz_hilo_net": haz_hilo_net,
                    "haz_err_net": haz_err_net,
                    # Normative
                    "rep_corr_norm": rep_corr_norm,
                    "haz_acc_norm": haz_acc_norm,
                    "haz_hilo_norm": haz_hilo_norm,
                    "haz_err_norm": haz_err_norm,
                    "loss_norm": loss_total_norm,
                }
            )
    return recs

# ───────────────────────── aggregation helpers

def aggregate(records: List[Dict]):
    """Group by (sigma, hazard) conditions and compute mean stats."""
    df = pd.DataFrame(records)
    grouped = (
        df.groupby(["sigma", "hazard"])
        .agg(
            N=("rep_corr_net", "size"),
            report_acc_net=("rep_corr_net", "mean"),
            hazard_acc_net=("haz_acc_net", "mean"),
            hazard_hilo_net=("haz_hilo_net", "mean"),
            report_acc_norm=("rep_corr_norm", "mean"),
            hazard_acc_norm=("haz_acc_norm", "mean"),
            hazard_hilo_norm=("haz_hilo_norm", "mean"),
            loss_norm=("loss_norm", "mean"),
        )
        .reset_index()
        .sort_values(["sigma", "hazard"])
    )
    return grouped, df

# ───────────────────────── main entry‑point

def main(max_variants: int = 40):
    model = load_model(get_default_hp())
    all_records: List[Dict] = []

    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            break
        all_records.extend(evaluate_variant(model, csv_path))

    table, raw_df = aggregate(all_records)

    # ---- print per‑condition table ----------------------------------------
    print("\nPer‑condition performance (LSTM vs Normative)")
    print("=" * 110)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    # ---- weighted overall metrics ----------------------------------------
    weights = table["N"]
    overall = {
        "LSTM": {
            "report_acc": np.average(table["report_acc_net"], weights=weights),
            "hazard_acc": np.average(table["hazard_acc_net"], weights=weights),
            "hazard_hilo": np.average(table["hazard_hilo_net"], weights=weights),
        },
        "Norm": {
            "report_acc": np.average(table["report_acc_norm"], weights=weights),
            "hazard_acc": np.average(table["hazard_acc_norm"], weights=weights),
            "hazard_hilo": np.average(table["hazard_hilo_norm"], weights=weights),
        },
    }
    print("\nOverall performance")
    for m, s in overall.items():
        print(
            f"{m}: report {s['report_acc']:.3%} | hazard ±0.10 {s['hazard_acc']:.3%} | hazard Hi/Lo {s['hazard_hilo']:.3%}"
        )

    mean_loss_norm = raw_df["loss_norm"].mean()
    print(f"\nIdeal observer mean BCE loss: {mean_loss_norm:.4e}")

    # ---- histogram of hazard‑prediction errors ---------------------------
    bins = np.arange(-1.0, 1.0001, 0.05)
    plt.figure(figsize=(8, 4.5))
    plt.hist(
        raw_df["haz_err_norm"],
        bins=bins,
        alpha=0.6,
        label="Bayesian",
        edgecolor="black",
    )
    plt.hist(
        raw_df["haz_err_net"],
        bins=bins,
        alpha=0.6,
        label="LSTM",
        edgecolor="black",
    )
    plt.axvline(0, color="k", linewidth=1)
    plt.xlabel("Prediction error: (predicted − true hazard)")
    plt.ylabel("Count")
    plt.title("Hazard prediction error distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

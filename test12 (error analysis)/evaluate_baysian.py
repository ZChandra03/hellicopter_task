"""evaluate_bayesian.py – Evaluate the Bayesian ideal observer only
-------------------------------------------------------------------
This script loops over pre‑generated task variants, runs the ideal Bayesian
observer on each one, and summarises its performance. It no longer loads or
examines any neural network model.
"""

import os
import ast
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # already present in the env

# ── NORMATIVE MODEL IMPORT ---------------------------------------------------
from NormativeModel import BayesianObserver

# ───────────────────────── configuration -------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR: str = os.path.join(BASE_DIR, "variants")
HS_GRID: np.ndarray = np.arange(0.0, 1.005, 0.005)
MU1, MU2 = -1, 1
EPS: float = 1e-10  # numerical safety for log / div

# ───────────────────────── evaluation per CSV variant ------------------------

def evaluate_variant(csv_path: str) -> List[Dict]:
    """Run the Bayesian observer on one task‑variant CSV and collect metrics."""
    df = pd.read_csv(csv_path)
    recs: List[Dict] = []

    for _, row in df.iterrows():
        ev_list = ast.literal_eval(row["evidence"])
        sigma = float(row["sigma"])
        haz_true = float(row["trueHazard"])
        rep_true = int(row["trueReport"])

        # ------------- Normative (ideal Bayesian) model ---------------------
        L_haz, L_state, rep_norm, _ = BayesianObserver(
            ev_list, MU1, MU2, sigma, HS_GRID.copy()
        )

        #  • report probability and hazard posterior
        p_state2_norm = float(L_state[1, -1])  # P(state = 2)
        p_state2_norm = np.clip(p_state2_norm, EPS, 1 - EPS)

        posterior = L_haz[:, -1]              # posterior over hazard grid
        haz_est_norm = float(np.dot(HS_GRID, posterior))  # E[h]
        p_haz_norm = np.clip(haz_est_norm, EPS, 1 - EPS)

        # ------------- summary statistics ----------------------------------
        rep_corr_norm = int(rep_norm == rep_true)
        haz_err_norm = haz_est_norm - haz_true
        haz_acc_norm = int(abs(haz_err_norm) <= 0.10)
        haz_hilo_norm = int((haz_est_norm > 0.5) == (haz_true > 0.5))

        # ------------- Ideal‑observer BCE loss ------------------------------
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
                "rep_corr_norm": rep_corr_norm,
                "haz_acc_norm": haz_acc_norm,
                "haz_hilo_norm": haz_hilo_norm,
                "haz_err_norm": haz_err_norm,
                "loss_norm": loss_total_norm,
            }
        )
    return recs

# ───────────────────────── aggregation helpers ------------------------------

def aggregate(records: List[Dict]):
    """Group by (sigma, hazard) and compute mean stats."""
    df = pd.DataFrame(records)
    grouped = (
        df.groupby(["sigma", "hazard"])
        .agg(
            N=("rep_corr_norm", "size"),
            report_acc_norm=("rep_corr_norm", "mean"),
            hazard_acc_norm=("haz_acc_norm", "mean"),
            hazard_hilo_norm=("haz_hilo_norm", "mean"),
            loss_norm=("loss_norm", "mean"),
        )
        .reset_index()
        .sort_values(["sigma", "hazard"])
    )
    return grouped, df

# ───────────────────────── main entry‑point ----------------------------------

def main(max_variants: int = 2):
    all_records: List[Dict] = []

    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            break
        all_records.extend(evaluate_variant(csv_path))
        print(f"Processed variant {k + 1}...")
    table, raw_df = aggregate(all_records)

    # ---- print per‑condition table ---------------------------------------
    print("\nPer‑condition performance (Bayesian observer)")
    print("=" * 90)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    # ---- weighted overall metrics ---------------------------------------
    weights = table["N"]
    overall = {
        "Bayesian": {
            "report_acc": np.average(table["report_acc_norm"], weights=weights),
            "hazard_acc": np.average(table["hazard_acc_norm"], weights=weights),
            "hazard_hilo": np.average(table["hazard_hilo_norm"], weights=weights),
        }
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
    plt.figure(figsize=(7, 4.0))
    plt.hist(
        raw_df["haz_err_norm"],
        bins=bins,
        alpha=0.75,
        label="Bayesian",
        edgecolor="black",
    )
    plt.axvline(0, color="k", linewidth=1)
    plt.xlabel("Prediction error: (predicted − true hazard)")
    plt.ylabel("Count")
    plt.title("Hazard prediction error – Bayesian observer")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

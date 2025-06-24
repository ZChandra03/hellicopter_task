"""evaluate.py – compare GRU vs Normative model and plot error histograms
-----------------------------------------------------------------------"""
import os, ast
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt             # ← NEW import

from gru_model import GRUModel
from NormativeModel import BayesianObserver

# ───────────────────────── configuration
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "gru_trained", "checkpoint.pt")
VARIANT_DIR  = os.path.join(BASE_DIR, "variants")
HS_GRID      = np.arange(0, 1.0, 0.05)
MU1, MU2     = -1, 1

# ───────────────────────── helpers
def get_default_hp() -> Dict[str, int]:
    return {"n_input": 1, "n_rnn": 128}

def load_model(hp: Dict[str, int]) -> GRUModel:
    m = GRUModel(hp).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m

def _row_to_tensor(ev_str: str) -> torch.Tensor:
    x = torch.tensor(ast.literal_eval(ev_str), dtype=torch.float32)
    return x.unsqueeze(0).unsqueeze(-1)      # (1, T, 1)

# ───────────────────────── evaluation
def evaluate_variant(model: GRUModel, csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    recs = []
    with torch.no_grad():
        for _, row in df.iterrows():
            ev_list  = ast.literal_eval(row["evidence"])
            sigma    = float(row["sigma"])
            haz_true = float(row["trueHazard"])
            rep_true = int(row["trueReport"])

            # ---- GRU -------------------------------------------------------
            x = torch.tensor(ev_list, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            loc_logits, haz_logits = model(x)

            loc_pred       =  1 if loc_logits[0, -1, 0].item() > 0 else -1
            rep_corr_gru   = int(loc_pred == rep_true)

            haz_pred_prob  = torch.sigmoid(haz_logits)[0, 0].item()
            haz_err_gru    = haz_pred_prob - haz_true
            haz_acc_gru    = int(abs(haz_err_gru) <= 0.10)
            haz_hilo_gru   = int((haz_pred_prob > 0.5) == (haz_true > 0.5))

            # ---- Normative model ------------------------------------------
            L_haz, _, rep_norm, _ = BayesianObserver(ev_list, MU1, MU2, sigma, HS_GRID.copy())
            rep_corr_norm = int(rep_norm == rep_true)

            posterior      = L_haz[:, -1]
            haz_est_norm   = float(np.dot(HS_GRID, posterior))
            haz_err_norm   = haz_est_norm - haz_true
            haz_acc_norm   = int(abs(haz_err_norm) <= 0.10)
            haz_hilo_norm  = int((haz_est_norm > 0.5) == (haz_true > 0.5))

            recs.append({
                "sigma"            : sigma,
                "hazard"           : haz_true,
                "rep_corr_gru"     : rep_corr_gru,
                "haz_acc_gru"      : haz_acc_gru,
                "haz_hilo_gru"     : haz_hilo_gru,
                "haz_err_gru"      : haz_err_gru,     # ← NEW
                "rep_corr_norm"    : rep_corr_norm,
                "haz_acc_norm"     : haz_acc_norm,
                "haz_hilo_norm"    : haz_hilo_norm,
                "haz_err_norm"     : haz_err_norm,    # ← NEW
            })
    return recs

# ───────────────────────── aggregation
def aggregate(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    grouped = (
        df.groupby(["sigma", "hazard"])
          .agg(N                = ("rep_corr_gru", "size"),
               report_acc_gru   = ("rep_corr_gru",  "mean"),
               hazard_acc_gru   = ("haz_acc_gru",   "mean"),
               hazard_hilo_gru  = ("haz_hilo_gru",  "mean"),
               report_acc_norm  = ("rep_corr_norm", "mean"),
               hazard_acc_norm  = ("haz_acc_norm",  "mean"),
               hazard_hilo_norm = ("haz_hilo_norm", "mean"))
          .reset_index()
          .sort_values(["sigma", "hazard"])
    )
    return grouped, df            # return raw df for histograms too

# ───────────────────────── main
def main(max_variants: int = 40):
    model = load_model(get_default_hp())
    all_recs: List[Dict] = []

    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            break
        all_recs.extend(evaluate_variant(model, csv_path))

    table, raw_df = aggregate(all_recs)

    # ---- print tables ------------------------------------------------------
    print("\nPer-condition performance (GRU vs Normative)")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    weights = table["N"]
    overall = {
        "GRU" : {
            "report_acc": np.average(table["report_acc_gru"],  weights=weights),
            "hazard_acc": np.average(table["hazard_acc_gru"],  weights=weights),
            "hazard_hilo":np.average(table["hazard_hilo_gru"], weights=weights),
        },
        "Norm": {
            "report_acc": np.average(table["report_acc_norm"], weights=weights),
            "hazard_acc": np.average(table["hazard_acc_norm"], weights=weights),
            "hazard_hilo":np.average(table["hazard_hilo_norm"],weights=weights),
        }
    }
    print("\nOverall performance")
    for m, s in overall.items():
        print(f"{m}: report {s['report_acc']:.3%} | hazard ±0.10 {s['hazard_acc']:.3%} | hazard Hi/Lo {s['hazard_hilo']:.3%}")

    # ---- plot histogram ----------------------------------------------------
    bins = np.arange(-1.0, 1.0001, 0.05)
    plt.figure(figsize=(8, 4.5))
    plt.hist(raw_df["haz_err_norm"], bins=bins, alpha=0.6, label="Bayesian", edgecolor="black")
    plt.hist(raw_df["haz_err_gru"],  bins=bins, alpha=0.6, label="GRU",      edgecolor="black")
    plt.axvline(0, color="k", linewidth=1)
    plt.xlabel("Prediction error: (predicted − true hazard)")
    plt.ylabel("Count")
    plt.title("Hazard prediction error distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────
if __name__ == "__main__":
    main()

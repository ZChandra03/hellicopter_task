"""evaluate.py – location accuracy & hazard accuracy/MAE of a trained GRUModel"""
import os, ast
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from gru_model import GRUModel

# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gru_trained", "checkpoint.pt")
VARIANT_DIR = os.path.join(BASE_DIR, "variants")

# -----------------------------------------------------------------------------
def get_default_hp() -> Dict[str, int]:
    return {"n_input": 1, "n_rnn": 128}

# -----------------------------------------------------------------------------
def load_model(hp: Dict[str, int], ckpt_path: str = MODEL_PATH) -> GRUModel:
    model = GRUModel(hp).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model

# -----------------------------------------------------------------------------
def _row_to_tensor(evidence_str: str) -> torch.Tensor:
    x = torch.tensor(ast.literal_eval(evidence_str), dtype=torch.float32)
    return x.unsqueeze(0).unsqueeze(-1)        # (1, T, 1)

# ──────────────────────────────────────────────────────────────────────────────
def evaluate_variant(model: GRUModel, csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    records = []
    with torch.no_grad():
        for _, row in df.iterrows():
            x = _row_to_tensor(row["evidence"]).to(DEVICE)
            loc_logits, haz_logits = model(x)

            # location head
            loc_pred =  1 if loc_logits[0, -1, 0].item() > 0 else -1
            rep_correct = int(loc_pred == row["trueReport"])

            # hazard head
            haz_pred_prob = torch.sigmoid(haz_logits)[0, 0].item()
            haz_true      = float(row["trueHazard"])
            haz_abs_err   = abs(haz_pred_prob - haz_true)
            haz_correct   = int(haz_abs_err <= 0.10)

            # NEW: high/low correctness (threshold 0.5)
            haz_hilo_correct = int((haz_pred_prob > 0.5) == (haz_true > 0.5))

            records.append({
                "sigma"          : row["sigma"],
                "hazard"         : haz_true,
                "rep_correct"    : rep_correct,
                "haz_abs_err"    : haz_abs_err,
                "haz_correct"    : haz_correct,
                "haz_hilo_corr"  : haz_hilo_correct,   # ← added
            })
    return records

# ──────────────────────────────────────────────────────────────────────────────
def aggregate(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No evaluation data found – check testConfig_var*.csv")

    grouped = (
        df.groupby(["sigma", "hazard"])
          .agg(N            =("rep_correct", "size"),
               report_acc   =("rep_correct", "mean"),
               hazard_acc   =("haz_correct", "mean"),
               hazard_hilo  =("haz_hilo_corr", "mean"),   # ← added
               hazard_mae   =("haz_abs_err", "mean"))
          .reset_index()
          .sort_values(["sigma", "hazard"])
    )
    return grouped

# ──────────────────────────────────────────────────────────────────────────────
def main(max_variants: int = 40):
    model = load_model(get_default_hp())

    all_records: List[Dict] = []
    for k in range(max_variants):
        csv_path = os.path.join(VARIANT_DIR, f"testConfig_var{k}.csv")
        if not os.path.isfile(csv_path):
            break
        all_records.extend(evaluate_variant(model, csv_path))

    table = aggregate(all_records)
    print("\nPer-condition performance")
    print("=" * 60)
    print(table.to_string(index=False, float_format="{:.3f}".format))

    overall = table.agg({
        "N"           : "sum",
        "report_acc"  : lambda x: np.average(x, weights=table["N"]),
        "hazard_acc"  : lambda x: np.average(x, weights=table["N"]),
        "hazard_hilo" : lambda x: np.average(x, weights=table["N"]),
        "hazard_mae"  : lambda x: np.average(x, weights=table["N"])})

    print("\nOverall performance:")
    print("  location accuracy        = {:.3%}".format(overall["report_acc"]))
    print("  hazard accuracy (±0.10)  = {:.3%}".format(overall["hazard_acc"]))
    print("  hazard high/low accuracy = {:.3%}".format(overall["hazard_hilo"]))
    print("  hazard MAE               = {:.4f}".format(overall["hazard_mae"]))


if __name__ == "__main__":
    main()
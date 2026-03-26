# table.py
# Compute simple (unweighted) mean accuracy across bins for each model
# Input : figures_MTS/binned_acc_uniform_means.csv
# Output: figures_MTS/overall_accuracy_by_model.csv

import os, re, json
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
IN_CSV  = HERE / "figures_OTS" / "binned_acc_uniform_means.csv"
OUT_CSV = HERE / "figures_OTS" / "overall_accuracy_by_model.csv"
EXPLAIN = HERE / "figures_OTS" / "overall_accuracy_explanation.json"

def _is_num(col_series): return pd.api.types.is_numeric_dtype(col_series)

def _is_likely_bin_name(col: str) -> bool:
    cl = col.lower()
    if "bin" in cl: return True
    if re.search(r"\d+(\.\d+)?\s*-\s*\d+(\.\d+)?", col): return True   # e.g. "0.3-0.7"
    if re.search(r"[\(\[\]\)]", col) and re.search(r"\d", col): return True  # e.g. "[0.3,0.4)"
    return False

def compute_overall_accuracy_per_model(df: pd.DataFrame):
    # 1) pick model column
    model_col = None
    for c in ["model", "Model", "model_name", "name", "run_name", "arch", "variant"]:
        if c in df.columns:
            model_col = c; break
    if model_col is None:
        non_num = [c for c in df.columns if not _is_num(df[c])]
        if non_num: model_col = non_num[0]
        else:
            model_col = "_model_idx"
            df[model_col] = np.arange(len(df))

    # 2) detect long vs wide
    acc_singletons = ["accuracy","acc","binned_accuracy","bin_accuracy","mean_accuracy","avg_accuracy","accuracy_mean"]
    singleton_acc = next((c for c in df.columns if c.lower() in acc_singletons), None)
    bin_col = next((c for c in df.columns if c.lower() in ["bin","bucket","bin_id","bin_index","range"]), None)

    acc_like_cols = [c for c in df.columns if "acc" in c.lower()]
    numeric_cols  = [c for c in df.columns if _is_num(df[c])]
    exclude = {model_col, "seed", "epoch", "step", "fold"}
    likely_bin_cols = [c for c in df.columns if c not in exclude and _is_likely_bin_name(c)]

    explain = {"model_column_used": model_col}

    if singleton_acc and bin_col:
        fmt = "long"
        used_acc_cols = [singleton_acc]
        overall = df.groupby(model_col, dropna=False)[singleton_acc].mean(numeric_only=True)
    elif singleton_acc and len(acc_like_cols) == 1:
        fmt = "long"
        used_acc_cols = [singleton_acc]
        overall = df.groupby(model_col, dropna=False)[singleton_acc].mean(numeric_only=True)
    else:
        fmt = "wide"
        if len(acc_like_cols) > 1:
            used_acc_cols = acc_like_cols
        elif likely_bin_cols:
            used_acc_cols = [c for c in likely_bin_cols if c in numeric_cols] or \
                            [c for c in numeric_cols if c not in exclude]
        else:
            # fallback: all numeric except obvious non-bin cols
            used_acc_cols = [c for c in numeric_cols if c not in exclude]

        # coerce numeric just in case
        for c in used_acc_cols:
            if not _is_num(df[c]): df[c] = pd.to_numeric(df[c], errors="coerce")
        df["_row_bin_mean"] = df[used_acc_cols].mean(axis=1, skipna=True)
        overall = df.groupby(model_col, dropna=False)["_row_bin_mean"].mean()

    explain.update({
        "format": fmt,
        "accuracy_columns_used": used_acc_cols if isinstance(used_acc_cols, list) else [used_acc_cols],
        "bin_column_used": bin_col
    })

    res = overall.reset_index()
    res.columns = [model_col, "overall_accuracy_mean_across_bins"]
    res = res.sort_values("overall_accuracy_mean_across_bins", ascending=False)
    return res, explain

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Expected CSV at: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    result, info = compute_overall_accuracy_per_model(df)

    # round for readability and save
    result_disp = result.copy()
    result_disp["overall_accuracy_mean_across_bins"] = result_disp["overall_accuracy_mean_across_bins"].round(6)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_disp.to_csv(OUT_CSV, index=False)
    with open(EXPLAIN, "w") as f:
        json.dump(info, f, indent=2)

    # pretty print
    print("\n=== Overall Accuracy by Model (mean across bins) ===")
    print(result_disp.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")
    print(f"Details: {EXPLAIN}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# compare_to_bayes.py ----------------------------------------------------------
# Aggregate similarity stats and plots for all 7 models across 40 variants
# -----------------------------------------------------------------------------
import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────── paths
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))      # folder *this* file lives in
OUT_DIR:  str = BASE_DIR                                        # where figures will be saved
VARIANT_DIR: str = os.path.join(BASE_DIR, "variants")           # adjust if you move CSVs
VARIANT_GLOB = os.path.join(VARIANT_DIR, "testConfig_var*_preds.csv")
MODELS = ["norm", "gru", "lstm", "rnn", "gru1", "lstm1", "rnn1"]

# ───────────────────────────────── load & concat
files = sorted(glob.glob(VARIANT_GLOB))
if not files:
    raise FileNotFoundError("No *_preds.csv files found – run add_predictions.py first")

df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Loaded {len(df_all):,} trials from {len(files)} variant files\n")

# ───────────────────────────────── metrics + plots
metrics = []
for m in MODELS:
    if m == "norm":   # Bayes vs itself: perfect reference line
        continue
    ok_cols = {f"rep_{m}", f"haz_{m}", "rep_norm", "haz_norm"}
    if not ok_cols.issubset(df_all.columns):
        print(f"[warn] columns for model {m} missing – skipping")
        continue

    site_agree = (df_all[f"rep_{m}"] == df_all["rep_norm"]).mean()
    haz_rmse   = np.sqrt(((df_all[f"haz_{m}"] - df_all["haz_norm"])**2).mean())
    haz_corr   = np.corrcoef(df_all[f"haz_{m}"], df_all["haz_norm"])[0, 1]

    metrics.append((m, site_agree, haz_rmse, haz_corr))

    # ───────── scatter plot
    plt.figure(figsize=(4, 4))
    plt.scatter(df_all["haz_norm"], df_all[f"haz_{m}"], s=8, alpha=0.25)
    plt.plot([0, 1], [0, 1], ls="--")          # identity line
    plt.xlabel("Bayesian hazard")
    plt.ylabel(f"{m.upper()} hazard")
    plt.title(f"{m.upper()} vs Bayes\nr = {haz_corr:.3f}, RMSE = {haz_rmse:.3f}")
    plt.tight_layout()

    # save alongside this script, not in the working dir
    fig_path = os.path.join(OUT_DIR, f"scatter_{m}.png")
    plt.savefig(fig_path, dpi=180)
    plt.show()
    print(f"[saved] {fig_path}")

# ───────────────────────────────── print summary
print("\nSummary across all variants\n"
      "model   siteAgree  hazardRMSE  hazardCorr")
for m, acc, rmse, r in metrics:
    print(f"{m:6}  {acc:9.3f}  {rmse:10.3f}  {r:11.3f}")

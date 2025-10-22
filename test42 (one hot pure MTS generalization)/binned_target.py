#!/usr/bin/env python3
# plot_03_07_from_csv_seeds.py
# Makes 4 plots using:
#  - per-seed CSV for TARGET group (Uniform 0.3–0.7)
#  - means CSV for BASELINE group (Uniform 0.0–1.0) for delta plots
#
# Defaults:
#   PER_SEED_CSV = figures/binned_acc_uniform_per_seed.csv
#   MEANS_CSV    = figures/binned_acc_uniform_means.csv
#
# CLI:
#   python plot_03_07_from_csv_seeds.py [PER_SEED_CSV] [MEANS_CSV]

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "figures_MTS")
os.makedirs(OUT_DIR, exist_ok=True)

# Defaults (override via argv)
PER_SEED_CSV = os.path.join(BASE_DIR, "figures_MTS", "binned_acc_uniform_per_seed.csv")
MEANS_CSV    = os.path.join(BASE_DIR, "figures_MTS", "binned_acc_uniform_means.csv")

TARGET_KEY   = "hz_0p3_0p7"             # Uniform 0.3–0.7
BASELINE_KEY = "hz_flat_0_1"            # Uniform 0.0–1.0 (flat)
TARGET_LABEL = "Uniform 0.3–0.7"
BASE_LABEL   = "Uniform 0.0–1.0 (flat)"

THIN_ALPHA   = 0.30
THIN_LW      = 0.9
MEAN_LW      = 2.6

def infer_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"CSV missing required column; tried {candidates}")
    return None

def load_per_seed(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Per-seed CSV not found: {path}")
    df = pd.read_csv(path)

    # Flexible column names
    group_col  = infer_col(df, ["group_folder", "group", "folder"])
    seed_col   = infer_col(df, ["seed", "seed_id"])
    bin_col    = infer_col(df, ["bin_center", "bin", "hazard_bin_center"])
    rep_col    = infer_col(df, ["report_acc", "report_accuracy", "report_acc_mean"])
    haz_col    = infer_col(df, ["hazard_acc", "hazard_accuracy", "hazard_acc_mean"])

    df = df[[group_col, seed_col, bin_col, rep_col, haz_col]].rename(columns={
        group_col: "group_folder",
        seed_col:  "seed",
        bin_col:   "bin_center",
        rep_col:   "report_acc",
        haz_col:   "hazard_acc",
    })
    return df

def load_means(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Flexible columns
    group_col  = infer_col(df, ["group_folder", "group", "folder"])
    bin_col    = infer_col(df, ["bin_center", "bin", "hazard_bin_center"])
    rep_mean   = infer_col(df, ["report_acc_mean", "report_mean", "report_accuracy_mean"])
    haz_mean   = infer_col(df, ["hazard_acc_mean", "hazard_mean", "hazard_accuracy_mean"])
    df = df[[group_col, bin_col, rep_mean, haz_mean]].rename(columns={
        group_col: "group_folder",
        bin_col:   "bin_center",
        rep_mean:  "report_acc_mean",
        haz_mean:  "hazard_acc_mean",
    })
    return df

def plot_abs(x, curves, mean_curve, ylabel, title, outfile):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    # per-seed thin lines
    for y in curves:
        ax.plot(x, y, alpha=THIN_ALPHA, linewidth=THIN_LW)
    # bold mean
    ax.plot(x, mean_curve, linewidth=MEAN_LW, label=TARGET_LABEL)
    ax.set_xlabel("True hazard (bin centers)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{outfile}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def plot_delta(x, seed_delta_curves, mean_delta, ylabel, title, outfile):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for y in seed_delta_curves:
        ax.plot(x, y, alpha=THIN_ALPHA, linewidth=THIN_LW)
    ax.plot(x, mean_delta, linewidth=MEAN_LW, label=f"{TARGET_LABEL} − {BASE_LABEL}")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.6)
    ax.set_xlabel("True hazard (bin centers)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{outfile}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def main():
    # CLI override
    per_seed_csv = PER_SEED_CSV if len(sys.argv) < 2 else sys.argv[1]
    means_csv    = MEANS_CSV    if len(sys.argv) < 3 else sys.argv[2]

    df_seeds = load_per_seed(per_seed_csv)
    df_tgt   = df_seeds[df_seeds["group_folder"] == TARGET_KEY]
    if df_tgt.empty:
        raise RuntimeError(f"No rows for group_folder='{TARGET_KEY}' in per-seed CSV.")

    # Ensure deterministic bin order
    df_tgt = df_tgt.sort_values(["bin_center", "seed"])
    x = np.sort(df_tgt["bin_center"].unique())

    # Build per-seed curves aligned on bin centers
    rep_curves, haz_curves = [], []
    for seed, sub in df_tgt.groupby("seed"):
        sub = sub.sort_values("bin_center")
        # align by merging with x to handle any missing bins
        aligned = pd.DataFrame({"bin_center": x}).merge(sub, on="bin_center", how="left")
        rep_curves.append(aligned["report_acc"].to_numpy())
        haz_curves.append(aligned["hazard_acc"].to_numpy())
    rep_curves = np.vstack(rep_curves)   # (n_seeds, n_bins)
    haz_curves = np.vstack(haz_curves)

    # Mean across seeds
    rep_mean = np.nanmean(rep_curves, axis=0)
    haz_mean = np.nanmean(haz_curves, axis=0)

    # 1) Report abs (0.3–0.7 only)
    plot_abs(
        x, rep_curves, rep_mean,
        ylabel="Report-head accuracy",
        title="Accuracy vs hazard — Uniform 0.3–0.7 (per-seed + mean)",
        outfile="report_head_accuracy_by_hazard_uniform_03_07_only"
    )

    # 2) Hazard abs (0.3–0.7 only)
    plot_abs(
        x, haz_curves, haz_mean,
        ylabel="Hazard-head accuracy (threshold 0.5)",
        title="Hazard-head vs hazard — Uniform 0.3–0.7 (per-seed + mean)",
        outfile="hazard_head_accuracy_by_hazard_uniform_03_07_only"
    )

    # 3–4) Delta vs flat baseline means
    df_means = load_means(means_csv)
    if df_means is None:
        print("[warn] Means CSV not found; skipping delta plots.")
        return
    base = df_means[df_means["group_folder"] == BASELINE_KEY].sort_values("bin_center")
    if base.empty:
        print(f"[warn] No baseline rows for '{BASELINE_KEY}' in means CSV; skipping delta plots.")
        return

    # Align baseline to x
    base_aligned = pd.DataFrame({"bin_center": x}).merge(base, on="bin_center", how="left")
    base_rep = base_aligned["report_acc_mean"].to_numpy()
    base_haz = base_aligned["hazard_acc_mean"].to_numpy()

    # Per-seed deltas (seed curve minus baseline mean)
    rep_delta_curves = rep_curves - base_rep
    haz_delta_curves = haz_curves - base_haz
    rep_delta_mean   = rep_mean - base_rep
    haz_delta_mean   = haz_mean - base_haz

    plot_delta(
        x, rep_delta_curves, rep_delta_mean,
        ylabel="Δ Report accuracy vs flat (mean baseline)",
        title="Report-head: Δ accuracy — Uniform 0.3–0.7 vs flat",
        outfile="report_head_accuracy_delta_vs_flat_03_07"
    )
    plot_delta(
        x, haz_delta_curves, haz_delta_mean,
        ylabel="Δ Hazard-head accuracy vs flat (mean baseline)",
        title="Hazard-head: Δ accuracy — Uniform 0.3–0.7 vs flat",
        outfile="hazard_head_accuracy_delta_vs_flat_03_07"
    )

if __name__ == "__main__":
    main()

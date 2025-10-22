#!/usr/bin/env python3
# make_hazard_head_from_per_seed_only.py
# Rebuilds hazard_head_accuracy_by_hazard_beta1.png using ONLY:
#   ./binned_acc_beta1_per_seed_hazard.csv
#
# Expects columns: model_folder, label, seed, bin_center, hazard_acc

import itertools
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH   = SCRIPT_DIR / "binned_acc_beta1_per_seed_hazard.csv"
OUT_PNG    = SCRIPT_DIR / "hazard_head_accuracy_by_hazard_beta1_TEST.png"

# Preferred colors (fallbacks will be auto-assigned if new labels appear)
PREFERRED_COLOR = {
    "Beta(0.1,0.1)": "tab:blue",
    "Beta(0.5,0.5)": "tab:orange",
    "Beta(1.0,1.0)": "tab:green",
    "Beta(2.0,2.0)": "tab:red",
    "Beta(10,10)" : "tab:purple",
}

def main():
    if not CSV_PATH.exists():
        # Helpful error that shows what's actually in the folder
        found = "\n".join(sorted(p.name for p in SCRIPT_DIR.glob("*.csv")))
        raise FileNotFoundError(
            f"Couldn't find {CSV_PATH.name} next to this script.\n"
            f"CSV files here:\n{found if found else '(none)'}"
        )

    df = pd.read_csv(CSV_PATH)

    required = {"model_folder", "label", "seed", "bin_center", "hazard_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    # Keep only needed columns & sort for nice lines
    df = df[["label", "seed", "bin_center", "hazard_acc"]].copy()
    df = df.sort_values(["label", "seed", "bin_center"])

    # Build plotting order: honor preferred labels first, then any extras
    labels_in_csv = list(df["label"].dropna().unique())
    ordered_labels = [lbl for lbl in PREFERRED_COLOR if lbl in labels_in_csv]
    ordered_labels += [lbl for lbl in labels_in_csv if lbl not in ordered_labels]

    # Assign colors: preferred first, then cycle through matplotlib defaults
    cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    color_for = {lbl: PREFERRED_COLOR.get(lbl, next(cycle)) for lbl in ordered_labels}

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for lbl in ordered_labels:
        sub = df[df["label"] == lbl]

        # Per-seed faint lines
        for seed, g in sub.groupby("seed"):
            ax.plot(
                g["bin_center"].values,
                g["hazard_acc"].values,
                linewidth=0.9,
                alpha=0.30,
                color=color_for[lbl],
            )

        # Bold mean line across seeds
        mean_df = (
            sub.groupby("bin_center", as_index=False)["hazard_acc"]
               .mean()
               .sort_values("bin_center")
        )
        ax.plot(
            mean_df["bin_center"].values,
            mean_df["hazard_acc"].values,
            linewidth=2.6,
            label=lbl,
            color=color_for[lbl],
        )

    ax.set_xlabel("True hazard (bin centers, width = 0.05)")
    ax.set_ylabel("Hazard-head accuracy (threshold 0.5)")
    ax.set_title("Hazard-head vs hazard — tested on beta_1p0, per-seed + mean")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_val_trials.py
==================
Plot the first 5 trials from valConfig_0.csv for a hardcoded sigma group.

Layout:
- horizontal axis: evidence value
- vertical axis: time step (increasing upward)

The first 5 trials are plotted side by side in a single figure.
"""

from __future__ import annotations

import ast
import os

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(BASE_DIR, "variants")

HARDCODED_SIGMA = 3
GROUP_KEY = f"sigma_{HARDCODED_SIGMA}"
VAL_INDEX = 0
N_TRIALS = 5
SAVE_PLOTS = False
OUTPUT_DIR = os.path.join(BASE_DIR, "trial_plots")

SIDE_COLORS = {
    -1: "blue",
    1: "red",
}


def parse_evidence(value) -> list[float]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    raise TypeError(f"Unsupported evidence type: {type(value)}")


def load_val_df(group_key: str, val_index: int) -> pd.DataFrame:
    path = os.path.join(VARIANTS_DIR, group_key, f"valConfig_{val_index}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing validation file: {path}")
    return pd.read_csv(path)


def side_color(value: int) -> str:
    return SIDE_COLORS[1 if int(value) >= 0 else -1]


def plot_trials(df: pd.DataFrame, group_key: str) -> None:
    n_to_plot = min(N_TRIALS, len(df))
    sigma = HARDCODED_SIGMA
    xlim = 2 * sigma + 1
    fig, axes = plt.subplots(1, n_to_plot, figsize=(4 * n_to_plot, 7), sharey=True)

    if n_to_plot == 1:
        axes = [axes]

    for trial_idx, ax in enumerate(axes):
        row = df.iloc[trial_idx]
        evidence = parse_evidence(row["evidence"])
        true_report = int(row["trueReport"])
        true_predict = int(row["truePredict"])
        time_steps = list(range(len(evidence)))

        states = parse_evidence(row["states"])
        if len(states) != len(evidence):
            raise ValueError(
                f"Length mismatch in trial {trial_idx}: "
                f"len(states)={len(states)} vs len(evidence)={len(evidence)}"
            )
        point_colors = [side_color(s) for s in states]

        ax.axvspan(-xlim, 0, alpha=0.08, color="blue")
        ax.axvspan(0, xlim, alpha=0.08, color="red")
        ax.plot(evidence, time_steps, linewidth=1)
        ax.scatter(evidence, time_steps, c=point_colors, s=35)
        ax.axvline(-1, linestyle="--", linewidth=1)
        ax.axvline(1, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle=":", linewidth=1)

        ax.set_xlabel("Evidence")
        ax.set_title(f"trial {trial_idx}")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-0.5, len(evidence) - 0.5)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02,
            0.98,
            f"rep: {true_report}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=side_color(true_report),
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.92,
            f"pred: {true_predict}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=side_color(true_predict),
            fontweight="bold",
        )

    axes[0].set_ylabel("Time step")
    fig.suptitle(f"{group_key} | valConfig_0 | first {n_to_plot} trials")
    fig.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{group_key}_val0_first_{n_to_plot}_trials.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()


def main() -> None:
    df = load_val_df(GROUP_KEY, VAL_INDEX)
    plot_trials(df, GROUP_KEY)


if __name__ == "__main__":
    main()

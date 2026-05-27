#!/usr/bin/env python3
"""
plot_report_head_during_trial.py
"""

from __future__ import annotations

import os
import ast
import json
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from rnn_models import GRUModel


# ----------------------------- root path setup -----------------------------
# Root = folder this script is in. If __file__ is unavailable, fall back to cwd.
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()

# Model and test file paths built relative to ROOT
MODEL_DIR = os.path.join(ROOT, "hz_flat_0_1", "seed_0")
TEST_CSV  = os.path.join(ROOT, "variants", "sigma_1", "testConfig_0.csv")

N_SAMPLES = 5
RANDOM_SEED = 0
SAVE_PATH = os.path.join(ROOT, "report_head_5_trials.png")   # or None
# -------------------------------------------------------------------------


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_list_cell(x):
    if isinstance(x, list):
        return x
    return ast.literal_eval(str(x))


def load_model(model_dir: str):
    hp_path = os.path.join(model_dir, "hp.json")
    ckpt_path = os.path.join(model_dir, "final.pt")

    if not os.path.exists(hp_path):
        raise FileNotFoundError(f"hp.json not found: {hp_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"final.pt not found: {ckpt_path}")

    with open(hp_path, "r") as f:
        hp = json.load(f)

    model = GRUModel(hp).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, hp


@torch.no_grad()
def run_one_trial(model, evidence: List[float]) -> Dict[str, np.ndarray]:
    x = torch.tensor(evidence, dtype=torch.float32, device=DEVICE).view(1, -1, 1)
    loc_logits, _ = model(x)   # (1, T, 1)

    logits = loc_logits[0, :, 0].cpu().numpy()
    p_plus = 1.0 / (1.0 + np.exp(-logits))
    signed = 2.0 * p_plus - 1.0

    return {
        "logits": logits,
        "p_plus": p_plus,
        "signed": signed,
    }


def sample_trials(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    idx = list(df.index)
    n = min(n, len(idx))
    chosen = rng.sample(idx, n)
    return df.loc[chosen].reset_index(drop=True)


def plot_trials(df: pd.DataFrame, model, save_path: str | None = None):
    n = len(df)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 3.6 * n), squeeze=False)

    for i, row in df.iterrows():
        evidence = parse_list_cell(row["evidence"])
        states = parse_list_cell(row["states"])

        out = run_one_trial(model, evidence)

        T = len(evidence)
        t = np.arange(1, T + 1)

        ax_e = axes[i, 0]
        ax_s = axes[i, 1]

        ax_e.plot(t, evidence, marker="o", linewidth=1.5)
        ax_e.axhline(0.0, linewidth=1.0, alpha=0.5)
        ax_e.set_xlim(1, T)
        ax_e.set_xlabel("Datapoint")
        ax_e.set_ylabel("Evidence")
        ax_e.set_title(
            f"Trial {i+1} | trueHazard={row.get('trueHazard', np.nan):.3f} | "
            f"trueReport={row.get('trueReport', np.nan):.0f}"
        )

        ax_s.step(t, states, where="mid", linewidth=2.0, label="True state (-1/+1)")
        ax_s.plot(
            t, out["signed"], marker="o", linewidth=1.5,
            label="Report head reading (2*sigmoid(logit)-1)"
        )
        ax_s.axhline(0.0, linewidth=1.0, alpha=0.5)
        ax_s.set_xlim(1, T)
        ax_s.set_ylim(-1.1, 1.1)
        ax_s.set_xlabel("Datapoint")
        ax_s.set_ylabel("State / signed reading")
        ax_s.set_title("Online report-head reading vs true state")

        ax_p = ax_s.twinx()
        ax_p.plot(t, out["p_plus"], linestyle="--", alpha=0.7, label="P(state=+1)")
        ax_p.set_ylim(-0.02, 1.02)
        ax_p.set_ylabel("P(state=+1)")

        lines1, labels1 = ax_s.get_legend_handles_labels()
        lines2, labels2 = ax_p.get_legend_handles_labels()
        ax_s.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.suptitle("Report head reading during trial for 5 sampled test cases", y=0.995)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


def main():
    print(f"ROOT     : {ROOT}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"TEST_CSV : {TEST_CSV}")

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    model, hp = load_model(MODEL_DIR)
    print(f"Loaded model from: {MODEL_DIR}")
    print(f"Using hp: {hp}")

    df = pd.read_csv(TEST_CSV)
    df_small = sample_trials(df, N_SAMPLES, RANDOM_SEED)

    plot_trials(df_small, model, SAVE_PATH)


if __name__ == "__main__":
    main()

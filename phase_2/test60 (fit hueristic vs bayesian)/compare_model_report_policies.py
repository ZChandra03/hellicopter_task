#!/usr/bin/env python3
"""
Compare trained report heads against heuristic and normative report policies.

This evaluates the two sigma_1 model sets:
    models_OTS/bce_rep/sigma_1_true
    models_OTS/bce_rep/sigma_1_hueristic

For each seed, the script loads a checkpoint, runs the GRU report head on the
held-out testConfig_*.csv files, and measures how closely the model's final
report response matches:
    1. heuristic report: sign of the final evidence sample
    2. normative report: BayesianObserver(...).resp_Rep

Outputs are saved in policy_match_results/ by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from NormativeModel import BayesianObserver
from rnn_models import GRUModel


BASE_DIR = Path(__file__).resolve().parent
VARIANTS_DIR = BASE_DIR / "variants"
MODELS_ROOT = BASE_DIR / "models_OTS" / "bce_rep"

DEFAULT_MODEL_SETS = {
    "bayesian_trained": MODELS_ROOT / "sigma_1_true",
    "heuristic_trained": MODELS_ROOT / "sigma_1_hueristic",
}

GROUP_KEY = "sigma_1"
TEST_IDXS = range(20)
SEEDS = range(10)
CHECKPOINT_NAME = "checkpoint_best.pt"
RESULTS_DIR = BASE_DIR / "policy_match_results"
FINAL_EVIDENCE_WINDOW = 0.2

MU1 = -1
MU2 = 1
HS = np.arange(0, 1, 0.05)
BIAS = 0
EPS = 1e-7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_evidence(value) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        return [float(v) for v in ast.literal_eval(value)]
    raise TypeError(f"Unsupported evidence type: {type(value)}")


def sign_to_label01(value: int | float) -> int:
    return 1 if float(value) > 0 else 0


def label01_to_sign(value: int | float) -> int:
    return 1 if int(value) == 1 else -1


def heuristic_report_sign(evidence: list[float]) -> int:
    return 1 if evidence[-1] >= 0 else -1


def normative_report_sign(evidence: list[float], sigma: float) -> int:
    _, _, resp_rep, _ = BayesianObserver(
        evidence,
        mu1=MU1,
        mu2=MU2,
        sigma=sigma,
        hs=HS,
        bias=BIAS,
    )
    return int(resp_rep)


def test_paths(group_key: str, idxs: Iterable[int]) -> list[Path]:
    paths = []
    for idx in idxs:
        path = VARIANTS_DIR / group_key / f"testConfig_{idx}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing test file: {path}")
        paths.append(path)
    return paths


def load_test_df(group_key: str) -> pd.DataFrame:
    dfs = []
    for path in test_paths(group_key, TEST_IDXS):
        df = pd.read_csv(path)
        df["source_file"] = path.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def add_reference_policies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    evidence = [parse_evidence(v) for v in df["evidence"]]
    sigmas = df["sigma"].astype(float).tolist()

    df["final_evidence"] = [ev[-1] for ev in evidence]
    df["final_evidence_abs"] = df["final_evidence"].abs()
    df["heuristic_report"] = [heuristic_report_sign(ev) for ev in evidence]
    df["normative_report"] = [
        normative_report_sign(ev, sigma) for ev, sigma in zip(evidence, sigmas)
    ]
    df["true_report"] = df["trueReport"].astype(float).astype(int)
    df["heuristic_report01"] = df["heuristic_report"].map(sign_to_label01)
    df["normative_report01"] = df["normative_report"].map(sign_to_label01)
    df["true_report01"] = df["true_report"].map(sign_to_label01)
    df["_evidence_parsed"] = evidence
    return df


class EvidenceDataset(Dataset):
    def __init__(self, evidence: list[list[float]]):
        self.xs = [
            torch.tensor(ev, dtype=torch.float32).unsqueeze(-1)
            for ev in evidence
        ]

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.xs[idx]


def load_model(model_dir: Path, checkpoint_name: str) -> GRUModel:
    hp_path = model_dir / "hp.json"
    ckpt_path = model_dir / checkpoint_name
    if not hp_path.exists():
        raise FileNotFoundError(f"Missing hp.json: {hp_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    with hp_path.open("r") as f:
        hp = json.load(f)

    model = GRUModel(hp).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def model_report_probabilities(
    model: GRUModel,
    evidence: list[list[float]],
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(EvidenceDataset(evidence), batch_size=batch_size, shuffle=False)
    probs = []
    for x in loader:
        x = x.to(DEVICE)
        loc_logits, _ = model(x)
        final_logits = loc_logits[:, -1, 0]
        probs.append(torch.sigmoid(final_logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def binary_cross_entropy(prob: np.ndarray, target01: np.ndarray) -> float:
    prob = np.clip(prob, EPS, 1.0 - EPS)
    return float(-(target01 * np.log(prob) + (1 - target01) * np.log(1 - prob)).mean())


def evaluate_one_seed(
    model_set_name: str,
    model_set_dir: Path,
    seed: int,
    checkpoint_name: str,
    df: pd.DataFrame,
    batch_size: int,
    final_evidence_window: float,
) -> dict:
    model_dir = model_set_dir / f"seed_{seed}"
    model = load_model(model_dir, checkpoint_name)

    probs = model_report_probabilities(model, df["_evidence_parsed"].tolist(), batch_size)
    pred01 = (probs >= 0.5).astype(int)
    pred_sign = np.array([label01_to_sign(v) for v in pred01])

    heuristic01 = df["heuristic_report01"].to_numpy(dtype=int)
    normative01 = df["normative_report01"].to_numpy(dtype=int)
    true01 = df["true_report01"].to_numpy(dtype=int)

    row = {
        "model_set": model_set_name,
        "seed": seed,
        "checkpoint": checkpoint_name,
        "n_trials": int(len(df)),
        "match_heuristic_report_acc": float((pred01 == heuristic01).mean()),
        "match_normative_report_acc": float((pred01 == normative01).mean()),
        "match_true_report_acc": float((pred01 == true01).mean()),
        "bce_to_heuristic_report": binary_cross_entropy(probs, heuristic01),
        "bce_to_normative_report": binary_cross_entropy(probs, normative01),
        "bce_to_true_report": binary_cross_entropy(probs, true01),
        "mean_p_report_plus": float(probs.mean()),
        "n_model_plus": int((pred_sign == 1).sum()),
        "n_model_minus": int((pred_sign == -1).sum()),
    }

    near_final = df["final_evidence_abs"].to_numpy(dtype=float) <= final_evidence_window
    row["near_final_abs_threshold"] = float(final_evidence_window)
    row["near_final_n_trials"] = int(near_final.sum())
    if near_final.any():
        row.update(
            {
                "near_final_match_heuristic_report_acc": float(
                    (pred01[near_final] == heuristic01[near_final]).mean()
                ),
                "near_final_match_normative_report_acc": float(
                    (pred01[near_final] == normative01[near_final]).mean()
                ),
                "near_final_match_true_report_acc": float(
                    (pred01[near_final] == true01[near_final]).mean()
                ),
                "near_final_bce_to_heuristic_report": binary_cross_entropy(
                    probs[near_final], heuristic01[near_final]
                ),
                "near_final_bce_to_normative_report": binary_cross_entropy(
                    probs[near_final], normative01[near_final]
                ),
                "near_final_bce_to_true_report": binary_cross_entropy(
                    probs[near_final], true01[near_final]
                ),
                "near_final_mean_p_report_plus": float(probs[near_final].mean()),
            }
        )
    else:
        row.update(
            {
                "near_final_match_heuristic_report_acc": np.nan,
                "near_final_match_normative_report_acc": np.nan,
                "near_final_match_true_report_acc": np.nan,
                "near_final_bce_to_heuristic_report": np.nan,
                "near_final_bce_to_normative_report": np.nan,
                "near_final_bce_to_true_report": np.nan,
                "near_final_mean_p_report_plus": np.nan,
            }
        )

    return row


def summarize(seed_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "match_heuristic_report_acc",
        "match_normative_report_acc",
        "match_true_report_acc",
        "bce_to_heuristic_report",
        "bce_to_normative_report",
        "bce_to_true_report",
        "mean_p_report_plus",
        "near_final_match_heuristic_report_acc",
        "near_final_match_normative_report_acc",
        "near_final_match_true_report_acc",
        "near_final_bce_to_heuristic_report",
        "near_final_bce_to_normative_report",
        "near_final_bce_to_true_report",
        "near_final_mean_p_report_plus",
    ]
    summary = (
        seed_results.groupby("model_set")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def reference_summary(df: pd.DataFrame, final_evidence_window: float) -> dict:
    heuristic01 = df["heuristic_report01"].to_numpy(dtype=int)
    normative01 = df["normative_report01"].to_numpy(dtype=int)
    true01 = df["true_report01"].to_numpy(dtype=int)
    near_final = df["final_evidence_abs"].to_numpy(dtype=float) <= final_evidence_window

    refs = {
        "n_trials": int(len(df)),
        "heuristic_matches_normative_report_acc": float((heuristic01 == normative01).mean()),
        "heuristic_matches_true_report_acc": float((heuristic01 == true01).mean()),
        "normative_matches_true_report_acc": float((normative01 == true01).mean()),
        "heuristic_plus_rate": float(heuristic01.mean()),
        "normative_plus_rate": float(normative01.mean()),
        "true_plus_rate": float(true01.mean()),
        "near_final_abs_threshold": float(final_evidence_window),
        "near_final_n_trials": int(near_final.sum()),
    }
    if near_final.any():
        refs.update(
            {
                "near_final_heuristic_matches_normative_report_acc": float(
                    (heuristic01[near_final] == normative01[near_final]).mean()
                ),
                "near_final_heuristic_matches_true_report_acc": float(
                    (heuristic01[near_final] == true01[near_final]).mean()
                ),
                "near_final_normative_matches_true_report_acc": float(
                    (normative01[near_final] == true01[near_final]).mean()
                ),
                "near_final_heuristic_plus_rate": float(heuristic01[near_final].mean()),
                "near_final_normative_plus_rate": float(normative01[near_final].mean()),
                "near_final_true_plus_rate": float(true01[near_final].mean()),
            }
        )
    return refs


def print_results(seed_results: pd.DataFrame, summary: pd.DataFrame, refs: dict) -> None:
    print()
    print("Reference policy agreement on testConfig_0.csv through testConfig_19.csv")
    print(pd.DataFrame([refs]).to_string(index=False))

    print()
    print("Per-seed model agreement")
    print(
        seed_results[
            [
                "model_set",
                "seed",
                "match_heuristic_report_acc",
                "match_normative_report_acc",
                "match_true_report_acc",
                "bce_to_heuristic_report",
                "bce_to_normative_report",
                "near_final_n_trials",
                "near_final_match_heuristic_report_acc",
                "near_final_match_normative_report_acc",
                "near_final_match_true_report_acc",
            ]
        ].to_string(index=False)
    )

    print()
    print("Across-seed summary")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sigma_1 report models to heuristic and normative report policies."
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--final-evidence-window",
        type=float,
        default=FINAL_EVIDENCE_WINDOW,
        help="Also report metrics where abs(final evidence) is at or below this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df = add_reference_policies(load_test_df(GROUP_KEY))
    refs = reference_summary(df, args.final_evidence_window)

    rows = []
    for model_set_name, model_set_dir in DEFAULT_MODEL_SETS.items():
        for seed in SEEDS:
            rows.append(
                evaluate_one_seed(
                    model_set_name=model_set_name,
                    model_set_dir=model_set_dir,
                    seed=seed,
                    checkpoint_name=args.checkpoint,
                    df=df,
                    batch_size=args.batch_size,
                    final_evidence_window=args.final_evidence_window,
                )
            )

    seed_results = pd.DataFrame(rows).sort_values(["model_set", "seed"])
    summary = summarize(seed_results)

    seed_results.to_csv(args.results_dir / "policy_match_by_seed.csv", index=False)
    summary.to_csv(args.results_dir / "policy_match_summary.csv", index=False)
    pd.DataFrame([refs]).to_csv(args.results_dir / "reference_policy_summary.csv", index=False)

    print_results(seed_results, summary, refs)
    print()
    print(f"Saved results to: {args.results_dir}")


if __name__ == "__main__":
    main()

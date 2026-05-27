#!/usr/bin/env python3
"""
eval_normative_accuracy.py
==========================
Evaluate the normative Bayesian observer on validation files
valConfig_0.csv through valConfig_4.csv for:
    variants/sigma_1
    variants/sigma_2
    variants/sigma_3

Prints both:
    - report accuracy
    - predict accuracy

Expected CSV columns:
    - evidence
    - trueReport
    - truePredict

This script imports BayesianObserver from NormativeModel.py.
"""

from __future__ import annotations

import ast
import os
from typing import Iterable

import numpy as np
import pandas as pd

from NormativeModel import BayesianObserver


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(BASE_DIR, "variants")
GROUP_KEYS = ["sigma_1", "sigma_2", "sigma_3"]
VAL_IDXS = range(5)  # valConfig_0.csv ... valConfig_4.csv

MU1 = -1
MU2 = 1
HS = np.arange(0, 1, 0.05)
BIAS = 0


def sigma_from_group(group_key: str) -> float:
    tail = group_key.split("_")[-1]
    return float(tail)


def val_paths_for_group(group_key: str) -> list[str]:
    paths = []
    for k in VAL_IDXS:
        path = os.path.join(VARIANTS_DIR, group_key, f"valConfig_{k}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing validation file: {path}")
        paths.append(path)
    return paths


def parse_evidence(value) -> list[float]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    raise TypeError(f"Unsupported evidence type: {type(value)}")


def final_entry_side_policy(evidence: list[float]) -> int:
    return 1 if evidence[-1] >= 0 else -1


def evaluate_df(df: pd.DataFrame, sigma: float) -> tuple[int, int, int, int, int]:
    correct_report = 0
    correct_predict = 0
    baseline_report = 0
    baseline_predict = 0
    n_trials = 0

    for _, row in df.iterrows():
        evidence = parse_evidence(row["evidence"])
        _, _, resp_rep, resp_pred = BayesianObserver(
            evidence,
            mu1=MU1,
            mu2=MU2,
            sigma=sigma,
            hs=HS,
            bias=BIAS,
        )

        final_side = final_entry_side_policy(evidence)
        true_report = int(row["trueReport"])
        true_predict = int(row["truePredict"])

        if int(resp_rep) == true_report:
            correct_report += 1
        if int(resp_pred) == true_predict:
            correct_predict += 1
        if final_side == true_report:
            baseline_report += 1
        if final_side == true_predict:
            baseline_predict += 1

        n_trials += 1

    return correct_report, correct_predict, baseline_report, baseline_predict, n_trials


def evaluate_group(group_key: str) -> dict:
    sigma = sigma_from_group(group_key)
    total_report = 0
    total_predict = 0
    baseline_report = 0
    baseline_predict = 0
    total_trials = 0

    for path in val_paths_for_group(group_key):
        df = pd.read_csv(path)
        c_rep, c_pred, b_rep, b_pred, n = evaluate_df(df, sigma)
        total_report += c_rep
        total_predict += c_pred
        baseline_report += b_rep
        baseline_predict += b_pred
        total_trials += n

    return {
        "group": group_key,
        "sigma": sigma,
        "n_trials": total_trials,
        "report_correct": total_report,
        "predict_correct": total_predict,
        "baseline_report_correct": baseline_report,
        "baseline_predict_correct": baseline_predict,
        "report_acc": total_report / total_trials,
        "predict_acc": total_predict / total_trials,
        "baseline_report_acc": baseline_report / total_trials,
        "baseline_predict_acc": baseline_predict / total_trials,
    }


def print_results(results: Iterable[dict]) -> None:
    print()
    print("Normative model validation accuracy")
    print("Using valConfig_0.csv through valConfig_4.csv for each sigma group")
    print("Baseline policy: predict the side of the final evidence entry (>= 0 -> +1, < 0 -> -1)")
    print()
    print(
        f"{'group':<10} {'sigma':>7} {'trials':>8} "
        f"{'norm_rep':>10} {'norm_pred':>10} {'last_rep':>10} {'last_pred':>10}"
    )
    print("-" * 76)
    for r in results:
        print(
            f"{r['group']:<10} "
            f"{r['sigma']:>7.3f} "
            f"{r['n_trials']:>8d} "
            f"{r['report_acc']:>10.4f} "
            f"{r['predict_acc']:>10.4f} "
            f"{r['baseline_report_acc']:>10.4f} "
            f"{r['baseline_predict_acc']:>10.4f}"
        )


def main() -> None:
    results = [evaluate_group(group_key) for group_key in GROUP_KEYS]
    print_results(results)


if __name__ == "__main__":
    main()

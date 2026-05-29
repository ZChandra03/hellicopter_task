#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from NormativeModel import BayesianObserver


DEFAULT_VARIANTS_ROOT = Path(__file__).resolve().parents[1] / "variants"
DEFAULT_OUTPUT = DEFAULT_VARIANTS_ROOT / "normative_model_test_accuracy.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the normative Bayesian observer on every variant folder's "
            "testConfig_*.csv files and write a compact accuracy summary."
        )
    )
    parser.add_argument(
        "--variants-root",
        type=Path,
        default=DEFAULT_VARIANTS_ROOT,
        help=f"Root folder to scan for variant test CSVs. Default: {DEFAULT_VARIANTS_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Summary CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--max-test-csvs",
        type=int,
        default=None,
        help="Optional cap on testConfig_*.csv files per variant folder.",
    )
    parser.add_argument(
        "--hazard-step",
        type=float,
        default=0.05,
        help="Grid spacing for possible hazard rates. Default: 0.05",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Bias parameter passed to BayesianObserver. Default: 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for rare tied normative responses. Default: 0",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def parse_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    parsed = ast.literal_eval(str(value))
    return [float(item) for item in parsed]


def load_mu(variant_dir: Path, default: float = 1.0) -> float:
    config_path = variant_dir / "TaskConfig.csv"
    if not config_path.exists():
        return default

    try:
        config = pd.read_csv(config_path, index_col=0)
        if "Mu" in config.index:
            return float(config.loc["Mu"].iloc[0])
    except Exception as exc:
        print(f"[warn] Could not read Mu from {config_path}: {exc}. Using {default}.")

    return default


def find_variant_dirs(variants_root: Path) -> list[Path]:
    test_csvs = variants_root.rglob("testConfig_*.csv")
    variant_dirs = sorted({path.parent for path in test_csvs}, key=lambda p: str(p).lower())
    if not variant_dirs:
        raise FileNotFoundError(f"No testConfig_*.csv files found under {variants_root}")
    return variant_dirs


def list_test_csvs(variant_dir: Path, max_test_csvs: int | None) -> list[Path]:
    csvs = sorted(variant_dir.glob("testConfig_*.csv"), key=natural_key)
    if max_test_csvs is not None:
        csvs = csvs[:max_test_csvs]
    if not csvs:
        raise FileNotFoundError(f"No testConfig_*.csv files found in {variant_dir}")
    return csvs


def evaluate_variant(
    variant_dir: Path,
    variants_root: Path,
    hs: np.ndarray,
    bias: float,
    max_test_csvs: int | None,
) -> dict[str, Any]:
    csvs = list_test_csvs(variant_dir, max_test_csvs)
    mu = load_mu(variant_dir)

    report_correct = 0
    predict_correct = 0
    total = 0

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            evidence = parse_list(row["evidence"])
            sigma = float(row["sigma"])
            true_report = float(row["trueReport"])
            true_predict = float(row["truePredict"])

            _, _, resp_report, resp_predict = BayesianObserver(
                evidence,
                mu1=-mu,
                mu2=mu,
                sigma=sigma,
                hs=hs,
                bias=bias,
            )

            report_correct += int(float(resp_report) == true_report)
            predict_correct += int(float(resp_predict) == true_predict)
            total += 1

    report_accuracy = report_correct / total
    predict_accuracy = predict_correct / total

    return {
        "variant_folder": variant_dir.relative_to(variants_root).as_posix(),
        "n_test_csvs": len(csvs),
        "n_trials": total,
        "report_correct": report_correct,
        "predict_correct": predict_correct,
        "report_accuracy": report_accuracy,
        "predict_accuracy": predict_accuracy,
        "combined_accuracy": 0.5 * (report_accuracy + predict_accuracy),
    }


def main() -> None:
    args = parse_args()
    variants_root = args.variants_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if args.hazard_step <= 0:
        raise ValueError("--hazard-step must be positive")

    np.random.seed(args.seed)
    hs = np.arange(0, 1, args.hazard_step)

    rows = []
    for variant_dir in find_variant_dirs(variants_root):
        print(f"Evaluating {variant_dir}...")
        rows.append(
            evaluate_variant(
                variant_dir=variant_dir,
                variants_root=variants_root,
                hs=hs,
                bias=args.bias,
                max_test_csvs=args.max_test_csvs,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

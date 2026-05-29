#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_VARIANTS_ROOT = Path(__file__).resolve().parents[1] / "variants"
DEFAULT_OUTPUT = DEFAULT_VARIANTS_ROOT / "last_evidence_report_heuristic_test_accuracy.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a report heuristic that reports the side indicated by the "
            "last evidence sample in each test trial."
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
        "--zero-report",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Report value to use if the last evidence is exactly zero. Default: 1",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def parse_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    parsed = ast.literal_eval(str(value))
    return [float(item) for item in parsed]


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


def report_from_last_evidence(evidence: list[float], zero_report: float) -> float:
    if not evidence:
        raise ValueError("Evidence sequence cannot be empty")
    last_evidence = evidence[-1]
    if last_evidence > 0:
        return 1.0
    if last_evidence < 0:
        return -1.0
    return zero_report


def evaluate_variant(
    variant_dir: Path,
    variants_root: Path,
    max_test_csvs: int | None,
    zero_report: float,
) -> dict[str, Any]:
    csvs = list_test_csvs(variant_dir, max_test_csvs)
    report_correct = 0
    total = 0

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            evidence = parse_list(row["evidence"])
            true_report = float(row["trueReport"])
            heuristic_report = report_from_last_evidence(evidence, zero_report)

            report_correct += int(heuristic_report == true_report)
            total += 1

    return {
        "variant_folder": variant_dir.relative_to(variants_root).as_posix(),
        "n_test_csvs": len(csvs),
        "n_trials": total,
        "report_correct": report_correct,
        "report_accuracy": report_correct / total,
    }


def main() -> None:
    args = parse_args()
    variants_root = args.variants_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    rows = []
    for variant_dir in find_variant_dirs(variants_root):
        print(f"Evaluating {variant_dir}...")
        rows.append(
            evaluate_variant(
                variant_dir=variant_dir,
                variants_root=variants_root,
                max_test_csvs=args.max_test_csvs,
                zero_report=float(args.zero_report),
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

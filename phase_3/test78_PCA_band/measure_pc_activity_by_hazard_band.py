#!/usr/bin/env python3
"""Measure PCA score activity within configurable true-hazard bands.

The primary activity measure is the mean absolute PC score.  The output also
reports signed means (direction), RMS activity (energy), and within-band
standard deviations.  By default, hazards are split into five equal-width
bands over [0, 1].

Example:
    python measure_pc_activity_by_hazard_band.py

Custom bands:
    python measure_pc_activity_by_hazard_band.py --band-edges 0,0.25,0.5,0.75,1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PCA_DIR = BASE_DIR / "pca_outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_PCA_DIR / "hazard_band_activity"
DEFAULT_BAND_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PC_COLUMN_RE = re.compile(r"^pc(\d+)$", flags=re.IGNORECASE)


def parse_band_edges(value: str) -> list[float]:
    try:
        edges = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Band edges must be comma-separated numbers, for example 0,0.2,0.4,1"
        ) from exc

    if len(edges) < 2:
        raise argparse.ArgumentTypeError("At least two band edges are required")
    if not all(math.isfinite(edge) for edge in edges):
        raise argparse.ArgumentTypeError("Band edges must all be finite")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise argparse.ArgumentTypeError("Band edges must be strictly increasing")
    return edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize signed and magnitude-based PC activity within true-hazard bands."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "PCA hidden-state CSV. By default, auto-detect the final-timestep "
            "or all-timestep export in ./pca_outputs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary CSVs and plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--band-edges",
        type=parse_band_edges,
        default=list(DEFAULT_BAND_EDGES),
        metavar="EDGE1,EDGE2,...",
        help="Hazard-band edges. Default: 0,0.2,0.4,0.6,0.8,1",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Keep only rows where combined_correct equals 1.",
    )
    parser.add_argument(
        "--timestep",
        default="all",
        metavar="ALL|FINAL|INTEGER",
        help=(
            "Use all exported states, only each trial's final state, or one integer "
            "timestep. Default: all."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Plot resolution. Default: 250",
    )
    return parser.parse_args()


def find_input_csv(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PCA input CSV does not exist: {path}")
        return path

    candidates = [
        DEFAULT_PCA_DIR / "pca_ep010_final_timestep_hidden_states.csv",
        DEFAULT_PCA_DIR / "pca_ep010_hidden_states.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()

    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not auto-detect a PCA hidden-state CSV. Run "
        "pca_checkpoint_ep010.py first or pass --input. Tried:\n  "
        f"{tried}"
    )


def pc_columns(df: pd.DataFrame) -> list[str]:
    numbered: list[tuple[int, str]] = []
    for column in df.columns:
        match = PC_COLUMN_RE.fullmatch(str(column))
        if match:
            numbered.append((int(match.group(1)), str(column)))
    columns = [column for _, column in sorted(numbered)]
    if not columns:
        raise ValueError("Input has no PC columns named pc1, pc2, ...")
    return columns


def validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    bad_counts = {
        column: int((~np.isfinite(result[column].to_numpy(dtype=float))).sum())
        for column in columns
    }
    bad_counts = {column: count for column, count in bad_counts.items() if count}
    if bad_counts:
        details = ", ".join(f"{column}={count}" for column, count in bad_counts.items())
        raise ValueError(f"Input contains missing or non-finite numeric values: {details}")
    return result


def trial_key_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["seed", "source_csv", "csv_trial"]
    if all(column in df.columns for column in preferred):
        return preferred
    if "global_trial" in df.columns:
        return ["global_trial"]
    return []


def filter_timesteps(df: pd.DataFrame, selection: str) -> tuple[pd.DataFrame, str]:
    normalized = selection.strip().lower()
    if normalized == "all":
        return df, "all"
    if "timestep" not in df.columns:
        raise ValueError("--timestep filtering requires a timestep column")

    result = df.copy()
    result["timestep"] = pd.to_numeric(result["timestep"], errors="coerce")
    if result["timestep"].isna().any():
        raise ValueError("The timestep column contains non-numeric values")

    if normalized == "final":
        keys = trial_key_columns(result)
        if keys:
            final_timestep = result.groupby(keys, observed=True)["timestep"].transform("max")
            result = result[result["timestep"] == final_timestep].copy()
        else:
            maximum = result["timestep"].max()
            result = result[result["timestep"] == maximum].copy()
        return result, "final"

    try:
        timestep = int(normalized)
    except ValueError as exc:
        raise ValueError("--timestep must be 'all', 'final', or an integer") from exc
    result = result[result["timestep"] == timestep].copy()
    return result, str(timestep)


def band_label(left: float, right: float, first: bool) -> str:
    opening = "[" if first else "("
    return f"{opening}{left:g}, {right:g}]"


def assign_hazard_bands(
    df: pd.DataFrame,
    hazard_column: str,
    edges: list[float],
) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    result[hazard_column] = pd.to_numeric(result[hazard_column], errors="coerce")
    hazards = result[hazard_column].to_numpy(dtype=float)
    if not np.isfinite(hazards).all():
        count = int((~np.isfinite(hazards)).sum())
        raise ValueError(f"{hazard_column} contains {count} missing or non-finite values")

    below = hazards < edges[0]
    above = hazards > edges[-1]
    if below.any() or above.any():
        raise ValueError(
            f"{int(below.sum() + above.sum())} hazard values fall outside the requested "
            f"range [{edges[0]:g}, {edges[-1]:g}]. Observed range: "
            f"[{hazards.min():g}, {hazards.max():g}]"
        )

    labels = [
        band_label(left, right, first=index == 0)
        for index, (left, right) in enumerate(zip(edges, edges[1:]))
    ]
    result["hazard_band_index"] = pd.cut(
        result[hazard_column],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    ).astype(int)
    result["hazard_band"] = result["hazard_band_index"].map(dict(enumerate(labels)))
    return result, labels


def count_trials(group: pd.DataFrame, key_columns: list[str]) -> int:
    if not key_columns:
        return len(group)
    return int(group[key_columns].drop_duplicates().shape[0])


def sample_stats(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    absolute = np.abs(values)
    signed_mean = float(np.mean(values))
    absolute_mean = float(np.mean(absolute))
    if count > 1:
        signed_std = float(np.std(values, ddof=1))
        absolute_std = float(np.std(absolute, ddof=1))
        signed_ci = 1.96 * signed_std / math.sqrt(count)
        absolute_ci = 1.96 * absolute_std / math.sqrt(count)
    else:
        signed_std = math.nan
        absolute_std = math.nan
        signed_ci = math.nan
        absolute_ci = math.nan

    return {
        "signed_mean": signed_mean,
        "signed_std": signed_std,
        "signed_ci95_half_width": signed_ci,
        "mean_absolute_activity": absolute_mean,
        "absolute_std": absolute_std,
        "absolute_ci95_half_width": absolute_ci,
        "rms_activity": float(np.sqrt(np.mean(np.square(values)))),
    }


def summarize_activity(
    df: pd.DataFrame,
    pcs: list[str],
    edges: list[float],
    labels: list[str],
    hazard_column: str = "true_hazard",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    key_columns = trial_key_columns(df)

    for band_index, label in enumerate(labels):
        group = df[df["hazard_band_index"] == band_index]
        count_row = {
            "hazard_band_index": band_index,
            "hazard_band": label,
            "hazard_lower_edge": edges[band_index],
            "hazard_upper_edge": edges[band_index + 1],
            "n_states": len(group),
            "n_trials": count_trials(group, key_columns),
            "observed_hazard_min": (
                float(group[hazard_column].min()) if not group.empty else math.nan
            ),
            "observed_hazard_max": (
                float(group[hazard_column].max()) if not group.empty else math.nan
            ),
            "observed_hazard_mean": (
                float(group[hazard_column].mean()) if not group.empty else math.nan
            ),
        }
        count_rows.append(count_row)

        for pc in pcs:
            row = {**count_row, "pc": pc.lower()}
            if group.empty:
                row.update(
                    {
                        "signed_mean": math.nan,
                        "signed_std": math.nan,
                        "signed_ci95_half_width": math.nan,
                        "mean_absolute_activity": math.nan,
                        "absolute_std": math.nan,
                        "absolute_ci95_half_width": math.nan,
                        "rms_activity": math.nan,
                    }
                )
            else:
                row.update(sample_stats(group[pc].to_numpy(dtype=float)))
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def plot_metric(
    summary: pd.DataFrame,
    labels: list[str],
    metric: str,
    ci_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    zero_line: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7.5, 1.25 * len(labels)), 5.2))
    x = np.arange(len(labels))

    for pc, group in summary.groupby("pc", sort=False):
        ordered = group.sort_values("hazard_band_index")
        ax.errorbar(
            x,
            ordered[metric].to_numpy(dtype=float),
            yerr=ordered[ci_column].to_numpy(dtype=float),
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=3,
            label=pc.upper(),
        )

    if zero_line:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_xlabel("True-hazard band")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncols=min(4, summary["pc"].nunique()))
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = find_input_csv(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "true_hazard" not in df.columns:
        raise ValueError("Input is missing the required true_hazard column")
    pcs = pc_columns(df)
    df = validate_numeric_columns(df, ["true_hazard", *pcs])

    if args.correct_only:
        if "combined_correct" not in df.columns:
            raise ValueError("--correct-only requires a combined_correct column")
        correct = pd.to_numeric(df["combined_correct"], errors="coerce")
        df = df[correct == 1].copy()

    df, timestep_selection = filter_timesteps(df, args.timestep)
    if df.empty:
        raise ValueError("No PCA rows remain after applying the requested filters")

    df, labels = assign_hazard_bands(df, "true_hazard", args.band_edges)
    summary, counts = summarize_activity(df, pcs, args.band_edges, labels)

    summary_path = output_dir / "pc_activity_by_hazard_band.csv"
    counts_path = output_dir / "hazard_band_counts.csv"
    absolute_plot_path = output_dir / "pc_mean_absolute_activity_by_hazard_band.png"
    signed_plot_path = output_dir / "pc_signed_mean_by_hazard_band.png"
    metadata_path = output_dir / "pc_activity_by_hazard_band_metadata.json"

    summary.to_csv(summary_path, index=False)
    counts.to_csv(counts_path, index=False)
    plot_metric(
        summary,
        labels,
        metric="mean_absolute_activity",
        ci_column="absolute_ci95_half_width",
        ylabel="Mean absolute PC score",
        title="PC activity by true-hazard band",
        output_path=absolute_plot_path,
        dpi=args.dpi,
    )
    plot_metric(
        summary,
        labels,
        metric="signed_mean",
        ci_column="signed_ci95_half_width",
        ylabel="Mean signed PC score",
        title="Signed PC score by true-hazard band",
        output_path=signed_plot_path,
        dpi=args.dpi,
        zero_line=True,
    )

    metadata = {
        "input_csv": str(input_path),
        "rows_analyzed": len(df),
        "pc_columns": [pc.lower() for pc in pcs],
        "hazard_column": "true_hazard",
        "hazard_band_edges": args.band_edges,
        "correct_only": args.correct_only,
        "timestep_selection": timestep_selection,
        "primary_activity_measure": "mean absolute PC score",
        "confidence_intervals": "normal-approximation 95% CI across PCA rows",
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Loaded {len(df):,} PCA rows from {input_path}")
    print(f"PC columns: {', '.join(pc.lower() for pc in pcs)}")
    print(f"Hazard bands: {', '.join(labels)}")
    print(f"Saved activity summary to {summary_path}")
    print(f"Saved band counts to {counts_path}")
    print(f"Saved plots to {absolute_plot_path} and {signed_plot_path}")
    print(f"Saved analysis metadata to {metadata_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from manifold_utils import (
    BASE_DIR,
    DEFAULT_CONFIG,
    checkpoint_label,
    choose_device,
    collect_evidence_states,
    encode_evidence,
    evidence_timestep_indices,
    load_config,
    load_model,
    model_specs,
    parse_model_filter,
    resolve_checkpoints,
    set_seeds,
    sigmoid,
    write_run_config,
)


DEFAULT_OUTPUT_DIR = BASE_DIR / "history_switch_outputs"
CONDITION_ORDER = ["low_to_low", "high_to_low", "low_to_high", "high_to_high"]
CONDITION_COLORS = {
    "low_to_low": "#4c78a8",
    "high_to_low": "#f58518",
    "low_to_high": "#54a24b",
    "high_to_high": "#e45756",
}
PAIR_DEFINITIONS = {
    "low_suffix": ("high_to_low", "low_to_low"),
    "high_suffix": ("high_to_high", "low_to_high"),
}


@dataclass(frozen=True)
class SequenceRow:
    condition: str
    replicate: int
    mirror: int
    evidence: np.ndarray
    states: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal history-switch experiment: append the exact same low- or "
            "high-hazard suffix to low- and high-hazard histories and measure relaxation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all")
    parser.add_argument("--checkpoints", default="1,5,7,10,final")
    parser.add_argument("--n-sequences", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--low-hazard", type=float, default=0.1)
    parser.add_argument("--high-hazard", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-trajectory-replicates",
        type=int,
        default=64,
        help="Replicates retained in the trajectory table; contrasts still use all replicates.",
    )
    parser.add_argument("--max-pca-points", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=8401)
    return parser.parse_args()


def latent_path(
    hazard: float,
    length: int,
    rng: np.random.Generator,
    start_state: int | None = None,
    force_final_positive: bool = False,
) -> np.ndarray:
    if start_state is None:
        current = 1 if rng.random() >= 0.5 else -1
    else:
        current = int(start_state)
    states = np.empty(length, dtype=np.float32)
    for index in range(length):
        states[index] = current
        if rng.random() < hazard:
            current = -current
    if force_final_positive and states[-1] < 0:
        states *= -1.0
    return states


def generate_sequence_bank(args: argparse.Namespace) -> list[SequenceRow]:
    rng = np.random.default_rng(args.random_seed)
    rows: list[SequenceRow] = []
    for replicate in range(args.n_sequences):
        low_prefix_states = latent_path(
            args.low_hazard,
            args.block_length,
            rng,
            force_final_positive=True,
        )
        high_prefix_states = latent_path(
            args.high_hazard,
            args.block_length,
            rng,
            force_final_positive=True,
        )
        low_suffix_states = latent_path(
            args.low_hazard,
            args.block_length,
            rng,
            start_state=1,
        )
        high_suffix_states = latent_path(
            args.high_hazard,
            args.block_length,
            rng,
            start_state=1,
        )

        low_prefix_evidence = low_prefix_states + rng.normal(
            0.0, args.sigma, args.block_length
        )
        high_prefix_evidence = high_prefix_states + rng.normal(
            0.0, args.sigma, args.block_length
        )
        # The suffix realization, including observation noise, is exactly shared
        # within each causal pair. Only the preceding history differs.
        low_suffix_evidence = low_suffix_states + rng.normal(
            0.0, args.sigma, args.block_length
        )
        high_suffix_evidence = high_suffix_states + rng.normal(
            0.0, args.sigma, args.block_length
        )
        condition_parts = {
            "low_to_low": (
                low_prefix_evidence,
                low_suffix_evidence,
                low_prefix_states,
                low_suffix_states,
            ),
            "high_to_low": (
                high_prefix_evidence,
                low_suffix_evidence,
                high_prefix_states,
                low_suffix_states,
            ),
            "low_to_high": (
                low_prefix_evidence,
                high_suffix_evidence,
                low_prefix_states,
                high_suffix_states,
            ),
            "high_to_high": (
                high_prefix_evidence,
                high_suffix_evidence,
                high_prefix_states,
                high_suffix_states,
            ),
        }
        for condition, parts in condition_parts.items():
            evidence = np.concatenate(parts[:2]).astype(np.float32)
            states = np.concatenate(parts[2:]).astype(np.float32)
            for mirror in (1, -1):
                rows.append(
                    SequenceRow(
                        condition=condition,
                        replicate=replicate,
                        mirror=mirror,
                        evidence=mirror * evidence,
                        states=mirror * states,
                    )
                )
    return rows


def build_bank_inputs(rows: list[SequenceRow], n_null: int) -> torch.Tensor:
    return torch.stack([encode_evidence(row.evidence, n_null) for row in rows])


def checkpoint_position(label: str) -> float:
    if label == "init":
        return 0.0
    if label.startswith("ep"):
        return float(int(label[2:]))
    if label == "final":
        return 11.0
    if label == "best":
        return 10.5
    return float("nan")


def exponential(time: np.ndarray, amplitude: float, tau: float, offset: float) -> np.ndarray:
    return offset + amplitude * np.exp(-time / tau)


def fit_relaxation(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    time = np.arange(len(values), dtype=float)
    if len(values) < 3 or not np.all(np.isfinite(values)):
        return {"amplitude": np.nan, "tau_evidence": np.nan, "offset": np.nan, "r2": np.nan}
    initial = float(values[0])
    offset_guess = float(np.mean(values[-max(2, len(values) // 5) :]))
    try:
        parameters, _ = curve_fit(
            exponential,
            time,
            values,
            p0=[max(initial - offset_guess, 0.0), max(len(values) / 3.0, 1.0), offset_guess],
            bounds=([0.0, 0.05, 0.0], [np.inf, 1e3, np.inf]),
            maxfev=20000,
        )
        prediction = exponential(time, *parameters)
        total = float(np.sum((values - np.mean(values)) ** 2))
        residual = float(np.sum((values - prediction) ** 2))
        fit_r2 = 1.0 - residual / total if total > 0 else np.nan
        return {
            "amplitude": float(parameters[0]),
            "tau_evidence": float(parameters[1]),
            "offset": float(parameters[2]),
            "r2": fit_r2,
        }
    except (RuntimeError, ValueError):
        return {"amplitude": np.nan, "tau_evidence": np.nan, "offset": np.nan, "r2": np.nan}


def first_threshold_crossing(values: np.ndarray, threshold: float, direction: str) -> float:
    if direction == "below":
        indices = np.flatnonzero(values < threshold)
    else:
        indices = np.flatnonzero(values > threshold)
    return float(indices[0]) if len(indices) else float("nan")


def paired_contrast_rows(
    model_key: str,
    checkpoint: str,
    order: float,
    rows: list[SequenceRow],
    hidden: np.ndarray,
    report_prob: np.ndarray,
    hazard_prob: np.ndarray,
    block_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row.condition, row.replicate, row.mirror): index for index, row in enumerate(rows)
    }
    trajectory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    suffix_slice = slice(block_length, 2 * block_length)

    for pair_name, (high_history_condition, low_history_condition) in PAIR_DEFINITIONS.items():
        hazard_differences = []
        report_differences = []
        hidden_distances = []
        for replicate in range(max(row.replicate for row in rows) + 1):
            for mirror in (1, -1):
                high_index = lookup[(high_history_condition, replicate, mirror)]
                low_index = lookup[(low_history_condition, replicate, mirror)]
                hazard_difference = (
                    hazard_prob[high_index, suffix_slice] - hazard_prob[low_index, suffix_slice]
                )
                report_difference = np.abs(
                    report_prob[high_index, suffix_slice] - report_prob[low_index, suffix_slice]
                )
                hidden_distance = np.linalg.norm(
                    hidden[high_index, suffix_slice] - hidden[low_index, suffix_slice], axis=1
                ) / np.sqrt(hidden.shape[-1])
                hazard_differences.append(hazard_difference)
                report_differences.append(report_difference)
                hidden_distances.append(hidden_distance)

        hazard_array = np.asarray(hazard_differences)
        report_array = np.asarray(report_differences)
        hidden_array = np.asarray(hidden_distances)
        mean_abs_hazard = np.mean(np.abs(hazard_array), axis=0)
        mean_hidden = np.mean(hidden_array, axis=0)
        for suffix_step in range(block_length):
            trajectory_rows.append(
                {
                    "model": model_key,
                    "checkpoint": checkpoint,
                    "checkpoint_order": order,
                    "pair": pair_name,
                    "suffix_step": suffix_step + 1,
                    "hazard_contrast_mean": float(np.mean(hazard_array[:, suffix_step])),
                    "hazard_abs_contrast_mean": float(mean_abs_hazard[suffix_step]),
                    "report_abs_contrast_mean": float(np.mean(report_array[:, suffix_step])),
                    "hidden_rms_distance_mean": float(mean_hidden[suffix_step]),
                }
            )
        for signal_name, signal_values in {
            "hazard_abs_contrast": mean_abs_hazard,
            "hidden_rms_distance": mean_hidden,
        }.items():
            fit = fit_relaxation(signal_values)
            asymptote = fit["offset"] if np.isfinite(fit["offset"]) else float(signal_values[-1])
            half_level = asymptote + 0.5 * (float(signal_values[0]) - asymptote)
            half_life = first_threshold_crossing(signal_values, half_level, "below")
            metric_rows.append(
                {
                    "model": model_key,
                    "checkpoint": checkpoint,
                    "checkpoint_order": order,
                    "pair": pair_name,
                    "signal": signal_name,
                    "initial": float(signal_values[0]),
                    "final": float(signal_values[-1]),
                    "auc": float(np.trapezoid(signal_values)),
                    "empirical_half_life_evidence": half_life,
                    "fit_amplitude": fit["amplitude"],
                    "fit_tau_evidence": fit["tau_evidence"],
                    "fit_offset": fit["offset"],
                    "fit_r2": fit["r2"],
                }
            )
    return trajectory_rows, metric_rows


def condition_summary_rows(
    model_key: str,
    checkpoint: str,
    order: float,
    rows: list[SequenceRow],
    report_prob: np.ndarray,
    hazard_prob: np.ndarray,
    block_length: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        indices = [index for index, row in enumerate(rows) if row.condition == condition]
        condition_hazard = hazard_prob[indices]
        condition_report = report_prob[indices]
        mean_hazard = np.mean(condition_hazard, axis=0)
        target_high_second = condition.endswith("high")
        threshold_direction = "above" if target_high_second else "below"
        crossing = first_threshold_crossing(
            mean_hazard[block_length:], 0.5, threshold_direction
        )
        output.append(
            {
                "model": model_key,
                "checkpoint": checkpoint,
                "checkpoint_order": order,
                "condition": condition,
                "n_sequences_including_mirrors": len(indices),
                "hazard_prob_after_first_evidence": float(mean_hazard[0]),
                "hazard_prob_pre_switch": float(mean_hazard[block_length - 1]),
                "hazard_prob_first_suffix": float(mean_hazard[block_length]),
                "hazard_prob_final": float(mean_hazard[-1]),
                "report_mirror_balanced_mean_final": float(
                    np.mean(condition_report[:, -1])
                ),
                "threshold_crossing_suffix_step": crossing + 1 if np.isfinite(crossing) else np.nan,
            }
        )
    return output


def condition_timecourse_rows(
    model_key: str,
    checkpoint: str,
    order: float,
    rows: list[SequenceRow],
    report_prob: np.ndarray,
    hazard_prob: np.ndarray,
    block_length: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        indices = [index for index, row in enumerate(rows) if row.condition == condition]
        hazard_mean = np.mean(hazard_prob[indices], axis=0)
        hazard_sem = np.std(hazard_prob[indices], axis=0, ddof=1) / np.sqrt(len(indices))
        report_mean = np.mean(report_prob[indices], axis=0)
        for evidence_index in range(2 * block_length):
            output.append(
                {
                    "model": model_key,
                    "checkpoint": checkpoint,
                    "checkpoint_order": order,
                    "condition": condition,
                    "evidence_index": evidence_index + 1,
                    "block": "prefix" if evidence_index < block_length else "suffix",
                    "hazard_prob_mean": float(hazard_mean[evidence_index]),
                    "hazard_prob_sem": float(hazard_sem[evidence_index]),
                    "report_prob_mean": float(report_mean[evidence_index]),
                }
            )
    return output


def plot_relaxation(
    model_key: str,
    condition_timecourse: pd.DataFrame,
    contrast_frame: pd.DataFrame,
    output_dir: Path,
    block_length: int,
) -> None:
    model_conditions = condition_timecourse[condition_timecourse["model"] == model_key]
    model_contrasts = contrast_frame[contrast_frame["model"] == model_key]
    if model_conditions.empty:
        return
    checkpoints = (
        model_conditions[["checkpoint", "checkpoint_order"]]
        .drop_duplicates()
        .sort_values("checkpoint_order")
    )
    fig, axes = plt.subplots(2, len(checkpoints), figsize=(4.0 * len(checkpoints), 7.2), squeeze=False)
    for column, checkpoint in enumerate(checkpoints["checkpoint"]):
        condition_subset = model_conditions[model_conditions["checkpoint"] == checkpoint]
        for condition, condition_df in condition_subset.groupby("condition"):
            condition_df = condition_df.sort_values("evidence_index")
            axes[0, column].plot(
                condition_df["evidence_index"],
                condition_df["hazard_prob_mean"],
                color=CONDITION_COLORS[condition],
                label=condition,
            )
            axes[0, column].fill_between(
                condition_df["evidence_index"].to_numpy(dtype=float),
                (condition_df["hazard_prob_mean"] - condition_df["hazard_prob_sem"]).to_numpy(dtype=float),
                (condition_df["hazard_prob_mean"] + condition_df["hazard_prob_sem"]).to_numpy(dtype=float),
                color=CONDITION_COLORS[condition],
                alpha=0.12,
                linewidth=0,
            )
        axes[0, column].axvline(block_length + 0.5, color="black", linestyle="--", linewidth=1)
        axes[0, column].axhline(0.5, color="0.5", linewidth=1)
        axes[0, column].set_ylim(0, 1)
        axes[0, column].set_title(checkpoint)

        contrast_subset = model_contrasts[
            model_contrasts["checkpoint"] == checkpoint
        ]
        for pair, pair_df in contrast_subset.groupby("pair"):
            pair_df = pair_df.sort_values("suffix_step")
            axes[1, column].plot(
                pair_df["suffix_step"],
                pair_df["hazard_abs_contrast_mean"],
                marker="o",
                markersize=3,
                label=f"{pair}: hazard",
            )
            axes[1, column].plot(
                pair_df["suffix_step"],
                pair_df["hidden_rms_distance_mean"],
                linestyle="--",
                label=f"{pair}: hidden",
            )
        axes[1, column].set_xlabel("Evidence items into shared suffix")
        axes[1, column].grid(True, alpha=0.25)
        axes[0, column].grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Hazard-head P(high)")
    axes[1, 0].set_ylabel("Paired history contrast")
    axes[0, -1].legend(frameon=False, fontsize=7)
    axes[1, -1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model_key}: matched-suffix history relaxation")
    fig.tight_layout()
    fig.savefig(output_dir / f"history_relaxation_{model_key}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.low_hazard < 0.5 < args.high_hazard <= 1.0):
        raise ValueError("Expected low_hazard < 0.5 < high_hazard")
    set_seeds(args.random_seed)
    cfg = load_config(args.config)
    specs = model_specs(cfg, parse_model_filter(args.models))
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = generate_sequence_bank(args)
    condition_rows: list[dict[str, Any]] = []
    condition_time_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    relaxation_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    print(f"Device: {device}")
    print(f"Generated {len(rows)} sequences ({args.n_sequences} paired replicates)")

    for spec in specs:
        hp = json.loads((spec.seed_dir / "hp.json").read_text(encoding="utf-8"))
        n_null = int(hp.get("n_null_timesteps", 4))
        inputs = build_bank_inputs(rows, n_null)
        indices = evidence_timestep_indices(2 * args.block_length, n_null)
        checkpoints = resolve_checkpoints(spec.seed_dir, args.checkpoints)
        print(f"\nModel {spec.key}: {len(checkpoints)} checkpoints")
        for checkpoint_path in checkpoints:
            label = checkpoint_label(checkpoint_path)
            order = checkpoint_position(label)
            print(f"  {label}")
            model, _ = load_model(spec, checkpoint_path, device)
            hidden, report_logits, hazard_logits = collect_evidence_states(
                model, inputs, indices, args.batch_size, device
            )
            report_prob = sigmoid(report_logits)
            hazard_prob = sigmoid(hazard_logits)
            condition_rows.extend(
                condition_summary_rows(
                    spec.key,
                    label,
                    order,
                    rows,
                    report_prob,
                    hazard_prob,
                    args.block_length,
                )
            )
            condition_time_rows.extend(
                condition_timecourse_rows(
                    spec.key,
                    label,
                    order,
                    rows,
                    report_prob,
                    hazard_prob,
                    args.block_length,
                )
            )
            checkpoint_contrasts, checkpoint_metrics = paired_contrast_rows(
                spec.key,
                label,
                order,
                rows,
                hidden,
                report_prob,
                hazard_prob,
                args.block_length,
            )
            contrast_rows.extend(checkpoint_contrasts)
            relaxation_rows.extend(checkpoint_metrics)

            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            pca_rng = np.random.default_rng(args.random_seed)
            if len(flat_hidden) > args.max_pca_points:
                pca_fit_indices = pca_rng.choice(
                    len(flat_hidden), size=args.max_pca_points, replace=False
                )
                pca_fit_states = flat_hidden[pca_fit_indices]
            else:
                pca_fit_states = flat_hidden
            pca = PCA(n_components=3, svd_solver="randomized", random_state=args.random_seed)
            pca.fit(pca_fit_states)
            for row_index, sequence_row in enumerate(rows):
                if sequence_row.replicate >= args.max_trajectory_replicates:
                    continue
                pc = pca.transform(hidden[row_index])
                for evidence_index in range(2 * args.block_length):
                    trajectory_rows.append(
                        {
                            "model": spec.key,
                            "checkpoint": label,
                            "checkpoint_order": order,
                            "condition": sequence_row.condition,
                            "replicate": sequence_row.replicate,
                            "mirror": sequence_row.mirror,
                            "evidence_index": evidence_index + 1,
                            "block": "prefix" if evidence_index < args.block_length else "suffix",
                            "evidence": float(sequence_row.evidence[evidence_index]),
                            "latent_state": float(sequence_row.states[evidence_index]),
                            "report_prob": float(report_prob[row_index, evidence_index]),
                            "hazard_prob": float(hazard_prob[row_index, evidence_index]),
                            "pc1": float(pc[evidence_index, 0]),
                            "pc2": float(pc[evidence_index, 1]),
                            "pc3": float(pc[evidence_index, 2]),
                        }
                    )
            del model, hidden
            if device.type == "cuda":
                torch.cuda.empty_cache()

    condition_df = pd.DataFrame(condition_rows)
    condition_time_df = pd.DataFrame(condition_time_rows)
    contrast_df = pd.DataFrame(contrast_rows)
    relaxation_df = pd.DataFrame(relaxation_rows)
    trajectory_df = pd.DataFrame(trajectory_rows)
    condition_df.to_csv(output_dir / "condition_summary.csv", index=False)
    condition_time_df.to_csv(output_dir / "condition_timecourses.csv", index=False)
    contrast_df.to_csv(output_dir / "paired_suffix_contrasts.csv", index=False)
    relaxation_df.to_csv(output_dir / "relaxation_metrics.csv", index=False)
    trajectory_df.to_csv(
        output_dir / "controlled_trajectories.csv.gz", index=False, compression="gzip"
    )
    np.savez_compressed(
        output_dir / "sequence_bank.npz",
        evidence=np.stack([row.evidence for row in rows]),
        states=np.stack([row.states for row in rows]),
        condition=np.asarray([row.condition for row in rows]),
        replicate=np.asarray([row.replicate for row in rows]),
        mirror=np.asarray([row.mirror for row in rows]),
    )
    for spec in specs:
        plot_relaxation(
            spec.key,
            condition_time_df,
            contrast_df,
            output_dir,
            args.block_length,
        )
    write_run_config(
        output_dir / "run_config.json",
        {
            **vars(args),
            "device_resolved": device,
            "seed": int(cfg.get("seed", 0)),
            "models_resolved": [spec.key for spec in specs],
            "pairing": (
                "Within each suffix type, high- and low-history conditions receive "
                "identical suffix latent states, evidence, and observation noise."
            ),
        },
    )
    print(f"\nSaved history-switch experiment to {output_dir}")


if __name__ == "__main__":
    main()

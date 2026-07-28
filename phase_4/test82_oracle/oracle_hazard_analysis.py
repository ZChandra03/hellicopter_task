#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import betainc, logsumexp, ndtr


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config.json"
DEFAULT_PREDICTIONS = (
    BASE_DIR.parent
    / "test82_input_length"
    / "prefix_length_outputs"
    / "prefix_trial_predictions.csv.gz"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "oracle_outputs"
METHOD_LABELS = {
    "model": "GRU (mean across seeds)",
    "evidence_oracle": "Evidence-matched oracle",
    "state_oracle": "Latent-state oracle",
}
METHOD_COLORS = {
    "model": "#0072B2",
    "evidence_oracle": "#009E73",
    "state_oracle": "#D55E00",
}
METRICS = ("accuracy", "nll", "brier", "p_correct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prefix-length hazard predictions with a Bayes-optimal "
            "observer of noisy evidence and an oracle that sees latent state switches."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help=(
            "Per-trial model predictions from prefix_length_analysis.py. "
            f"Default: {DEFAULT_PREDICTIONS}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of prefix lengths. Default: all lengths in predictions.",
    )
    parser.add_argument(
        "--quadrature-points",
        type=int,
        default=64,
        help="Gauss-Legendre points in each hazard half-interval. Default: 64",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1.0,
        help="Absolute latent evidence mean used by the generator. Default: 1",
    )
    parser.add_argument(
        "--x-limit",
        type=float,
        default=5.0,
        help="Absolute truncation bound used by the evidence generator. Default: 5",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of equal-width probability bins. Default: 10",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional trial cap for smoke testing.",
    )
    return parser.parse_args()


def parse_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item) for item in ast.literal_eval(str(value))]


def signed_int(value: Any) -> int:
    return 1 if float(value) > 0 else -1


def sem(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / math.sqrt(array.size))


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    with config_path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    model_root = Path(raw["model_root"]).expanduser().resolve()
    variant_root = Path(raw["variant_root"]).expanduser().resolve()
    variant_subdir = raw.get("variant_subdir") or raw.get("sigma") or model_root.name
    return {
        **raw,
        "model_root": model_root,
        "variant_root": variant_root,
        "variant_subdir": str(variant_subdir),
        "variant_dir": variant_root / str(variant_subdir),
    }


def load_model_predictions(
    predictions_path: Path,
    requested_lengths: list[int] | None,
    max_trials: int | None,
) -> tuple[pd.DataFrame, list[int]]:
    predictions_path = predictions_path.expanduser().resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing model predictions: {predictions_path}\n"
            "Run phase_4/test82_input_length/prefix_length_analysis.py first, "
            "or pass --predictions."
        )

    predictions = pd.read_csv(predictions_path)
    required = {
        "seed",
        "prefix_length",
        "source_csv",
        "csv_trial",
        "global_trial",
        "true_hazard",
        "true_predict",
        "hazard_prob_high",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions file is missing columns: {missing}")

    available_lengths = sorted(
        int(length) for length in predictions["prefix_length"].dropna().unique()
    )
    if requested_lengths is None:
        prefix_lengths = available_lengths
    else:
        prefix_lengths = sorted(dict.fromkeys(int(length) for length in requested_lengths))
        unavailable = sorted(set(prefix_lengths) - set(available_lengths))
        if unavailable:
            raise ValueError(
                f"Requested prefix lengths {unavailable} are absent from predictions; "
                f"available lengths are {available_lengths}"
            )
    predictions = predictions[predictions["prefix_length"].isin(prefix_lengths)].copy()

    if max_trials is not None:
        if max_trials <= 0:
            raise ValueError("--max-trials must be positive")
        keep_trials = (
            predictions[["source_csv", "csv_trial", "global_trial"]]
            .drop_duplicates()
            .sort_values("global_trial")
            .head(max_trials)
        )
        predictions = predictions.merge(
            keep_trials,
            on=["source_csv", "csv_trial", "global_trial"],
            how="inner",
            validate="many_to_one",
        )

    duplicate_key = ["seed", "prefix_length", "source_csv", "csv_trial"]
    if predictions.duplicated(duplicate_key).any():
        raise ValueError(f"Predictions contain duplicate rows for key {duplicate_key}")

    predictions["prefix_length"] = predictions["prefix_length"].astype(int)
    predictions["seed"] = predictions["seed"].astype(int)
    predictions["csv_trial"] = predictions["csv_trial"].astype(int)
    predictions["true_predict"] = predictions["true_predict"].map(signed_int)
    predictions["target_high"] = (predictions["true_predict"] > 0).astype(int)
    predictions["hazard_prob_high"] = predictions["hazard_prob_high"].clip(0.0, 1.0)
    return predictions, prefix_lengths


def load_trials(
    predictions: pd.DataFrame,
    variant_dir: Path,
    max_prefix_length: int,
) -> pd.DataFrame:
    trial_keys = (
        predictions[["source_csv", "csv_trial", "global_trial"]]
        .drop_duplicates()
        .sort_values("global_trial")
    )
    rows: list[dict[str, Any]] = []

    for source_csv, source_keys in trial_keys.groupby("source_csv", sort=False):
        csv_path = variant_dir / str(source_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not match prediction source to trial CSV: {csv_path}")
        source_df = pd.read_csv(csv_path).reset_index(drop=True)
        for key in source_keys.itertuples(index=False):
            csv_trial = int(key.csv_trial)
            if csv_trial < 0 or csv_trial >= len(source_df):
                raise IndexError(f"{csv_path.name} has no row {csv_trial}")
            row = source_df.iloc[csv_trial]
            evidence = parse_list(row["evidence"])
            states = parse_list(row["states"])
            if len(evidence) != len(states):
                raise ValueError(
                    f"{csv_path.name} row {csv_trial}: evidence and state lengths differ"
                )
            if len(evidence) < max_prefix_length:
                raise ValueError(
                    f"{csv_path.name} row {csv_trial} has length {len(evidence)}, "
                    f"shorter than requested prefix {max_prefix_length}"
                )
            rows.append(
                {
                    "source_csv": str(source_csv),
                    "csv_trial": csv_trial,
                    "global_trial": int(key.global_trial),
                    "sigma": float(row["sigma"]),
                    "true_hazard": float(row["trueHazard"]),
                    "true_predict": signed_int(row["truePredict"]),
                    "target_high": int(float(row["truePredict"]) > 0),
                    "evidence": evidence,
                    "states": states,
                }
            )

    trials = pd.DataFrame(rows).sort_values("global_trial").reset_index(drop=True)
    if len(trials) != len(trial_keys):
        raise RuntimeError("Failed to load exactly one trial for every prediction key")
    if (trials["sigma"] <= 0).any():
        raise ValueError("The evidence-matched oracle currently requires sigma > 0")
    return trials


def truncated_normal_logpdf(
    evidence: np.ndarray,
    mean: float,
    sigma: np.ndarray,
    x_limit: float,
) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    z = (evidence - mean) / sigma
    upper = (x_limit - mean) / sigma
    lower = (-x_limit - mean) / sigma
    normalization = np.maximum(ndtr(upper) - ndtr(lower), np.finfo(float).tiny)
    return (
        -0.5 * z * z
        - np.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - np.log(normalization)
    )


def quadrature_grid(points_per_half: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_per_half < 4:
        raise ValueError("--quadrature-points must be at least 4")
    base_nodes, base_weights = leggauss(points_per_half)
    low_hazard = 0.25 * (base_nodes + 1.0)
    high_hazard = 0.5 + 0.25 * (base_nodes + 1.0)
    half_weights = 0.25 * base_weights
    hazards = np.concatenate([low_hazard, high_hazard])
    weights = np.concatenate([half_weights, half_weights])
    high_mask = np.concatenate(
        [
            np.zeros(points_per_half, dtype=bool),
            np.ones(points_per_half, dtype=bool),
        ]
    )
    return hazards, weights, high_mask


def evidence_oracle_posteriors(
    evidence: np.ndarray,
    sigma: np.ndarray,
    prefix_lengths: list[int],
    mu: float,
    x_limit: float,
    quadrature_points: int,
) -> dict[int, np.ndarray]:
    hazards, weights, high_mask = quadrature_grid(quadrature_points)
    hazard_row = hazards[None, :]
    log_weights = np.log(weights)

    log_emission_neg = truncated_normal_logpdf(
        evidence, -abs(mu), sigma[:, None], x_limit
    )
    log_emission_pos = truncated_normal_logpdf(
        evidence, abs(mu), sigma[:, None], x_limit
    )
    emission_neg = np.exp(log_emission_neg)
    emission_pos = np.exp(log_emission_pos)

    alpha_neg = 0.5 * emission_neg[:, 0, None] * np.ones_like(hazard_row)
    alpha_pos = 0.5 * emission_pos[:, 0, None] * np.ones_like(hazard_row)
    scale = np.maximum(alpha_neg + alpha_pos, np.finfo(float).tiny)
    alpha_neg /= scale
    alpha_pos /= scale
    log_likelihood = np.log(scale)

    requested = set(prefix_lengths)
    results: dict[int, np.ndarray] = {}

    for timestep in range(evidence.shape[1]):
        if timestep > 0:
            previous_neg = alpha_neg
            previous_pos = alpha_pos
            alpha_neg = (
                (1.0 - hazard_row) * previous_neg + hazard_row * previous_pos
            ) * emission_neg[:, timestep, None]
            alpha_pos = (
                hazard_row * previous_neg + (1.0 - hazard_row) * previous_pos
            ) * emission_pos[:, timestep, None]
            scale = np.maximum(alpha_neg + alpha_pos, np.finfo(float).tiny)
            alpha_neg /= scale
            alpha_pos /= scale
            log_likelihood += np.log(scale)

        prefix_length = timestep + 1
        if prefix_length in requested:
            log_low = logsumexp(
                log_likelihood[:, ~high_mask] + log_weights[~high_mask],
                axis=1,
            )
            log_high = logsumexp(
                log_likelihood[:, high_mask] + log_weights[high_mask],
                axis=1,
            )
            normalizer = np.logaddexp(log_low, log_high)
            probability_high = np.exp(log_high - normalizer)
            probability_high[np.isclose(probability_high, 0.5, atol=1e-12)] = 0.5
            results[prefix_length] = probability_high

    missing = sorted(requested - set(results))
    if missing:
        raise RuntimeError(f"Failed to calculate evidence oracle for lengths {missing}")
    return results


def state_oracle_probability(switch_count: np.ndarray, opportunities: int) -> np.ndarray:
    switch_count = np.asarray(switch_count, dtype=float)
    posterior_low = betainc(
        switch_count + 1.0,
        opportunities - switch_count + 1.0,
        0.5,
    )
    return 1.0 - posterior_low


def calculate_oracles(
    trials: pd.DataFrame,
    prefix_lengths: list[int],
    mu: float,
    x_limit: float,
    quadrature_points: int,
) -> pd.DataFrame:
    max_length = max(prefix_lengths)
    evidence = np.asarray(
        [values[:max_length] for values in trials["evidence"]], dtype=float
    )
    states = np.asarray(
        [values[:max_length] for values in trials["states"]], dtype=float
    )
    sigma = trials["sigma"].to_numpy(dtype=float)

    evidence_probs = evidence_oracle_posteriors(
        evidence,
        sigma,
        prefix_lengths,
        mu,
        x_limit,
        quadrature_points,
    )
    switch_events = states[:, 1:] != states[:, :-1]
    cumulative_switches = np.cumsum(switch_events, axis=1)

    rows: list[pd.DataFrame] = []
    identity_cols = [
        "source_csv",
        "csv_trial",
        "global_trial",
        "sigma",
        "true_hazard",
        "true_predict",
        "target_high",
    ]
    for prefix_length in prefix_lengths:
        if prefix_length == 1:
            switch_count = np.zeros(len(trials), dtype=int)
        else:
            switch_count = cumulative_switches[:, prefix_length - 2].astype(int)
        opportunities = prefix_length - 1
        frame = trials[identity_cols].copy()
        frame["prefix_length"] = int(prefix_length)
        frame["transition_opportunities"] = int(opportunities)
        frame["switch_count"] = switch_count
        frame["switch_fraction"] = np.divide(
            switch_count,
            opportunities,
            out=np.full(len(trials), 0.5, dtype=float),
            where=opportunities > 0,
        )
        frame["state_oracle_prob_high"] = state_oracle_probability(
            switch_count, opportunities
        )
        frame["evidence_oracle_prob_high"] = evidence_probs[prefix_length]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float).copy()
    probability[np.isclose(probability, 0.5, atol=1e-12)] = 0.5
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    correct = np.where(
        probability > 0.5,
        target == 1,
        np.where(probability < 0.5, target == 0, 0.5),
    ).astype(float)
    p_correct = np.where(target == 1, probability, 1.0 - probability)
    return {
        "accuracy": float(np.mean(correct)),
        "nll": float(
            np.mean(
                -(target * np.log(clipped) + (1 - target) * np.log(1.0 - clipped))
            )
        ),
        "brier": float(np.mean((probability - target) ** 2)),
        "p_correct": float(np.mean(p_correct)),
    }


def merge_model_and_oracles(
    predictions: pd.DataFrame, oracle: pd.DataFrame
) -> pd.DataFrame:
    keys = ["source_csv", "csv_trial", "global_trial", "prefix_length"]
    oracle_columns = keys + [
        "sigma",
        "true_hazard",
        "true_predict",
        "target_high",
        "transition_opportunities",
        "switch_count",
        "switch_fraction",
        "state_oracle_prob_high",
        "evidence_oracle_prob_high",
    ]
    prediction_columns = [
        "seed",
        *keys,
        "true_hazard",
        "true_predict",
        "target_high",
        "hazard_prob_high",
    ]
    merged = predictions[prediction_columns].merge(
        oracle[oracle_columns],
        on=keys,
        how="left",
        suffixes=("_model", "_oracle"),
        validate="many_to_one",
    )
    if merged["evidence_oracle_prob_high"].isna().any():
        raise RuntimeError("Some model prediction rows did not match an oracle trial")
    if not np.array_equal(
        merged["target_high_model"].to_numpy(),
        merged["target_high_oracle"].to_numpy(),
    ):
        raise RuntimeError("Model and oracle hazard targets disagree")
    merged["target_high"] = merged["target_high_oracle"].astype(int)
    merged["true_hazard"] = merged["true_hazard_oracle"].astype(float)
    merged["true_predict"] = merged["true_predict_oracle"].astype(int)
    merged = merged.drop(
        columns=[
            "target_high_model",
            "target_high_oracle",
            "true_hazard_model",
            "true_hazard_oracle",
            "true_predict_model",
            "true_predict_oracle",
        ]
    )
    return merged


def summarize_comparison(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed_rows: list[dict[str, Any]] = []
    for (seed, prefix_length), group in merged.groupby(["seed", "prefix_length"]):
        target = group["target_high"].to_numpy(dtype=int)
        model_metrics = binary_metrics(
            target, group["hazard_prob_high"].to_numpy(dtype=float)
        )
        evidence_metrics = binary_metrics(
            target, group["evidence_oracle_prob_high"].to_numpy(dtype=float)
        )
        state_metrics = binary_metrics(
            target, group["state_oracle_prob_high"].to_numpy(dtype=float)
        )
        row: dict[str, Any] = {
            "seed": int(seed),
            "prefix_length": int(prefix_length),
            "n_trials": len(group),
        }
        for metric in METRICS:
            row[f"model_{metric}"] = model_metrics[metric]
            row[f"evidence_oracle_{metric}"] = evidence_metrics[metric]
            row[f"state_oracle_{metric}"] = state_metrics[metric]
            row[f"model_minus_evidence_{metric}"] = (
                model_metrics[metric] - evidence_metrics[metric]
            )
            row[f"model_minus_state_{metric}"] = (
                model_metrics[metric] - state_metrics[metric]
            )
        by_seed_rows.append(row)
    by_seed = pd.DataFrame(by_seed_rows).sort_values(["seed", "prefix_length"])

    summary_rows: list[dict[str, Any]] = []
    for prefix_length, group in by_seed.groupby("prefix_length"):
        row = {
            "prefix_length": int(prefix_length),
            "n_seeds": int(group["seed"].nunique()),
            "n_trials_per_seed": float(group["n_trials"].mean()),
        }
        for metric in METRICS:
            row[f"model_{metric}_mean"] = float(group[f"model_{metric}"].mean())
            row[f"model_{metric}_sem"] = sem(group[f"model_{metric}"])
            row[f"evidence_oracle_{metric}"] = float(
                group[f"evidence_oracle_{metric}"].mean()
            )
            row[f"state_oracle_{metric}"] = float(
                group[f"state_oracle_{metric}"].mean()
            )
            row[f"model_minus_evidence_{metric}_mean"] = float(
                group[f"model_minus_evidence_{metric}"].mean()
            )
            row[f"model_minus_evidence_{metric}_sem"] = sem(
                group[f"model_minus_evidence_{metric}"]
            )
            row[f"model_minus_state_{metric}_mean"] = float(
                group[f"model_minus_state_{metric}"].mean()
            )
            row[f"model_minus_state_{metric}_sem"] = sem(
                group[f"model_minus_state_{metric}"]
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("prefix_length")
    return by_seed, summary


def make_trial_mean_table(merged: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "prefix_length",
        "source_csv",
        "csv_trial",
        "global_trial",
        "target_high",
        "true_hazard",
        "true_predict",
        "sigma",
        "transition_opportunities",
        "switch_count",
        "switch_fraction",
        "state_oracle_prob_high",
        "evidence_oracle_prob_high",
    ]
    return (
        merged.groupby(keys, as_index=False, dropna=False)
        .agg(
            model_prob_high=("hazard_prob_high", "mean"),
            model_prob_high_seed_sd=("hazard_prob_high", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["prefix_length", "global_trial"])
    )


def build_state_oracle_lookup(prefix_lengths: list[int]) -> pd.DataFrame:
    rows = []
    for prefix_length in prefix_lengths:
        opportunities = prefix_length - 1
        for switch_count in range(opportunities + 1):
            probability = state_oracle_probability(
                np.asarray([switch_count]), opportunities
            )[0]
            rows.append(
                {
                    "prefix_length": int(prefix_length),
                    "transition_opportunities": int(opportunities),
                    "switch_count": int(switch_count),
                    "switch_fraction": (
                        float(switch_count / opportunities)
                        if opportunities
                        else 0.5
                    ),
                    "state_oracle_prob_high": float(probability),
                }
            )
    return pd.DataFrame(rows)


def build_calibration_tables(
    trial_mean: pd.DataFrame, n_bins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bins < 2:
        raise ValueError("--calibration-bins must be at least 2")
    method_columns = {
        "model": "model_prob_high",
        "evidence_oracle": "evidence_oracle_prob_high",
        "state_oracle": "state_oracle_prob_high",
    }
    rows: list[dict[str, Any]] = []
    for method, probability_column in method_columns.items():
        for prefix_length, group in trial_mean.groupby("prefix_length"):
            probabilities = group[probability_column].to_numpy(dtype=float)
            targets = group["target_high"].to_numpy(dtype=float)
            bin_index = np.minimum(
                (np.clip(probabilities, 0.0, 1.0) * n_bins).astype(int),
                n_bins - 1,
            )
            for current_bin in range(n_bins):
                mask = bin_index == current_bin
                if not np.any(mask):
                    continue
                mean_probability = float(np.mean(probabilities[mask]))
                observed_rate = float(np.mean(targets[mask]))
                rows.append(
                    {
                        "method": method,
                        "prefix_length": int(prefix_length),
                        "calibration_bin": int(current_bin),
                        "bin_low": current_bin / n_bins,
                        "bin_high": (current_bin + 1) / n_bins,
                        "n_trials": int(np.sum(mask)),
                        "mean_probability": mean_probability,
                        "observed_high_rate": observed_rate,
                        "calibration_gap": observed_rate - mean_probability,
                        "absolute_calibration_gap": abs(
                            observed_rate - mean_probability
                        ),
                    }
                )
    calibration = pd.DataFrame(rows)

    summary_rows = []
    for (method, prefix_length), group in calibration.groupby(
        ["method", "prefix_length"]
    ):
        weights = group["n_trials"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "method": method,
                "prefix_length": int(prefix_length),
                "n_trials": int(weights.sum()),
                "expected_calibration_error": float(
                    np.average(group["absolute_calibration_gap"], weights=weights)
                ),
            }
        )
    calibration_summary = pd.DataFrame(summary_rows).sort_values(
        ["method", "prefix_length"]
    )
    return calibration, calibration_summary


def build_agreement_summary(trial_mean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix_length, group in trial_mean.groupby("prefix_length"):
        model = group["model_prob_high"].to_numpy(dtype=float)
        evidence = group["evidence_oracle_prob_high"].to_numpy(dtype=float)
        state = group["state_oracle_prob_high"].to_numpy(dtype=float)
        rows.append(
            {
                "prefix_length": int(prefix_length),
                "n_trials": len(group),
                "model_evidence_pearson_r": float(np.corrcoef(model, evidence)[0, 1])
                if np.std(model) > 0 and np.std(evidence) > 0
                else np.nan,
                "model_evidence_mae": float(np.mean(np.abs(model - evidence))),
                "model_evidence_rmse": float(
                    np.sqrt(np.mean((model - evidence) ** 2))
                ),
                "model_state_pearson_r": float(np.corrcoef(model, state)[0, 1])
                if np.std(model) > 0 and np.std(state) > 0
                else np.nan,
                "model_state_mae": float(np.mean(np.abs(model - state))),
                "evidence_state_pearson_r": float(
                    np.corrcoef(evidence, state)[0, 1]
                )
                if np.std(evidence) > 0 and np.std(state) > 0
                else np.nan,
                "evidence_state_mae": float(np.mean(np.abs(evidence - state))),
            }
        )
    return pd.DataFrame(rows).sort_values("prefix_length")


def build_hazard_bin_summary(trial_mean: pd.DataFrame) -> pd.DataFrame:
    frame = trial_mean.copy()
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    frame["hazard_bin"] = pd.cut(
        frame["true_hazard"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=labels,
        include_lowest=True,
    )
    method_columns = {
        "model": "model_prob_high",
        "evidence_oracle": "evidence_oracle_prob_high",
        "state_oracle": "state_oracle_prob_high",
    }
    rows = []
    for method, probability_column in method_columns.items():
        for (prefix_length, hazard_bin), group in frame.groupby(
            ["prefix_length", "hazard_bin"], observed=True
        ):
            metrics = binary_metrics(
                group["target_high"].to_numpy(dtype=int),
                group[probability_column].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "method": method,
                    "prefix_length": int(prefix_length),
                    "hazard_bin": str(hazard_bin),
                    "n_trials": len(group),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "prefix_length", "hazard_bin"])


def plot_line_with_model_sem(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    x = summary["prefix_length"].to_numpy(dtype=float)
    for method in ["model", "evidence_oracle", "state_oracle"]:
        if method == "model":
            y = summary[f"model_{metric}_mean"].to_numpy(dtype=float)
            error = summary[f"model_{metric}_sem"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                y - error,
                y + error,
                color=METHOD_COLORS[method],
                alpha=0.18,
                linewidth=0,
            )
        else:
            y = summary[f"{method}_{metric}"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Evidence prefix length")
    ax.set_xticks(x.astype(int))
    ax.grid(True, alpha=0.25)


def plot_main_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    plot_line_with_model_sem(
        axes[0, 0],
        summary,
        "accuracy",
        "Classification accuracy (0.5 credit for ties)",
        "Accuracy",
    )
    axes[0, 0].set_ylim(0.45, 1.01)
    plot_line_with_model_sem(
        axes[0, 1],
        summary,
        "nll",
        "Negative log-likelihood",
        "NLL (lower is better)",
    )
    plot_line_with_model_sem(
        axes[1, 0],
        summary,
        "brier",
        "Brier score",
        "Brier score (lower is better)",
    )

    ax = axes[1, 1]
    x = summary["prefix_length"].to_numpy(dtype=float)
    y = summary["model_minus_evidence_nll_mean"].to_numpy(dtype=float)
    error = summary["model_minus_evidence_nll_sem"].to_numpy(dtype=float)
    ax.axhline(0.0, color="0.35", linewidth=1.2)
    ax.fill_between(
        x,
        y - error,
        y + error,
        color=METHOD_COLORS["model"],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(x, y, marker="o", linewidth=2.0, color=METHOD_COLORS["model"])
    ax.set_title("Model NLL above the evidence oracle")
    ax.set_ylabel("Model NLL - oracle NLL")
    ax.set_xlabel("Evidence prefix length")
    ax.set_xticks(x.astype(int))
    ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Hazard inference: trained GRU versus Bayesian observers", y=1.055)
    fig.savefig(
        output_dir / "oracle_model_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_state_oracle_lookup(lookup: pd.DataFrame, output_dir: Path) -> None:
    prefix_lengths = sorted(lookup["prefix_length"].unique())
    max_switches = int(lookup["switch_count"].max())
    matrix = np.full((len(prefix_lengths), max_switches + 1), np.nan)
    for row_idx, prefix_length in enumerate(prefix_lengths):
        group = lookup[lookup["prefix_length"] == prefix_length]
        for row in group.itertuples(index=False):
            matrix[row_idx, int(row.switch_count)] = float(
                row.state_oracle_prob_high
            )

    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(15.0, 5.2))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(np.arange(max_switches + 1))
    ax.set_yticks(np.arange(len(prefix_lengths)))
    ax.set_yticklabels(prefix_lengths)
    ax.set_xlabel("Observed latent-state switches (k)")
    ax.set_ylabel("Evidence prefix length (L)")
    ax.set_title("Exact P(high hazard | k switches in L-1 opportunities)")
    for row_idx in range(matrix.shape[0]):
        for column_idx in range(matrix.shape[1]):
            value = matrix[row_idx, column_idx]
            if np.isfinite(value):
                text_color = "white" if value < 0.18 or value > 0.82 else "black"
                ax.text(
                    column_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )
    fig.colorbar(image, ax=ax, label="Posterior P(high hazard)")
    fig.tight_layout()
    fig.savefig(
        output_dir / "state_oracle_posterior_lookup.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def choose_display_lengths(prefix_lengths: list[int]) -> list[int]:
    desired = [3, 10, max(prefix_lengths)]
    chosen: list[int] = []
    for target in desired:
        closest = min(prefix_lengths, key=lambda length: (abs(length - target), length))
        if closest not in chosen:
            chosen.append(closest)
    return chosen


def plot_calibration(
    calibration: pd.DataFrame,
    prefix_lengths: list[int],
    output_dir: Path,
) -> None:
    display_lengths = choose_display_lengths(prefix_lengths)
    fig, axes = plt.subplots(
        1,
        len(display_lengths),
        figsize=(5.0 * len(display_lengths), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, prefix_length in zip(axes, display_lengths):
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1.2)
        for method in ["model", "evidence_oracle", "state_oracle"]:
            group = calibration[
                (calibration["prefix_length"] == prefix_length)
                & (calibration["method"] == method)
            ].sort_values("mean_probability")
            ax.plot(
                group["mean_probability"],
                group["observed_high_rate"],
                marker="o",
                linewidth=1.8,
                markersize=5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"L = {prefix_length}")
        ax.set_xlabel("Mean predicted P(high)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Observed high-hazard fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle("Probability calibration", y=1.035)
    fig.savefig(
        output_dir / "oracle_calibration.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_probability_agreement(
    trial_mean: pd.DataFrame,
    agreement: pd.DataFrame,
    prefix_lengths: list[int],
    output_dir: Path,
) -> None:
    display_lengths = choose_display_lengths(prefix_lengths)
    fig, axes = plt.subplots(
        1,
        len(display_lengths),
        figsize=(5.0 * len(display_lengths), 4.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, prefix_length in zip(axes, display_lengths):
        group = trial_mean[trial_mean["prefix_length"] == prefix_length]
        correlation_row = agreement[agreement["prefix_length"] == prefix_length].iloc[0]
        hexbin = ax.hexbin(
            group["evidence_oracle_prob_high"],
            group["model_prob_high"],
            gridsize=30,
            extent=(0, 1, 0, 1),
            mincnt=1,
            bins="log",
            cmap="Blues",
        )
        ax.plot([0, 1], [0, 1], color="#D55E00", linewidth=1.3)
        correlation = correlation_row["model_evidence_pearson_r"]
        ax.set_title(f"L = {prefix_length}, r = {correlation:.2f}")
        ax.set_xlabel("Evidence oracle P(high)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.colorbar(hexbin, ax=ax, label="Trial count")
    axes[0].set_ylabel("Mean GRU P(high)")
    fig.suptitle("Trial-level agreement with the evidence-matched oracle", y=1.035)
    fig.savefig(
        output_dir / "model_vs_evidence_oracle_probability.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_hazard_bin_accuracy(hazard_summary: pd.DataFrame, output_dir: Path) -> None:
    methods = ["model", "evidence_oracle", "state_oracle"]
    prefix_lengths = sorted(hazard_summary["prefix_length"].unique())
    hazard_bins = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.0),
        sharey=True,
        constrained_layout=True,
    )
    for ax, method in zip(axes, methods):
        group = hazard_summary[hazard_summary["method"] == method]
        pivot = (
            group.pivot(
                index="prefix_length", columns="hazard_bin", values="accuracy"
            )
            .reindex(index=prefix_lengths, columns=hazard_bins)
        )
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("True hazard")
        ax.set_xticks(np.arange(len(hazard_bins)))
        ax.set_xticklabels(hazard_bins, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(prefix_lengths)))
        ax.set_yticklabels(prefix_lengths)
        for row_idx, prefix_length in enumerate(prefix_lengths):
            for column_idx, hazard_bin in enumerate(hazard_bins):
                value = pivot.loc[prefix_length, hazard_bin]
                if np.isfinite(value):
                    color = "white" if value < 0.35 else "black"
                    ax.text(
                        column_idx,
                        row_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Evidence prefix length")
    fig.suptitle(
        "Hazard classification accuracy by true generating hazard "
        "(0.5 credit for exact ties)"
    )
    fig.savefig(
        output_dir / "oracle_accuracy_by_hazard_bin.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_run_config(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    predictions: pd.DataFrame,
    prefix_lengths: list[int],
    output_dir: Path,
) -> None:
    serializable = {
        "config": str(args.config.expanduser().resolve()),
        "predictions": str(args.predictions.expanduser().resolve()),
        "model_root": str(cfg["model_root"]),
        "variant_dir": str(cfg["variant_dir"]),
        "output_dir": str(output_dir),
        "prefix_lengths": prefix_lengths,
        "seeds": sorted(int(seed) for seed in predictions["seed"].unique()),
        "n_trials": int(predictions["global_trial"].nunique()),
        "quadrature_points_per_hazard_half": int(args.quadrature_points),
        "mu": float(args.mu),
        "x_limit": float(args.x_limit),
        "calibration_bins": int(args.calibration_bins),
        "max_trials": args.max_trials,
        "accuracy_tie_rule": (
            "Predictions exactly equal to 0.5 receive 0.5 accuracy credit; "
            "NLL, Brier score, and p(correct) use the unmodified probability."
        ),
        "state_oracle": (
            "Closed-form posterior from latent switch count; privileged-information ceiling."
        ),
        "evidence_oracle": (
            "HMM likelihood integrated over a uniform hazard prior using Gauss-Legendre "
            "quadrature; receives the same noisy evidence prefixes as the model."
        ),
    }
    with (output_dir / "oracle_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    predictions, prefix_lengths = load_model_predictions(
        args.predictions, args.prefix_lengths, args.max_trials
    )
    trials = load_trials(predictions, cfg["variant_dir"], max(prefix_lengths))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args, cfg, predictions, prefix_lengths, output_dir)

    print(
        f"Loaded {len(trials)} trials, {predictions['seed'].nunique()} seeds, "
        f"and prefix lengths {prefix_lengths}"
    )
    print("Computing latent-state and evidence-matched Bayesian observers")
    oracle = calculate_oracles(
        trials,
        prefix_lengths,
        args.mu,
        args.x_limit,
        args.quadrature_points,
    )
    oracle.to_csv(output_dir / "oracle_trial_predictions.csv.gz", index=False)

    merged = merge_model_and_oracles(predictions, oracle)
    merged.to_csv(output_dir / "model_oracle_trial_predictions.csv.gz", index=False)
    by_seed, summary = summarize_comparison(merged)
    by_seed.to_csv(output_dir / "oracle_comparison_by_seed.csv", index=False)
    summary.to_csv(output_dir / "oracle_comparison_summary.csv", index=False)

    trial_mean = make_trial_mean_table(merged)
    trial_mean.to_csv(output_dir / "oracle_trial_model_mean.csv.gz", index=False)
    state_lookup = build_state_oracle_lookup(prefix_lengths)
    state_lookup.to_csv(output_dir / "state_oracle_posterior_lookup.csv", index=False)
    calibration, calibration_summary = build_calibration_tables(
        trial_mean, args.calibration_bins
    )
    calibration.to_csv(output_dir / "oracle_calibration.csv", index=False)
    calibration_summary.to_csv(
        output_dir / "oracle_calibration_summary.csv", index=False
    )
    agreement = build_agreement_summary(trial_mean)
    agreement.to_csv(output_dir / "oracle_probability_agreement.csv", index=False)
    hazard_summary = build_hazard_bin_summary(trial_mean)
    hazard_summary.to_csv(output_dir / "oracle_by_hazard_bin.csv", index=False)

    plot_main_comparison(summary, output_dir)
    plot_state_oracle_lookup(state_lookup, output_dir)
    plot_calibration(calibration, prefix_lengths, output_dir)
    plot_probability_agreement(
        trial_mean, agreement, prefix_lengths, output_dir
    )
    plot_hazard_bin_accuracy(hazard_summary, output_dir)

    print(f"Saved oracle comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()

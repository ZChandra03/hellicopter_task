from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
TEST84_DIR = BASE_DIR.parent / "test84_manifold"
if str(TEST84_DIR) not in sys.path:
    sys.path.insert(0, str(TEST84_DIR))

from manifold_utils import (  # noqa: E402
    build_input_tensor,
    checkpoint_label,
    choose_device,
    collect_evidence_states,
    compute_normative_features,
    evidence_timestep_indices,
    load_config,
    load_model,
    load_trials,
    model_specs,
    resolve_checkpoints,
    set_seeds,
    sigmoid,
    write_run_config,
)
from slow_point_tracking import (  # noqa: E402
    apply_map,
    classify_stability,
    deduplicate,
    jacobian_eigenvalues,
    map_definitions,
    point_type,
)


DEFAULT_CONFIG = BASE_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Epoch-10 task-aligned manifold, hazard-to-report gating, vector-field, "
            "and input-conditioned fixed-point analysis."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument("--model", default="bayesian")
    parser.add_argument("--checkpoint", default="10")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-csvs", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seed", type=int, default=8510)
    parser.add_argument("--probe-trials", type=int, default=400)
    parser.add_argument("--probe-evidence-min", type=float, default=-2.0)
    parser.add_argument("--probe-evidence-max", type=float, default=2.0)
    parser.add_argument("--probe-evidence-count", type=int, default=61)
    parser.add_argument("--local-structure-points", type=int, default=900)
    parser.add_argument("--local-neighbors", type=int, default=48)
    parser.add_argument(
        "--fixed-evidence-grid",
        default="-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,1.5",
    )
    parser.add_argument("--fixed-seeds", type=int, default=48)
    parser.add_argument("--fixed-opt-steps", type=int, default=3000)
    parser.add_argument("--fixed-learning-rate", type=float, default=0.025)
    parser.add_argument("--fixed-patience", type=int, default=800)
    parser.add_argument("--fixed-dedup-eps", type=float, default=0.02)
    parser.add_argument("--fixed-max-per-evidence", type=int, default=10)
    parser.add_argument("--fixed-jacobians-per-evidence", type=int, default=3)
    parser.add_argument("--fixed-tol", type=float, default=1e-5)
    parser.add_argument("--slow-tol", type=float, default=1e-3)
    parser.add_argument("--eig-tol", type=float, default=0.01)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_float_grid(text: str) -> np.ndarray:
    values = np.asarray([float(part.strip()) for part in text.split(",")], dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("A non-empty comma-separated evidence grid is required")
    return np.unique(values)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def model_axes(
    model,
    states: np.ndarray,
    last_evidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build orthonormal report, hazard-exclusive, and residual-variance axes."""
    report_weight = model.loc_head.weight.detach().cpu().numpy().reshape(-1).astype(float)
    hazard_weight = model.haz_head.weight.detach().cpu().numpy().reshape(-1).astype(float)
    report_axis = report_weight / np.linalg.norm(report_weight)
    hazard_residual = hazard_weight - np.dot(hazard_weight, report_axis) * report_axis
    hazard_axis = hazard_residual / np.linalg.norm(hazard_residual)

    center = states.mean(axis=0, dtype=np.float64)
    centered = states.astype(np.float64, copy=False) - center
    residual = centered.copy()
    residual -= np.outer(residual @ report_axis, report_axis)
    residual -= np.outer(residual @ hazard_axis, hazard_axis)
    covariance = residual.T @ residual / max(len(residual) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    residual_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    evidence_corr = np.corrcoef(residual @ residual_axis, last_evidence)[0, 1]
    if np.isfinite(evidence_corr) and evidence_corr < 0:
        residual_axis = -residual_axis

    axes = np.stack([report_axis, hazard_axis, residual_axis], axis=1)
    raw = states @ axes
    means = raw.mean(axis=0)
    scales = raw.std(axis=0, ddof=1)
    if np.any(scales <= 1e-12):
        raise RuntimeError(f"Degenerate task-aligned scale: {scales}")

    total_variance = float(np.var(states, axis=0, ddof=1).sum())
    task_variance = float(np.var(raw, axis=0, ddof=1).sum())
    report_hazard_cosine = float(
        np.dot(report_weight, hazard_weight)
        / (np.linalg.norm(report_weight) * np.linalg.norm(hazard_weight))
    )
    report_boundary_z = float(
        ((0.0 / np.linalg.norm(report_weight)) - means[0]) / scales[0]
    )
    metadata = {
        "coordinate_names": [
            "report_readout",
            "hazard_exclusive",
            "residual_variance",
        ],
        "coordinate_means": means.tolist(),
        "coordinate_scales": scales.tolist(),
        "report_weight_norm": float(np.linalg.norm(report_weight)),
        "hazard_weight_norm": float(np.linalg.norm(hazard_weight)),
        "report_hazard_weight_cosine": report_hazard_cosine,
        "hazard_head_gain_along_exclusive_axis": float(np.dot(hazard_weight, hazard_axis)),
        "report_decision_boundary_z": report_boundary_z,
        "task_aligned_variance_fraction": task_variance / total_variance,
        "residual_axis_last_evidence_correlation": float(
            np.corrcoef((states - center) @ residual_axis, last_evidence)[0, 1]
        ),
    }
    return axes, means, scales, metadata


def project_states(
    states: np.ndarray,
    axes: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    return (np.asarray(states) @ axes - means) / scales


def stratified_indices(
    x: np.ndarray,
    y: np.ndarray,
    count: int,
    rng: np.random.Generator,
    bins: int = 5,
) -> np.ndarray:
    if count >= len(x):
        return np.arange(len(x), dtype=int)
    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    xb = np.clip(np.digitize(x, x_edges[1:-1]), 0, bins - 1)
    yb = np.clip(np.digitize(y, y_edges[1:-1]), 0, bins - 1)
    groups = [np.flatnonzero((xb == i) & (yb == j)) for i in range(bins) for j in range(bins)]
    groups = [group for group in groups if len(group)]
    chosen: list[int] = []
    while len(chosen) < count and groups:
        next_groups: list[np.ndarray] = []
        for group in groups:
            available = np.setdiff1d(group, np.asarray(chosen, dtype=int), assume_unique=False)
            if len(available):
                chosen.append(int(rng.choice(available)))
                next_groups.append(group)
                if len(chosen) >= count:
                    break
        groups = next_groups
    return np.asarray(chosen, dtype=int)


def trajectory_frame(
    trials,
    hidden: np.ndarray,
    report_logits: np.ndarray,
    hazard_logits: np.ndarray,
    normative: dict[str, np.ndarray],
    coords: np.ndarray,
) -> pd.DataFrame:
    n_trials, n_evidence, _ = hidden.shape
    frame = pd.DataFrame(
        {
            "trial": np.repeat(np.arange(n_trials), n_evidence),
            "evidence_index": np.tile(np.arange(1, n_evidence + 1), n_trials),
            "evidence": np.concatenate([trial.evidence for trial in trials]),
            "true_hazard": np.repeat([trial.true_hazard for trial in trials], n_evidence),
            "true_report": np.repeat([trial.true_report for trial in trials], n_evidence),
            "report_logit": report_logits.reshape(-1),
            "hazard_logit": hazard_logits.reshape(-1),
            "report_prob": sigmoid(report_logits.reshape(-1)),
            "hazard_prob": sigmoid(hazard_logits.reshape(-1)),
            "bayes_state_belief": normative["bayes_state_belief"].reshape(-1),
            "bayes_hazard_mean": normative["bayes_hazard_mean"].reshape(-1),
            "coord_report": coords[..., 0].reshape(-1),
            "coord_hazard": coords[..., 1].reshape(-1),
            "coord_residual": coords[..., 2].reshape(-1),
        }
    )
    return frame


def choose_example_trials(trials, count: int, rng: np.random.Generator) -> np.ndarray:
    final_evidence = np.asarray([trial.evidence[-1] for trial in trials])
    true_report = np.asarray([trial.true_report for trial in trials])
    true_hazard = np.asarray([trial.true_hazard for trial in trials])
    final_sign = np.where(final_evidence > 0, 1, -1)
    categories = [
        np.flatnonzero(np.abs(final_evidence) <= 0.2),
        np.flatnonzero(final_sign != true_report),
        np.flatnonzero(true_hazard <= 0.2),
        np.flatnonzero(true_hazard >= 0.8),
    ]
    chosen: list[int] = []
    while len(chosen) < count:
        progressed = False
        for category in categories:
            remaining = np.setdiff1d(category, np.asarray(chosen, dtype=int), assume_unique=False)
            if len(remaining):
                chosen.append(int(rng.choice(remaining)))
                progressed = True
                if len(chosen) >= count:
                    break
        if not progressed:
            break
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(len(trials)), np.asarray(chosen, dtype=int))
        chosen.extend(rng.choice(remaining, size=count - len(chosen), replace=False).tolist())
    return np.asarray(chosen, dtype=int)


def local_structure(
    states: np.ndarray,
    coords: np.ndarray,
    trajectory: pd.DataFrame,
    sample_count: int,
    neighbors: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_count = min(sample_count, len(states))
    sample_indices = rng.choice(len(states), size=sample_count, replace=False)
    k = min(neighbors, len(states))
    nn = NearestNeighbors(n_neighbors=k).fit(states)
    _, indices = nn.kneighbors(states[sample_indices])
    rows: list[dict[str, Any]] = []
    for sample_index, neighbor_indices in zip(sample_indices, indices):
        neighborhood = states[neighbor_indices]
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = singular**2 / max(len(neighborhood) - 1, 1)
        total = float(eigenvalues.sum())
        pr = float(total**2 / np.sum(eigenvalues**2)) if total > 0 else np.nan
        point = trajectory.iloc[int(sample_index)]
        rows.append(
            {
                "flat_state_index": int(sample_index),
                "trial": int(point["trial"]),
                "evidence_index": int(point["evidence_index"]),
                "coord_report": coords[sample_index, 0],
                "coord_hazard": coords[sample_index, 1],
                "coord_residual": coords[sample_index, 2],
                "local_participation_ratio": pr,
                "local_top2_variance_fraction": float(eigenvalues[:2].sum() / total),
                "local_top3_variance_fraction": float(eigenvalues[:3].sum() / total),
            }
        )
    local = pd.DataFrame(rows)
    sheet = bin_manifold_sheet(trajectory, local)
    return local, sheet


def bin_manifold_sheet(trajectory: pd.DataFrame, local: pd.DataFrame, bins: int = 12) -> pd.DataFrame:
    report_edges = np.linspace(
        trajectory["coord_report"].quantile(0.01),
        trajectory["coord_report"].quantile(0.99),
        bins + 1,
    )
    hazard_edges = np.linspace(
        trajectory["coord_hazard"].quantile(0.01),
        trajectory["coord_hazard"].quantile(0.99),
        bins + 1,
    )
    work = trajectory.copy()
    work["report_bin"] = np.digitize(work["coord_report"], report_edges[1:-1])
    work["hazard_bin"] = np.digitize(work["coord_hazard"], hazard_edges[1:-1])
    local_work = local.copy()
    local_work["report_bin"] = np.digitize(local_work["coord_report"], report_edges[1:-1])
    local_work["hazard_bin"] = np.digitize(local_work["coord_hazard"], hazard_edges[1:-1])
    local_summary = (
        local_work.groupby(["report_bin", "hazard_bin"], observed=True)
        .agg(
            local_participation_ratio=("local_participation_ratio", "mean"),
            local_top3_variance_fraction=("local_top3_variance_fraction", "mean"),
        )
        .reset_index()
    )
    summary = (
        work.groupby(["report_bin", "hazard_bin"], observed=True)
        .agg(
            n=("coord_residual", "size"),
            coord_report=("coord_report", "mean"),
            coord_hazard=("coord_hazard", "mean"),
            residual_mean=("coord_residual", "mean"),
            residual_sd=("coord_residual", "std"),
            report_prob=("report_prob", "mean"),
            hazard_prob=("hazard_prob", "mean"),
            bayes_hazard_mean=("bayes_hazard_mean", "mean"),
        )
        .reset_index()
        .merge(local_summary, on=["report_bin", "hazard_bin"], how="left")
    )
    return summary


def batch_cycle(model, hidden: torch.Tensor, evidence: torch.Tensor, n_null: int) -> torch.Tensor:
    """Advance post-evidence states through null steps and one evidence event."""
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape (batch, units)")
    evidence = evidence.reshape(-1)
    if len(evidence) == 1 and len(hidden) != 1:
        evidence = evidence.expand(len(hidden))
    if len(evidence) != len(hidden):
        raise ValueError("evidence length must be one or equal hidden batch size")
    sequence = torch.zeros(
        (len(hidden), n_null + 1, 2), dtype=hidden.dtype, device=hidden.device
    )
    sequence[:, -1, 0] = evidence
    sequence[:, -1, 1] = 1.0
    _, final = model.rnn.gru(sequence, hidden.unsqueeze(0))
    return final.squeeze(0)


def probe_grid(
    model,
    prefix_states: np.ndarray,
    evidence_grid: np.ndarray,
    axes: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    n_null: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    n_trials = len(prefix_states)
    n_evidence = len(evidence_grid)
    report_logits = np.empty((n_trials, n_evidence), dtype=np.float32)
    hazard_logits = np.empty_like(report_logits)
    post_coords = np.empty((n_trials, n_evidence, 3), dtype=np.float32)
    hazard_gain = np.empty_like(report_logits)
    hazard_probability_gain = np.empty_like(report_logits)
    report_memory_gain = np.empty_like(report_logits)
    residual_gain = np.empty_like(report_logits)

    axes_tensor = torch.tensor(axes, dtype=torch.float32, device=device)
    scale_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
    hazard_head_gain = float(
        model.haz_head.weight.detach().reshape(-1) @ axes_tensor[:, 1]
    )
    if abs(hazard_head_gain) < 1e-8:
        raise RuntimeError("Hazard-exclusive axis has negligible hazard head gain")

    was_training = model.rnn.gru.training
    model.rnn.gru.train(True)
    for evidence_index, evidence_value in enumerate(evidence_grid):
        hidden = torch.tensor(
            prefix_states, dtype=torch.float32, device=device, requires_grad=True
        )
        evidence = torch.full((n_trials,), float(evidence_value), device=device)
        post = batch_cycle(model, hidden, evidence, n_null)
        logits = model.loc_head(post).squeeze(-1)
        haz = model.haz_head(post).squeeze(-1)
        gradient = torch.autograd.grad(logits.sum(), hidden, retain_graph=False)[0]
        gain_per_hazard_logit = (gradient @ axes_tensor[:, 1]) / hazard_head_gain
        p = torch.sigmoid(logits)
        report_logits[:, evidence_index] = logits.detach().cpu().numpy()
        hazard_logits[:, evidence_index] = haz.detach().cpu().numpy()
        post_coords[:, evidence_index] = project_states(
            post.detach().cpu().numpy(), axes, means, scales
        )
        hazard_gain[:, evidence_index] = gain_per_hazard_logit.detach().cpu().numpy()
        hazard_probability_gain[:, evidence_index] = (
            p * (1.0 - p) * gain_per_hazard_logit
        ).detach().cpu().numpy()
        report_memory_gain[:, evidence_index] = (
            gradient @ (axes_tensor[:, 0] * scale_tensor[0])
        ).detach().cpu().numpy()
        residual_gain[:, evidence_index] = (
            gradient @ (axes_tensor[:, 2] * scale_tensor[2])
        ).detach().cpu().numpy()
    model.rnn.gru.train(was_training)
    input_gain = np.gradient(report_logits.astype(float), evidence_grid, axis=1)
    return {
        "report_logit": report_logits,
        "hazard_logit": hazard_logits,
        "report_prob": sigmoid(report_logits),
        "post_coords": post_coords,
        "hazard_to_report_logit_gain": hazard_gain,
        "hazard_to_report_probability_gain": hazard_probability_gain,
        "report_memory_gain": report_memory_gain,
        "residual_memory_gain": residual_gain,
        "input_evidence_gain": input_gain,
    }


def zero_crossing(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    sign_changes = np.flatnonzero(y[:-1] * y[1:] <= 0)
    if len(sign_changes):
        candidates: list[tuple[float, int]] = []
        for index in sign_changes:
            y0, y1 = y[index], y[index + 1]
            if y1 == y0:
                crossing = float((x[index] + x[index + 1]) / 2)
            else:
                crossing = float(x[index] - y0 * (x[index + 1] - x[index]) / (y1 - y0))
            candidates.append((crossing, int(index)))
        return min(candidates, key=lambda value: abs(value[0]))
    nearest = int(np.argmin(np.abs(y)))
    return float("nan"), nearest


def summarize_probe(
    trials,
    trial_indices: np.ndarray,
    prefix_coords: np.ndarray,
    prefix_report_logits: np.ndarray,
    prefix_hazard_logits: np.ndarray,
    normative: dict[str, np.ndarray],
    evidence_grid: np.ndarray,
    probe: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_trials = len(trial_indices)
    prefix_hazard_prob = sigmoid(prefix_hazard_logits)
    quartile = pd.qcut(prefix_hazard_prob, 4, labels=False, duplicates="drop").astype(int)
    input_gain = probe["input_evidence_gain"]
    transitions: list[dict[str, Any]] = []
    for row_index, trial_index in enumerate(trial_indices):
        threshold, nearest = zero_crossing(evidence_grid, probe["report_logit"][row_index])
        trial = trials[int(trial_index)]
        transitions.append(
            {
                "trial": int(trial_index),
                "true_hazard": trial.true_hazard,
                "true_report": trial.true_report,
                "prefix_report_logit": prefix_report_logits[row_index],
                "prefix_hazard_logit": prefix_hazard_logits[row_index],
                "prefix_hazard_prob": prefix_hazard_prob[row_index],
                "prefix_hazard_quartile": int(quartile[row_index]),
                "prefix_bayes_state_belief": normative["bayes_state_belief"][trial_index, -2],
                "prefix_bayes_hazard_mean": normative["bayes_hazard_mean"][trial_index, -2],
                "prefix_coord_report": prefix_coords[row_index, 0],
                "prefix_coord_hazard": prefix_coords[row_index, 1],
                "prefix_coord_residual": prefix_coords[row_index, 2],
                "report_transition_evidence": threshold,
                "nearest_grid_evidence": evidence_grid[nearest],
                "transition_hazard_to_report_logit_gain": probe[
                    "hazard_to_report_logit_gain"
                ][row_index, nearest],
                "transition_hazard_to_report_probability_gain": probe[
                    "hazard_to_report_probability_gain"
                ][row_index, nearest],
                "transition_input_evidence_gain": input_gain[row_index, nearest],
                "transition_coord_report": probe["post_coords"][row_index, nearest, 0],
                "transition_coord_hazard": probe["post_coords"][row_index, nearest, 1],
                "transition_coord_residual": probe["post_coords"][row_index, nearest, 2],
            }
        )
    transition_frame = pd.DataFrame(transitions)

    long_rows: list[pd.DataFrame] = []
    for evidence_index, evidence_value in enumerate(evidence_grid):
        long_rows.append(
            pd.DataFrame(
                {
                    "probe_row": np.arange(n_trials),
                    "trial": trial_indices,
                    "evidence": evidence_value,
                    "prefix_hazard_quartile": quartile,
                    "prefix_report_logit": prefix_report_logits,
                    "prefix_hazard_prob": prefix_hazard_prob,
                    "report_prob": probe["report_prob"][:, evidence_index],
                    "report_logit": probe["report_logit"][:, evidence_index],
                    "hazard_to_report_logit_gain": probe[
                        "hazard_to_report_logit_gain"
                    ][:, evidence_index],
                    "hazard_to_report_probability_gain": probe[
                        "hazard_to_report_probability_gain"
                    ][:, evidence_index],
                    "input_evidence_gain": input_gain[:, evidence_index],
                    "post_coord_report": probe["post_coords"][:, evidence_index, 0],
                    "post_coord_hazard": probe["post_coords"][:, evidence_index, 1],
                    "post_coord_residual": probe["post_coords"][:, evidence_index, 2],
                }
            )
        )
    probe_long = pd.concat(long_rows, ignore_index=True)
    probe_long["prefix_prior_sign"] = np.where(
        probe_long["prefix_report_logit"] >= 0, 1, -1
    )
    neutral_cutoff = float(np.quantile(np.abs(prefix_report_logits), 0.5))
    neutral = probe_long[np.abs(probe_long["prefix_report_logit"]) <= neutral_cutoff]
    psychometric = (
        neutral.groupby(
            ["prefix_prior_sign", "prefix_hazard_quartile", "evidence"],
            observed=True,
        )
        .agg(
            n=("trial", "size"),
            report_prob=("report_prob", "mean"),
            report_prob_sem=("report_prob", lambda value: value.std(ddof=1) / math.sqrt(len(value))),
            hazard_to_report_probability_gain=(
                "hazard_to_report_probability_gain",
                "mean",
            ),
        )
        .reset_index()
    )
    probe_long["prefix_report_bin"] = pd.qcut(
        probe_long["prefix_report_logit"], 8, labels=False, duplicates="drop"
    )
    surface = (
        probe_long.groupby(["prefix_report_bin", "evidence"], observed=True)
        .agg(
            n=("trial", "size"),
            prefix_report_logit=("prefix_report_logit", "mean"),
            prefix_hazard_prob=("prefix_hazard_prob", "mean"),
            report_prob=("report_prob", "mean"),
            hazard_to_report_logit_gain=("hazard_to_report_logit_gain", "mean"),
            hazard_to_report_probability_gain=(
                "hazard_to_report_probability_gain",
                "mean",
            ),
            input_evidence_gain=("input_evidence_gain", "mean"),
            post_coord_report=("post_coord_report", "mean"),
            post_coord_hazard=("post_coord_hazard", "mean"),
            post_coord_residual=("post_coord_residual", "mean"),
        )
        .reset_index()
    )
    return transition_frame, probe_long, psychometric, surface


@torch.inference_mode()
def controlled_vector_field(
    model,
    states: np.ndarray,
    coords: np.ndarray,
    evidence_values: list[float],
    axes: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    n_null: int,
    device: torch.device,
    rng: np.random.Generator,
    maximum: int = 6000,
    bins: int = 12,
) -> pd.DataFrame:
    if len(states) > maximum:
        selected = rng.choice(len(states), size=maximum, replace=False)
        states = states[selected]
        coords = coords[selected]
    report_edges = np.linspace(np.quantile(coords[:, 0], 0.01), np.quantile(coords[:, 0], 0.99), bins + 1)
    hazard_edges = np.linspace(np.quantile(coords[:, 1], 0.01), np.quantile(coords[:, 1], 0.99), bins + 1)
    rows: list[pd.DataFrame] = []
    hidden = torch.tensor(states, dtype=torch.float32, device=device)
    for evidence_value in evidence_values:
        next_parts: list[np.ndarray] = []
        for start in range(0, len(states), 1024):
            part = hidden[start : start + 1024]
            evidence = torch.full((len(part),), evidence_value, device=device)
            next_parts.append(batch_cycle(model, part, evidence, n_null).cpu().numpy())
        next_states = np.concatenate(next_parts)
        next_coords = project_states(next_states, axes, means, scales)
        delta = next_coords - coords
        frame = pd.DataFrame(
            {
                "evidence": evidence_value,
                "coord_report": coords[:, 0],
                "coord_hazard": coords[:, 1],
                "coord_residual": coords[:, 2],
                "delta_report": delta[:, 0],
                "delta_hazard": delta[:, 1],
                "delta_residual": delta[:, 2],
                "speed_3d": np.linalg.norm(delta, axis=1),
                "report_bin": np.digitize(coords[:, 0], report_edges[1:-1]),
                "hazard_bin": np.digitize(coords[:, 1], hazard_edges[1:-1]),
            }
        )
        rows.append(
            frame.groupby(["evidence", "report_bin", "hazard_bin"], observed=True)
            .agg(
                n=("speed_3d", "size"),
                coord_report=("coord_report", "mean"),
                coord_hazard=("coord_hazard", "mean"),
                coord_residual=("coord_residual", "mean"),
                delta_report=("delta_report", "mean"),
                delta_hazard=("delta_hazard", "mean"),
                delta_residual=("delta_residual", "mean"),
                speed_3d=("speed_3d", "mean"),
            )
            .reset_index()
        )
    return pd.concat(rows, ignore_index=True)


def farthest_seeds(states: np.ndarray, coords: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(states))
    selected = [int(np.argmax(np.linalg.norm(coords, axis=1)))]
    min_distance = np.linalg.norm(coords - coords[selected[0]], axis=1)
    for _ in range(1, count):
        index = int(np.argmax(min_distance))
        selected.append(index)
        min_distance = np.minimum(min_distance, np.linalg.norm(coords - coords[index], axis=1))
    return states[np.asarray(selected, dtype=int)]


def optimize_conditioned_points(
    model,
    initial_states: np.ndarray,
    evidence_values: np.ndarray,
    n_null: int,
    device: torch.device,
    steps: int,
    learning_rate: float,
    patience: int,
) -> tuple[np.ndarray, np.ndarray]:
    was_training = model.rnn.gru.training
    model.rnn.gru.train(True)
    hidden = torch.tensor(initial_states, dtype=torch.float32, device=device, requires_grad=True)
    evidence = torch.tensor(evidence_values, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([hidden], lr=learning_rate)
    best = math.inf
    stale = 0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = batch_cycle(model, hidden, evidence, n_null) - hidden
        q = 0.5 * torch.sum(delta * delta, dim=1)
        loss = q.mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best * (1.0 - 1e-7):
            best = value
            stale = 0
        else:
            stale += 1
        if patience and stale >= patience:
            break
    with torch.no_grad():
        speed = torch.linalg.norm(batch_cycle(model, hidden, evidence, n_null) - hidden, dim=1)
    model.rnn.gru.train(was_training)
    return hidden.detach().cpu().numpy(), speed.detach().cpu().numpy()


def connect_branches(frame: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["branch"] = -1
    next_branch = 0
    previous_rows: np.ndarray | None = None
    for evidence in sorted(result["evidence"].unique()):
        current_rows = result.index[result["evidence"] == evidence].to_numpy()
        if previous_rows is None:
            for row in current_rows:
                result.loc[row, "branch"] = next_branch
                next_branch += 1
        else:
            used_previous: set[int] = set()
            for row in current_rows:
                distances = np.linalg.norm(states[previous_rows] - states[row], axis=1)
                for order in np.argsort(distances):
                    previous = int(previous_rows[order])
                    if previous not in used_previous:
                        result.loc[row, "branch"] = int(result.loc[previous, "branch"])
                        used_previous.add(previous)
                        break
                if result.loc[row, "branch"] < 0:
                    result.loc[row, "branch"] = next_branch
                    next_branch += 1
        previous_rows = current_rows
    result["branch"] = result["branch"].astype(int)
    return result


def conditioned_fixed_points(
    model,
    empirical_states: np.ndarray,
    empirical_coords: np.ndarray,
    mirrored_states: np.ndarray,
    mirrored_coords: np.ndarray,
    evidence_grid: np.ndarray,
    axes: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    n_null: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray]:
    pool_states = np.concatenate([empirical_states, mirrored_states], axis=0)
    pool_coords = np.concatenate([empirical_coords, mirrored_coords], axis=0)
    seeds = farthest_seeds(pool_states, pool_coords, args.fixed_seeds)
    repeated_states = np.concatenate([seeds.copy() for _ in evidence_grid], axis=0)
    repeated_evidence = np.repeat(evidence_grid, len(seeds))
    candidates, speeds = optimize_conditioned_points(
        model,
        repeated_states,
        repeated_evidence,
        n_null,
        device,
        args.fixed_opt_steps,
        args.fixed_learning_rate,
        args.fixed_patience,
    )
    nn = NearestNeighbors(n_neighbors=2).fit(pool_states)
    empirical_neighbor = nn.kneighbors(pool_states[: min(5000, len(pool_states))])[0][:, 1]
    neighbor_scale = float(np.median(empirical_neighbor[empirical_neighbor > 1e-12]))
    nearest = NearestNeighbors(n_neighbors=1).fit(pool_states)

    kept_states: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for evidence_index, evidence_value in enumerate(evidence_grid):
        start = evidence_index * len(seeds)
        stop = start + len(seeds)
        local_candidates = candidates[start:stop]
        local_speeds = speeds[start:stop]
        kept = deduplicate(local_candidates, local_speeds, args.fixed_dedup_eps)[
            : args.fixed_max_per_evidence
        ]
        for local_rank, candidate_index in enumerate(kept):
            point = local_candidates[candidate_index]
            coord = project_states(point[None, :], axes, means, scales)[0]
            tensor = torch.tensor(point, dtype=torch.float32, device=device)
            with torch.no_grad():
                report_prob = float(torch.sigmoid(model.loc_head(tensor)).item())
                hazard_prob = float(torch.sigmoid(model.haz_head(tensor)).item())
            distance = float(nearest.kneighbors(point[None, :])[0][0, 0])
            row = {
                "evidence": float(evidence_value),
                "point_rank": local_rank,
                "source_seed": int(candidate_index),
                "speed": float(local_speeds[candidate_index]),
                "point_type": point_type(
                    float(local_speeds[candidate_index]), args.fixed_tol, args.slow_tol
                ),
                "report_prob": report_prob,
                "hazard_prob": hazard_prob,
                "coord_report": coord[0],
                "coord_hazard": coord[1],
                "coord_residual": coord[2],
                "distance_to_real": distance,
                "distance_to_real_over_neighbor": distance / neighbor_scale,
                "stability": "not_computed",
                "spectral_radius": np.nan,
                "n_unstable_eigenvalues": np.nan,
                "n_near_unit_eigenvalues": np.nan,
            }
            rows.append(row)
            kept_states.append(point)
    frame = pd.DataFrame(rows)
    state_array = np.stack(kept_states)
    frame = connect_branches(frame, state_array)

    for evidence_value in evidence_grid:
        subset = frame.index[frame["evidence"] == evidence_value].to_numpy()
        subset = subset[np.argsort(frame.loc[subset, "speed"].to_numpy())]
        for row_index in subset[: args.fixed_jacobians_per_evidence]:
            map_inputs = map_definitions(n_null)["zero_cycle"].clone()
            map_inputs[-1, 0] = float(evidence_value)
            eigenvalues = jacobian_eigenvalues(
                model, state_array[row_index], map_inputs.to(device), device
            )
            stability = classify_stability(eigenvalues, args.eig_tol)
            for key, value in stability.items():
                frame.loc[row_index, key] = value
    return frame, state_array


@torch.inference_mode()
def zero_cycle_basins(
    model,
    prefix_states: np.ndarray,
    prefix_coords: np.ndarray,
    trials,
    fixed: pd.DataFrame,
    fixed_states: np.ndarray,
    axes: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    n_null: int,
    device: torch.device,
    steps: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zero = fixed[np.isclose(fixed["evidence"], 0.0)]
    stable = zero[zero["stability"].str.startswith("stable")]
    if len(stable) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    low_row = int(stable["hazard_prob"].idxmin())
    high_row = int(stable["hazard_prob"].idxmax())
    target_indices = {"low_hazard": low_row, "high_hazard": high_row}
    target_states = np.stack([fixed_states[low_row], fixed_states[high_row]])

    save_steps = {0, 1, 2, 5, 10, 20, 50, steps}
    hidden = torch.tensor(prefix_states, dtype=torch.float32, device=device)
    relaxation_rows: list[pd.DataFrame] = []
    for step in range(steps + 1):
        if step in save_steps:
            state_np = hidden.cpu().numpy()
            coord = project_states(state_np, axes, means, scales)
            relaxation_rows.append(
                pd.DataFrame(
                    {
                        "trial": np.arange(len(prefix_states)),
                        "cycle": step,
                        "coord_report": coord[:, 0],
                        "coord_hazard": coord[:, 1],
                        "coord_residual": coord[:, 2],
                        "report_prob": torch.sigmoid(model.loc_head(hidden).squeeze(-1)).cpu().numpy(),
                        "hazard_prob": torch.sigmoid(model.haz_head(hidden).squeeze(-1)).cpu().numpy(),
                    }
                )
            )
        if step < steps:
            hidden = batch_cycle(
                model, hidden, torch.zeros(len(hidden), device=device), n_null
            )
    terminal = hidden.cpu().numpy()
    next_hidden = batch_cycle(
        model, hidden, torch.zeros(len(hidden), device=device), n_null
    )
    terminal_speed = torch.linalg.norm(next_hidden - hidden, dim=1).cpu().numpy()
    distances = np.linalg.norm(terminal[:, None, :] - target_states[None, :, :], axis=2)
    assignment_index = np.argmin(distances, axis=1)
    labels = np.asarray(["low_hazard", "high_hazard"], dtype=object)[assignment_index]
    margin = (distances[:, 0] - distances[:, 1]) / np.maximum(
        distances[:, 0] + distances[:, 1], 1e-12
    )
    assignments = pd.DataFrame(
        {
            "trial": np.arange(len(prefix_states)),
            "true_hazard": [trial.true_hazard for trial in trials],
            "true_report": [trial.true_report for trial in trials],
            "initial_coord_report": prefix_coords[:, 0],
            "initial_coord_hazard": prefix_coords[:, 1],
            "initial_coord_residual": prefix_coords[:, 2],
            "basin": labels,
            "signed_high_basin_margin": margin,
            "distance_to_low_fixed": distances[:, 0],
            "distance_to_high_fixed": distances[:, 1],
            "terminal_speed": terminal_speed,
            "terminal_report_prob": torch.sigmoid(model.loc_head(hidden).squeeze(-1)).cpu().numpy(),
            "terminal_hazard_prob": torch.sigmoid(model.haz_head(hidden).squeeze(-1)).cpu().numpy(),
        }
    )
    basin_classifier = LogisticRegression(
        C=10.0, class_weight="balanced", random_state=0, max_iter=2000
    ).fit(prefix_coords, (labels == "high_hazard").astype(int))
    assignments["separatrix_score"] = basin_classifier.decision_function(prefix_coords)
    assignments["high_basin_probability"] = basin_classifier.predict_proba(prefix_coords)[:, 1]
    relaxation = pd.concat(relaxation_rows, ignore_index=True).merge(
        assignments[["trial", "basin"]], on="trial", how="left"
    )
    summary = (
        assignments.groupby("basin", observed=True)
        .agg(
            n=("trial", "size"),
            fraction_true_high_hazard=("true_hazard", lambda value: np.mean(value >= 0.5)),
            fraction_positive_report=("true_report", lambda value: np.mean(value > 0)),
            initial_report_mean=("initial_coord_report", "mean"),
            initial_hazard_mean=("initial_coord_hazard", "mean"),
            terminal_report_prob=("terminal_report_prob", "mean"),
            terminal_hazard_prob=("terminal_hazard_prob", "mean"),
            terminal_speed=("terminal_speed", "mean"),
        )
        .reset_index()
    )
    summary["fraction"] = summary["n"] / len(assignments)
    summary["target_fixed_row"] = summary["basin"].map(target_indices)
    summary["separatrix_coef_report"] = float(basin_classifier.coef_[0, 0])
    summary["separatrix_coef_hazard"] = float(basin_classifier.coef_[0, 1])
    summary["separatrix_coef_residual"] = float(basin_classifier.coef_[0, 2])
    summary["separatrix_intercept"] = float(basin_classifier.intercept_[0])
    summary["separatrix_training_accuracy"] = float(
        basin_classifier.score(prefix_coords, (labels == "high_hazard").astype(int))
    )
    return assignments, relaxation, summary


def plot_zero_cycle_basins(
    assignments: pd.DataFrame,
    relaxation: pd.DataFrame,
    fixed: pd.DataFrame,
    output_dir: Path,
) -> None:
    if assignments.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"low_hazard": "tab:blue", "high_hazard": "tab:orange"}
    for label, subset in assignments.groupby("basin"):
        axes[0].scatter(
            subset["initial_coord_report"],
            subset["initial_coord_hazard"],
            s=10,
            alpha=0.45,
            color=colors[label],
            label=label.replace("_", " "),
        )
    boundary = assignments[
        np.abs(assignments["separatrix_score"])
        <= assignments["separatrix_score"].abs().quantile(0.08)
    ]
    axes[0].scatter(
        boundary["initial_coord_report"],
        boundary["initial_coord_hazard"],
        s=22,
        facecolor="none",
        edgecolor="black",
        linewidth=0.7,
        label="candidate separatrix",
    )
    zero_stable = fixed[
        np.isclose(fixed["evidence"], 0.0)
        & fixed["stability"].str.startswith("stable")
    ]
    axes[0].scatter(
        zero_stable["coord_report"], zero_stable["coord_hazard"],
        marker="X", s=85, color="black", label="stable slow point"
    )
    axes[0].set(
        xlabel="initial report-readout coordinate",
        ylabel="initial hazard-exclusive coordinate",
        title="100-cycle basin assignment",
    )
    axes[0].legend(frameon=False, fontsize=8)

    mean_paths = (
        relaxation.groupby(["basin", "cycle"], observed=True)
        .agg(coord_report=("coord_report", "mean"), coord_hazard=("coord_hazard", "mean"))
        .reset_index()
    )
    for label, subset in mean_paths.groupby("basin"):
        subset = subset.sort_values("cycle")
        axes[1].plot(
            subset["coord_report"], subset["coord_hazard"], marker="o",
            color=colors[label], label=label.replace("_", " ")
        )
        for _, row in subset.iterrows():
            axes[1].annotate(int(row["cycle"]), (row["coord_report"], row["coord_hazard"]), fontsize=7)
    axes[1].scatter(
        zero_stable["coord_report"], zero_stable["coord_hazard"], marker="X", s=85, color="black"
    )
    axes[1].set(
        xlabel="report-readout coordinate",
        ylabel="hazard-exclusive coordinate",
        title="Mean relaxation paths (labels are cycle count)",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Epoch 10: input-conditioned zero-evidence basins", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "zero_cycle_basins.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def pivot_grid(frame: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot(index="prefix_report_bin", columns="evidence", values=value)
    return pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float), pivot.to_numpy(dtype=float)


def plot_gating(
    psychometric: pd.DataFrame,
    surface: pd.DataFrame,
    transitions: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap = plt.get_cmap("coolwarm")
    quartiles = sorted(psychometric["prefix_hazard_quartile"].unique())
    extreme_quartiles = [quartiles[0], quartiles[-1]]
    for prior_sign, linestyle in [(-1, "--"), (1, "-")]:
        for quartile in extreme_quartiles:
            subset = psychometric[
                (psychometric["prefix_hazard_quartile"] == quartile)
                & (psychometric["prefix_prior_sign"] == prior_sign)
            ]
            if subset.empty:
                continue
            prior_label = "positive prior" if prior_sign > 0 else "negative prior"
            hazard_label = "high hazard" if quartile == extreme_quartiles[-1] else "low hazard"
            color_position = 0.85 if quartile == extreme_quartiles[-1] else 0.15
            axes[0, 0].plot(
                subset["evidence"], subset["report_prob"], marker=".",
                color=cmap(color_position), linestyle=linestyle,
                label=f"{prior_label}, {hazard_label}"
            )
    axes[0, 0].axhline(0.5, color="0.4", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Matched-near-neutral prefix psychometrics", xlabel="Controlled final evidence", ylabel="P(report +)")
    axes[0, 0].legend(frameon=False, ncol=2)

    x, y, z = pivot_grid(surface, "hazard_to_report_probability_gain")
    limit = np.nanmax(np.abs(z))
    image = axes[0, 1].imshow(
        z, origin="lower", aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit,
        extent=[x.min(), x.max(), y.min(), y.max()]
    )
    axes[0, 1].set(title="Local hazard-to-report gain", xlabel="Controlled final evidence", ylabel="Prefix report quantile bin")
    fig.colorbar(image, ax=axes[0, 1], label="d P(report +) / d prefix hazard logit")

    valid = transitions.dropna(subset=["report_transition_evidence"])
    valid = valid.copy()
    valid["signed_transition_evidence"] = (
        valid["report_transition_evidence"] * np.sign(valid["prefix_report_logit"])
    )
    grouped = [
        valid.loc[valid["prefix_hazard_quartile"] == quartile, "signed_transition_evidence"].to_numpy()
        for quartile in sorted(valid["prefix_hazard_quartile"].unique())
    ]
    axes[1, 0].boxplot(grouped, tick_labels=[f"Q{q + 1}" for q in sorted(valid["prefix_hazard_quartile"].unique())], showfliers=False)
    axes[1, 0].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="High hazard erases the signed prior",
        xlabel="Prefix network hazard",
        ylabel="Transition evidence x sign(prefix report)",
    )

    x2, y2, z2 = pivot_grid(surface, "input_evidence_gain")
    image2 = axes[1, 1].imshow(
        z2, origin="lower", aspect="auto", cmap="magma",
        extent=[x2.min(), x2.max(), y2.min(), y2.max()]
    )
    axes[1, 1].set(title="Final-evidence gain", xlabel="Controlled final evidence", ylabel="Prefix report quantile bin")
    fig.colorbar(image2, ax=axes[1, 1], label="d final report logit / d evidence")
    for ax in axes.flat:
        ax.grid(False)
    fig.suptitle("Epoch 10: where hazard state changes report computation", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "hazard_to_report_gating.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_branches(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    norm = plt.Normalize(frame["evidence"].min(), frame["evidence"].max())
    cmap = plt.get_cmap("coolwarm")
    minimum_speed_locus = frame.loc[frame.groupby("evidence")["speed"].idxmin()].sort_values("evidence")
    axes[0].plot(minimum_speed_locus["evidence"], minimum_speed_locus["report_prob"], color="0.2", linewidth=2, label="minimum-speed locus")
    axes[1].plot(minimum_speed_locus["evidence"], minimum_speed_locus["hazard_prob"], color="0.2", linewidth=2, label="minimum-speed locus")
    axes[2].plot(minimum_speed_locus["coord_report"], minimum_speed_locus["coord_hazard"], color="0.2", linewidth=2, label="minimum-speed locus")
    colors = cmap(norm(frame["evidence"]))
    slow = frame["point_type"].isin(["fixed", "slow"])
    stable = frame["stability"].str.startswith("stable") & slow
    saddle = frame["stability"].str.contains("saddle") & slow
    unknown = ~(stable | saddle)
    for ax, x, y in [
        (axes[0], "evidence", "report_prob"),
        (axes[1], "evidence", "hazard_prob"),
        (axes[2], "coord_report", "coord_hazard"),
    ]:
        ax.scatter(frame.loc[unknown & ~slow, x], frame.loc[unknown & ~slow, y], c=colors[unknown & ~slow], s=20, alpha=0.22)
        ax.scatter(frame.loc[slow & ~stable & ~saddle, x], frame.loc[slow & ~stable & ~saddle, y], c=colors[slow & ~stable & ~saddle], marker="o", edgecolor="black", s=44)
        ax.scatter(frame.loc[stable, x], frame.loc[stable, y], c=colors[stable], marker="o", edgecolor="black", s=58)
        ax.scatter(frame.loc[saddle, x], frame.loc[saddle, y], c=colors[saddle], marker="X", edgecolor="black", s=72)
    axes[0].axhline(0.5, color="0.25", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Repeated evidence value", ylabel="P(report +)", title="Input-conditioned report loci")
    axes[1].set(xlabel="Repeated evidence value", ylabel="P(high hazard)", title="Input-conditioned hazard loci")
    axes[2].set(xlabel="Report-readout coordinate", ylabel="Hazard-exclusive coordinate", title="Locus geometry")
    legend = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="black", label="stable (tested)"),
        Line2D([0], [0], marker="X", color="none", markeredgecolor="black", label="saddle/unstable (tested)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.6", markeredgecolor="black", label="slow; stability not tested"),
        Line2D([0], [0], color="0.2", linewidth=2, label="minimum-speed locus"),
    ]
    axes[2].legend(handles=legend, frameon=False, fontsize=8)
    fig.suptitle("Epoch 10 stroboscopic fixed/slow-point loci", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_point_branches.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_atlas(
    trajectory: pd.DataFrame,
    examples: np.ndarray,
    sheet: pd.DataFrame,
    vector_field: pd.DataFrame,
    fixed: pd.DataFrame,
    transitions: pd.DataFrame,
    report_boundary: float,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(examples)))
    for color, trial in zip(colors, examples):
        subset = trajectory[trajectory["trial"] == int(trial)].sort_values("evidence_index")
        ax3d.plot(subset["coord_report"], subset["coord_hazard"], subset["coord_residual"], color=color, alpha=0.8, linewidth=1.2)
        ax3d.scatter(subset["coord_report"].iloc[-1], subset["coord_hazard"].iloc[-1], subset["coord_residual"].iloc[-1], color=color, s=15)
    zero_fixed = fixed[np.isclose(fixed["evidence"], 0.0)]
    ax3d.scatter(zero_fixed["coord_report"], zero_fixed["coord_hazard"], zero_fixed["coord_residual"], marker="X", color="black", s=52, label="zero-evidence slow/fixed")
    ax3d.set(xlabel="report readout (SD)", ylabel="hazard-exclusive (SD)", zlabel="residual (SD)", title="Representative task trajectories")
    ax3d.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(2, 2, 2)
    field = vector_field[np.isclose(vector_field["evidence"], 0.0)]
    scale = np.quantile(np.hypot(field["delta_report"], field["delta_hazard"]), 0.9)
    ax.quiver(field["coord_report"], field["coord_hazard"], field["delta_report"], field["delta_hazard"], field["speed_3d"], cmap="viridis", angles="xy", scale_units="xy", scale=max(scale * 8, 1e-6), width=0.004)
    ax.scatter(zero_fixed["coord_report"], zero_fixed["coord_hazard"], c="black", marker="X", s=45)
    ax.scatter(transitions["transition_coord_report"], transitions["transition_coord_hazard"], c=transitions["transition_hazard_to_report_probability_gain"], cmap="coolwarm", s=9, alpha=0.35)
    ax.axvline(report_boundary, color="0.3", linestyle="--", linewidth=1, label="report boundary")
    ax.set(xlabel="report readout (SD)", ylabel="hazard-exclusive (SD)", title="Zero-evidence flow and empirical transition line")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(2, 2, 3)
    scatter = ax.scatter(sheet["coord_report"], sheet["coord_hazard"], c=sheet["residual_mean"], s=np.clip(sheet["n"], 5, 180), cmap="coolwarm", alpha=0.85)
    fig.colorbar(scatter, ax=ax, label="mean residual coordinate")
    ax.set(xlabel="report readout (SD)", ylabel="hazard-exclusive (SD)", title="Occupied manifold sheet / fold")

    ax = fig.add_subplot(2, 2, 4)
    valid = sheet.dropna(subset=["local_participation_ratio"])
    scatter = ax.scatter(valid["coord_report"], valid["coord_hazard"], c=valid["local_participation_ratio"], s=np.clip(valid["n"], 5, 180), cmap="magma", alpha=0.85)
    fig.colorbar(scatter, ax=ax, label="local participation ratio")
    ax.set(xlabel="report readout (SD)", ylabel="hazard-exclusive (SD)", title="Locally expanded / transition regions")
    fig.suptitle("Epoch 10 task-aligned manifold atlas", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "task_aligned_manifold_atlas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_points_of_interest(
    output_dir: Path,
    fixed: pd.DataFrame,
    transitions: pd.DataFrame,
    surface: pd.DataFrame,
    local: pd.DataFrame,
    basins: pd.DataFrame,
) -> pd.DataFrame:
    """Collect the point, line, and region landmarks used by the atlas."""
    rows: list[dict[str, Any]] = []
    for _, point in fixed[fixed["point_type"].isin(["fixed", "slow"])].iterrows():
        rows.append(
            {
                "interest_type": "conditioned_fixed_or_slow_point",
                "coord_report": point["coord_report"],
                "coord_hazard": point["coord_hazard"],
                "coord_residual": point["coord_residual"],
                "control_value": point["evidence"],
                "metric": point["speed"],
                "description": (
                    f"{point['point_type']} point of the repeated-evidence cycle; "
                    f"stability={point['stability']}"
                ),
            }
        )

    transition_work = transitions.copy()
    transition_work["report_bin"] = pd.qcut(
        transition_work["prefix_report_logit"], 6, labels=False, duplicates="drop"
    )
    transition_work["hazard_bin"] = pd.qcut(
        transition_work["prefix_hazard_logit"], 4, labels=False, duplicates="drop"
    )
    transition_centers = (
        transition_work.groupby(["report_bin", "hazard_bin"], observed=True)
        .agg(
            coord_report=("transition_coord_report", "mean"),
            coord_hazard=("transition_coord_hazard", "mean"),
            coord_residual=("transition_coord_residual", "mean"),
            control_value=("report_transition_evidence", "median"),
            metric=("transition_hazard_to_report_probability_gain", "mean"),
            n=("trial", "size"),
        )
        .reset_index()
    )
    for _, point in transition_centers.iterrows():
        rows.append(
            {
                "interest_type": "empirical_report_transition_line",
                "coord_report": point["coord_report"],
                "coord_hazard": point["coord_hazard"],
                "coord_residual": point["coord_residual"],
                "control_value": point["control_value"],
                "metric": point["metric"],
                "description": (
                    f"centroid of report-logit-zero transitions (n={int(point['n'])})"
                ),
            }
        )

    ridge_indices = surface.groupby("prefix_report_bin")[
        "hazard_to_report_probability_gain"
    ].apply(lambda values: values.abs().idxmax())
    for _, point in surface.loc[ridge_indices.to_numpy()].iterrows():
        rows.append(
            {
                "interest_type": "maximum_hazard_leverage_ridge",
                "coord_report": point["post_coord_report"],
                "coord_hazard": point["post_coord_hazard"],
                "coord_residual": point["post_coord_residual"],
                "control_value": point["evidence"],
                "metric": point["hazard_to_report_probability_gain"],
                "description": "largest mean local hazard-to-report gain within a prefix-report bin",
            }
        )

    cutoff = local["local_participation_ratio"].quantile(0.98)
    for _, point in local[local["local_participation_ratio"] >= cutoff].iterrows():
        rows.append(
            {
                "interest_type": "local_dimension_expansion",
                "coord_report": point["coord_report"],
                "coord_hazard": point["coord_hazard"],
                "coord_residual": point["coord_residual"],
                "control_value": point["evidence_index"],
                "metric": point["local_participation_ratio"],
                "description": "top two percent of local participation-ratio estimates",
            }
        )
    if not basins.empty:
        boundary = basins.loc[
            basins["separatrix_score"].abs().nsmallest(
                max(12, int(0.02 * len(basins)))
            ).index
        ]
        for _, point in boundary.iterrows():
            rows.append(
                {
                    "interest_type": "zero_cycle_candidate_separatrix",
                    "coord_report": point["initial_coord_report"],
                    "coord_hazard": point["initial_coord_hazard"],
                    "coord_residual": point["initial_coord_residual"],
                    "control_value": 0.0,
                    "metric": point["separatrix_score"],
                    "description": "empirical prefix state near the fitted boundary between zero-cycle endpoint basins",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "points_of_interest.csv", index=False)
    return result


def write_summary(
    output_dir: Path,
    axis_metadata: dict[str, Any],
    transitions: pd.DataFrame,
    surface: pd.DataFrame,
    fixed: pd.DataFrame,
    local: pd.DataFrame,
    probe_long: pd.DataFrame,
    basins: pd.DataFrame,
) -> dict[str, Any]:
    valid = transitions.dropna(subset=["report_transition_evidence"])
    quartile_thresholds = (
        valid.groupby("prefix_hazard_quartile")["report_transition_evidence"]
        .median()
        .to_dict()
    )
    maximum_gain_row = surface.iloc[
        int(np.nanargmax(np.abs(surface["hazard_to_report_probability_gain"])))
    ]
    tested = fixed[fixed["stability"] != "not_computed"]
    valid = valid.copy()
    valid["signed_transition_evidence"] = (
        valid["report_transition_evidence"] * np.sign(valid["prefix_report_logit"])
    )
    signed_quartile_thresholds = (
        valid.groupby("prefix_hazard_quartile")["signed_transition_evidence"]
        .median()
        .to_dict()
    )
    valid["prior_weakening_gain"] = -np.sign(valid["prefix_report_logit"]) * valid[
        "transition_hazard_to_report_probability_gain"
    ]
    zero_probe = probe_long[np.isclose(probe_long["evidence"], 0.0)].copy()
    zero_probe["prior_weakening_gain"] = -np.sign(zero_probe["prefix_report_logit"]) * zero_probe[
        "hazard_to_report_probability_gain"
    ]
    standardized_hazard = (
        valid["prefix_hazard_logit"] - valid["prefix_hazard_logit"].mean()
    ) / valid["prefix_hazard_logit"].std(ddof=1)
    standardized_prior_magnitude = (
        valid["prefix_report_logit"].abs() - valid["prefix_report_logit"].abs().mean()
    ) / valid["prefix_report_logit"].abs().std(ddof=1)
    design = np.column_stack(
        [
            np.ones(len(valid)),
            standardized_hazard,
            standardized_prior_magnitude,
            standardized_hazard * standardized_prior_magnitude,
        ]
    )
    coefficients = np.linalg.lstsq(
        design, valid["signed_transition_evidence"].to_numpy(), rcond=None
    )[0]
    slow = fixed[fixed["point_type"].isin(["fixed", "slow"])]
    summary = {
        "task_aligned_variance_fraction": axis_metadata["task_aligned_variance_fraction"],
        "report_hazard_weight_cosine": axis_metadata["report_hazard_weight_cosine"],
        "fraction_probe_trials_with_report_crossing": float(len(valid) / len(transitions)),
        "median_report_transition_evidence_by_prefix_hazard_quartile": {
            str(int(key)): float(value) for key, value in quartile_thresholds.items()
        },
        "median_signed_transition_evidence_by_prefix_hazard_quartile": {
            str(int(key)): float(value)
            for key, value in signed_quartile_thresholds.items()
        },
        "signed_transition_spearman_with_prefix_hazard": float(
            valid[["signed_transition_evidence", "prefix_hazard_logit"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "signed_transition_regression": {
            "standardized_hazard_coefficient": float(coefficients[1]),
            "standardized_abs_prior_coefficient": float(coefficients[2]),
            "hazard_x_abs_prior_coefficient": float(coefficients[3]),
        },
        "strongest_mean_hazard_to_report_probability_gain": {
            "gain": float(maximum_gain_row["hazard_to_report_probability_gain"]),
            "controlled_final_evidence": float(maximum_gain_row["evidence"]),
            "prefix_report_logit": float(maximum_gain_row["prefix_report_logit"]),
            "prefix_report_bin": int(maximum_gain_row["prefix_report_bin"]),
        },
        "median_abs_transition_hazard_to_report_probability_gain": float(
            np.nanmedian(np.abs(valid["transition_hazard_to_report_probability_gain"]))
        ),
        "fraction_transition_hazard_gain_weakens_prior": float(
            np.mean(valid["prior_weakening_gain"] > 0)
        ),
        "fraction_zero_evidence_hazard_gain_weakens_prior": float(
            np.mean(zero_probe["prior_weakening_gain"] > 0)
        ),
        "median_zero_evidence_prior_weakening_gain": float(
            zero_probe["prior_weakening_gain"].median()
        ),
        "median_transition_input_evidence_gain": float(
            np.nanmedian(valid["transition_input_evidence_gain"])
        ),
        "local_participation_ratio_median": float(local["local_participation_ratio"].median()),
        "local_participation_ratio_95th_percentile": float(local["local_participation_ratio"].quantile(0.95)),
        "conditioned_points": int(len(fixed)),
        "fixed_points": int((fixed["point_type"] == "fixed").sum()),
        "slow_points": int(fixed["point_type"].isin(["fixed", "slow"]).sum()),
        "tested_stable_points": int(tested["stability"].str.startswith("stable").sum()),
        "tested_saddle_or_unstable_points": int(tested["stability"].str.contains("saddle").sum()),
        "median_conditioned_point_distance_over_neighbor": float(fixed["distance_to_real_over_neighbor"].median()),
        "median_slow_point_distance_over_neighbor": float(
            slow["distance_to_real_over_neighbor"].median()
        ),
    }
    if not basins.empty:
        basin_counts = basins["basin"].value_counts(normalize=True).to_dict()
        summary["zero_cycle_basin_fraction"] = {
            str(key): float(value) for key, value in basin_counts.items()
        }
        summary["zero_cycle_basin_assignment_spearman_with_initial_hazard"] = float(
            basins[["signed_high_basin_margin", "initial_coord_hazard"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        basin_features = basins[
            [
                "initial_coord_report",
                "initial_coord_hazard",
                "initial_coord_residual",
            ]
        ]
        basin_target = (basins["basin"] == "high_hazard").astype(int)
        basin_classifier = LogisticRegression(
            C=10.0, class_weight="balanced", random_state=0, max_iter=2000
        ).fit(basin_features, basin_target)
        summary["zero_cycle_separatrix"] = {
            "coef_report": float(basin_classifier.coef_[0, 0]),
            "coef_hazard": float(basin_classifier.coef_[0, 1]),
            "coef_residual": float(basin_classifier.coef_[0, 2]),
            "intercept": float(basin_classifier.intercept_[0]),
            "training_accuracy": float(
                basin_classifier.score(basin_features, basin_target)
            ),
        }
    save_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_csvs = 1
        args.max_trials = 80
        args.probe_trials = 32
        args.probe_evidence_count = 11
        args.local_structure_points = 60
        args.fixed_seeds = 12
        args.fixed_opt_steps = 20
        args.fixed_patience = 0
        args.fixed_jacobians_per_evidence = 0
    set_seeds(args.random_seed)
    rng = np.random.default_rng(args.random_seed)
    cfg = load_config(args.config)
    specs = model_specs(cfg, {args.model})
    if len(specs) != 1:
        raise RuntimeError(f"Expected exactly one model, got {len(specs)}")
    spec = specs[0]
    checkpoint_path = resolve_checkpoints(spec.seed_dir, args.checkpoint)[0]
    checkpoint = checkpoint_label(checkpoint_path)
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    trials, source_paths = load_trials(
        cfg, split=args.split, max_csvs=args.max_csvs, max_trials=args.max_trials
    )
    model, hp = load_model(spec, checkpoint_path, device)
    n_null = int(hp.get("n_null_timesteps", 4))
    inputs = build_input_tensor(trials, n_null)
    mirrored_inputs = build_input_tensor(trials, n_null, mirrored=True)
    evidence_indices = evidence_timestep_indices(len(trials[0].evidence), n_null)
    print(f"Device: {device}; model: {spec.key}; checkpoint: {checkpoint}; trials: {len(trials)}")
    hidden, report_logits, hazard_logits = collect_evidence_states(
        model, inputs, evidence_indices, args.batch_size, device
    )
    mirrored_hidden, _, _ = collect_evidence_states(
        model, mirrored_inputs, evidence_indices, args.batch_size, device
    )
    normative = compute_normative_features(trials, Path(cfg["normative_model"]))
    flat_states = hidden.reshape(-1, hidden.shape[-1])
    flat_mirrored = mirrored_hidden.reshape(-1, mirrored_hidden.shape[-1])
    flat_last_evidence = np.concatenate([trial.evidence for trial in trials])
    axes, means, scales, axis_metadata = model_axes(
        model, flat_states, flat_last_evidence
    )
    np.savez_compressed(
        output_dir / "task_aligned_axes.npz",
        axes=axes,
        means=means,
        scales=scales,
        report_weight=model.loc_head.weight.detach().cpu().numpy(),
        hazard_weight=model.haz_head.weight.detach().cpu().numpy(),
    )
    save_json(output_dir / "task_aligned_axes.json", axis_metadata)
    coords = project_states(flat_states, axes, means, scales).reshape(*hidden.shape[:2], 3)
    mirrored_coords = project_states(flat_mirrored, axes, means, scales)
    trajectory = trajectory_frame(
        trials, hidden, report_logits, hazard_logits, normative, coords
    )
    trajectory.to_csv(output_dir / "trajectory_states.csv.gz", index=False, compression="gzip")
    examples = choose_example_trials(trials, min(48, len(trials)), rng)
    example_frame = trajectory[trajectory["trial"].isin(examples)].copy()
    example_frame.to_csv(output_dir / "trajectory_examples.csv", index=False)

    print("Estimating local manifold structure")
    local, sheet = local_structure(
        flat_states,
        coords.reshape(-1, 3),
        trajectory,
        args.local_structure_points,
        args.local_neighbors,
        rng,
    )
    local.to_csv(output_dir / "local_structure.csv", index=False)
    sheet.to_csv(output_dir / "manifold_sheet.csv", index=False)

    prefix_index = hidden.shape[1] - 2
    prefix_all = hidden[:, prefix_index]
    prefix_report_all = report_logits[:, prefix_index]
    prefix_hazard_all = hazard_logits[:, prefix_index]
    prefix_coords_all = coords[:, prefix_index]
    probe_indices = stratified_indices(
        prefix_report_all, prefix_hazard_all, min(args.probe_trials, len(trials)), rng
    )
    evidence_grid = np.linspace(
        args.probe_evidence_min,
        args.probe_evidence_max,
        args.probe_evidence_count,
    )
    print(f"Running controlled final-evidence probes on {len(probe_indices)} prefix states")
    probe = probe_grid(
        model,
        prefix_all[probe_indices],
        evidence_grid,
        axes,
        means,
        scales,
        n_null,
        device,
    )
    transitions, probe_long, psychometric, surface = summarize_probe(
        trials,
        probe_indices,
        prefix_coords_all[probe_indices],
        prefix_report_all[probe_indices],
        prefix_hazard_all[probe_indices],
        normative,
        evidence_grid,
        probe,
    )
    transitions.to_csv(output_dir / "report_transition_line.csv", index=False)
    probe_long.to_csv(output_dir / "controlled_probe_points.csv.gz", index=False, compression="gzip")
    psychometric.to_csv(output_dir / "psychometric_by_prefix_hazard.csv", index=False)
    surface.to_csv(output_dir / "hazard_to_report_surface.csv", index=False)
    np.savez_compressed(
        output_dir / "controlled_probe_arrays.npz",
        trial_indices=probe_indices,
        evidence_grid=evidence_grid,
        **probe,
    )

    print("Computing controlled vector fields")
    vector_field = controlled_vector_field(
        model,
        flat_states,
        coords.reshape(-1, 3),
        [-1.0, 0.0, 1.0],
        axes,
        means,
        scales,
        n_null,
        device,
        rng,
    )
    vector_field.to_csv(output_dir / "controlled_vector_field.csv", index=False)

    fixed_evidence = parse_float_grid(args.fixed_evidence_grid)
    print(f"Searching input-conditioned fixed/slow points at {len(fixed_evidence)} evidence values")
    fixed, fixed_states = conditioned_fixed_points(
        model,
        flat_states,
        coords.reshape(-1, 3),
        flat_mirrored,
        mirrored_coords,
        fixed_evidence,
        axes,
        means,
        scales,
        n_null,
        device,
        args,
    )
    fixed.to_csv(output_dir / "conditioned_fixed_points.csv", index=False)
    np.savez_compressed(
        output_dir / "conditioned_fixed_point_states.npz", states=fixed_states
    )

    print("Mapping finite-horizon zero-evidence basins and separatrix candidates")
    basins, relaxation, basin_summary = zero_cycle_basins(
        model,
        prefix_all,
        prefix_coords_all,
        trials,
        fixed,
        fixed_states,
        axes,
        means,
        scales,
        n_null,
        device,
    )
    basins.to_csv(output_dir / "zero_cycle_basin_assignments.csv", index=False)
    relaxation.to_csv(output_dir / "zero_cycle_relaxation.csv.gz", index=False, compression="gzip")
    basin_summary.to_csv(output_dir / "zero_cycle_basin_summary.csv", index=False)

    write_points_of_interest(output_dir, fixed, transitions, surface, local, basins)
    plot_gating(psychometric, surface, transitions, output_dir)
    plot_fixed_branches(fixed, output_dir)
    plot_zero_cycle_basins(basins, relaxation, fixed, output_dir)
    plot_atlas(
        trajectory,
        examples[:16],
        sheet,
        vector_field,
        fixed,
        transitions,
        axis_metadata["report_decision_boundary_z"],
        output_dir,
    )
    summary = write_summary(
        output_dir,
        axis_metadata,
        transitions,
        surface,
        fixed,
        local,
        probe_long,
        basins,
    )
    write_run_config(
        output_dir / "run_config.json",
        {
            **vars(args),
            "config": str(args.config.expanduser().resolve()),
            "output_dir": str(output_dir),
            "model_root": str(spec.model_root),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_label": checkpoint,
            "device_resolved": str(device),
            "source_csvs": [str(path) for path in source_paths],
            "elapsed_seconds": time.time() - start_time,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

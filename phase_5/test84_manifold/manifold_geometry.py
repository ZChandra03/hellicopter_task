#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes, subspace_angles
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from manifold_utils import (
    BASE_DIR,
    DEFAULT_CONFIG,
    build_input_tensor,
    checkpoint_label,
    choose_device,
    collect_evidence_states,
    compute_normative_features,
    covariance_eigendecomposition,
    evidence_timestep_indices,
    linear_cka,
    load_config,
    load_model,
    load_trials,
    model_specs,
    parse_model_filter,
    participation_ratio,
    resolve_checkpoints,
    set_seeds,
    sigmoid,
    twonn_dimension,
    variance_dimension,
    write_run_config,
)


DEFAULT_OUTPUT_DIR = BASE_DIR / "geometry_outputs"
DECODER_TARGETS = [
    "bayes_state_belief",
    "bayes_hazard_mean",
    "bayes_hazard_sd",
    "bayes_state_entropy",
    "last_evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track intrinsic dimension, subspace motion, mirror symmetry, and "
            "Bayesian-variable decodability through seed-0 training."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all", help="all or comma-separated model keys")
    parser.add_argument(
        "--checkpoints",
        default="all",
        help="all or comma list such as init,1,5,7,10,final",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-csvs", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-id-points", type=int, default=5000)
    parser.add_argument("--max-cv-trials", type=int, default=500)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-comparison-points", type=int, default=5000)
    parser.add_argument("--max-atlas-points", type=int, default=2500)
    parser.add_argument(
        "--atlas-checkpoints",
        default="1,5,7,10,final",
        help="Comma-separated checkpoints to show in the 3-D atlas",
    )
    parser.add_argument("--hazard-step", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def checkpoint_position(label: str) -> float:
    if label == "init":
        return 0.0
    if label.startswith("ep"):
        return float(int(label[2:]))
    if label == "best":
        return 10.5
    if label == "final":
        return 11.0
    return float("nan")


def choose_fixed_indices(total: int, maximum: int, seed: int) -> np.ndarray:
    if total <= maximum:
        return np.arange(total, dtype=int)
    return np.sort(np.random.default_rng(seed).choice(total, size=maximum, replace=False))


def cv_manifold_and_decoding(
    states: np.ndarray,
    targets: dict[str, np.ndarray],
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return [], []
    n_splits = min(n_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    reconstruction_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(states, groups=groups)):
        x_train = states[train_index]
        x_test = states[test_index]
        max_components = min(10, x_train.shape[1], len(x_train) - 1)
        pca = PCA(n_components=max_components, svd_solver="randomized", random_state=seed + fold)
        z_train = pca.fit_transform(x_train)
        z_test = pca.transform(x_test)
        centered_test = x_test - pca.mean_
        total_ss = float(np.sum(centered_test * centered_test))

        for n_components in (2, 3):
            if n_components > max_components:
                continue
            reconstruction = z_test[:, :n_components] @ pca.components_[:n_components] + pca.mean_
            residual_ss = float(np.sum((x_test - reconstruction) ** 2))
            reconstruction_rows.append(
                {
                    "fold": fold,
                    "n_components": n_components,
                    "heldout_reconstruction_r2": 1.0 - residual_ss / total_ss,
                }
            )

        for target_name, target_values in targets.items():
            y_train = target_values[train_index]
            y_test = target_values[test_index]
            for n_components in (2, 3, min(10, max_components)):
                if n_components > max_components:
                    continue
                train_features = z_train[:, :n_components]
                test_features = z_test[:, :n_components]
                scale = np.std(train_features, axis=0, ddof=1)
                scale[scale < 1e-12] = 1.0
                train_features = (train_features - np.mean(train_features, axis=0)) / scale
                test_features = (test_features - np.mean(z_train[:, :n_components], axis=0)) / scale
                decoder = Ridge(alpha=1.0).fit(train_features, y_train)
                prediction = decoder.predict(test_features)
                decoder_rows.append(
                    {
                        "fold": fold,
                        "target": target_name,
                        "n_components": n_components,
                        "heldout_r2": float(r2_score(y_test, prediction)),
                    }
                )
    return reconstruction_rows, decoder_rows


def behavior_rows(
    model_key: str,
    checkpoint: str,
    order: float,
    report_logits: np.ndarray,
    hazard_logits: np.ndarray,
    trials,
    normative: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    report_prediction = np.where(report_logits[:, -1] >= 0.0, 1, -1)
    hazard_prediction = np.where(hazard_logits[:, -1] >= 0.0, 1, -1)
    true_report = np.asarray([trial.true_report for trial in trials])
    true_predict = np.asarray([trial.true_predict for trial in trials])
    final_evidence = np.asarray([trial.evidence[-1] for trial in trials])
    last_prediction = np.where(final_evidence >= 0.0, 1, -1)
    masks = {
        "all": np.ones(len(trials), dtype=bool),
        "ambiguous_final_abs_le_0p2": np.abs(final_evidence) <= 0.2,
        "last_evidence_conflict": last_prediction != true_report,
    }
    rows: list[dict[str, Any]] = []
    for subset, mask in masks.items():
        if not np.any(mask):
            continue
        rows.append(
            {
                "model": model_key,
                "checkpoint": checkpoint,
                "checkpoint_order": order,
                "subset": subset,
                "n_trials": int(np.sum(mask)),
                "report_accuracy": float(np.mean(report_prediction[mask] == true_report[mask])),
                "predict_accuracy": float(np.mean(hazard_prediction[mask] == true_predict[mask])),
                "report_matches_last_evidence": float(
                    np.mean(report_prediction[mask] == last_prediction[mask])
                ),
                "report_matches_bayes": float(
                    np.mean(report_prediction[mask] == normative["bayes_report"][mask])
                ),
                "hazard_matches_bayes": float(
                    np.mean(hazard_prediction[mask] == normative["bayes_predict"][mask])
                ),
            }
        )
    return rows


def mirror_symmetry_metrics(
    states: np.ndarray,
    mirror_states: np.ndarray,
    report_logits: np.ndarray,
    mirror_report_logits: np.ndarray,
    hazard_logits: np.ndarray,
    mirror_hazard_logits: np.ndarray,
    groups: np.ndarray,
    max_points: int,
    seed: int,
) -> dict[str, float]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    split = max(1, int(round(0.7 * len(unique_groups))))
    train_groups = set(unique_groups[:split].tolist())
    train_mask = np.asarray([group in train_groups for group in groups])
    test_mask = ~train_mask
    if not np.any(test_mask):
        test_mask = train_mask.copy()

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    if len(train_indices) > max_points:
        train_indices = rng.choice(train_indices, size=max_points, replace=False)
    if len(test_indices) > max_points:
        test_indices = rng.choice(test_indices, size=max_points, replace=False)

    x_mean = np.mean(states[train_indices], axis=0)
    y_mean = np.mean(mirror_states[train_indices], axis=0)
    x_train = states[train_indices] - x_mean
    y_train = mirror_states[train_indices] - y_mean
    rotation, _ = orthogonal_procrustes(x_train, y_train)
    prediction = (states[test_indices] - x_mean) @ rotation + y_mean
    residual_rms = float(np.sqrt(np.mean((prediction - mirror_states[test_indices]) ** 2)))
    target_rms = float(
        np.sqrt(np.mean((mirror_states[test_indices] - y_mean) ** 2))
    )
    return {
        "mirror_report_antisymmetry_mae": float(
            np.mean(np.abs(report_logits + mirror_report_logits))
        ),
        "mirror_hazard_symmetry_mae": float(
            np.mean(np.abs(hazard_logits - mirror_hazard_logits))
        ),
        "mirror_report_probability_complement_mae": float(
            np.mean(np.abs(sigmoid(report_logits) + sigmoid(mirror_report_logits) - 1.0))
        ),
        "mirror_hazard_probability_mae": float(
            np.mean(np.abs(sigmoid(hazard_logits) - sigmoid(mirror_hazard_logits)))
        ),
        "mirror_procrustes_nrmse": residual_rms / max(target_rms, 1e-12),
    }


def procrustes_shape_disparity(
    x: np.ndarray,
    x_basis: np.ndarray,
    y: np.ndarray,
    y_basis: np.ndarray,
    n_components: int = 10,
) -> float:
    n_components = min(n_components, x_basis.shape[1], y_basis.shape[1])
    x_score = (x - np.mean(x, axis=0)) @ x_basis[:, :n_components]
    y_score = (y - np.mean(y, axis=0)) @ y_basis[:, :n_components]
    x_score /= max(np.linalg.norm(x_score, "fro"), 1e-12)
    y_score /= max(np.linalg.norm(y_score, "fro"), 1e-12)
    rotation, _ = orthogonal_procrustes(x_score, y_score)
    return float(np.linalg.norm(x_score @ rotation - y_score, "fro"))


def alignment_rows_for_model(cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cache:
        return []
    early_index = next(
        (index for index, item in enumerate(cache) if item["checkpoint"].startswith("ep")),
        0,
    )
    final_index = next(
        (index for index, item in enumerate(cache) if item["checkpoint"] == "final"),
        len(cache) - 1,
    )
    rows: list[dict[str, Any]] = []
    for current_index, current in enumerate(cache):
        references = {
            "early": cache[early_index],
            "final": cache[final_index],
        }
        if current_index > 0:
            references["previous"] = cache[current_index - 1]
        for reference_name, reference in references.items():
            angles_2 = np.degrees(
                subspace_angles(current["basis"][:, :2], reference["basis"][:, :2])
            )
            angles_3 = np.degrees(
                subspace_angles(current["basis"][:, :3], reference["basis"][:, :3])
            )
            current_centered = current["states"] - current["mean"]
            early_basis = cache[early_index]["basis"][:, :2]
            total_variance = float(np.sum(current_centered * current_centered))
            projected = current_centered @ early_basis
            variance_outside_early = 1.0 - float(np.sum(projected * projected)) / max(
                total_variance, 1e-12
            )
            rows.append(
                {
                    "model": current["model"],
                    "checkpoint": current["checkpoint"],
                    "checkpoint_order": current["checkpoint_order"],
                    "reference": reference_name,
                    "reference_checkpoint": reference["checkpoint"],
                    "linear_cka": linear_cka(current["states"], reference["states"]),
                    "procrustes_disparity_top10": procrustes_shape_disparity(
                        current["states"],
                        current["basis"],
                        reference["states"],
                        reference["basis"],
                    ),
                    "top2_angle_mean_deg": float(np.mean(angles_2)),
                    "top2_angle_max_deg": float(np.max(angles_2)),
                    "top3_angle_mean_deg": float(np.mean(angles_3)),
                    "top3_angle_max_deg": float(np.max(angles_3)),
                    "variance_outside_early_top2": variance_outside_early,
                    "pc3_fraction_inside_early_top2": float(
                        np.sum((early_basis.T @ current["basis"][:, 2]) ** 2)
                    ),
                }
            )
    return rows


def aggregate_fold_rows(frame: pd.DataFrame, value_column: str, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return (
        frame.groupby(group_columns)[value_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": value_column, "std": f"{value_column}_std"})
    )


def plot_model_summary(
    model_key: str,
    dimension: pd.DataFrame,
    eigenspectrum: pd.DataFrame,
    reconstruction: pd.DataFrame,
    behavior: pd.DataFrame,
    output_dir: Path,
) -> None:
    dim = dimension[
        (dimension["model"] == model_key) & (dimension["scope"] == "all_evidence")
    ].sort_values("checkpoint_order")
    eig = eigenspectrum[
        (eigenspectrum["model"] == model_key)
        & (eigenspectrum["scope"] == "all_evidence")
    ]
    rec = reconstruction[
        (reconstruction["model"] == model_key)
        & (reconstruction["scope"] == "all_evidence")
    ]
    beh = behavior[
        (behavior["model"] == model_key) & (behavior["subset"] == "all")
    ].sort_values("checkpoint_order")
    if dim.empty:
        return
    labels = dim["checkpoint"].tolist()
    x = np.arange(len(labels))
    position = dict(zip(labels, x))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for pc in range(1, 6):
        subset = eig[eig["pc"] == pc].sort_values("checkpoint_order")
        ax.plot([position[v] for v in subset["checkpoint"]], subset["explained_variance_ratio"], marker="o", label=f"PC{pc}")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Variance spectrum")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    ax.plot(x, dim["participation_ratio"], marker="o", label="participation ratio")
    ax.plot(x, dim["twonn_dimension"], marker="o", label="TWO-NN")
    ax.plot(x, dim["dimension_95pct"], marker="o", label="95% PCA dimension")
    ax.set_ylabel("Dimension estimate")
    ax.set_title("Global and local dimensionality")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for n_components in (2, 3):
        subset = rec[rec["n_components"] == n_components].sort_values("checkpoint_order")
        ax.plot([position[v] for v in subset["checkpoint"]], subset["heldout_reconstruction_r2"], marker="o", label=f"{n_components} PCs")
    ax.set_ylabel("Held-out reconstruction $R^2$")
    ax.set_title("Does a third dimension generalize?")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(x, beh["report_accuracy"], marker="o", label="report accuracy")
    ax.plot(x, beh["predict_accuracy"], marker="o", label="predict accuracy")
    ax.plot(x, beh["report_matches_last_evidence"], marker="o", label="matches last evidence")
    ax.plot(x, beh["report_matches_bayes"], marker="o", label="matches Bayes")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Fraction")
    ax.set_title("Behavior on the same trials")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{model_key}: manifold and behavior through training", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / f"manifold_summary_{model_key}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_gains(model_key: str, decoder: pd.DataFrame, output_dir: Path) -> None:
    subset = decoder[
        (decoder["model"] == model_key) & (decoder["scope"] == "all_evidence")
    ]
    if subset.empty:
        return
    pivot = subset.pivot_table(
        index=["checkpoint", "checkpoint_order", "target"],
        columns="n_components",
        values="heldout_r2",
    ).reset_index()
    if 2 not in pivot or 3 not in pivot:
        return
    pivot["pc3_gain"] = pivot[3] - pivot[2]
    labels = (
        pivot[["checkpoint", "checkpoint_order"]]
        .drop_duplicates()
        .sort_values("checkpoint_order")["checkpoint"]
        .tolist()
    )
    x = np.arange(len(labels))
    position = dict(zip(labels, x))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for target, target_df in pivot.groupby("target"):
        target_df = target_df.sort_values("checkpoint_order")
        ax.plot(
            [position[value] for value in target_df["checkpoint"]],
            target_df["pc3_gain"],
            marker="o",
            label=target,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Held-out $R^2$(3 PCs) - $R^2$(2 PCs)")
    ax.set_title(f"{model_key}: information specifically unlocked by PC3")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"pc3_decoder_gain_{model_key}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_alignment(model_key: str, alignment: pd.DataFrame, output_dir: Path) -> None:
    subset = alignment[alignment["model"] == model_key]
    early = subset[subset["reference"] == "early"].sort_values("checkpoint_order")
    final = subset[subset["reference"] == "final"].sort_values("checkpoint_order")
    if early.empty:
        return
    labels = early["checkpoint"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(x, early["linear_cka"], marker="o", label="vs early")
    axes[0].plot(x, final["linear_cka"], marker="o", label="vs final")
    axes[0].set_ylabel("Linear CKA")
    axes[0].legend(frameon=False)
    axes[1].plot(x, early["top2_angle_mean_deg"], marker="o", label="top-2")
    axes[1].plot(x, early["top3_angle_mean_deg"], marker="o", label="top-3")
    axes[1].set_ylabel("Mean principal angle to early (deg)")
    axes[1].legend(frameon=False)
    axes[2].plot(x, early["variance_outside_early_top2"], marker="o")
    axes[2].set_ylabel("Variance outside early top-2 plane")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{model_key}: representational morphing")
    fig.tight_layout()
    fig.savefig(output_dir / f"alignment_{model_key}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def normalize_atlas_tokens(value: str) -> list[str]:
    labels = []
    for token in [part.strip().lower() for part in value.split(",") if part.strip()]:
        if token in {"init", "final", "best"}:
            labels.append(token)
        elif token.startswith("ep"):
            labels.append(f"ep{int(token[2:]):03d}")
        else:
            labels.append(f"ep{int(token):03d}")
    return labels


def plot_atlas(
    model_key: str,
    cache: list[dict[str, Any]],
    requested_labels: list[str],
    hazard_values: np.ndarray,
    max_points: int,
    output_dir: Path,
) -> None:
    selected = [item for label in requested_labels for item in cache if item["checkpoint"] == label]
    if not selected:
        return
    indices = choose_fixed_indices(len(hazard_values), max_points, seed=19)
    fig = plt.figure(figsize=(4.5 * len(selected), 4.2))
    scatter = None
    for panel, item in enumerate(selected, start=1):
        ax = fig.add_subplot(1, len(selected), panel, projection="3d")
        scores = (item["states"] - item["mean"]) @ item["basis"][:, :3]
        scatter = ax.scatter(
            scores[indices, 0],
            scores[indices, 1],
            scores[indices, 2],
            c=hazard_values[indices],
            cmap="viridis",
            s=3,
            alpha=0.35,
            linewidths=0,
        )
        ax.set_title(item["checkpoint"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.view_init(elev=23, azim=-55)
    if scatter is not None:
        color_axis = fig.add_axes([0.92, 0.20, 0.012, 0.58])
        fig.colorbar(scatter, cax=color_axis, label="Bayesian hazard mean")
    fig.suptitle(f"{model_key}: local 3-D manifold atlas")
    fig.subplots_adjust(left=0.03, right=0.89, bottom=0.05, top=0.88, wspace=0.05)
    fig.savefig(output_dir / f"manifold_atlas_{model_key}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seeds(0)
    cfg = load_config(args.config)
    specs = model_specs(cfg, parse_model_filter(args.models))
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trials, source_paths = load_trials(
        cfg, split=args.split, max_csvs=args.max_csvs, max_trials=args.max_trials
    )
    normative_path = Path(cfg["normative_model"]).expanduser().resolve()
    normative = compute_normative_features(trials, normative_path, args.hazard_step)
    n_evidence = len(trials[0].evidence)
    trial_groups = np.repeat(np.arange(len(trials), dtype=int), n_evidence)
    flat_targets = {name: normative[name].reshape(-1) for name in DECODER_TARGETS}
    comparison_indices = choose_fixed_indices(
        len(trial_groups), args.max_comparison_points, seed=13
    )

    rng = np.random.default_rng(23)
    cv_trial_ids = np.arange(len(trials), dtype=int)
    if len(cv_trial_ids) > args.max_cv_trials:
        cv_trial_ids = np.sort(
            rng.choice(cv_trial_ids, size=args.max_cv_trials, replace=False)
        )
    cv_row_mask = np.isin(trial_groups, cv_trial_ids)

    dimension_rows: list[dict[str, Any]] = []
    eigenspectrum_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    behavior_all_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    atlas_labels = normalize_atlas_tokens(args.atlas_checkpoints)

    print(f"Device: {device}")
    print(f"Trials: {len(trials)} from {len(source_paths)} {args.split} CSVs")
    print(f"Evidence states per checkpoint: {len(trials) * n_evidence:,}")

    for spec in specs:
        hp = json.loads((spec.seed_dir / "hp.json").read_text(encoding="utf-8"))
        n_null = int(hp.get("n_null_timesteps", 4))
        inputs = build_input_tensor(trials, n_null, mirrored=False)
        mirrored_inputs = build_input_tensor(trials, n_null, mirrored=True)
        evidence_indices = evidence_timestep_indices(n_evidence, n_null)
        checkpoints = resolve_checkpoints(spec.seed_dir, args.checkpoints)
        cache: list[dict[str, Any]] = []
        print(f"\nModel {spec.key}: {len(checkpoints)} checkpoints")

        for checkpoint_index, checkpoint_path in enumerate(checkpoints):
            label = checkpoint_label(checkpoint_path)
            order = checkpoint_position(label)
            print(f"  {label}: collecting states")
            model, _ = load_model(spec, checkpoint_path, device)
            states, report_logits, hazard_logits = collect_evidence_states(
                model, inputs, evidence_indices, args.batch_size, device
            )
            mirror_states, mirror_report, mirror_hazard = collect_evidence_states(
                model, mirrored_inputs, evidence_indices, args.batch_size, device
            )
            flat_states = states.reshape(-1, states.shape[-1])
            flat_mirror = mirror_states.reshape(-1, mirror_states.shape[-1])
            mean, eigenvalues, basis = covariance_eigendecomposition(flat_states)
            total_variance = float(np.sum(eigenvalues))
            ratios = eigenvalues / max(total_variance, 1e-12)
            twonn, twonn_n = twonn_dimension(
                flat_states, args.max_id_points, seed=100 + checkpoint_index
            )
            symmetry = mirror_symmetry_metrics(
                flat_states,
                flat_mirror,
                report_logits.reshape(-1),
                mirror_report.reshape(-1),
                hazard_logits.reshape(-1),
                mirror_hazard.reshape(-1),
                trial_groups,
                args.max_comparison_points,
                seed=200 + checkpoint_index,
            )
            dimension_rows.append(
                {
                    "model": spec.key,
                    "model_label": spec.label,
                    "checkpoint": label,
                    "checkpoint_order": order,
                    "scope": "all_evidence",
                    "n_states": len(flat_states),
                    "participation_ratio": participation_ratio(eigenvalues),
                    "twonn_dimension": twonn,
                    "twonn_n_fit": twonn_n,
                    "dimension_90pct": variance_dimension(eigenvalues, 0.90),
                    "dimension_95pct": variance_dimension(eigenvalues, 0.95),
                    "dimension_99pct": variance_dimension(eigenvalues, 0.99),
                    "pc3_explained_variance": float(ratios[2]),
                    "variance_outside_top2": float(1.0 - np.sum(ratios[:2])),
                    **symmetry,
                }
            )
            for pc_index in range(min(20, len(ratios))):
                eigenspectrum_rows.append(
                    {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "scope": "all_evidence",
                        "pc": pc_index + 1,
                        "eigenvalue": float(eigenvalues[pc_index]),
                        "explained_variance_ratio": float(ratios[pc_index]),
                        "cumulative_explained_variance_ratio": float(
                            np.sum(ratios[: pc_index + 1])
                        ),
                    }
                )

            cv_reconstruction, cv_decoder = cv_manifold_and_decoding(
                flat_states[cv_row_mask],
                {name: values[cv_row_mask] for name, values in flat_targets.items()},
                trial_groups[cv_row_mask],
                args.cv_folds,
                seed=300 + checkpoint_index,
            )
            for row in cv_reconstruction:
                reconstruction_rows.append(
                    {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "scope": "all_evidence",
                        **row,
                    }
                )
            for row in cv_decoder:
                decoder_rows.append(
                    {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "scope": "all_evidence",
                        **row,
                    }
                )
            behavior_all_rows.extend(
                behavior_rows(
                    spec.key,
                    label,
                    order,
                    report_logits,
                    hazard_logits,
                    trials,
                    normative,
                )
            )

            report_weight = model.loc_head.weight.detach().cpu().numpy().reshape(-1)
            hazard_weight = model.haz_head.weight.detach().cpu().numpy().reshape(-1)
            readout_cosine = float(
                report_weight @ hazard_weight
                / max(np.linalg.norm(report_weight) * np.linalg.norm(hazard_weight), 1e-12)
            )
            for pc_index in range(min(10, basis.shape[1])):
                pc = basis[:, pc_index]
                readout_rows.append(
                    {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "scope": "all_evidence",
                        "pc": pc_index + 1,
                        "report_head_alignment_sq": float(
                            (pc @ report_weight) ** 2
                            / max(report_weight @ report_weight, 1e-12)
                        ),
                        "hazard_head_alignment_sq": float(
                            (pc @ hazard_weight) ** 2
                            / max(hazard_weight @ hazard_weight, 1e-12)
                        ),
                        "report_hazard_head_cosine": readout_cosine,
                    }
                )

            # Repeat the dimensionality and decoding tests at the final evidence
            # endpoint. This rules out apparent dimensions caused only by pooling
            # different positions along the 20-item trajectory.
            final_states = states[:, -1]
            final_mirror_states = mirror_states[:, -1]
            final_groups = np.arange(len(trials), dtype=int)
            _, final_eigenvalues, _ = covariance_eigendecomposition(final_states)
            final_total_variance = float(np.sum(final_eigenvalues))
            final_ratios = final_eigenvalues / max(final_total_variance, 1e-12)
            final_twonn, final_twonn_n = twonn_dimension(
                final_states,
                min(args.max_id_points, len(final_states)),
                seed=400 + checkpoint_index,
            )
            final_symmetry = mirror_symmetry_metrics(
                final_states,
                final_mirror_states,
                report_logits[:, -1],
                mirror_report[:, -1],
                hazard_logits[:, -1],
                mirror_hazard[:, -1],
                final_groups,
                min(args.max_comparison_points, len(final_states)),
                seed=500 + checkpoint_index,
            )
            dimension_rows.append(
                {
                    "model": spec.key,
                    "model_label": spec.label,
                    "checkpoint": label,
                    "checkpoint_order": order,
                    "scope": "final_evidence",
                    "n_states": len(final_states),
                    "participation_ratio": participation_ratio(final_eigenvalues),
                    "twonn_dimension": final_twonn,
                    "twonn_n_fit": final_twonn_n,
                    "dimension_90pct": variance_dimension(final_eigenvalues, 0.90),
                    "dimension_95pct": variance_dimension(final_eigenvalues, 0.95),
                    "dimension_99pct": variance_dimension(final_eigenvalues, 0.99),
                    "pc3_explained_variance": float(final_ratios[2]),
                    "variance_outside_top2": float(1.0 - np.sum(final_ratios[:2])),
                    **final_symmetry,
                }
            )
            for pc_index in range(min(20, len(final_ratios))):
                eigenspectrum_rows.append(
                    {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "scope": "final_evidence",
                        "pc": pc_index + 1,
                        "eigenvalue": float(final_eigenvalues[pc_index]),
                        "explained_variance_ratio": float(final_ratios[pc_index]),
                        "cumulative_explained_variance_ratio": float(
                            np.sum(final_ratios[: pc_index + 1])
                        ),
                    }
                )
            final_cv_mask = np.isin(final_groups, cv_trial_ids)
            final_reconstruction, final_decoder = cv_manifold_and_decoding(
                final_states[final_cv_mask],
                {
                    name: normative[name][:, -1][final_cv_mask]
                    for name in DECODER_TARGETS
                },
                final_groups[final_cv_mask],
                args.cv_folds,
                seed=600 + checkpoint_index,
            )
            for row in final_reconstruction:
                reconstruction_rows.append(
                    {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "scope": "final_evidence",
                        **row,
                    }
                )
            for row in final_decoder:
                decoder_rows.append(
                    {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "scope": "final_evidence",
                        **row,
                    }
                )

            cache.append(
                {
                    "model": spec.key,
                    "checkpoint": label,
                    "checkpoint_order": order,
                    "states": flat_states[comparison_indices].astype(np.float32),
                    "mean": mean.astype(np.float32),
                    "basis": basis[:, :10].astype(np.float32),
                }
            )
            del model, states, mirror_states, flat_states, flat_mirror
            if device.type == "cuda":
                torch.cuda.empty_cache()

        alignment_rows.extend(alignment_rows_for_model(cache))
        plot_atlas(
            spec.key,
            cache,
            atlas_labels,
            normative["bayes_hazard_mean"].reshape(-1)[comparison_indices],
            args.max_atlas_points,
            output_dir,
        )

    dimension_df = pd.DataFrame(dimension_rows)
    eigenspectrum_df = pd.DataFrame(eigenspectrum_rows)
    reconstruction_fold_df = pd.DataFrame(reconstruction_rows)
    decoder_fold_df = pd.DataFrame(decoder_rows)
    behavior_df = pd.DataFrame(behavior_all_rows)
    readout_df = pd.DataFrame(readout_rows)
    alignment_df = pd.DataFrame(alignment_rows)
    reconstruction_df = aggregate_fold_rows(
        reconstruction_fold_df,
        "heldout_reconstruction_r2",
        ["model", "checkpoint", "checkpoint_order", "scope", "n_components"],
    )
    decoder_df = aggregate_fold_rows(
        decoder_fold_df,
        "heldout_r2",
        ["model", "checkpoint", "checkpoint_order", "scope", "target", "n_components"],
    )

    outputs = {
        "dimension_metrics.csv": dimension_df,
        "eigenspectrum.csv": eigenspectrum_df,
        "reconstruction_fold_metrics.csv": reconstruction_fold_df,
        "reconstruction_metrics.csv": reconstruction_df,
        "decoder_fold_metrics.csv": decoder_fold_df,
        "decoder_metrics.csv": decoder_df,
        "behavior_metrics.csv": behavior_df,
        "readout_alignment.csv": readout_df,
        "alignment_metrics.csv": alignment_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    for spec in specs:
        plot_model_summary(
            spec.key,
            dimension_df,
            eigenspectrum_df,
            reconstruction_df,
            behavior_df,
            output_dir,
        )
        plot_decoder_gains(spec.key, decoder_df, output_dir)
        plot_alignment(spec.key, alignment_df, output_dir)

    write_run_config(
        output_dir / "run_config.json",
        {
            **vars(args),
            "device_resolved": device,
            "seed": int(cfg.get("seed", 0)),
            "n_trials": len(trials),
            "source_csvs": [str(path) for path in source_paths],
            "models_resolved": [spec.key for spec in specs],
            "normative_model": normative_path,
            "state_scopes": [
                "all hidden states immediately after evidence items",
                "final post-evidence hidden state per trial",
            ],
        },
    )
    print(f"\nSaved geometry analysis to {output_dir}")


if __name__ == "__main__":
    main()

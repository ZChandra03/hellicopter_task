#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from manifold_utils import (
    BASE_DIR,
    DEFAULT_CONFIG,
    build_input_tensor,
    checkpoint_label,
    choose_device,
    collect_evidence_states,
    evidence_timestep_indices,
    load_config,
    load_model,
    load_trials,
    model_specs,
    parse_model_filter,
    resolve_checkpoints,
    set_seeds,
    sigmoid,
    write_run_config,
)


DEFAULT_OUTPUT_DIR = BASE_DIR / "slow_point_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track fixed/slow points and autonomous endpoint basins through training. "
            "Evidence maps are phase-matched from one post-evidence state to the next."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="all")
    parser.add_argument("--checkpoints", default="1,5,7,10,final")
    parser.add_argument(
        "--maps",
        default="null,zero_cycle",
        help="Comma list from null,neg_cycle,zero_cycle,pos_cycle",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-csvs", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-inits", type=int, default=128)
    parser.add_argument("--opt-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--cluster-eps", type=float, default=1e-3)
    parser.add_argument("--fixed-tol", type=float, default=1e-6)
    parser.add_argument("--slow-tol", type=float, default=1e-3)
    parser.add_argument("--eig-tol", type=float, default=1e-2)
    parser.add_argument("--max-points-per-map", type=int, default=20)
    parser.add_argument("--basin-inits", type=int, default=256)
    parser.add_argument("--basin-steps", type=int, default=200)
    parser.add_argument("--basin-tol", type=float, default=1e-4)
    parser.add_argument("--basin-cluster-eps", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=8421)
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


def map_definitions(n_null: int) -> dict[str, torch.Tensor]:
    null = [[0.0, 0.0]]
    # Real states are sampled immediately after evidence. A phase-matched map
    # therefore applies the four null steps first, then the next evidence item.
    return {
        "null": torch.tensor(null, dtype=torch.float32),
        "neg_cycle": torch.tensor([[0.0, 0.0]] * n_null + [[-1.0, 1.0]], dtype=torch.float32),
        "zero_cycle": torch.tensor([[0.0, 0.0]] * n_null + [[0.0, 1.0]], dtype=torch.float32),
        "pos_cycle": torch.tensor([[0.0, 0.0]] * n_null + [[1.0, 1.0]], dtype=torch.float32),
    }


def apply_map(model, hidden: torch.Tensor, map_inputs: torch.Tensor) -> torch.Tensor:
    squeeze = hidden.ndim == 1
    hidden_batch = hidden.unsqueeze(0) if squeeze else hidden
    sequence = map_inputs.to(device=hidden.device, dtype=hidden.dtype)
    sequence = sequence.unsqueeze(0).expand(hidden_batch.shape[0], -1, -1)
    _, final = model.rnn.gru(sequence, hidden_batch.unsqueeze(0))
    result = final.squeeze(0)
    return result.squeeze(0) if squeeze else result


def optimize_slow_points(
    model,
    initial_states: np.ndarray,
    map_inputs: torch.Tensor,
    device: torch.device,
    steps: int,
    learning_rate: float,
    patience: int,
) -> tuple[np.ndarray, np.ndarray]:
    # cuDNN only retains the GRU intermediates needed for backward in training
    # mode. This one-layer GRU has no dropout, and all weights remain frozen, so
    # train/eval produce the same map while train mode enables d q / d h.
    was_training = model.rnn.gru.training
    model.rnn.gru.train(True)
    hidden = torch.tensor(initial_states, dtype=torch.float32, device=device).requires_grad_(True)
    optimizer = torch.optim.Adam([hidden], lr=learning_rate)
    best = math.inf
    stale = 0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = apply_map(model, hidden, map_inputs) - hidden
        q = 0.5 * torch.sum(delta * delta, dim=1)
        loss = torch.mean(q)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best * (1.0 - 1e-7):
            best = value
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break
    with torch.no_grad():
        delta = apply_map(model, hidden, map_inputs) - hidden
        speed = torch.linalg.norm(delta, dim=1)
    model.rnn.gru.train(was_training)
    return hidden.detach().cpu().numpy(), speed.detach().cpu().numpy()


def deduplicate(points: np.ndarray, speeds: np.ndarray, eps: float) -> list[int]:
    kept: list[int] = []
    for index in np.argsort(speeds):
        if all(np.linalg.norm(points[index] - points[other]) > eps for other in kept):
            kept.append(int(index))
    return kept


def point_type(speed: float, fixed_tol: float, slow_tol: float) -> str:
    if speed < fixed_tol:
        return "fixed"
    if speed < slow_tol:
        return "slow"
    return "not_slow"


def jacobian_eigenvalues(
    model, point: np.ndarray, map_inputs: torch.Tensor, device: torch.device
) -> np.ndarray:
    was_training = model.rnn.gru.training
    model.rnn.gru.train(True)
    hidden = torch.tensor(point, dtype=torch.float32, device=device, requires_grad=True)

    def map_function(value: torch.Tensor) -> torch.Tensor:
        return apply_map(model, value, map_inputs)

    # cuDNN GRU backward does not implement the batched VJP used by
    # vectorize=True. The unvectorized 128-output Jacobian is slower but robust.
    jacobian = torch.autograd.functional.jacobian(map_function, hidden, vectorize=False)
    eigenvalues = torch.linalg.eigvals(jacobian).detach().cpu().numpy()
    model.rnn.gru.train(was_training)
    return eigenvalues


def nearest_distance(points: np.ndarray, point: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(points - point[None, :], axis=1)))


def empirical_neighbor_scale(points: np.ndarray, maximum: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    sample = points
    if len(sample) > maximum:
        sample = sample[rng.choice(len(sample), size=maximum, replace=False)]
    distances, _ = NearestNeighbors(n_neighbors=2).fit(sample).kneighbors(sample)
    positive = distances[:, 1][distances[:, 1] > 1e-12]
    return float(np.median(positive)) if len(positive) else float("nan")


@torch.inference_mode()
def empirical_map_speeds(
    model,
    points: np.ndarray,
    map_inputs: torch.Tensor,
    device: torch.device,
    maximum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample = points
    if len(sample) > maximum:
        sample = sample[rng.choice(len(sample), size=maximum, replace=False)]
    output: list[np.ndarray] = []
    for start in range(0, len(sample), 1024):
        hidden = torch.tensor(sample[start : start + 1024], dtype=torch.float32, device=device)
        speed = torch.linalg.norm(apply_map(model, hidden, map_inputs) - hidden, dim=1)
        output.append(speed.cpu().numpy())
    return np.concatenate(output)


def classify_stability(eigenvalues: np.ndarray, tolerance: float) -> dict[str, Any]:
    radii = np.abs(eigenvalues)
    unstable = int(np.sum(radii > 1.0 + tolerance))
    marginal = int(np.sum(np.abs(radii - 1.0) <= tolerance))
    if unstable and marginal:
        label = "saddle_with_slow_directions"
    elif unstable:
        label = "saddle_or_unstable"
    elif marginal:
        label = "stable_with_slow_directions"
    else:
        label = "stable"
    return {
        "stability": label,
        "spectral_radius": float(np.max(radii)),
        "n_unstable_eigenvalues": unstable,
        "n_near_unit_eigenvalues": marginal,
        "top_eigenvalue_abs": json.dumps(
            [float(value) for value in np.sort(radii)[::-1][:10]]
        ),
    }


@torch.inference_mode()
def rollout_map(
    model,
    initial_states: np.ndarray,
    map_inputs: torch.Tensor,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = torch.tensor(initial_states, dtype=torch.float32, device=device)
    for _ in range(steps):
        hidden = apply_map(model, hidden, map_inputs)
    next_hidden = apply_map(model, hidden, map_inputs)
    speeds = torch.linalg.norm(next_hidden - hidden, dim=1)
    return hidden.cpu().numpy(), speeds.cpu().numpy()


def basin_census(
    model_key: str,
    checkpoint: str,
    order: float,
    model,
    final_states: np.ndarray,
    trials,
    map_inputs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(args.random_seed + int(order if np.isfinite(order) else 0))
    count = min(args.basin_inits, len(final_states))
    indices = np.sort(rng.choice(len(final_states), size=count, replace=False))
    endpoints, speeds = rollout_map(
        model, final_states[indices], map_inputs, args.basin_steps, device
    )
    converged = speeds < args.basin_tol
    cluster_labels = np.full(count, -1, dtype=int)
    if np.any(converged):
        cluster_labels[converged] = DBSCAN(
            eps=args.basin_cluster_eps, min_samples=1
        ).fit_predict(endpoints[converged])
    endpoint_tensor = torch.tensor(endpoints, dtype=torch.float32, device=device)
    with torch.no_grad():
        report_prob = sigmoid(model.loc_head(endpoint_tensor).squeeze(-1).cpu().numpy())
        hazard_prob = sigmoid(model.haz_head(endpoint_tensor).squeeze(-1).cpu().numpy())

    rows: list[dict[str, Any]] = []
    for local_index, trial_index in enumerate(indices):
        trial = trials[int(trial_index)]
        rows.append(
            {
                "model": model_key,
                "checkpoint": checkpoint,
                "checkpoint_order": order,
                "trial_index": int(trial_index),
                "source_true_hazard": trial.true_hazard,
                "source_hazard_class": "high" if trial.true_hazard >= 0.5 else "low",
                "source_true_report": trial.true_report,
                "endpoint_speed": float(speeds[local_index]),
                "converged": bool(converged[local_index]),
                "basin_cluster": int(cluster_labels[local_index]),
                "endpoint_report_prob": float(report_prob[local_index]),
                "endpoint_hazard_prob": float(hazard_prob[local_index]),
            }
        )

    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for cluster, cluster_frame in frame.groupby("basin_cluster"):
        summaries.append(
            {
                "model": model_key,
                "checkpoint": checkpoint,
                "checkpoint_order": order,
                "basin_cluster": int(cluster),
                "interpretation": "no_converged_discrete_basin" if cluster == -1 else "converged_endpoint_cluster",
                "n_initial_states": len(cluster_frame),
                "fraction_of_initial_states": len(cluster_frame) / len(frame),
                "fraction_source_high_hazard": float(
                    np.mean(cluster_frame["source_hazard_class"] == "high")
                ),
                "fraction_source_positive_report": float(
                    np.mean(cluster_frame["source_true_report"] == 1)
                ),
                "endpoint_speed_mean": float(cluster_frame["endpoint_speed"].mean()),
                "endpoint_report_prob_mean": float(
                    cluster_frame["endpoint_report_prob"].mean()
                ),
                "endpoint_hazard_prob_mean": float(
                    cluster_frame["endpoint_hazard_prob"].mean()
                ),
            }
        )
    return rows, summaries


def plot_state_space(
    model_key: str,
    checkpoint: str,
    map_name: str,
    model,
    real_states: np.ndarray,
    slow_points: np.ndarray,
    point_rows: list[dict[str, Any]],
    map_inputs: torch.Tensor,
    output_dir: Path,
    device: torch.device,
) -> None:
    if len(real_states) < 3:
        return
    rng = np.random.default_rng(73)
    sample_indices = (
        rng.choice(len(real_states), size=min(2500, len(real_states)), replace=False)
    )
    sample = real_states[sample_indices]
    pca = PCA(n_components=3, svd_solver="randomized", random_state=73).fit(real_states)
    sample_pc = pca.transform(sample)
    with torch.no_grad():
        sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
        next_states = apply_map(model, sample_tensor, map_inputs).cpu().numpy()
    delta_pc = pca.transform(next_states) - sample_pc

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    speed = np.linalg.norm(delta_pc[:, :2], axis=1)
    scatter = axes[0].scatter(
        sample_pc[:, 0], sample_pc[:, 1], c=np.log10(speed + 1e-8), s=4, alpha=0.35, cmap="magma"
    )
    quiver_indices = np.arange(0, len(sample_pc), max(1, len(sample_pc) // 250))
    axes[0].quiver(
        sample_pc[quiver_indices, 0],
        sample_pc[quiver_indices, 1],
        delta_pc[quiver_indices, 0],
        delta_pc[quiver_indices, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="0.25",
        alpha=0.4,
        width=0.002,
    )
    fig.colorbar(scatter, ax=axes[0], label="log10 projected speed")
    axes[0].set_title("On-trajectory projected drift")

    axes[1].scatter(sample_pc[:, 0], sample_pc[:, 1], s=3, c="0.75", alpha=0.2)
    if len(slow_points):
        slow_pc = pca.transform(slow_points)
        colors = [row["hazard_prob"] for row in point_rows]
        scatter_points = axes[1].scatter(
            slow_pc[:, 0],
            slow_pc[:, 1],
            c=colors,
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=75,
            edgecolors="black",
            linewidths=0.5,
        )
        fig.colorbar(scatter_points, ax=axes[1], label="slow-point P(high)")
    axes[1].set_title("Optimized fixed/slow candidates")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{model_key} {checkpoint}: {map_name}")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"state_space_{model_key}_{checkpoint}_{map_name}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_tracking_summary(points: pd.DataFrame, basins: pd.DataFrame, output_dir: Path) -> None:
    if points.empty:
        return
    summary = (
        points.groupby(["model", "checkpoint", "checkpoint_order", "map"], as_index=False)
        .agg(
            n_candidates=("point_index", "size"),
            n_slow=("point_type", lambda values: int(np.sum(np.isin(values, ["fixed", "slow"])))),
            median_speed=("speed", "median"),
            max_spectral_radius=("spectral_radius", "max"),
        )
    )
    for model_key, model_frame in summary.groupby("model"):
        maps = sorted(model_frame["map"].unique())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for map_name in maps:
            subset = model_frame[model_frame["map"] == map_name].sort_values("checkpoint_order")
            axes[0].plot(subset["checkpoint"], subset["n_slow"], marker="o", label=map_name)
            axes[1].plot(subset["checkpoint"], subset["median_speed"], marker="o", label=map_name)
            axes[2].plot(subset["checkpoint"], subset["max_spectral_radius"], marker="o", label=map_name)
        axes[0].set_ylabel("Fixed + slow candidates")
        axes[1].set_ylabel("Median candidate speed")
        axes[1].set_yscale("log")
        axes[2].set_ylabel("Largest spectral radius")
        axes[2].axhline(1.0, color="black", linewidth=1)
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.25)
        axes[-1].legend(frameon=False)
        fig.suptitle(f"{model_key}: slow-point topology through training")
        fig.tight_layout()
        fig.savefig(output_dir / f"slow_point_tracking_{model_key}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if not basins.empty:
        basin_summary = (
            basins.groupby(["model", "checkpoint", "checkpoint_order"], as_index=False)
            .agg(
                converged_fraction=("converged", "mean"),
                n_endpoint_clusters=("basin_cluster", lambda values: len(set(values) - {-1})),
            )
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for model_key, subset in basin_summary.groupby("model"):
            subset = subset.sort_values("checkpoint_order")
            axes[0].plot(subset["checkpoint"], subset["converged_fraction"], marker="o", label=model_key)
            axes[1].plot(subset["checkpoint"], subset["n_endpoint_clusters"], marker="o", label=model_key)
        axes[0].set_ylabel("Fraction converged after null rollout")
        axes[1].set_ylabel("Discrete endpoint clusters")
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "null_basin_census.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seeds(args.random_seed)
    cfg = load_config(args.config)
    specs = model_specs(cfg, parse_model_filter(args.models))
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials, source_paths = load_trials(
        cfg, split=args.split, max_csvs=args.max_csvs, max_trials=args.max_trials
    )
    requested_maps = [part.strip() for part in args.maps.split(",") if part.strip()]
    rng = np.random.default_rng(args.random_seed)
    point_rows_all: list[dict[str, Any]] = []
    basin_rows_all: list[dict[str, Any]] = []
    basin_summary_all: list[dict[str, Any]] = []
    state_arrays: dict[str, np.ndarray] = {}

    print(f"Device: {device}")
    print(f"Trials: {len(trials)}")
    for spec in specs:
        hp = json.loads((spec.seed_dir / "hp.json").read_text(encoding="utf-8"))
        n_null = int(hp.get("n_null_timesteps", 4))
        definitions = map_definitions(n_null)
        unknown_maps = sorted(set(requested_maps) - set(definitions))
        if unknown_maps:
            raise ValueError(f"Unknown maps: {unknown_maps}; choices are {sorted(definitions)}")
        inputs = build_input_tensor(trials, n_null)
        evidence_indices = evidence_timestep_indices(len(trials[0].evidence), n_null)
        checkpoints = resolve_checkpoints(spec.seed_dir, args.checkpoints)
        print(f"\nModel {spec.key}: {len(checkpoints)} checkpoints")

        for checkpoint_index, checkpoint_path in enumerate(checkpoints):
            label = checkpoint_label(checkpoint_path)
            order = checkpoint_position(label)
            print(f"  {label}: collecting empirical initial states")
            model, _ = load_model(spec, checkpoint_path, device)
            hidden, _, _ = collect_evidence_states(
                model, inputs, evidence_indices, args.batch_size, device
            )
            all_real_states = hidden.reshape(-1, hidden.shape[-1])
            final_states = hidden[:, -1]
            neighbor_scale = empirical_neighbor_scale(
                all_real_states, maximum=5000, seed=args.random_seed + checkpoint_index
            )
            init_count = min(args.n_inits, len(all_real_states))
            init_indices = rng.choice(len(all_real_states), size=init_count, replace=False)
            initial_states = all_real_states[init_indices]
            initial_states = np.concatenate(
                [initial_states, np.zeros((1, hidden.shape[-1]), dtype=np.float32)], axis=0
            )

            for map_name in requested_maps:
                print(f"    optimizing {map_name}")
                map_inputs = definitions[map_name].to(device)
                real_speeds = empirical_map_speeds(
                    model,
                    all_real_states,
                    map_inputs,
                    device,
                    maximum=5000,
                    seed=args.random_seed + 100 + checkpoint_index,
                )
                candidates, speeds = optimize_slow_points(
                    model,
                    initial_states,
                    map_inputs,
                    device,
                    args.opt_steps,
                    args.learning_rate,
                    args.patience,
                )
                kept = deduplicate(candidates, speeds, args.cluster_eps)[
                    : args.max_points_per_map
                ]
                checkpoint_point_rows: list[dict[str, Any]] = []
                checkpoint_points: list[np.ndarray] = []
                for point_index, candidate_index in enumerate(kept):
                    point = candidates[candidate_index]
                    speed = float(speeds[candidate_index])
                    eigenvalues = jacobian_eigenvalues(
                        model, point, map_inputs, device
                    )
                    stability = classify_stability(eigenvalues, args.eig_tol)
                    point_tensor = torch.tensor(point, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        report_prob = float(
                            sigmoid(model.loc_head(point_tensor).cpu().numpy())[0]
                        )
                        hazard_prob = float(
                            sigmoid(model.haz_head(point_tensor).cpu().numpy())[0]
                        )
                    distance_to_real = nearest_distance(all_real_states, point)
                    row = {
                        "model": spec.key,
                        "checkpoint": label,
                        "checkpoint_order": order,
                        "map": map_name,
                        "point_index": point_index,
                        "source_candidate_index": candidate_index,
                        "speed": speed,
                        "point_type": point_type(speed, args.fixed_tol, args.slow_tol),
                        "distance_to_nearest_empirical_state": distance_to_real,
                        "empirical_median_nearest_neighbor_distance": neighbor_scale,
                        "distance_to_real_over_median_neighbor": (
                            distance_to_real / neighbor_scale
                            if np.isfinite(neighbor_scale) and neighbor_scale > 0
                            else np.nan
                        ),
                        "empirical_speed_percentile": float(
                            100.0 * np.mean(real_speeds <= speed)
                        ),
                        "report_prob": report_prob,
                        "hazard_prob": hazard_prob,
                        **stability,
                    }
                    point_rows_all.append(row)
                    checkpoint_point_rows.append(row)
                    checkpoint_points.append(point)
                point_array = (
                    np.stack(checkpoint_points)
                    if checkpoint_points
                    else np.zeros((0, hidden.shape[-1]), dtype=np.float32)
                )
                state_arrays[f"{spec.key}__{label}__{map_name}"] = point_array
                plot_state_space(
                    spec.key,
                    label,
                    map_name,
                    model,
                    all_real_states,
                    point_array,
                    checkpoint_point_rows,
                    map_inputs,
                    output_dir,
                    device,
                )

                if map_name == "null":
                    basin_rows, basin_summaries = basin_census(
                        spec.key,
                        label,
                        order,
                        model,
                        final_states,
                        trials,
                        map_inputs,
                        args,
                        device,
                    )
                    basin_rows_all.extend(basin_rows)
                    basin_summary_all.extend(basin_summaries)
            del model, hidden
            if device.type == "cuda":
                torch.cuda.empty_cache()

    points_df = pd.DataFrame(point_rows_all)
    basins_df = pd.DataFrame(basin_rows_all)
    basin_summary_df = pd.DataFrame(basin_summary_all)
    points_df.to_csv(output_dir / "slow_points.csv", index=False)
    basins_df.to_csv(output_dir / "null_basin_assignments.csv", index=False)
    basin_summary_df.to_csv(output_dir / "null_basin_summary.csv", index=False)
    np.savez_compressed(output_dir / "slow_point_states.npz", **state_arrays)
    plot_tracking_summary(points_df, basins_df, output_dir)
    write_run_config(
        output_dir / "run_config.json",
        {
            **vars(args),
            "device_resolved": device,
            "seed": int(cfg.get("seed", 0)),
            "models_resolved": [spec.key for spec in specs],
            "source_csvs": [str(path) for path in source_paths],
            "map_phase": (
                "cycle maps start at a post-evidence hidden state, apply n_null null "
                "updates, then apply the next canonical evidence update"
            ),
        },
    )
    print(f"\nSaved slow-point analysis to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

import pca_checkpoint_ep010 as pca_impl


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "representation_similarity_outputs"
DEFAULT_PCA_COMPONENTS = (2, 3, 5, 10)
DEFAULT_CONDITION_COLS = ("true_report", "true_predict")
SEED_RE = re.compile(r"seed_(\d+)$")


@dataclass
class SeedRepresentation:
    seed: int
    seed_name: str
    condition_means: np.ndarray
    rdm_vector: np.ndarray
    pca_coords_by_n: dict[int, np.ndarray]
    state_sample: np.ndarray
    explained_variance_ratio: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare checkpoint_ep010 hidden-state representations across seeds with "
            "Procrustes PCA trajectories, RSA/RDM, linear CKA, and SVCCA."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.json. Default: ./config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for metric CSVs. Default: ./representation_similarity_outputs",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=pca_impl.CHECKPOINT_NAME,
        help=f"Checkpoint file to compare. Default: {pca_impl.CHECKPOINT_NAME}",
    )
    parser.add_argument(
        "--seeds",
        default="all",
        help='Comma-separated seed ids, e.g. "0,1,2", or "all". Default: all',
    )
    parser.add_argument(
        "--variant-split",
        default=pca_impl.DEFAULT_VARIANT_SPLIT,
        help=f"Variant split prefix to load. Default: {pca_impl.DEFAULT_VARIANT_SPLIT}",
    )
    parser.add_argument(
        "--max-variant-csvs",
        type=int,
        default=pca_impl.DEFAULT_MAX_VARIANT_CSVS,
        help="Optional cap on variant CSVs for faster smoke runs. Default: no cap",
    )
    parser.add_argument(
        "--condition-cols",
        default=",".join(DEFAULT_CONDITION_COLS),
        help=(
            "Comma-separated trial metadata columns defining task conditions for "
            "condition-time means. Default: true_report,true_predict"
        ),
    )
    parser.add_argument(
        "--pca-components",
        default=",".join(str(n) for n in DEFAULT_PCA_COMPONENTS),
        help="Comma-separated top PC counts for trajectory shape. Default: 2,3,5,10",
    )
    parser.add_argument(
        "--max-state-rows",
        type=int,
        default=50000,
        help=(
            "Matched trial-time hidden-state rows to sample for full-state CKA/SVCCA. "
            "Use <=0 for all rows. Default: 50000"
        ),
    )
    parser.add_argument(
        "--rdm-metric",
        choices=("correlation", "euclidean"),
        default="correlation",
        help="Distance metric for condition-time RDMs. Default: correlation",
    )
    parser.add_argument(
        "--rsa-correlation",
        choices=("spearman", "pearson"),
        default="spearman",
        help="Correlation method for RDM vectors. Default: spearman",
    )
    parser.add_argument(
        "--svcca-var-threshold",
        type=float,
        default=0.99,
        help="Variance threshold for SVCCA PCA reduction. Default: 0.99",
    )
    parser.add_argument(
        "--no-svcca",
        action="store_true",
        help="Skip SVCCA and only compute linear CKA for full hidden states.",
    )
    parser.add_argument(
        "--model-class",
        default=pca_impl.DEFAULT_MODEL_CLASS,
        help=f"Model class in rnn_models.py. Default: {pca_impl.DEFAULT_MODEL_CLASS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for hidden-state row sampling. Default: 0",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def parse_condition_cols(value: str) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one condition column.")
    return parsed


def load_analysis_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = pca_impl.load_config(args.config)
    model_subdir = cfg.get("model_subdir") or pca_impl.infer_model_label(cfg["model_root"])
    variant_subdir = cfg.get("variant_subdir") or cfg.get("sigma") or cfg["model_root"].name
    cfg.update(
        {
            "model_subdir": model_subdir,
            "variant_subdir": variant_subdir,
            "variant_split": args.variant_split,
            "max_variant_csvs": args.max_variant_csvs,
            "model_class": args.model_class,
            "checkpoint_name": args.checkpoint_name,
            "output_dir": args.output_dir.expanduser().resolve(),
            "model_dir": cfg["model_root"],
            "variant_dir": cfg["variant_root"] / variant_subdir,
        }
    )
    return cfg


def normalize_condition_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value.is_integer():
            return int(value)
    return value


def get_seed_dirs(model_dir: Path, seeds: str, checkpoint_name: str) -> list[Path]:
    if seeds.strip().lower() == "all":
        seed_dirs = []
        for path in model_dir.iterdir():
            match = SEED_RE.fullmatch(path.name)
            if path.is_dir() and match and (path / checkpoint_name).exists():
                seed_dirs.append(path)
        seed_dirs.sort(key=lambda path: int(SEED_RE.fullmatch(path.name).group(1)))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed_* directories with {checkpoint_name} in {model_dir}")
        return seed_dirs

    seed_dirs = []
    for seed in parse_csv_ints(seeds):
        seed_dirs.append(pca_impl.get_seed_dir(model_dir, seed, checkpoint_name))
    return seed_dirs


def infer_n_time(dataset: pca_impl.HelicopterPCADataset) -> int:
    n_time = int(dataset[0][0].shape[0])
    for i in range(1, len(dataset)):
        if int(dataset[i][0].shape[0]) != n_time:
            raise ValueError("All encoded trials must have the same number of timesteps.")
    return n_time


def build_condition_index(
    dataset: pca_impl.HelicopterPCADataset,
    condition_cols: list[str],
    n_time: int,
) -> tuple[list[tuple[Any, ...]], dict[tuple[Any, ...], int], pd.DataFrame]:
    missing = [
        col
        for col in condition_cols
        if any(col not in meta for meta in dataset.trial_meta)
    ]
    if missing:
        raise KeyError(f"Condition columns are missing from trial metadata: {missing}")

    condition_tuples = sorted(
        {
            tuple(normalize_condition_value(meta[col]) for col in condition_cols)
            for meta in dataset.trial_meta
        }
    )
    condition_to_offset = {
        condition: condition_idx * n_time
        for condition_idx, condition in enumerate(condition_tuples)
    }

    rows = []
    for condition_idx, condition in enumerate(condition_tuples):
        for timestep in range(n_time):
            row = {
                "condition_time_index": condition_idx * n_time + timestep,
                "condition_index": condition_idx,
                "timestep": timestep,
            }
            row.update({col: condition[i] for i, col in enumerate(condition_cols)})
            rows.append(row)

    return condition_tuples, condition_to_offset, pd.DataFrame(rows)


def choose_state_sample_indices(
    total_rows: int,
    max_state_rows: int,
    seed: int,
) -> np.ndarray:
    if max_state_rows <= 0 or max_state_rows >= total_rows:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=max_state_rows, replace=False)).astype(np.int64)


def collect_selected_state_rows(
    sample_states: np.ndarray,
    sample_indices: np.ndarray,
    hidden: np.ndarray,
    trial_indices: np.ndarray,
    n_time: int,
) -> None:
    for batch_pos, trial_idx in enumerate(trial_indices):
        start = int(trial_idx) * n_time
        stop = start + n_time
        left = int(np.searchsorted(sample_indices, start, side="left"))
        right = int(np.searchsorted(sample_indices, stop, side="left"))
        if left == right:
            continue
        local_timesteps = sample_indices[left:right] - start
        sample_states[left:right] = hidden[batch_pos, local_timesteps, :]


def condition_tuple_for_trial(
    meta: dict[str, Any],
    condition_cols: list[str],
) -> tuple[Any, ...]:
    return tuple(normalize_condition_value(meta[col]) for col in condition_cols)


def rdm_upper_vector(points: np.ndarray, metric: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if metric == "correlation":
        row_centered = points - points.mean(axis=1, keepdims=True)
        row_norms = np.linalg.norm(row_centered, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        normalized = row_centered / row_norms
        distances = 1.0 - normalized @ normalized.T
    elif metric == "euclidean":
        squared = np.sum(points * points, axis=1, keepdims=True)
        distances = np.sqrt(np.maximum(squared + squared.T - 2.0 * points @ points.T, 0.0))
    else:
        raise ValueError(f"Unsupported RDM metric: {metric}")

    upper = np.triu_indices(points.shape[0], k=1)
    return distances[upper]


@torch.inference_mode()
def collect_seed_representation(
    model_cls,
    seed_dir: Path,
    dataset: pca_impl.HelicopterPCADataset,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    condition_cols: list[str],
    condition_to_offset: dict[tuple[Any, ...], int],
    n_condition_time: int,
    n_time: int,
    sample_indices: np.ndarray,
    max_pcs: int,
    rdm_metric: str,
) -> SeedRepresentation:
    seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
    model = pca_impl.load_model(model_cls, seed_dir, cfg["checkpoint_name"], device)
    pca = IncrementalPCA(n_components=max_pcs)

    condition_sums: np.ndarray | None = None
    condition_counts = np.zeros(n_condition_time, dtype=np.int64)
    sample_states: np.ndarray | None = None

    print(f"Collecting hidden states for {seed_dir.name}/{cfg['checkpoint_name']}")
    for hidden, trial_indices, _, _ in pca_impl.iter_hidden_batches(model, dataloader, device):
        hidden = np.asarray(hidden, dtype=np.float32)
        n_hidden = hidden.shape[-1]
        if condition_sums is None:
            condition_sums = np.zeros((n_condition_time, n_hidden), dtype=np.float64)
            sample_states = np.zeros((len(sample_indices), n_hidden), dtype=np.float32)

        pca.partial_fit(hidden.reshape(-1, n_hidden))

        for batch_pos, trial_idx in enumerate(trial_indices):
            meta = dataset.trial_meta[int(trial_idx)]
            condition = condition_tuple_for_trial(meta, condition_cols)
            offset = condition_to_offset[condition]
            condition_sums[offset : offset + n_time] += hidden[batch_pos].astype(np.float64)
            condition_counts[offset : offset + n_time] += 1

        collect_selected_state_rows(
            sample_states,
            sample_indices,
            hidden,
            trial_indices,
            n_time,
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if condition_sums is None or sample_states is None:
        raise ValueError(f"No hidden states were collected for {seed_dir.name}.")
    if np.any(condition_counts == 0):
        missing = np.flatnonzero(condition_counts == 0)[:10].tolist()
        raise ValueError(f"Missing trials for condition-time rows in {seed_dir.name}: {missing}")

    condition_means = condition_sums / condition_counts[:, None]
    pca_condition_coords = pca.transform(condition_means)
    pca_components = {
        n: pca_condition_coords[:, :n].astype(np.float64, copy=True)
        for n in range(1, max_pcs + 1)
    }

    return SeedRepresentation(
        seed=seed,
        seed_name=seed_dir.name,
        condition_means=condition_means.astype(np.float64),
        rdm_vector=rdm_upper_vector(condition_means, rdm_metric),
        pca_coords_by_n=pca_components,
        state_sample=sample_states.astype(np.float64),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )


def procrustes_distance(reference: np.ndarray, target: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if reference.shape != target.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {target.shape}")

    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    tgt_centered = target - target.mean(axis=0, keepdims=True)
    ref_norm = np.linalg.norm(ref_centered)
    tgt_norm = np.linalg.norm(tgt_centered)
    if ref_norm == 0.0 or tgt_norm == 0.0:
        return {"procrustes_distance": np.nan, "procrustes_rms": np.nan, "procrustes_disparity": np.nan}

    ref_scaled = ref_centered / ref_norm
    tgt_scaled = tgt_centered / tgt_norm
    u, _, vt = np.linalg.svd(tgt_scaled.T @ ref_scaled, full_matrices=False)
    rotation = u @ vt
    aligned = tgt_scaled @ rotation
    diff = ref_scaled - aligned
    disparity = float(np.sum(diff * diff))
    return {
        "procrustes_distance": float(np.sqrt(disparity)),
        "procrustes_rms": float(np.sqrt(disparity / reference.shape[0])),
        "procrustes_disparity": disparity,
    }


def rankdata_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def vector_correlation(a: np.ndarray, b: np.ndarray, method: str) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return np.nan, np.nan

    try:
        from scipy import stats

        if method == "spearman":
            result = stats.spearmanr(a, b)
        elif method == "pearson":
            result = stats.pearsonr(a, b)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        if method == "spearman":
            a = rankdata_average(a)
            b = rankdata_average(b)
        elif method != "pearson":
            raise ValueError(f"Unsupported correlation method: {method}")

        a = a - a.mean()
        b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            return np.nan, np.nan
        return float(a @ b / denom), np.nan


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denom = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    if denom == 0.0:
        return np.nan
    return float(np.sum(xy * xy) / denom)


def pca_reduce_for_svcca(x: np.ndarray, var_threshold: float) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    variance = singular_values * singular_values
    total = float(np.sum(variance))
    if total == 0.0:
        return x[:, :1], 1

    cumulative = np.cumsum(variance) / total
    keep = int(np.searchsorted(cumulative, var_threshold, side="left") + 1)
    keep = max(1, min(keep, singular_values.size, x.shape[0] - 1))
    return u[:, :keep] * singular_values[:keep], keep


def invsqrt_symmetric(matrix: np.ndarray, eps: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    return (vectors / np.sqrt(values)) @ vectors.T


def svcca_similarity(
    x: np.ndarray,
    y: np.ndarray,
    var_threshold: float,
    eps: float = 1e-10,
) -> dict[str, float]:
    x_reduced, x_dims = pca_reduce_for_svcca(x, var_threshold)
    y_reduced, y_dims = pca_reduce_for_svcca(y, var_threshold)
    n_obs = min(x_reduced.shape[0], y_reduced.shape[0])
    if n_obs < 3:
        return {
            "svcca_mean_corr": np.nan,
            "svcca_median_corr": np.nan,
            "svcca_min_corr": np.nan,
            "svcca_dims_a": float(x_dims),
            "svcca_dims_b": float(y_dims),
        }

    x_centered = x_reduced - x_reduced.mean(axis=0, keepdims=True)
    y_centered = y_reduced - y_reduced.mean(axis=0, keepdims=True)
    scale = max(n_obs - 1, 1)
    cxx = x_centered.T @ x_centered / scale
    cyy = y_centered.T @ y_centered / scale
    cxy = x_centered.T @ y_centered / scale
    cca_matrix = invsqrt_symmetric(cxx, eps) @ cxy @ invsqrt_symmetric(cyy, eps)
    corrs = np.linalg.svd(cca_matrix, compute_uv=False)
    corrs = np.clip(corrs, 0.0, 1.0)
    return {
        "svcca_mean_corr": float(np.mean(corrs)),
        "svcca_median_corr": float(np.median(corrs)),
        "svcca_min_corr": float(np.min(corrs)),
        "svcca_dims_a": float(x_dims),
        "svcca_dims_b": float(y_dims),
    }


def write_seed_pca_variance(representations: list[SeedRepresentation], out_path: Path) -> None:
    rows = []
    for rep in representations:
        cumulative = 0.0
        for i, ratio in enumerate(rep.explained_variance_ratio, start=1):
            cumulative += float(ratio)
            rows.append(
                {
                    "seed": rep.seed,
                    "component": i,
                    "explained_variance_ratio": float(ratio),
                    "cumulative_explained_variance_ratio": cumulative,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def compute_pairwise_metrics(
    representations: list[SeedRepresentation],
    pca_components: list[int],
    rsa_correlation: str,
    rdm_metric: str,
    svcca_var_threshold: float,
    skip_svcca: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    procrustes_rows = []
    rsa_rows = []
    full_state_rows = []
    combined_rows = []

    for rep_a, rep_b in itertools.combinations(representations, 2):
        combined = {
            "seed_a": rep_a.seed,
            "seed_b": rep_b.seed,
            "n_condition_time_points": rep_a.condition_means.shape[0],
            "n_full_state_rows": rep_a.state_sample.shape[0],
        }

        for n_components in pca_components:
            scores = procrustes_distance(
                rep_a.pca_coords_by_n[n_components],
                rep_b.pca_coords_by_n[n_components],
            )
            row = {
                "seed_a": rep_a.seed,
                "seed_b": rep_b.seed,
                "n_components": n_components,
                "n_condition_time_points": rep_a.condition_means.shape[0],
                **scores,
            }
            procrustes_rows.append(row)
            combined[f"pca_procrustes_rms_pc{n_components}"] = scores["procrustes_rms"]
            combined[f"pca_procrustes_distance_pc{n_components}"] = scores["procrustes_distance"]

        rsa_r, rsa_p = vector_correlation(rep_a.rdm_vector, rep_b.rdm_vector, rsa_correlation)
        rsa_row = {
            "seed_a": rep_a.seed,
            "seed_b": rep_b.seed,
            "rdm_metric": rdm_metric,
            "correlation_method": rsa_correlation,
            "rsa_correlation": rsa_r,
            "p_value": rsa_p,
            "n_rdm_edges": int(rep_a.rdm_vector.size),
        }
        rsa_rows.append(rsa_row)
        combined["rsa_rdm_correlation"] = rsa_r

        state_row = {
            "seed_a": rep_a.seed,
            "seed_b": rep_b.seed,
            "n_full_state_rows": rep_a.state_sample.shape[0],
            "linear_cka": linear_cka(rep_a.state_sample, rep_b.state_sample),
        }
        if not skip_svcca:
            state_row.update(
                svcca_similarity(rep_a.state_sample, rep_b.state_sample, svcca_var_threshold)
            )
        full_state_rows.append(state_row)
        combined.update(
            {
                key: value
                for key, value in state_row.items()
                if key not in {"seed_a", "seed_b", "n_full_state_rows"}
            }
        )

        combined_rows.append(combined)

    return (
        pd.DataFrame(procrustes_rows),
        pd.DataFrame(rsa_rows),
        pd.DataFrame(full_state_rows),
        pd.DataFrame(combined_rows),
    )


def write_summary(
    procrustes_df: pd.DataFrame,
    rsa_df: pd.DataFrame,
    full_state_df: pd.DataFrame,
    out_path: Path,
) -> None:
    rows = []
    if not procrustes_df.empty:
        for n_components, group in procrustes_df.groupby("n_components"):
            rows.append(
                {
                    "metric": f"pca_procrustes_rms_pc{n_components}",
                    "mean": group["procrustes_rms"].mean(),
                    "std": group["procrustes_rms"].std(),
                    "min": group["procrustes_rms"].min(),
                    "max": group["procrustes_rms"].max(),
                    "n_pairs": len(group),
                }
            )
    if not rsa_df.empty:
        rows.append(
            {
                "metric": "rsa_rdm_correlation",
                "mean": rsa_df["rsa_correlation"].mean(),
                "std": rsa_df["rsa_correlation"].std(),
                "min": rsa_df["rsa_correlation"].min(),
                "max": rsa_df["rsa_correlation"].max(),
                "n_pairs": len(rsa_df),
            }
        )
    if not full_state_df.empty:
        for metric in ["linear_cka", "svcca_mean_corr", "svcca_median_corr"]:
            if metric not in full_state_df:
                continue
            rows.append(
                {
                    "metric": metric,
                    "mean": full_state_df[metric].mean(),
                    "std": full_state_df[metric].std(),
                    "min": full_state_df[metric].min(),
                    "max": full_state_df[metric].max(),
                    "n_pairs": len(full_state_df),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_run_config(args: argparse.Namespace, cfg: dict[str, Any], out_path: Path) -> None:
    serializable = {
        "args": vars(args),
        "resolved_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cfg.items()
        },
    }
    serializable["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in serializable["args"].items()
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_analysis_config(args)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    save_run_config(args, cfg, cfg["output_dir"] / "run_config.json")

    pca_components = sorted(set(parse_csv_ints(args.pca_components)))
    max_pcs = max(pca_components)
    condition_cols = parse_condition_cols(args.condition_cols)

    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    seed_dirs = get_seed_dirs(cfg["model_dir"], args.seeds, cfg["checkpoint_name"])
    first_hp = pca_impl.load_hp(seed_dirs[0])
    batch_size = int(first_hp.get("batch_size", 256))
    csvs = pca_impl.list_eval_csvs(cfg)
    dataset = pca_impl.HelicopterPCADataset(
        csvs,
        int(first_hp["n_input"]),
        int(first_hp["n_null_timesteps"]),
    )
    n_time = infer_n_time(dataset)
    _, condition_to_offset, condition_time_df = build_condition_index(
        dataset,
        condition_cols,
        n_time,
    )
    condition_time_df.to_csv(cfg["output_dir"] / "condition_time_points.csv", index=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pca_impl.collate_batch,
    )
    total_state_rows = len(dataset) * n_time
    sample_indices = choose_state_sample_indices(total_state_rows, args.max_state_rows, args.seed)
    pd.DataFrame({"global_trial_time_index": sample_indices}).to_csv(
        cfg["output_dir"] / "full_state_sample_indices.csv",
        index=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = pca_impl.import_model_class(cfg["model_root"], cfg["model_class"])

    print(f"Using device: {device}")
    print(f"Loaded {len(dataset)} trials from {len(csvs)} {cfg['variant_split']} CSVs")
    print(f"Comparing seeds: {', '.join(path.name for path in seed_dirs)}")
    print(f"Encoded timesteps per trial: {n_time}")
    print(f"Condition columns: {', '.join(condition_cols)}")
    print(f"Condition-time points: {len(condition_time_df)}")
    print(f"Full-state sampled rows: {len(sample_indices)} of {total_state_rows}")

    representations = []
    for seed_dir in seed_dirs:
        hp = pca_impl.load_hp(seed_dir)
        if int(hp["n_input"]) != int(first_hp["n_input"]):
            raise ValueError(f"{seed_dir.name} has different n_input than {seed_dirs[0].name}")
        if int(hp["n_null_timesteps"]) != int(first_hp["n_null_timesteps"]):
            raise ValueError(
                f"{seed_dir.name} has different n_null_timesteps than {seed_dirs[0].name}"
            )
        representations.append(
            collect_seed_representation(
                model_cls=model_cls,
                seed_dir=seed_dir,
                dataset=dataset,
                dataloader=dataloader,
                cfg=cfg,
                device=device,
                condition_cols=condition_cols,
                condition_to_offset=condition_to_offset,
                n_condition_time=len(condition_time_df),
                n_time=n_time,
                sample_indices=sample_indices,
                max_pcs=max_pcs,
                rdm_metric=args.rdm_metric,
            )
        )

    write_seed_pca_variance(
        representations,
        cfg["output_dir"] / "seed_pca_explained_variance.csv",
    )
    procrustes_df, rsa_df, full_state_df, combined_df = compute_pairwise_metrics(
        representations=representations,
        pca_components=pca_components,
        rsa_correlation=args.rsa_correlation,
        rdm_metric=args.rdm_metric,
        svcca_var_threshold=args.svcca_var_threshold,
        skip_svcca=args.no_svcca,
    )

    procrustes_path = cfg["output_dir"] / "procrustes_pca_trajectory_distance.csv"
    rsa_path = cfg["output_dir"] / "rsa_rdm_correlation.csv"
    full_state_path = cfg["output_dir"] / "full_state_linear_similarity.csv"
    combined_path = cfg["output_dir"] / "combined_pairwise_metrics.csv"
    summary_path = cfg["output_dir"] / "metric_summary.csv"

    procrustes_df.to_csv(procrustes_path, index=False)
    rsa_df.to_csv(rsa_path, index=False)
    full_state_df.to_csv(full_state_path, index=False)
    combined_df.to_csv(combined_path, index=False)
    write_summary(procrustes_df, rsa_df, full_state_df, summary_path)

    print(f"Saved Procrustes PCA distances to {procrustes_path}")
    print(f"Saved RSA/RDM correlations to {rsa_path}")
    print(f"Saved full-state linear metrics to {full_state_path}")
    print(f"Saved combined pairwise metrics to {combined_path}")
    print(f"Saved metric summary to {summary_path}")


if __name__ == "__main__":
    main()

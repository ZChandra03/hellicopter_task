#!/usr/bin/env python3
"""Compare readout heads and hidden-state PCs across two model roots.

The most portable comparison is the within-root head/PC alignment profile:
fit PCA separately for each root, compare each readout head with PC ranks inside
that root, then compare those scalar profiles across roots. The script also
writes direct head/head, PC/PC, and cross-root head/PC vector comparisons; those
direct vector comparisons assume both roots use compatible hidden-unit
coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

from pca_checkpoint_ep010 import (
    BASE_DIR,
    CHECKPOINT_NAME,
    DEFAULT_CONFIG,
    DEFAULT_MODEL_CLASS,
    DEFAULT_VARIANT_SPLIT,
    HelicopterPCADataset,
    collate_batch,
    get_seed_dir,
    import_model_class,
    infer_model_label,
    list_eval_csvs,
    load_hp,
    load_model,
)


SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
HEADS = {
    "report": "loc_head",
    "hazard": "haz_head",
}


@dataclass(frozen=True)
class ModelSpec:
    role: str
    label: str
    root: Path
    code_class: type


@dataclass
class ModelBasis:
    spec: ModelSpec
    seed: int
    pca: IncrementalPCA
    pca_sample_count: int
    heads: dict[str, np.ndarray]

    @property
    def hidden_dim(self) -> int:
        return int(self.pca.components_.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit hidden-state PCA for model_root and model_comp_root, then compare "
            "within-root head/PC profiles, activation-space similarity, direct "
            "readout heads, PC directions, cross head/PC alignments, and top-k "
            "PC subspaces."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config containing model_root/model_comp_root/variant_root. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["1"],
        help="Seed numbers to compare in both roots, or 'all'. Default: 1",
    )
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        default=None,
        help="Optional seed numbers for model_root only, or 'all'. Defaults to --seeds.",
    )
    parser.add_argument(
        "--model-comp-seeds",
        nargs="+",
        default=None,
        help="Optional seed numbers for model_comp_root only, or 'all'. Defaults to --seeds.",
    )
    parser.add_argument(
        "--pairing",
        choices=["matched", "all-pairs"],
        default="matched",
        help="Compare matching seed numbers or every model/comp seed pair. Default: matched",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of leading PCs to fit and compare. Default: 3",
    )
    parser.add_argument(
        "--fit-timestep-mode",
        choices=["all", "final"],
        default="all",
        help="Hidden states used to fit PCA. Default: all",
    )
    parser.add_argument(
        "--representation-timestep-mode",
        choices=["final", "all", "none"],
        default="final",
        help=(
            "Matched hidden-state rows used for CKA/SVCCA representation similarity. "
            "Use 'none' to skip. Default: final"
        ),
    )
    parser.add_argument(
        "--max-representation-samples",
        type=int,
        default=50000,
        help="Reservoir sample size for activation-space metrics. Default: 50000",
    )
    parser.add_argument(
        "--svcca-components",
        type=int,
        default=20,
        help="Maximum sample-space components for SVCCA-like correlations. Default: 20",
    )
    parser.add_argument(
        "--variant-split",
        default=DEFAULT_VARIANT_SPLIT,
        help=f"Variant split prefix to load. Default: {DEFAULT_VARIANT_SPLIT}",
    )
    parser.add_argument(
        "--max-variant-csvs",
        type=int,
        default=None,
        help="Limit number of variant CSVs for faster exploratory runs.",
    )
    parser.add_argument(
        "--model-class",
        default=DEFAULT_MODEL_CLASS,
        help=f"Model class to import from each rnn_models.py. Default: {DEFAULT_MODEL_CLASS}",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=CHECKPOINT_NAME,
        help=f"Checkpoint file in each seed directory. Default: {CHECKPOINT_NAME}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "model_root_comparison",
        help="Directory for CSVs and plots. Default: ./model_root_comparison",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Plot resolution. Default: 250",
    )
    return parser.parse_args()


def load_compare_config(args: argparse.Namespace) -> dict[str, Any]:
    with args.config.expanduser().resolve().open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    required = ["model_root", "model_comp_root", "variant_root"]
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    model_root = Path(cfg["model_root"]).expanduser().resolve()
    model_comp_root = Path(cfg["model_comp_root"]).expanduser().resolve()
    variant_root = Path(cfg["variant_root"]).expanduser().resolve()
    variant_subdir = cfg.get("variant_subdir") or cfg.get("sigma") or model_root.name

    cfg.update(
        {
            "config_path": args.config.expanduser().resolve(),
            "model_root": model_root,
            "model_comp_root": model_comp_root,
            "variant_root": variant_root,
            "model_label": cfg.get("model_label")
            or cfg.get("model_subdir")
            or infer_model_label(model_root),
            "model_comp_label": cfg.get("model_comp_label")
            or cfg.get("model_comp_subdir")
            or infer_model_label(model_comp_root),
            "variant_subdir": variant_subdir,
            "variant_split": args.variant_split,
            "max_variant_csvs": args.max_variant_csvs,
            "model_class": args.model_class,
            "checkpoint_name": args.checkpoint_name,
            "n_components": args.n_components,
            "fit_timestep_mode": args.fit_timestep_mode,
            "representation_timestep_mode": args.representation_timestep_mode,
            "max_representation_samples": args.max_representation_samples,
            "svcca_components": args.svcca_components,
            "output_dir": args.output_dir.expanduser().resolve(),
            "dpi": args.dpi,
        }
    )
    cfg["variant_dir"] = variant_root / variant_subdir
    return cfg


def available_seeds(model_dir: Path, checkpoint_name: str) -> list[int]:
    seeds = []
    for seed_dir in sorted(model_dir.glob("seed_*")):
        match = SEED_DIR_RE.fullmatch(seed_dir.name)
        if match and (seed_dir / checkpoint_name).is_file():
            seeds.append(int(match.group(1)))
    return seeds


def parse_seed_spec(seed_args: list[str], model_dir: Path, checkpoint_name: str) -> list[int]:
    if len(seed_args) == 1 and seed_args[0].lower() == "all":
        seeds = available_seeds(model_dir, checkpoint_name)
        if not seeds:
            raise FileNotFoundError(
                f"No seed directories with {checkpoint_name} found in {model_dir}"
            )
        return seeds

    seeds = []
    for raw in seed_args:
        try:
            seed = int(raw)
        except ValueError as exc:
            raise ValueError("Seed specs must be integers or the single value 'all'") from exc
        seeds.append(seed)
    return list(dict.fromkeys(seeds))


def build_seed_pairs(
    model_seeds: list[int],
    comp_seeds: list[int],
    pairing: str,
) -> list[tuple[int, int]]:
    if pairing == "all-pairs":
        return [(seed_a, seed_b) for seed_a in model_seeds for seed_b in comp_seeds]
    if pairing != "matched":
        raise ValueError(f"Unsupported pairing: {pairing}")

    comp_seed_set = set(comp_seeds)
    pairs = [(seed, seed) for seed in model_seeds if seed in comp_seed_set]
    if not pairs:
        raise ValueError(
            "No matching seed numbers were found across model_root and model_comp_root. "
            "Use --pairing all-pairs or pass overlapping seeds."
        )
    return pairs


def select_pca_samples(hidden: np.ndarray, mode: str) -> np.ndarray:
    if mode == "all":
        return hidden.reshape(-1, hidden.shape[-1])
    if mode == "final":
        return hidden[:, -1, :]
    raise ValueError(f"Unsupported fit-timestep-mode: {mode}")


@torch.inference_mode()
def fit_hidden_pca(
    model,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[IncrementalPCA, int]:
    n_components = int(cfg["n_components"])
    pca = IncrementalPCA(n_components=n_components)
    rows_seen = 0

    for x, _ in dataloader:
        hidden = model.rnn(x.to(device)).detach().cpu().numpy()
        samples = select_pca_samples(hidden, str(cfg["fit_timestep_mode"]))
        if samples.shape[0] < n_components:
            raise ValueError(
                f"PCA batch has {samples.shape[0]} samples, fewer than "
                f"n_components={n_components}. Increase batch size or use all timesteps."
            )
        pca.partial_fit(samples)
        rows_seen += int(samples.shape[0])

    if rows_seen < n_components:
        raise ValueError(f"Only {rows_seen} hidden states were available for PCA")
    return pca, rows_seen


def iter_head_vectors(model) -> Iterable[tuple[str, str, int, np.ndarray]]:
    for head_label, module_name in HEADS.items():
        if not hasattr(model, module_name):
            raise AttributeError(f"Model has no expected readout head: {module_name}")
        module = getattr(model, module_name)
        if not hasattr(module, "weight"):
            raise AttributeError(f"{module_name} has no weight tensor")

        weight = module.weight.detach().cpu().numpy()
        if weight.ndim == 1:
            weight = weight[None, :]
        elif weight.ndim != 2:
            raise ValueError(f"{module_name}.weight has unsupported shape {weight.shape}")

        for output_index, vector in enumerate(weight):
            label = head_label if weight.shape[0] == 1 else f"{head_label}_{output_index}"
            yield label, module_name, output_index, np.asarray(vector, dtype=float)


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 2:
        return math.nan

    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = math.sqrt(float(np.sum(a_centered**2) * np.sum(b_centered**2)))
    if denominator == 0:
        return math.nan
    return float(np.sum(a_centered * b_centered) / denominator)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return math.nan
    return float(np.dot(a, b) / denominator)


def compare_vectors(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compare vectors with shapes {a.shape} and {b.shape}")
    corr = pearson_r(a, b)
    cosine = cosine_similarity(a, b)
    return {
        "pearson_r": corr,
        "abs_pearson_r": abs(corr) if math.isfinite(corr) else math.nan,
        "cosine_similarity": cosine,
        "abs_cosine_similarity": abs(cosine) if math.isfinite(cosine) else math.nan,
        "vector_l2_norm_a": float(np.linalg.norm(a)),
        "vector_l2_norm_b": float(np.linalg.norm(b)),
    }


def analyze_basis(
    spec: ModelSpec,
    seed: int,
    csvs: list[Path],
    cfg: dict[str, Any],
    device: torch.device,
) -> ModelBasis:
    seed_dir = get_seed_dir(spec.root, seed, cfg["checkpoint_name"])
    hp = load_hp(seed_dir)
    batch_size = int(hp.get("batch_size", 256))
    dataset = HelicopterPCADataset(
        csvs,
        int(hp["n_input"]),
        int(hp["n_null_timesteps"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    model = load_model(spec.code_class, seed_dir, cfg["checkpoint_name"], device)

    print(
        f"{spec.role} seed_{seed}: {len(dataset)} trials, "
        f"n_input={hp['n_input']}, n_null_timesteps={hp['n_null_timesteps']}"
    )
    pca, pca_sample_count = fit_hidden_pca(model, dataloader, cfg, device)
    heads = {head: vector for head, _, _, vector in iter_head_vectors(model)}

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ModelBasis(spec=spec, seed=seed, pca=pca, pca_sample_count=pca_sample_count, heads=heads)


def center_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = center_columns(x)
    y = center_columns(y)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = float(np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro"))
    if denominator == 0:
        return math.nan
    return float(np.linalg.norm(xy, ord="fro") ** 2 / denominator)


def top_left_singular_vectors(x: np.ndarray, max_components: int) -> np.ndarray:
    x = center_columns(x)
    max_rank = min(x.shape[0] - 1, x.shape[1], max_components)
    if max_rank <= 0:
        return np.empty((x.shape[0], 0), dtype=float)

    u, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((x.shape[0], 0), dtype=float)

    tolerance = np.finfo(float).eps * max(x.shape) * float(singular_values[0])
    nonzero_count = int(np.sum(singular_values > tolerance))
    keep = min(max_rank, nonzero_count)
    return u[:, :keep]


def svcca_like_correlations(
    x: np.ndarray,
    y: np.ndarray,
    max_components: int,
) -> np.ndarray:
    ux = top_left_singular_vectors(x, max_components)
    uy = top_left_singular_vectors(y, max_components)
    keep = min(ux.shape[1], uy.shape[1])
    if keep <= 0:
        return np.asarray([], dtype=float)
    return np.linalg.svd(ux[:, :keep].T @ uy[:, :keep], compute_uv=False)


def orthogonal_procrustes_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.shape[1] != y.shape[1]:
        return {
            "orthogonal_procrustes_rms": math.nan,
            "orthogonal_procrustes_normalized_rms": math.nan,
            "orthogonal_procrustes_scale": math.nan,
        }

    x = center_columns(x)
    y = center_columns(y)
    u, singular_values, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    rotation = u @ vt
    x_energy = float(np.sum(x**2))
    scale = float(np.sum(singular_values) / x_energy) if x_energy > 0 else math.nan
    aligned = scale * (x @ rotation) if math.isfinite(scale) else x @ rotation
    rms = float(np.sqrt(np.mean((aligned - y) ** 2)))
    y_rms = float(np.sqrt(np.mean(y**2)))
    normalized = rms / y_rms if y_rms > 0 else math.nan
    return {
        "orthogonal_procrustes_rms": rms,
        "orthogonal_procrustes_normalized_rms": normalized,
        "orthogonal_procrustes_scale": scale,
    }


def append_pair_reservoir(
    samples_a: list[np.ndarray],
    samples_b: list[np.ndarray],
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    rows_seen: int,
    max_samples: int,
    rng: np.random.Generator,
) -> int:
    for row_a, row_b in zip(rows_a, rows_b):
        rows_seen += 1
        if len(samples_a) < max_samples:
            samples_a.append(np.asarray(row_a, dtype=np.float32).copy())
            samples_b.append(np.asarray(row_b, dtype=np.float32).copy())
            continue

        replace_index = int(rng.integers(0, rows_seen))
        if replace_index < max_samples:
            samples_a[replace_index] = np.asarray(row_a, dtype=np.float32).copy()
            samples_b[replace_index] = np.asarray(row_b, dtype=np.float32).copy()
    return rows_seen


@torch.inference_mode()
def collect_matched_hidden_samples(
    model_basis: ModelBasis,
    comp_basis: ModelBasis,
    csvs: list[Path],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    seed_dir_model = get_seed_dir(
        model_basis.spec.root,
        model_basis.seed,
        cfg["checkpoint_name"],
    )
    seed_dir_comp = get_seed_dir(
        comp_basis.spec.root,
        comp_basis.seed,
        cfg["checkpoint_name"],
    )
    hp_model = load_hp(seed_dir_model)
    hp_comp = load_hp(seed_dir_comp)
    batch_size = min(int(hp_model.get("batch_size", 256)), int(hp_comp.get("batch_size", 256)))

    dataset_model = HelicopterPCADataset(
        csvs,
        int(hp_model["n_input"]),
        int(hp_model["n_null_timesteps"]),
    )
    dataset_comp = HelicopterPCADataset(
        csvs,
        int(hp_comp["n_input"]),
        int(hp_comp["n_null_timesteps"]),
    )
    if len(dataset_model) != len(dataset_comp):
        raise ValueError(
            f"Matched representation comparison needs equal trial counts, got "
            f"{len(dataset_model)} and {len(dataset_comp)}"
        )

    loader_model = DataLoader(
        dataset_model,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    loader_comp = DataLoader(
        dataset_comp,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    model = load_model(
        model_basis.spec.code_class,
        seed_dir_model,
        cfg["checkpoint_name"],
        device,
    )
    comp_model = load_model(
        comp_basis.spec.code_class,
        seed_dir_comp,
        cfg["checkpoint_name"],
        device,
    )

    samples_model: list[np.ndarray] = []
    samples_comp: list[np.ndarray] = []
    rows_seen = 0
    rng = np.random.default_rng(0)
    mode = str(cfg["representation_timestep_mode"])
    max_samples = int(cfg["max_representation_samples"])

    for (x_model, idx_model), (x_comp, idx_comp) in zip(loader_model, loader_comp):
        if not np.array_equal(idx_model.numpy(), idx_comp.numpy()):
            raise ValueError("Dataloader trial order diverged during representation sampling")

        hidden_model = model.rnn(x_model.to(device)).detach().cpu().numpy()
        hidden_comp = comp_model.rnn(x_comp.to(device)).detach().cpu().numpy()
        rows_model = select_pca_samples(hidden_model, mode)
        rows_comp = select_pca_samples(hidden_comp, mode)
        if rows_model.shape[0] != rows_comp.shape[0]:
            raise ValueError(
                "Representation rows do not line up. This can happen when using "
                "--representation-timestep-mode all with models that have different "
                "sequence lengths; try --representation-timestep-mode final."
            )

        rows_seen = append_pair_reservoir(
            samples_model,
            samples_comp,
            rows_model,
            rows_comp,
            rows_seen,
            max_samples,
            rng,
        )

    del model, comp_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not samples_model:
        raise ValueError("No hidden-state rows were collected for representation comparison")

    return np.vstack(samples_model), np.vstack(samples_comp), rows_seen


def compute_activation_similarity_rows(
    model_basis: ModelBasis,
    comp_basis: ModelBasis,
    csvs: list[Path],
    cfg: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    if cfg["representation_timestep_mode"] == "none":
        return []

    x, y, rows_seen = collect_matched_hidden_samples(
        model_basis,
        comp_basis,
        csvs,
        cfg,
        device,
    )
    canonical_corrs = svcca_like_correlations(
        x,
        y,
        int(cfg["svcca_components"]),
    )
    procrustes = orthogonal_procrustes_metrics(x, y)

    return [
        {
            **basis_meta("model", model_basis),
            **basis_meta("comp", comp_basis),
            "checkpoint": cfg["checkpoint_name"],
            "representation_timestep_mode": cfg["representation_timestep_mode"],
            "rows_seen_before_sampling": rows_seen,
            "sample_count": int(x.shape[0]),
            "model_hidden_dim": int(x.shape[1]),
            "comp_hidden_dim": int(y.shape[1]),
            "linear_cka": linear_cka(x, y),
            "svcca_component_count": int(canonical_corrs.size),
            "svcca_mean_corr": (
                float(np.mean(canonical_corrs)) if canonical_corrs.size else math.nan
            ),
            "svcca_median_corr": (
                float(np.median(canonical_corrs)) if canonical_corrs.size else math.nan
            ),
            "svcca_top_corr": (
                float(np.max(canonical_corrs)) if canonical_corrs.size else math.nan
            ),
            "svcca_min_corr": (
                float(np.min(canonical_corrs)) if canonical_corrs.size else math.nan
            ),
            "metric_note": (
                "CKA and SVCCA compare matched activation rows and are invariant "
                "to hidden-unit permutations/orthogonal rotations. Procrustes "
                "fits the best orthogonal map when hidden dimensions match."
            ),
            **procrustes,
        }
    ]


def basis_meta(prefix: str, basis: ModelBasis) -> dict[str, Any]:
    return {
        f"{prefix}_role": basis.spec.role,
        f"{prefix}_label": basis.spec.label,
        f"{prefix}_root": str(basis.spec.root),
        f"{prefix}_seed": basis.seed,
        f"{prefix}_pca_sample_count": basis.pca_sample_count,
    }


def compare_head_heads(a: ModelBasis, b: ModelBasis, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for head_a, vector_a in a.heads.items():
        for head_b, vector_b in b.heads.items():
            rows.append(
                {
                    **basis_meta("model", a),
                    **basis_meta("comp", b),
                    "checkpoint": cfg["checkpoint_name"],
                    "fit_timestep_mode": cfg["fit_timestep_mode"],
                    "coordinate_assumption": (
                        "direct vector comparison assumes hidden units share a coordinate system"
                    ),
                    "head_model": head_a,
                    "head_comp": head_b,
                    **compare_vectors(vector_a, vector_b),
                }
            )
    return rows


def compare_pc_pcs(a: ModelBasis, b: ModelBasis, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    components_a = np.asarray(a.pca.components_, dtype=float)
    components_b = np.asarray(b.pca.components_, dtype=float)
    if components_a.shape[1] != components_b.shape[1]:
        raise ValueError(
            f"Cannot compare PCs with hidden dims {components_a.shape[1]} and {components_b.shape[1]}"
        )

    for pc_a, vector_a in enumerate(components_a, start=1):
        for pc_b, vector_b in enumerate(components_b, start=1):
            rows.append(
                {
                    **basis_meta("model", a),
                    **basis_meta("comp", b),
                    "checkpoint": cfg["checkpoint_name"],
                    "fit_timestep_mode": cfg["fit_timestep_mode"],
                    "coordinate_assumption": (
                        "direct vector comparison assumes hidden units share a coordinate system"
                    ),
                    "pc_model": pc_a,
                    "pc_comp": pc_b,
                    "model_explained_variance_ratio": float(
                        a.pca.explained_variance_ratio_[pc_a - 1]
                    ),
                    "comp_explained_variance_ratio": float(
                        b.pca.explained_variance_ratio_[pc_b - 1]
                    ),
                    **compare_vectors(vector_a, vector_b),
                }
            )
    return rows


def head_pc_rows(
    head_basis: ModelBasis,
    pc_basis: ModelBasis,
    cfg: dict[str, Any],
    comparison: str,
) -> list[dict[str, Any]]:
    rows = []
    components = np.asarray(pc_basis.pca.components_, dtype=float)
    for head, head_vector in head_basis.heads.items():
        if head_vector.shape[0] != components.shape[1]:
            raise ValueError(
                f"{head_basis.spec.role} {head} head dim {head_vector.shape[0]} does not match "
                f"{pc_basis.spec.role} PC dim {components.shape[1]}"
            )

        for pc_index, pc_vector in enumerate(components, start=1):
            rows.append(
                {
                    "comparison": comparison,
                    **basis_meta("head", head_basis),
                    **basis_meta("pc", pc_basis),
                    "checkpoint": cfg["checkpoint_name"],
                    "fit_timestep_mode": cfg["fit_timestep_mode"],
                    "coordinate_assumption": (
                        "direct vector comparison assumes hidden units share a coordinate system"
                    ),
                    "head": head,
                    "pc": pc_index,
                    "pc_explained_variance_ratio": float(
                        pc_basis.pca.explained_variance_ratio_[pc_index - 1]
                    ),
                    **compare_vectors(head_vector, pc_vector),
                }
            )
    return rows


def compare_pc_subspaces(a: ModelBasis, b: ModelBasis, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    components_a = np.asarray(a.pca.components_, dtype=float)
    components_b = np.asarray(b.pca.components_, dtype=float)
    if components_a.shape[1] != components_b.shape[1]:
        raise ValueError(
            f"Cannot compare PC subspaces with hidden dims {components_a.shape[1]} "
            f"and {components_b.shape[1]}"
        )

    max_k = min(components_a.shape[0], components_b.shape[0])
    for k in range(1, max_k + 1):
        # PCA components are row-orthonormal; singular values of Qa^T Qb are
        # cosines of the principal angles between the top-k subspaces.
        qa = components_a[:k].T
        qb = components_b[:k].T
        singular_values = np.linalg.svd(qa.T @ qb, compute_uv=False)
        singular_values = np.clip(singular_values, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(singular_values))

        for angle_index, (singular_value, angle_deg) in enumerate(
            zip(singular_values, angles_deg),
            start=1,
        ):
            rows.append(
                {
                    **basis_meta("model", a),
                    **basis_meta("comp", b),
                    "checkpoint": cfg["checkpoint_name"],
                    "fit_timestep_mode": cfg["fit_timestep_mode"],
                    "coordinate_assumption": (
                        "principal angles compare PC subspaces in the hidden-unit coordinate system"
                    ),
                    "subspace_k": k,
                    "angle_index": angle_index,
                    "singular_value": float(singular_value),
                    "principal_angle_deg": float(angle_deg),
                    "mean_singular_value_for_k": float(np.mean(singular_values)),
                    "max_principal_angle_deg_for_k": float(np.max(angles_deg)),
                    "mean_principal_angle_deg_for_k": float(np.mean(angles_deg)),
                }
            )
    return rows


def compare_head_pc_alignment_profiles(
    within_head_pc: pd.DataFrame,
    seed_pairs: list[tuple[int, int]],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for seed_model, seed_comp in seed_pairs:
        model_df = within_head_pc[
            (within_head_pc["head_role"] == "model_root")
            & (within_head_pc["head_seed"] == seed_model)
        ].copy()
        comp_df = within_head_pc[
            (within_head_pc["head_role"] == "model_comp_root")
            & (within_head_pc["head_seed"] == seed_comp)
        ].copy()
        if model_df.empty or comp_df.empty:
            continue

        merged = model_df.merge(
            comp_df,
            on=["head", "pc"],
            suffixes=("_model", "_comp"),
            how="inner",
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "model_seed": seed_model,
                    "comp_seed": seed_comp,
                    "checkpoint": cfg["checkpoint_name"],
                    "fit_timestep_mode": cfg["fit_timestep_mode"],
                    "head": row["head"],
                    "pc": int(row["pc"]),
                    "model_abs_cosine_similarity": float(
                        row["abs_cosine_similarity_model"]
                    ),
                    "comp_abs_cosine_similarity": float(
                        row["abs_cosine_similarity_comp"]
                    ),
                    "delta_abs_cosine_similarity": float(
                        row["abs_cosine_similarity_comp"]
                        - row["abs_cosine_similarity_model"]
                    ),
                    "abs_delta_abs_cosine_similarity": float(
                        abs(
                            row["abs_cosine_similarity_comp"]
                            - row["abs_cosine_similarity_model"]
                        )
                    ),
                    "model_cosine_similarity": float(row["cosine_similarity_model"]),
                    "comp_cosine_similarity": float(row["cosine_similarity_comp"]),
                    "delta_cosine_similarity": float(
                        row["cosine_similarity_comp"] - row["cosine_similarity_model"]
                    ),
                    "model_abs_pearson_r": float(row["abs_pearson_r_model"]),
                    "comp_abs_pearson_r": float(row["abs_pearson_r_comp"]),
                    "delta_abs_pearson_r": float(
                        row["abs_pearson_r_comp"] - row["abs_pearson_r_model"]
                    ),
                    "abs_delta_abs_pearson_r": float(
                        abs(row["abs_pearson_r_comp"] - row["abs_pearson_r_model"])
                    ),
                    "model_pc_explained_variance_ratio": float(
                        row["pc_explained_variance_ratio_model"]
                    ),
                    "comp_pc_explained_variance_ratio": float(
                        row["pc_explained_variance_ratio_comp"]
                    ),
                    "delta_pc_explained_variance_ratio": float(
                        row["pc_explained_variance_ratio_comp"]
                        - row["pc_explained_variance_ratio_model"]
                    ),
                }
            )

    profile_deltas = pd.DataFrame(rows)
    if profile_deltas.empty:
        return profile_deltas, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for (seed_model, seed_comp, head), group in profile_deltas.groupby(
        ["model_seed", "comp_seed", "head"],
        sort=True,
    ):
        ordered = group.sort_values("pc")
        model_profile = ordered["model_abs_cosine_similarity"].to_numpy(dtype=float)
        comp_profile = ordered["comp_abs_cosine_similarity"].to_numpy(dtype=float)
        profile_corr = pearson_r(model_profile, comp_profile)
        model_best_row = ordered.loc[ordered["model_abs_cosine_similarity"].idxmax()]
        comp_best_row = ordered.loc[ordered["comp_abs_cosine_similarity"].idxmax()]
        summary_rows.append(
            {
                "model_seed": seed_model,
                "comp_seed": seed_comp,
                "checkpoint": cfg["checkpoint_name"],
                "fit_timestep_mode": cfg["fit_timestep_mode"],
                "head": head,
                "n_pcs": int(len(ordered)),
                "abs_cosine_profile_pearson_r": profile_corr,
                "mean_abs_delta_abs_cosine_similarity": float(
                    ordered["abs_delta_abs_cosine_similarity"].mean()
                ),
                "max_abs_delta_abs_cosine_similarity": float(
                    ordered["abs_delta_abs_cosine_similarity"].max()
                ),
                "model_best_pc": int(model_best_row["pc"]),
                "comp_best_pc": int(comp_best_row["pc"]),
                "same_best_pc": int(model_best_row["pc"] == comp_best_row["pc"]),
                "model_best_abs_cosine_similarity": float(
                    model_best_row["model_abs_cosine_similarity"]
                ),
                "comp_best_abs_cosine_similarity": float(
                    comp_best_row["comp_abs_cosine_similarity"]
                ),
            }
        )

    return profile_deltas, pd.DataFrame(summary_rows)


def ordered_pivot(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    row_order: list[Any],
    col_order: list[Any],
) -> pd.DataFrame:
    matrix = df.pivot_table(
        index=row_col,
        columns=col_col,
        values=value_col,
        aggfunc="mean",
        sort=False,
    )
    return matrix.reindex(index=row_order, columns=col_order)


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    colorbar_label: str,
    dpi: int,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> None:
    if matrix.empty:
        return

    fig_width = max(5.8, 1.05 * matrix.shape[1] + 2.8)
    fig_height = max(3.4, 0.55 * matrix.shape[0] + 1.9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.set_title(title)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix.iat[row_index, col_index]
            if pd.notna(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{float(value):.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if float(value) < 0.55 else "white",
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plots(
    output_dir: Path,
    head_head: pd.DataFrame,
    pc_pc: pd.DataFrame,
    within_head_pc: pd.DataFrame,
    cross_head_pc: pd.DataFrame,
    profile_deltas: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    dpi = int(cfg["dpi"])

    if not pc_pc.empty:
        pc_plot = pc_pc.copy()
        pc_plot["row_label"] = "model PC" + pc_plot["pc_model"].astype(str)
        pc_plot["col_label"] = "comp PC" + pc_plot["pc_comp"].astype(str)
        pc_rows = [f"model PC{i}" for i in sorted(pc_plot["pc_model"].unique())]
        pc_cols = [f"comp PC{i}" for i in sorted(pc_plot["pc_comp"].unique())]
        plot_heatmap(
            ordered_pivot(
                pc_plot,
                "row_label",
                "col_label",
                "abs_cosine_similarity",
                pc_rows,
                pc_cols,
            ),
            output_dir / "pc_pc_abs_cosine_heatmap.png",
            "PC direction alignment across roots",
            "mean abs(cosine)",
            dpi,
        )

    if not head_head.empty:
        head_plot = head_head.copy()
        head_plot["row_label"] = "model " + head_plot["head_model"].astype(str)
        head_plot["col_label"] = "comp " + head_plot["head_comp"].astype(str)
        head_rows = list(dict.fromkeys(head_plot["row_label"].tolist()))
        head_cols = list(dict.fromkeys(head_plot["col_label"].tolist()))
        plot_heatmap(
            ordered_pivot(
                head_plot,
                "row_label",
                "col_label",
                "abs_cosine_similarity",
                head_rows,
                head_cols,
            ),
            output_dir / "head_head_abs_cosine_heatmap.png",
            "Readout head alignment across roots",
            "mean abs(cosine)",
            dpi,
        )

    if not profile_deltas.empty:
        profile_plot = profile_deltas.copy()
        profile_plot["row_label"] = profile_plot["head"].astype(str)
        profile_plot["col_label"] = "PC" + profile_plot["pc"].astype(str)
        profile_rows = list(dict.fromkeys(profile_plot["row_label"].tolist()))
        profile_cols = [f"PC{i}" for i in sorted(profile_plot["pc"].unique())]
        plot_heatmap(
            ordered_pivot(
                profile_plot,
                "row_label",
                "col_label",
                "abs_delta_abs_cosine_similarity",
                profile_rows,
                profile_cols,
            ),
            output_dir / "head_pc_alignment_profile_abs_delta_heatmap.png",
            "Head/PC alignment profile difference across roots",
            "mean abs delta abs(cosine)",
            dpi,
            vmin=0.0,
            vmax=max(
                0.01,
                float(profile_plot["abs_delta_abs_cosine_similarity"].max()),
            ),
            cmap="magma",
        )

    if not within_head_pc.empty:
        within_plot = within_head_pc.copy()
        within_plot["row_label"] = within_plot["head_role"] + " " + within_plot["head"]
        within_plot["col_label"] = "PC" + within_plot["pc"].astype(str)
        within_rows = list(dict.fromkeys(within_plot["row_label"].tolist()))
        within_cols = [f"PC{i}" for i in sorted(within_plot["pc"].unique())]
        plot_heatmap(
            ordered_pivot(
                within_plot,
                "row_label",
                "col_label",
                "abs_cosine_similarity",
                within_rows,
                within_cols,
            ),
            output_dir / "within_root_head_pc_abs_cosine_heatmap.png",
            "Within-root head/PC alignment",
            "mean abs(cosine)",
            dpi,
        )

    if not cross_head_pc.empty:
        for comparison, group in cross_head_pc.groupby("comparison", sort=False):
            plot_df = group.copy()
            plot_df["row_label"] = plot_df["head_role"] + " " + plot_df["head"]
            plot_df["col_label"] = plot_df["pc_role"] + " PC" + plot_df["pc"].astype(str)
            row_order = list(dict.fromkeys(plot_df["row_label"].tolist()))
            col_order = list(dict.fromkeys(plot_df["col_label"].tolist()))
            safe_name = comparison.replace(" ", "_").replace("/", "_")
            plot_heatmap(
                ordered_pivot(
                    plot_df,
                    "row_label",
                    "col_label",
                    "abs_cosine_similarity",
                    row_order,
                    col_order,
                ),
                output_dir / f"{safe_name}_abs_cosine_heatmap.png",
                comparison.replace("_", " "),
                "mean abs(cosine)",
                dpi,
            )


def write_run_config(
    output_path: Path,
    cfg: dict[str, Any],
    model_seeds: list[int],
    comp_seeds: list[int],
    seed_pairs: list[tuple[int, int]],
    csvs: list[Path],
) -> None:
    serializable = {
        "config_path": str(cfg["config_path"]),
        "model_root": str(cfg["model_root"]),
        "model_comp_root": str(cfg["model_comp_root"]),
        "model_label": cfg["model_label"],
        "model_comp_label": cfg["model_comp_label"],
        "variant_dir": str(cfg["variant_dir"]),
        "variant_split": cfg["variant_split"],
        "max_variant_csvs": cfg["max_variant_csvs"],
        "variant_csv_count": len(csvs),
        "variant_csvs": [path.name for path in csvs],
        "model_class": cfg["model_class"],
        "checkpoint_name": cfg["checkpoint_name"],
        "n_components": cfg["n_components"],
        "fit_timestep_mode": cfg["fit_timestep_mode"],
        "representation_timestep_mode": cfg["representation_timestep_mode"],
        "max_representation_samples": cfg["max_representation_samples"],
        "svcca_components": cfg["svcca_components"],
        "model_seeds": model_seeds,
        "model_comp_seeds": comp_seeds,
        "seed_pairs": seed_pairs,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.n_components <= 0:
        raise ValueError("--n-components must be positive")
    if args.max_variant_csvs is not None and args.max_variant_csvs <= 0:
        raise ValueError("--max-variant-csvs must be positive when provided")
    if args.representation_timestep_mode != "none" and args.max_representation_samples <= 0:
        raise ValueError("--max-representation-samples must be positive")
    if args.representation_timestep_mode != "none" and args.svcca_components <= 0:
        raise ValueError("--svcca-components must be positive")

    cfg = load_compare_config(args)
    for key in ["model_root", "model_comp_root", "variant_dir"]:
        if not cfg[key].exists():
            raise FileNotFoundError(f"{key} does not exist: {cfg[key]}")

    model_seed_args = args.model_seeds or args.seeds
    comp_seed_args = args.model_comp_seeds or args.seeds
    model_seeds = parse_seed_spec(model_seed_args, cfg["model_root"], cfg["checkpoint_name"])
    comp_seeds = parse_seed_spec(comp_seed_args, cfg["model_comp_root"], cfg["checkpoint_name"])
    seed_pairs = build_seed_pairs(model_seeds, comp_seeds, args.pairing)

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    csv_cfg = {
        "variant_dir": cfg["variant_dir"],
        "variant_split": cfg["variant_split"],
        "max_variant_csvs": cfg["max_variant_csvs"],
    }
    csvs = list_eval_csvs(csv_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_spec = ModelSpec(
        role="model_root",
        label=cfg["model_label"],
        root=cfg["model_root"],
        code_class=import_model_class(cfg["model_root"], cfg["model_class"]),
    )
    comp_spec = ModelSpec(
        role="model_comp_root",
        label=cfg["model_comp_label"],
        root=cfg["model_comp_root"],
        code_class=import_model_class(cfg["model_comp_root"], cfg["model_class"]),
    )

    print(f"Using device: {device}")
    print(f"Loaded {len(csvs)} {cfg['variant_split']} CSVs from {cfg['variant_dir']}")
    print(f"Comparing seed pairs: {', '.join(f'{a}:{b}' for a, b in seed_pairs)}")
    print(f"Fitting PCA from {cfg['fit_timestep_mode']} hidden states")

    basis_cache: dict[tuple[str, int], ModelBasis] = {}
    for seed in sorted({seed for seed, _ in seed_pairs}):
        basis_cache[(model_spec.role, seed)] = analyze_basis(model_spec, seed, csvs, cfg, device)
    for seed in sorted({seed for _, seed in seed_pairs}):
        basis_cache[(comp_spec.role, seed)] = analyze_basis(comp_spec, seed, csvs, cfg, device)

    head_head_rows: list[dict[str, Any]] = []
    pc_pc_rows: list[dict[str, Any]] = []
    within_head_pc_rows: list[dict[str, Any]] = []
    cross_head_pc_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []

    seen_within: set[tuple[str, int]] = set()
    for seed_model, seed_comp in seed_pairs:
        model_basis = basis_cache[(model_spec.role, seed_model)]
        comp_basis = basis_cache[(comp_spec.role, seed_comp)]

        activation_rows.extend(
            compute_activation_similarity_rows(
                model_basis,
                comp_basis,
                csvs,
                cfg,
                device,
            )
        )

        if model_basis.hidden_dim == comp_basis.hidden_dim:
            head_head_rows.extend(compare_head_heads(model_basis, comp_basis, cfg))
            pc_pc_rows.extend(compare_pc_pcs(model_basis, comp_basis, cfg))
            cross_head_pc_rows.extend(
                head_pc_rows(
                    model_basis,
                    comp_basis,
                    cfg,
                    comparison="model_root_head_vs_model_comp_root_pc",
                )
            )
            cross_head_pc_rows.extend(
                head_pc_rows(
                    comp_basis,
                    model_basis,
                    cfg,
                    comparison="model_comp_root_head_vs_model_root_pc",
                )
            )
            subspace_rows.extend(compare_pc_subspaces(model_basis, comp_basis, cfg))
        else:
            print(
                f"Skipping direct vector/subspace comparisons for seed pair "
                f"{seed_model}:{seed_comp} because hidden dimensions differ "
                f"({model_basis.hidden_dim} vs {comp_basis.hidden_dim}). "
                "Activation-space metrics can still be used."
            )

        for basis in [model_basis, comp_basis]:
            cache_key = (basis.spec.role, basis.seed)
            if cache_key in seen_within:
                continue
            seen_within.add(cache_key)
            within_head_pc_rows.extend(
                head_pc_rows(
                    basis,
                    basis,
                    cfg,
                    comparison=f"{basis.spec.role}_head_vs_own_pc",
                )
            )

    output_dir = cfg["output_dir"]
    head_head = pd.DataFrame(head_head_rows)
    pc_pc = pd.DataFrame(pc_pc_rows)
    within_head_pc = pd.DataFrame(within_head_pc_rows)
    cross_head_pc = pd.DataFrame(cross_head_pc_rows)
    subspaces = pd.DataFrame(subspace_rows)
    activation_similarity = pd.DataFrame(activation_rows)
    profile_deltas, profile_summary = compare_head_pc_alignment_profiles(
        within_head_pc,
        seed_pairs,
        cfg,
    )

    head_head_path = output_dir / "head_head_comparisons.csv"
    pc_pc_path = output_dir / "pc_pc_comparisons.csv"
    within_head_pc_path = output_dir / "within_root_head_pc_comparisons.csv"
    cross_head_pc_path = output_dir / "cross_root_head_pc_comparisons.csv"
    subspace_path = output_dir / "pc_subspace_principal_angles.csv"
    activation_path = output_dir / "activation_representation_similarity.csv"
    profile_delta_path = output_dir / "head_pc_alignment_profile_deltas.csv"
    profile_summary_path = output_dir / "head_pc_alignment_profile_summary.csv"

    head_head.to_csv(head_head_path, index=False)
    pc_pc.to_csv(pc_pc_path, index=False)
    within_head_pc.to_csv(within_head_pc_path, index=False)
    cross_head_pc.to_csv(cross_head_pc_path, index=False)
    subspaces.to_csv(subspace_path, index=False)
    activation_similarity.to_csv(activation_path, index=False)
    profile_deltas.to_csv(profile_delta_path, index=False)
    profile_summary.to_csv(profile_summary_path, index=False)

    save_plots(
        output_dir,
        head_head,
        pc_pc,
        within_head_pc,
        cross_head_pc,
        profile_deltas,
        cfg,
    )
    write_run_config(output_dir / "run_config.json", cfg, model_seeds, comp_seeds, seed_pairs, csvs)

    print(f"Saved head/head comparisons to {head_head_path}")
    print(f"Saved PC/PC comparisons to {pc_pc_path}")
    print(f"Saved within-root head/PC comparisons to {within_head_pc_path}")
    print(f"Saved cross-root head/PC comparisons to {cross_head_pc_path}")
    print(f"Saved activation representation similarity to {activation_path}")
    print(f"Saved head/PC alignment profile deltas to {profile_delta_path}")
    print(f"Saved head/PC alignment profile summary to {profile_summary_path}")
    print(f"Saved PC subspace principal angles to {subspace_path}")
    print(f"Saved heatmaps and run config to {output_dir}")


if __name__ == "__main__":
    main()

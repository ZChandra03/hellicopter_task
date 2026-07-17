#!/usr/bin/env python3
"""Track readout-head correlations with hidden-state PCs across epochs.

For each requested seed and checkpoint_ep*.pt file, this script fits PCA to
that checkpoint's hidden states, compares each linear readout head with PC1-PC3,
and plots the correlations across training epochs.  Checkpoint aliases such as
checkpoint_best.pt and final.pt are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
    DEFAULT_CONFIG,
    DEFAULT_MODEL_CLASS,
    DEFAULT_VARIANT_SPLIT,
    HelicopterPCADataset,
    collate_batch,
    import_model_class,
    infer_model_label,
    list_eval_csvs,
    load_config,
    load_hp,
    load_model,
)


SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
CHECKPOINT_EPOCH_RE = re.compile(r"^checkpoint_ep(\d+)\.pt$")
DEFAULT_N_COMPONENTS = 5
DEFAULT_CORRELATION_PCS = 3
HEADS = {
    "report": "loc_head",
    "hazard": "haz_head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit hidden-state PCA at each epoch and plot how each trained "
            "readout head correlates with PC1-PC3 over training."
        )
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["all"],
        help="Seed numbers to analyze, or 'all'. Default: all",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        default=["all"],
        help="Epoch numbers to analyze, or 'all'. Default: all checkpoint_ep*.pt files",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help=(
            "Number of leading PCs to fit and save explained variance for. "
            f"Default: {DEFAULT_N_COMPONENTS}"
        ),
    )
    parser.add_argument(
        "--correlation-pcs",
        type=int,
        default=DEFAULT_CORRELATION_PCS,
        help=(
            "Number of leading PCs to compare with each head in correlation plots. "
            f"Default: {DEFAULT_CORRELATION_PCS}"
        ),
    )
    parser.add_argument(
        "--fit-timestep-mode",
        choices=["all", "final"],
        default="all",
        help="Hidden states used to fit PCA. Default: all",
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
        help=f"Model class to import from rnn_models.py. Default: {DEFAULT_MODEL_CLASS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "head_pc_correlations",
        help="Directory for CSVs and plots. Default: ./head_pc_correlations",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Plot resolution. Default: 250",
    )
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(DEFAULT_CONFIG)
    model_subdir = cfg.get("model_subdir") or infer_model_label(cfg["model_root"])
    variant_subdir = cfg.get("variant_subdir") or cfg.get("sigma") or cfg["model_root"].name
    cfg.update(
        {
            "model_subdir": model_subdir,
            "variant_subdir": variant_subdir,
            "variant_split": args.variant_split,
            "max_variant_csvs": args.max_variant_csvs,
            "model_class": args.model_class,
            "n_components": args.n_components,
            "correlation_pcs": args.correlation_pcs,
            "fit_timestep_mode": args.fit_timestep_mode,
            "output_dir": args.output_dir.expanduser().resolve(),
            "dpi": args.dpi,
        }
    )
    cfg["model_dir"] = cfg["model_root"]
    cfg["variant_dir"] = cfg["variant_root"] / cfg["variant_subdir"]
    return cfg


def discover_epoch_checkpoints(seed_dir: Path) -> list[tuple[int, str]]:
    checkpoints: list[tuple[int, str]] = []
    for path in seed_dir.glob("checkpoint_ep*.pt"):
        match = CHECKPOINT_EPOCH_RE.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path.name))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def parse_requested_seeds(seed_args: list[str], cfg: dict[str, Any]) -> list[int]:
    if len(seed_args) == 1 and seed_args[0].lower() == "all":
        seeds = []
        for seed_dir in sorted(cfg["model_dir"].glob("seed_*")):
            match = SEED_DIR_RE.fullmatch(seed_dir.name)
            if match and discover_epoch_checkpoints(seed_dir):
                seeds.append(int(match.group(1)))
        if not seeds:
            raise FileNotFoundError(
                f"No seed directories with checkpoint_ep*.pt files found in {cfg['model_dir']}"
            )
        return seeds

    seeds = []
    for raw in seed_args:
        try:
            seed = int(raw)
        except ValueError as exc:
            raise ValueError("--seeds must be integers or the single value 'all'") from exc
        seeds.append(seed)
    return list(dict.fromkeys(seeds))


def parse_requested_epochs(epoch_args: list[str], seed_dir: Path) -> list[tuple[int, str]]:
    available = discover_epoch_checkpoints(seed_dir)
    if not available:
        raise FileNotFoundError(f"No checkpoint_ep*.pt files found in {seed_dir}")

    if len(epoch_args) == 1 and epoch_args[0].lower() == "all":
        return available

    available_by_epoch = dict(available)
    selected: list[tuple[int, str]] = []
    missing = []
    for raw in epoch_args:
        try:
            epoch = int(raw)
        except ValueError as exc:
            raise ValueError("--epochs must be integers or the single value 'all'") from exc
        checkpoint_name = available_by_epoch.get(epoch)
        if checkpoint_name is None:
            missing.append(epoch)
        else:
            selected.append((epoch, checkpoint_name))

    if missing:
        missing_text = ", ".join(str(epoch) for epoch in missing)
        raise FileNotFoundError(f"{seed_dir.name} is missing requested epochs: {missing_text}")
    return list(dict.fromkeys(selected))


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
        raise ValueError(
            f"Only {rows_seen} hidden states were available for {n_components} PCs"
        )
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


def compute_head_pc_rows(
    model,
    pca: IncrementalPCA,
    seed: int,
    epoch: int,
    checkpoint_name: str,
    cfg: dict[str, Any],
    pca_sample_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    components = np.asarray(pca.components_, dtype=float)
    n_correlation_pcs = min(int(cfg["correlation_pcs"]), components.shape[0])

    for head_label, module_name, output_index, head_vector in iter_head_vectors(model):
        if head_vector.shape[0] != components.shape[1]:
            raise ValueError(
                f"{module_name} vector length {head_vector.shape[0]} does not match "
                f"PCA component length {components.shape[1]}"
            )

        for pc_index, pc_vector in enumerate(components[:n_correlation_pcs], start=1):
            corr = pearson_r(head_vector, pc_vector)
            cosine = cosine_similarity(head_vector, pc_vector)
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "checkpoint": checkpoint_name,
                    "model": cfg["model_subdir"],
                    "variant_split": cfg["variant_split"],
                    "fit_timestep_mode": cfg["fit_timestep_mode"],
                    "pca_sample_count": pca_sample_count,
                    "head": head_label,
                    "head_module": module_name,
                    "head_output_index": output_index,
                    "pc": pc_index,
                    "pearson_r": corr,
                    "abs_pearson_r": abs(corr) if math.isfinite(corr) else math.nan,
                    "cosine_similarity": cosine,
                    "abs_cosine_similarity": (
                        abs(cosine) if math.isfinite(cosine) else math.nan
                    ),
                    "head_l2_norm": float(np.linalg.norm(head_vector)),
                    "pc_l2_norm": float(np.linalg.norm(pc_vector)),
                }
            )
    return rows


def compute_variance_rows(
    pca: IncrementalPCA,
    seed: int,
    epoch: int,
    checkpoint_name: str,
    cfg: dict[str, Any],
    pca_sample_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for pc_index, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "checkpoint": checkpoint_name,
                "model": cfg["model_subdir"],
                "variant_split": cfg["variant_split"],
                "fit_timestep_mode": cfg["fit_timestep_mode"],
                "pca_sample_count": pca_sample_count,
                "pc": pc_index,
                "explained_variance_ratio": float(ratio),
            }
        )
    return rows


def plot_head_epoch_correlations(
    correlations: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_paths = []
    pc_values = sorted(correlations["pc"].unique())
    markers = ["o", "s", "^", "D", "v"]

    plot_specs = [
        (
            "pearson_r",
            "Pearson correlation",
            "correlation",
            "correlation with PC directions",
            (-1.0, 1.0),
            True,
        ),
        (
            "abs_pearson_r",
            "Absolute Pearson correlation",
            "abs_correlation",
            "absolute correlation with PC directions",
            (0.0, 1.0),
            False,
        ),
    ]

    for head, head_group in correlations.groupby("head", sort=True):
        seed_count = int(head_group["seed"].nunique())
        epochs = sorted(head_group["epoch"].unique())

        for metric, ylabel, file_metric, title_suffix, y_limits, show_zero in plot_specs:
            fig, ax = plt.subplots(figsize=(8.4, 5.2))

            for index, pc in enumerate(pc_values):
                pc_group = head_group[head_group["pc"] == pc]
                if pc_group.empty:
                    continue

                color = f"C{index}"
                for seed, seed_group in pc_group.groupby("seed", sort=True):
                    seed_group = seed_group.sort_values("epoch")
                    ax.plot(
                        seed_group["epoch"],
                        seed_group[metric],
                        color=color,
                        alpha=0.22,
                        linewidth=1.0,
                        marker=markers[index % len(markers)],
                        markersize=3,
                    )

                mean_group = (
                    pc_group.groupby("epoch", as_index=False)[metric]
                    .mean()
                    .sort_values("epoch")
                )
                ax.plot(
                    mean_group["epoch"],
                    mean_group[metric],
                    color=color,
                    marker=markers[index % len(markers)],
                    linewidth=3.0,
                    markersize=6,
                    label=f"PC{int(pc)} mean",
                )

            if show_zero:
                ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
            ax.set_xticks(epochs)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_ylim(*y_limits)
            ax.set_title(
                f"{head.capitalize()} head {title_suffix} "
                f"({seed_count} seeds; faint=seed, bold=mean)"
            )
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(frameon=False, title="PC")
            fig.tight_layout()

            output_path = output_dir / f"all_seeds_{head}_head_pc_{file_metric}_by_epoch.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(output_path)

    return output_paths


def plot_explained_variance_by_epoch(
    variance: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    output_paths = []
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    seed_count = int(variance["seed"].nunique())
    for index, pc in enumerate(sorted(variance["pc"].unique())):
        pc_group = variance[variance["pc"] == pc]
        color = f"C{index}"
        for _, seed_group in pc_group.groupby("seed", sort=True):
            seed_group = seed_group.sort_values("epoch")
            ax.plot(
                seed_group["epoch"],
                seed_group["explained_variance_ratio"],
                color=color,
                alpha=0.22,
                linewidth=1.0,
                marker=markers[index % len(markers)],
                markersize=3,
            )

        mean_group = (
            pc_group.groupby("epoch", as_index=False)["explained_variance_ratio"]
            .mean()
            .sort_values("epoch")
        )
        ax.plot(
            mean_group["epoch"],
            mean_group["explained_variance_ratio"],
            color=color,
            marker=markers[index % len(markers)],
            linewidth=3.0,
            markersize=6,
            label=f"PC{int(pc)} mean",
        )

    epochs = sorted(variance["epoch"].unique())
    observed_max = float(variance["explained_variance_ratio"].max())
    y_top = max(0.05, observed_max * 1.08)
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(bottom=0.0, top=y_top)
    ax.set_title(
        f"PCA explained variance over epochs ({seed_count} seeds; faint=seed, bold=mean)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, title="PC")
    fig.tight_layout()

    output_path = output_dir / "all_seeds_pca_explained_variance_by_epoch.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(output_path)

    return output_paths


def write_run_config(
    output_path: Path,
    cfg: dict[str, Any],
    seeds: list[int],
    epochs_by_seed: dict[int, list[int]],
    csvs: list[Path],
) -> None:
    serializable = {
        "model_root": str(cfg["model_root"]),
        "model_subdir": cfg["model_subdir"],
        "variant_dir": str(cfg["variant_dir"]),
        "variant_split": cfg["variant_split"],
        "max_variant_csvs": cfg["max_variant_csvs"],
        "model_class": cfg["model_class"],
        "seeds": seeds,
        "epochs_by_seed": {str(seed): epochs for seed, epochs in epochs_by_seed.items()},
        "n_components": cfg["n_components"],
        "correlation_pcs": cfg["correlation_pcs"],
        "fit_timestep_mode": cfg["fit_timestep_mode"],
        "variant_csv_count": len(csvs),
        "variant_csvs": [path.name for path in csvs],
        "checkpoint_filter": "checkpoint_ep*.pt only; checkpoint_best.pt and final.pt excluded",
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.n_components <= 0:
        raise ValueError("--n-components must be positive")
    if args.correlation_pcs <= 0:
        raise ValueError("--correlation-pcs must be positive")
    if args.correlation_pcs > args.n_components:
        raise ValueError("--correlation-pcs cannot exceed --n-components")
    if args.max_variant_csvs is not None and args.max_variant_csvs <= 0:
        raise ValueError("--max-variant-csvs must be positive when provided")

    cfg = build_run_config(args)
    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    seeds = parse_requested_seeds(args.seeds, cfg)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    csvs = list_eval_csvs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    all_correlation_rows: list[dict[str, Any]] = []
    all_variance_rows: list[dict[str, Any]] = []
    epochs_by_seed: dict[int, list[int]] = {}

    print(f"Using device: {device}")
    print(f"Loaded {len(csvs)} {cfg['variant_split']} CSVs from {cfg['variant_dir']}")
    print(f"Analyzing seeds: {', '.join(str(seed) for seed in seeds)}")
    print(
        f"Fitting {cfg['n_components']} PCs from {cfg['fit_timestep_mode']} hidden states; "
        f"plotting head correlations for PC1-PC{cfg['correlation_pcs']}"
    )

    for seed in seeds:
        seed_dir = cfg["model_dir"] / f"seed_{seed}"
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"Missing seed directory: {seed_dir}")

        checkpoints = parse_requested_epochs(args.epochs, seed_dir)
        epochs_by_seed[seed] = [epoch for epoch, _ in checkpoints]
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

        print(
            f"seed_{seed}: {len(dataset)} trials, {len(checkpoints)} epochs, "
            f"n_input={hp['n_input']}, n_null_timesteps={hp['n_null_timesteps']}"
        )

        for epoch, checkpoint_name in checkpoints:
            print(f"  epoch {epoch}: fitting PCA from {checkpoint_name}")
            model = load_model(model_cls, seed_dir, checkpoint_name, device)
            pca, pca_sample_count = fit_hidden_pca(model, dataloader, cfg, device)
            all_correlation_rows.extend(
                compute_head_pc_rows(
                    model=model,
                    pca=pca,
                    seed=seed,
                    epoch=epoch,
                    checkpoint_name=checkpoint_name,
                    cfg=cfg,
                    pca_sample_count=pca_sample_count,
                )
            )
            all_variance_rows.extend(
                compute_variance_rows(
                    pca=pca,
                    seed=seed,
                    epoch=epoch,
                    checkpoint_name=checkpoint_name,
                    cfg=cfg,
                    pca_sample_count=pca_sample_count,
                )
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    correlations = pd.DataFrame(all_correlation_rows).sort_values(
        ["seed", "head", "epoch", "pc"]
    )
    correlations_path = cfg["output_dir"] / "head_pc_correlations_by_epoch.csv"
    correlations.to_csv(correlations_path, index=False)

    variance = pd.DataFrame(all_variance_rows).sort_values(["seed", "epoch", "pc"])
    variance_path = cfg["output_dir"] / "pca_explained_variance_by_epoch.csv"
    variance.to_csv(variance_path, index=False)

    correlation_plots = plot_head_epoch_correlations(
        correlations,
        cfg["output_dir"],
        int(cfg["dpi"]),
    )
    variance_plots = plot_explained_variance_by_epoch(
        variance,
        cfg["output_dir"],
        int(cfg["dpi"]),
    )
    write_run_config(cfg["output_dir"] / "run_config.json", cfg, seeds, epochs_by_seed, csvs)

    print(f"Saved head/PC correlations to {correlations_path}")
    print(f"Saved explained variance to {variance_path}")
    print("Saved correlation plots:")
    for path in correlation_plots:
        print(f"  {path}")
    print("Saved explained-variance plots:")
    for path in variance_plots:
        print(f"  {path}")


if __name__ == "__main__":
    main()

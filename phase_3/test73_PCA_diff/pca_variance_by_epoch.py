#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

from pca_checkpoint_ep010 import (
    DEFAULT_MODEL_CLASS,
    DEFAULT_VARIANT_SPLIT,
    HelicopterPCADataset,
    collate_batch,
    import_model_class,
    infer_model_label,
    iter_hidden_batches,
    load_hp,
    load_model,
    natural_key,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "pca_variance_by_epoch_outputs"
CHECKPOINT_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")
SEED_RE = re.compile(r"seed_(\d+)$")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_root: Path
    variant_root: Path
    variant_subdir: str
    model_class: str

    @property
    def variant_dir(self) -> Path:
        return self.variant_root / self.variant_subdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA explained variance percent by training epoch."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config JSON. Default: ./config.json",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of PCs to fit and plot. Default: 5",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to include, e.g. 0,1,2. Default: all seeds found.",
    )
    parser.add_argument(
        "--variant-split",
        type=str,
        default=None,
        help=f"Variant split prefix. Default: config value or {DEFAULT_VARIANT_SPLIT!r}",
    )
    parser.add_argument(
        "--max-variant-csvs",
        type=int,
        default=None,
        help="Limit evaluation CSVs for quick runs. Default: use all matching CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSVs and plots. Default: ./pca_variance_by_epoch_outputs",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device. Default: auto",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def as_model_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(cfg.get("models"), list):
        entries = cfg["models"]
    elif isinstance(cfg.get("model_roots"), list):
        entries = cfg["model_roots"]
    elif isinstance(cfg.get("model_root"), list):
        entries = cfg["model_root"]
    elif cfg.get("model_root"):
        entries = [{"model_root": cfg["model_root"]}]
    else:
        raise ValueError("Config must define model_root, model_roots, or models.")

    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"model_root": entry})
        elif isinstance(entry, dict):
            normalized.append(entry)
        else:
            raise TypeError(f"Unsupported model entry in config: {entry!r}")
    return normalized


def build_model_specs(cfg: dict[str, Any]) -> list[ModelSpec]:
    if not cfg.get("variant_root"):
        raise ValueError("Config is missing required key: variant_root")

    specs = []
    default_variant_root = resolve_path(cfg["variant_root"])
    default_model_class = cfg.get("model_class", DEFAULT_MODEL_CLASS)

    for entry in as_model_entries(cfg):
        if not entry.get("model_root"):
            raise ValueError(f"Model entry is missing model_root: {entry}")

        model_root = resolve_path(entry["model_root"])
        variant_root = resolve_path(entry.get("variant_root", default_variant_root))
        variant_subdir = (
            entry.get("variant_subdir")
            or cfg.get("variant_subdir")
            or entry.get("sigma")
            or cfg.get("sigma")
            or model_root.name
        )
        label = entry.get("label") or entry.get("name") or infer_model_label(model_root)
        model_class = entry.get("model_class") or default_model_class

        specs.append(
            ModelSpec(
                label=str(label),
                model_root=model_root,
                variant_root=variant_root,
                variant_subdir=str(variant_subdir),
                model_class=str(model_class),
            )
        )

    return specs


def parse_seed_filter(seed_text: str | None) -> set[int] | None:
    if seed_text is None:
        return None
    seeds = set()
    for part in seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.add(int(part))
    return seeds


def seed_number(seed_dir: Path) -> int:
    match = SEED_RE.fullmatch(seed_dir.name)
    if match is None:
        raise ValueError(f"Not a seed directory: {seed_dir}")
    return int(match.group(1))


def discover_seed_dirs(model_root: Path, seed_filter: set[int] | None) -> list[Path]:
    seed_dirs = [p for p in model_root.glob("seed_*") if p.is_dir()]
    seed_dirs = sorted(seed_dirs, key=seed_number)
    if seed_filter is not None:
        seed_dirs = [p for p in seed_dirs if seed_number(p) in seed_filter]
    if not seed_dirs:
        seed_desc = "matching " if seed_filter is not None else ""
        raise FileNotFoundError(f"No {seed_desc}seed_* directories found in {model_root}")
    return seed_dirs


def checkpoint_epoch(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not an epoch checkpoint: {path}")
    return int(match.group(1))


def discover_epoch_checkpoints(seed_dir: Path) -> list[Path]:
    checkpoints = [p for p in seed_dir.glob("checkpoint_ep*.pt") if CHECKPOINT_RE.fullmatch(p.name)]
    checkpoints = sorted(checkpoints, key=checkpoint_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_ep*.pt files found in {seed_dir}")
    return checkpoints


def list_eval_csvs(
    variant_dir: Path,
    variant_split: str,
    max_variant_csvs: int | None,
) -> list[Path]:
    pattern = f"{variant_split}Config_*.csv"
    csvs = sorted(variant_dir.glob(pattern), key=natural_key)
    if max_variant_csvs is not None:
        csvs = csvs[:max_variant_csvs]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for {variant_dir / pattern}")
    return csvs


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def fit_checkpoint_pca(
    model_cls,
    seed_dir: Path,
    checkpoint: Path,
    dataloader: DataLoader,
    n_components: int,
    device: torch.device,
) -> np.ndarray:
    pca = IncrementalPCA(n_components=n_components)
    model = load_model(model_cls, seed_dir, checkpoint.name, device)

    for hidden, _, _, _ in iter_hidden_batches(model, dataloader, device):
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        pca.partial_fit(flat_hidden)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.asarray(pca.explained_variance_ratio_, dtype=float)


def run_model_spec(
    spec: ModelSpec,
    args: argparse.Namespace,
    variant_split: str,
    device: torch.device,
    seed_filter: set[int] | None,
) -> pd.DataFrame:
    if not spec.model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {spec.model_root}")
    if not spec.variant_dir.exists():
        raise FileNotFoundError(f"Variant directory does not exist: {spec.variant_dir}")

    model_cls = import_model_class(spec.model_root, spec.model_class)
    seed_dirs = discover_seed_dirs(spec.model_root, seed_filter)
    csvs = list_eval_csvs(spec.variant_dir, variant_split, args.max_variant_csvs)

    rows = []
    print(f"\nModel: {spec.label}")
    print(f"  model_root: {spec.model_root}")
    print(f"  variants: {len(csvs)} CSVs from {spec.variant_dir}")

    for seed_dir in seed_dirs:
        seed = seed_number(seed_dir)
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
        checkpoints = discover_epoch_checkpoints(seed_dir)
        print(f"  seed_{seed}: {len(checkpoints)} checkpoints, {len(dataset)} trials")

        for checkpoint in checkpoints:
            epoch = checkpoint_epoch(checkpoint)
            print(f"    fitting PCA for epoch {epoch:03d}")
            ratios = fit_checkpoint_pca(
                model_cls,
                seed_dir,
                checkpoint,
                dataloader,
                int(args.n_components),
                device,
            )
            cumulative = np.cumsum(ratios)
            for pc_idx, ratio in enumerate(ratios, start=1):
                rows.append(
                    {
                        "model": spec.label,
                        "model_root": str(spec.model_root),
                        "variant_dir": str(spec.variant_dir),
                        "seed": seed,
                        "epoch": epoch,
                        "checkpoint": checkpoint.name,
                        "pc": f"PC{pc_idx}",
                        "pc_index": pc_idx,
                        "explained_variance_ratio": float(ratio),
                        "explained_variance_pct": float(ratio * 100.0),
                        "cumulative_explained_variance_ratio": float(cumulative[pc_idx - 1]),
                        "cumulative_explained_variance_pct": float(cumulative[pc_idx - 1] * 100.0),
                    }
                )

    return pd.DataFrame(rows)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "model"


def plot_model_variance(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = plt.get_cmap("tab10").colors

    for pc_idx, (pc, pc_df) in enumerate(sorted(df.groupby("pc"), key=lambda item: item[0])):
        color = colors[pc_idx % len(colors)]
        for _, seed_df in pc_df.groupby("seed"):
            seed_df = seed_df.sort_values("epoch")
            ax.plot(
                seed_df["epoch"],
                seed_df["explained_variance_pct"],
                color=color,
                alpha=0.18,
                linewidth=1.0,
            )

        mean_df = (
            pc_df.groupby("epoch", as_index=False)["explained_variance_pct"]
            .mean()
            .sort_values("epoch")
        )
        ax.plot(
            mean_df["epoch"],
            mean_df["explained_variance_pct"],
            marker="o",
            color=color,
            linewidth=2.4,
            label=pc,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Component", frameon=False, ncol=min(5, df["pc"].nunique()))
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_all_models(df: pd.DataFrame, out_path: Path) -> None:
    models = list(df["model"].drop_duplicates())
    n_models = len(models)
    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(10, max(4.8, 3.8 * n_models)),
        sharex=True,
        squeeze=False,
    )
    colors = plt.get_cmap("tab10").colors

    for ax, model_label in zip(axes[:, 0], models):
        model_df = df[df["model"] == model_label]
        for pc_idx, (pc, pc_df) in enumerate(sorted(model_df.groupby("pc"), key=lambda item: item[0])):
            mean_df = (
                pc_df.groupby("epoch", as_index=False)["explained_variance_pct"]
                .mean()
                .sort_values("epoch")
            )
            ax.plot(
                mean_df["epoch"],
                mean_df["explained_variance_pct"],
                marker="o",
                linewidth=2.0,
                color=colors[pc_idx % len(colors)],
                label=pc,
            )

        ax.set_ylabel("Variance (%)")
        ax.set_title(model_label)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, ncol=min(5, model_df["pc"].nunique()))

    axes[-1, 0].set_xlabel("Epoch")
    fig.suptitle("PCA explained variance by epoch", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_outputs(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "pca_variance_by_epoch.csv"
    results.to_csv(long_csv, index=False)

    wide = results.pivot_table(
        index=["model", "seed", "epoch", "checkpoint"],
        columns="pc",
        values="explained_variance_pct",
    ).reset_index()
    wide_csv = output_dir / "pca_variance_by_epoch_wide.csv"
    wide.to_csv(wide_csv, index=False)

    for model_label, model_df in results.groupby("model", sort=False):
        plot_path = output_dir / f"pca_variance_by_epoch_{safe_filename(model_label)}.png"
        plot_model_variance(
            model_df,
            plot_path,
            title=f"{model_label}: first {model_df['pc'].nunique()} PCs",
        )
        print(f"Saved model plot: {plot_path}")

    all_models_plot = output_dir / "pca_variance_by_epoch_all_models.png"
    plot_all_models(results, all_models_plot)

    print(f"Saved long CSV: {long_csv}")
    print(f"Saved wide CSV: {wide_csv}")
    print(f"Saved combined plot: {all_models_plot}")


def main() -> None:
    args = parse_args()
    if int(args.n_components) < 1:
        raise ValueError("--n-components must be at least 1")

    cfg = load_json(args.config.expanduser().resolve())
    variant_split = args.variant_split or cfg.get("variant_split", DEFAULT_VARIANT_SPLIT)
    seed_filter = parse_seed_filter(args.seeds)
    device = choose_device(args.device)
    specs = build_model_specs(cfg)

    print(f"Using device: {device}")
    print(f"Found {len(specs)} model spec(s) in {args.config}")

    result_frames = [
        run_model_spec(spec, args, variant_split, device, seed_filter)
        for spec in specs
    ]
    results = pd.concat(result_frames, ignore_index=True)
    save_outputs(results, args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()

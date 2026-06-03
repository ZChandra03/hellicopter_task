#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config.json"
SEED_RE = re.compile(r"seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss history by epoch for each seed."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config JSON with model_root. Default: config.json",
    )
    parser.add_argument(
        "--model-subdir",
        default=None,
        help="Optional model run subdirectory under model_root. Default: use model_root directly.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs",
        help="Directory for CSV and plots. Default: ./outputs",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not cfg.get("model_root"):
        raise ValueError(f"Config is missing required key: model_root")

    cfg["model_root"] = Path(cfg["model_root"]).expanduser().resolve()
    return cfg


def infer_model_label(model_root: Path) -> str:
    if model_root.parent.name:
        return f"{model_root.parent.name}/{model_root.name}"
    return model_root.name


def list_seed_dirs(model_dir: Path) -> list[Path]:
    seed_dirs = [p for p in model_dir.iterdir() if p.is_dir() and SEED_RE.fullmatch(p.name)]
    seed_dirs.sort(key=lambda p: int(SEED_RE.fullmatch(p.name).group(1)))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {model_dir}")
    return seed_dirs


def load_history(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} must contain a non-empty JSON list")

    return [float(value) for value in history]


def collect_loss_history(model_dir: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    seed_dirs = list_seed_dirs(model_dir)

    for seed_dir in seed_dirs:
        seed_match = SEED_RE.fullmatch(seed_dir.name)
        seed = int(seed_match.group(1))
        loss_history = load_history(seed_dir / "loss_history.json")
        val_loss_history = load_history(seed_dir / "val_loss_history.json")

        if len(loss_history) != len(val_loss_history):
            raise ValueError(
                f"{seed_dir.name} has {len(loss_history)} loss values but "
                f"{len(val_loss_history)} val_loss values"
            )

        for epoch, (loss, val_loss) in enumerate(zip(loss_history, val_loss_history), start=1):
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "loss": loss,
                    "val_loss": val_loss,
                }
            )

    return sorted(rows, key=lambda row: (int(row["seed"]), int(row["epoch"])))


def write_csv(rows: list[dict[str, float | int]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "epoch", "loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict[str, float | int]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    seed_ids = sorted({int(row["seed"]) for row in rows})
    cmap = plt.get_cmap("tab10", len(seed_ids))

    for i, seed in enumerate(seed_ids):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        ax.plot(
            [int(row["epoch"]) for row in seed_rows],
            [float(row[metric]) for row in seed_rows],
            marker="o",
            linewidth=1.6,
            markersize=4,
            alpha=0.85,
            color=cmap(i),
            label=f"seed {seed}",
        )

    epochs = sorted({int(row["epoch"]) for row in rows})
    mean_values = [
        sum(float(row[metric]) for row in rows if int(row["epoch"]) == epoch)
        / sum(1 for row in rows if int(row["epoch"]) == epoch)
        for epoch in epochs
    ]
    ax.plot(
        epochs,
        mean_values,
        color="black",
        linewidth=3.0,
        marker="o",
        markersize=5,
        label="average",
        zorder=10,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config.expanduser().resolve())
    model_dir = cfg["model_root"] if args.model_subdir is None else cfg["model_root"] / args.model_subdir
    model_label = args.model_subdir or infer_model_label(cfg["model_root"])
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_loss_history(model_dir)
    csv_path = output_dir / "loss_history_by_seed.csv"
    loss_plot_path = output_dir / "loss_history_by_epoch.png"
    val_loss_plot_path = output_dir / "val_loss_history_by_epoch.png"

    write_csv(rows, csv_path)
    plot_metric(
        rows,
        metric="loss",
        ylabel="Loss",
        title=f"{model_label} loss by epoch",
        out_path=loss_plot_path,
    )
    plot_metric(
        rows,
        metric="val_loss",
        ylabel="Validation loss",
        title=f"{model_label} validation loss by epoch",
        out_path=val_loss_plot_path,
    )

    print(f"Saved loss history to {csv_path}")
    print(f"Saved loss plot to {loss_plot_path}")
    print(f"Saved validation loss plot to {val_loss_plot_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "accuracy_by_checkpoint_config.json"
CHECKPOINT_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")
SEED_RE = re.compile(r"seed_(\d+)$")


class HelicopterEvalDataset(Dataset):
    def __init__(self, csv_paths: list[Path]):
        xs = []
        y_report = []
        y_hazard = []

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                evidence = row["evidence"]
                if not isinstance(evidence, list):
                    evidence = ast.literal_eval(str(evidence))

                xs.append(torch.tensor(evidence, dtype=torch.float32).unsqueeze(-1))
                y_report.append(float((row["trueReport"] + 1) * 0.5))
                y_hazard.append(float((row["truePredict"] + 1) * 0.5))

        if not xs:
            raise ValueError("No evaluation trials were loaded.")

        self.x = xs
        self.y_report = torch.tensor(y_report, dtype=torch.float32).unsqueeze(1)
        self.y_hazard = torch.tensor(y_hazard, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_report[idx], self.y_hazard[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot report and predict accuracy by checkpoint for each seed model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to config JSON. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--model-subdir",
        default="bce_both/sigma_1",
        help="Experiment folder under model_root. Default: bce_both/sigma_1",
    )
    parser.add_argument(
        "--variant-subdir",
        default=None,
        help="Variant folder under variant_root. Defaults to the model-subdir leaf, e.g. sigma_1.",
    )
    parser.add_argument(
        "--variant-split",
        default="test",
        choices=["train", "val", "test"],
        help="Which variant CSV split to evaluate. Default: test",
    )
    parser.add_argument(
        "--max-variant-csvs",
        type=int,
        default=None,
        help="Optional cap on the number of variant CSVs to evaluate.",
    )
    parser.add_argument(
        "--model-class",
        default="GRUModel",
        help="Model class in model_root/rnn_models.py. Default: GRUModel",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size. Default: 256",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "outputs",
        help="Directory for CSV and plots. Default: ./outputs",
    )
    parser.add_argument(
        "--skip-best",
        action="store_true",
        help="Do not include checkpoint_best.pt in the plots.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Do not include final.pt in the plots.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["model_root", "variant_root"]
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    cfg["model_root"] = Path(cfg["model_root"]).expanduser().resolve()
    cfg["variant_root"] = Path(cfg["variant_root"]).expanduser().resolve()
    return cfg


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config.resolve())
    variant_subdir = args.variant_subdir or Path(args.model_subdir).name

    cfg.update(
        {
            "model_subdir": args.model_subdir,
            "variant_subdir": variant_subdir,
            "variant_split": args.variant_split,
            "max_variant_csvs": args.max_variant_csvs,
            "model_class": args.model_class,
            "batch_size": args.batch_size,
            "include_checkpoint_best": not args.skip_best,
            "include_final": not args.skip_final,
            "output_dir": args.output_dir.expanduser().resolve(),
        }
    )
    cfg["model_dir"] = cfg["model_root"] / cfg["model_subdir"]
    cfg["variant_dir"] = cfg["variant_root"] / cfg["variant_subdir"]
    return cfg


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def list_eval_csvs(cfg: dict[str, Any]) -> list[Path]:
    pattern = f"{cfg['variant_split']}Config_*.csv"
    csvs = sorted(cfg["variant_dir"].glob(pattern), key=natural_key)
    if cfg["max_variant_csvs"] is not None:
        csvs = csvs[: int(cfg["max_variant_csvs"])]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for {cfg['variant_dir'] / pattern}")
    return csvs


def import_model_class(model_root: Path, class_name: str):
    module_path = model_root / "rnn_models.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find {module_path}")

    spec = importlib.util.spec_from_file_location("ots_rnn_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} does not define {class_name}") from exc


def list_seed_dirs(model_dir: Path) -> list[Path]:
    seed_dirs = [p for p in model_dir.iterdir() if p.is_dir() and SEED_RE.fullmatch(p.name)]
    seed_dirs.sort(key=lambda p: int(SEED_RE.fullmatch(p.name).group(1)))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {model_dir}")
    return seed_dirs


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    if name == "checkpoint_init.pt":
        return (0, 0, name)
    match = CHECKPOINT_RE.fullmatch(name)
    if match:
        return (1, int(match.group(1)), name)
    if name == "final.pt":
        return (2, 0, name)
    if name == "checkpoint_best.pt":
        return (3, 0, name)
    return (4, 0, name)


def checkpoint_label(path: Path) -> str:
    name = path.name
    if name == "checkpoint_init.pt":
        return "init"
    if name == "checkpoint_best.pt":
        return "best"
    if name == "final.pt":
        return "final"
    match = CHECKPOINT_RE.fullmatch(name)
    if match:
        return f"ep{int(match.group(1)):03d}"
    return path.stem


def checkpoint_epoch(path: Path) -> int | None:
    if path.name == "checkpoint_init.pt":
        return 0
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match:
        return int(match.group(1))
    return None


def list_checkpoints(seed_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    ckpts = [seed_dir / "checkpoint_init.pt"]
    ckpts.extend(seed_dir.glob("checkpoint_ep*.pt"))
    if cfg["include_final"]:
        ckpts.append(seed_dir / "final.pt")
    if cfg["include_checkpoint_best"]:
        ckpts.append(seed_dir / "checkpoint_best.pt")

    existing = sorted({p for p in ckpts if p.exists()}, key=checkpoint_sort_key)
    if not existing:
        raise FileNotFoundError(f"No checkpoints found in {seed_dir}")
    return existing


def load_hp(seed_dir: Path) -> dict[str, Any]:
    hp_path = seed_dir / "hp.json"
    if hp_path.exists():
        with hp_path.open("r", encoding="utf-8") as f:
            hp = json.load(f)
    else:
        hp = {}

    hp.setdefault("n_input", 1)
    hp.setdefault("n_rnn", 128)
    hp.setdefault("batch_size", 25)
    hp.setdefault("train_heads", "both")
    return hp


def collate_batch(batch):
    xs, y_report, y_hazard = zip(*batch)
    return torch.stack(xs, 0), torch.stack(y_report, 0), torch.stack(y_hazard, 0)


@torch.inference_mode()
def evaluate_checkpoint(
    model_cls,
    checkpoint_path: Path,
    hp: dict[str, Any],
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model = model_cls(hp).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    report_correct = 0
    predict_correct = 0
    total = 0

    for x, y_report, y_predict in dataloader:
        x = x.to(device)
        y_report = y_report.to(device)
        y_predict = y_predict.to(device)

        loc_logits, predict_logits = model(x)
        report_pred = (torch.sigmoid(loc_logits[:, -1, :]) > 0.5).float()
        predict_pred = (torch.sigmoid(predict_logits) > 0.5).float()

        report_correct += int((report_pred == y_report).sum().item())
        predict_correct += int((predict_pred == y_predict).sum().item())
        total += int(y_report.numel())

    report_accuracy = report_correct / total
    predict_accuracy = predict_correct / total
    combined_accuracy = 0.5 * (report_accuracy + predict_accuracy)
    return {
        "report_accuracy": report_accuracy,
        "predict_accuracy": predict_accuracy,
        "combined_accuracy": combined_accuracy,
        "n_examples": total,
    }


def collect_results(cfg: dict[str, Any]) -> pd.DataFrame:
    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    csvs = list_eval_csvs(cfg)
    dataset = HelicopterEvalDataset(csvs)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(f"Evaluating {len(dataset)} trials from {len(csvs)} {cfg['variant_split']} CSVs")
    print(f"Using device: {device}")

    rows = []
    for seed_dir in list_seed_dirs(cfg["model_dir"]):
        seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
        hp = load_hp(seed_dir)
        for order, ckpt in enumerate(list_checkpoints(seed_dir, cfg)):
            metrics = evaluate_checkpoint(model_cls, ckpt, hp, dataloader, device)
            row = {
                "model": seed_dir.name,
                "seed": seed,
                "checkpoint": checkpoint_label(ckpt),
                "checkpoint_file": ckpt.name,
                "checkpoint_order": order,
                "epoch": checkpoint_epoch(ckpt),
                **metrics,
            }
            rows.append(row)
            print(
                f"{seed_dir.name} {row['checkpoint']}: "
                f"report={row['report_accuracy']:.4f} "
                f"predict={row['predict_accuracy']:.4f} "
                f"combined={row['combined_accuracy']:.4f}"
            )

    return pd.DataFrame(rows)


def plot_results(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    metric_column: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if metric_column not in df.columns:
        raise ValueError(f"Dataframe does not contain {metric_column!r}")

    checkpoint_table = (
        df[["checkpoint", "checkpoint_order"]]
        .drop_duplicates()
        .sort_values("checkpoint_order")
        .reset_index(drop=True)
    )
    x_by_checkpoint = {
        row.checkpoint: i for i, row in enumerate(checkpoint_table.itertuples(index=False))
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab10")

    for i, (model_name, model_df) in enumerate(df.groupby("model", sort=True)):
        model_df = model_df.sort_values("checkpoint_order")
        x = [x_by_checkpoint[label] for label in model_df["checkpoint"]]
        ax.plot(
            x,
            model_df[metric_column],
            color=cmap(i % 10),
            alpha=0.42,
            linewidth=1.2,
            marker="o",
            markersize=3.2,
            label=model_name,
        )

    avg = (
        df.groupby(["checkpoint", "checkpoint_order"], as_index=False)[metric_column]
        .mean()
        .sort_values("checkpoint_order")
    )
    avg_x = [x_by_checkpoint[label] for label in avg["checkpoint"]]
    ax.plot(
        avg_x,
        avg[metric_column],
        color="black",
        linewidth=3.4,
        marker="o",
        markersize=5,
        label="average",
        zorder=10,
    )

    ax.set_xticks(range(len(checkpoint_table)))
    ax.set_xticklabels(checkpoint_table["checkpoint"], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = build_run_config(args)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    df = collect_results(cfg)
    csv_path = cfg["output_dir"] / "accuracy_by_checkpoint.csv"
    report_plot_path = cfg["output_dir"] / "report_accuracy_by_checkpoint.png"
    predict_plot_path = cfg["output_dir"] / "predict_accuracy_by_checkpoint.png"

    df.to_csv(csv_path, index=False)
    plot_results(
        df,
        cfg,
        metric_column="report_accuracy",
        ylabel="Report accuracy",
        title=f"{cfg['model_subdir']} report accuracy by checkpoint",
        out_path=report_plot_path,
    )
    plot_results(
        df,
        cfg,
        metric_column="predict_accuracy",
        ylabel="Predict accuracy",
        title=f"{cfg['model_subdir']} predict accuracy by checkpoint",
        out_path=predict_plot_path,
    )
    print(f"Saved metrics to {csv_path}")
    print(f"Saved report plot to {report_plot_path}")
    print(f"Saved predict plot to {predict_plot_path}")


if __name__ == "__main__":
    main()

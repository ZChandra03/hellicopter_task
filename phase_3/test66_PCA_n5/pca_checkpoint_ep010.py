#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "accuracy_by_checkpoint_config.json"
SEED_RE = re.compile(r"seed_(\d+)$")
CHECKPOINT_NAME = "checkpoint_ep010.pt"


class HelicopterPCADataset(Dataset):
    def __init__(self, csv_paths: list[Path]):
        self.x = []
        self.trial_meta = []

        global_trial = 0
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            for csv_trial, row in df.reset_index(drop=True).iterrows():
                evidence = row["evidence"]
                if not isinstance(evidence, list):
                    evidence = ast.literal_eval(str(evidence))

                self.x.append(torch.tensor(evidence, dtype=torch.float32).unsqueeze(-1))
                self.trial_meta.append(
                    {
                        "source_csv": csv_path.name,
                        "csv_trial": int(csv_trial),
                        "global_trial": int(global_trial),
                        "trial_in_block": row.get("trialInBlock", np.nan),
                        "true_hazard": float(row["trueHazard"]),
                        "true_report": int(row["trueReport"]),
                        "true_predict": int(row["truePredict"]),
                    }
                )
                global_trial += 1

        if not self.x:
            raise ValueError("No PCA trials were loaded.")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA on seed 1 checkpoint_ep010 hidden states."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to root config JSON. Default: {DEFAULT_CONFIG}",
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
        help="Which variant CSV split to run through the models. Default: test",
    )
    parser.add_argument(
        "--max-variant-csvs",
        type=int,
        default=None,
        help="Optional cap on the number of variant CSVs.",
    )
    parser.add_argument(
        "--model-class",
        default="GRUModel",
        help="Model class in model_root/rnn_models.py. Default: GRUModel",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of PCA components. Default: 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to fit and transform. Default: 0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size in trials. Default: 256",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=50000,
        help="Maximum transformed rows to keep for scatter plots. Default: 50000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "pca_outputs",
        help="Directory for PCA CSVs and plots. Default: ./pca_outputs",
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
            "n_components": args.n_components,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_plot_points": args.max_plot_points,
            "output_dir": args.output_dir.expanduser().resolve(),
            "checkpoint_name": CHECKPOINT_NAME,
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


def get_seed_dir(model_dir: Path, seed: int, checkpoint_name: str) -> Path:
    seed_dir = model_dir / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
    if not (seed_dir / checkpoint_name).exists():
        raise FileNotFoundError(f"Missing checkpoint: {seed_dir / checkpoint_name}")
    return seed_dir


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
    xs, idxs = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(idxs, dtype=torch.long)


def load_model(model_cls, seed_dir: Path, checkpoint_name: str, device: torch.device):
    hp = load_hp(seed_dir)
    model = model_cls(hp).to(device)
    state = torch.load(seed_dir / checkpoint_name, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def iter_hidden_batches(model, dataloader: DataLoader, device: torch.device):
    for x, trial_idx in dataloader:
        x = x.to(device)
        hidden = model.rnn(x)
        loc_logits = model.loc_head(hidden)
        predict_logits = model.haz_head(hidden[:, -1])
        report_pred = (torch.sigmoid(loc_logits[:, -1, :]) > 0.5).cpu().numpy().astype(int)
        predict_pred = (torch.sigmoid(predict_logits) > 0.5).cpu().numpy().astype(int)
        yield hidden.detach().cpu().numpy(), trial_idx.numpy(), report_pred, predict_pred


def fit_seed_pca(
    model_cls,
    seed_dir: Path,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=int(cfg["n_components"]))

    print(f"Fitting PCA from {seed_dir.name}/{cfg['checkpoint_name']}")
    model = load_model(model_cls, seed_dir, cfg["checkpoint_name"], device)
    for hidden, _, _, _ in iter_hidden_batches(model, dataloader, device):
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        pca.partial_fit(flat_hidden)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pca


def write_transformed_csv(
    model_cls,
    pca: IncrementalPCA,
    seed_dir: Path,
    dataset: HelicopterPCADataset,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    out_path: Path,
) -> pd.DataFrame:
    pc_cols = [f"pc{i + 1}" for i in range(int(cfg["n_components"]))]
    fieldnames = [
        "model",
        "seed",
        "checkpoint",
        "source_csv",
        "csv_trial",
        "global_trial",
        "timestep",
        "trial_in_block",
        "true_hazard",
        "true_report",
        "true_predict",
        "report_pred",
        "predict_pred",
        "report_correct",
        "predict_correct",
        "combined_correct",
        *pc_cols,
    ]

    rng = np.random.default_rng(0)
    plot_rows = []
    max_plot_points = int(cfg["max_plot_points"])
    rows_seen = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
        print(f"Transforming {seed_dir.name}/{cfg['checkpoint_name']}")
        model = load_model(model_cls, seed_dir, cfg["checkpoint_name"], device)

        for hidden, trial_indices, report_pred, predict_pred in iter_hidden_batches(model, dataloader, device):
            transformed = pca.transform(hidden.reshape(-1, hidden.shape[-1]))
            n_batch, n_time, _ = hidden.shape
            transformed = transformed.reshape(n_batch, n_time, -1)

            for batch_pos, trial_idx in enumerate(trial_indices):
                meta = dataset.trial_meta[int(trial_idx)]
                true_report01 = 1 if meta["true_report"] > 0 else 0
                true_predict01 = 1 if meta["true_predict"] > 0 else 0
                batch_report_pred = int(report_pred[batch_pos, 0])
                batch_predict_pred = int(predict_pred[batch_pos, 0])
                report_correct = int(batch_report_pred == true_report01)
                predict_correct = int(batch_predict_pred == true_predict01)
                combined_correct = int(report_correct == 1 and predict_correct == 1)

                for timestep in range(n_time):
                    row = {
                        "model": seed_dir.name,
                        "seed": seed,
                        "checkpoint": cfg["checkpoint_name"],
                        "source_csv": meta["source_csv"],
                        "csv_trial": meta["csv_trial"],
                        "global_trial": meta["global_trial"],
                        "timestep": timestep,
                        "trial_in_block": meta["trial_in_block"],
                        "true_hazard": meta["true_hazard"],
                        "true_report": meta["true_report"],
                        "true_predict": meta["true_predict"],
                        "report_pred": 1 if batch_report_pred == 1 else -1,
                        "predict_pred": 1 if batch_predict_pred == 1 else -1,
                        "report_correct": report_correct,
                        "predict_correct": predict_correct,
                        "combined_correct": combined_correct,
                    }
                    for i, col in enumerate(pc_cols):
                        row[col] = float(transformed[batch_pos, timestep, i])

                    writer.writerow(row)
                    rows_seen += 1

                    if len(plot_rows) < max_plot_points:
                        plot_rows.append(row.copy())
                    else:
                        replace_idx = rng.integers(0, rows_seen)
                        if replace_idx < max_plot_points:
                            plot_rows[int(replace_idx)] = row.copy()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(plot_rows)


def save_explained_variance(pca: IncrementalPCA, out_path: Path) -> None:
    rows = []
    cumulative = 0.0
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        cumulative += float(ratio)
        rows.append(
            {
                "component": f"pc{i}",
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance_ratio": cumulative,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_explained_variance(pca: IncrementalPCA, out_path: Path) -> None:
    pcs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(pcs, explained, marker="o", linewidth=2.0, label="per PC")
    ax.plot(pcs, cumulative, marker="o", linewidth=2.0, label="cumulative")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Explained variance vs PC")
    ax.set_xticks(pcs)
    ax.set_ylim(0.0, min(1.05, max(1.0, float(cumulative[-1]) + 0.05)))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def scatter_pc1_pc2(
    df: pd.DataFrame,
    color_col: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    values = df[color_col]
    scatter = ax.scatter(
        df["pc1"],
        df["pc2"],
        c=values,
        s=4,
        alpha=0.35,
        linewidths=0,
        cmap=cmap,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label=color_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mean_trajectory(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label, group_df in sorted(df.groupby("true_predict")):
        trajectory = group_df.groupby("timestep", as_index=False)[["pc1", "pc2"]].mean()
        ax.plot(
            trajectory["pc1"],
            trajectory["pc2"],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            label=f"true_predict={label}",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Mean hidden-state trajectory by true_predict")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_plots(plot_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    if plot_df.empty:
        print("No sampled rows available for plotting.")
        return

    scatter_pc1_pc2(
        plot_df,
        color_col="true_hazard",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_hazard",
        out_path=cfg["output_dir"] / "pca_ep010_pc1_pc2_by_true_hazard.png",
    )
    scatter_pc1_pc2(
        plot_df,
        color_col="timestep",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by timestep",
        out_path=cfg["output_dir"] / "pca_ep010_pc1_pc2_by_timestep.png",
        cmap="plasma",
    )
    scatter_pc1_pc2(
        plot_df,
        color_col="true_report",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_report",
        out_path=cfg["output_dir"] / "pca_ep010_pc1_pc2_by_true_report.png",
        cmap="coolwarm",
    )
    scatter_pc1_pc2(
        plot_df,
        color_col="true_predict",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_predict",
        out_path=cfg["output_dir"] / "pca_ep010_pc1_pc2_by_true_predict.png",
        cmap="coolwarm",
    )
    plot_mean_trajectory(
        plot_df,
        out_path=cfg["output_dir"] / "pca_ep010_mean_trajectory_by_true_predict.png",
    )

    correct_df = plot_df[plot_df["combined_correct"] == 1].copy()
    if correct_df.empty:
        print("No combined-correct sampled rows available for correct-only plots.")
        return

    scatter_pc1_pc2(
        correct_df,
        color_col="true_hazard",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_hazard, correct only",
        out_path=cfg["output_dir"] / "pca_ep010_correct_only_pc1_pc2_by_true_hazard.png",
    )
    scatter_pc1_pc2(
        correct_df,
        color_col="timestep",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by timestep, correct only",
        out_path=cfg["output_dir"] / "pca_ep010_correct_only_pc1_pc2_by_timestep.png",
        cmap="plasma",
    )
    scatter_pc1_pc2(
        correct_df,
        color_col="true_report",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_report, correct only",
        out_path=cfg["output_dir"] / "pca_ep010_correct_only_pc1_pc2_by_true_report.png",
        cmap="coolwarm",
    )
    scatter_pc1_pc2(
        correct_df,
        color_col="true_predict",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_predict, correct only",
        out_path=cfg["output_dir"] / "pca_ep010_correct_only_pc1_pc2_by_true_predict.png",
        cmap="coolwarm",
    )
    plot_mean_trajectory(
        correct_df,
        out_path=cfg["output_dir"] / "pca_ep010_correct_only_mean_trajectory_by_true_predict.png",
    )


def main() -> None:
    args = parse_args()
    cfg = build_run_config(args)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    seed_dir = get_seed_dir(cfg["model_dir"], int(cfg["seed"]), cfg["checkpoint_name"])
    csvs = list_eval_csvs(cfg)
    dataset = HelicopterPCADataset(csvs)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(f"Using device: {device}")
    print(f"Loaded {len(dataset)} trials from {len(csvs)} {cfg['variant_split']} CSVs")
    print(f"Using {seed_dir.name}/{cfg['checkpoint_name']}")

    pca = fit_seed_pca(model_cls, seed_dir, dataloader, cfg, device)
    variance_path = cfg["output_dir"] / "pca_ep010_explained_variance.csv"
    save_explained_variance(pca, variance_path)
    variance_plot_path = cfg["output_dir"] / "pca_ep010_explained_variance.png"
    plot_explained_variance(pca, variance_plot_path)

    transformed_path = cfg["output_dir"] / "pca_ep010_hidden_states.csv"
    plot_df = write_transformed_csv(
        model_cls,
        pca,
        seed_dir,
        dataset,
        dataloader,
        cfg,
        device,
        transformed_path,
    )
    plot_sample_path = cfg["output_dir"] / "pca_ep010_plot_sample.csv"
    plot_df.to_csv(plot_sample_path, index=False)
    save_plots(plot_df, cfg)

    print(f"Saved transformed PCA rows to {transformed_path}")
    print(f"Saved plot sample to {plot_sample_path}")
    print(f"Saved explained variance to {variance_path}")
    print(f"Saved explained variance plot to {variance_plot_path}")


if __name__ == "__main__":
    main()

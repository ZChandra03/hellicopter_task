#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader, Dataset

from pca_checkpoint_ep010 import (
    DEFAULT_CONFIG,
    encode_evidence_sequence,
    get_seed_dir,
    import_model_class,
    infer_model_label,
    load_config,
    load_hp,
    natural_key,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "prefix_length_outputs"
DEFAULT_PREFIX_LENGTHS = [1, 2, 3, 5, 10, 15, 20]
DEFAULT_CHECKPOINT = "checkpoint_ep010.pt"
DEFAULT_MODEL_CLASS = "GRUModel"
SEED_RE = re.compile(r"seed_(\d+)$")


@dataclass(frozen=True)
class TrialRecord:
    source_csv: str
    csv_trial: int
    global_trial: int
    trial_in_block: float
    sigma: float
    true_hazard: float
    true_predict: int
    full_true_report: int
    evidence: list[float]
    states: list[float]


class HelicopterTrialTable:
    def __init__(self, csv_paths: list[Path]):
        self.trials: list[TrialRecord] = []
        global_trial = 0

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            for csv_trial, row in df.reset_index(drop=True).iterrows():
                evidence = parse_list(row["evidence"])
                states = parse_list(row["states"])
                if len(evidence) != len(states):
                    raise ValueError(
                        f"{csv_path.name} row {csv_trial} has evidence length "
                        f"{len(evidence)} but states length {len(states)}"
                    )
                if not evidence:
                    raise ValueError(f"{csv_path.name} row {csv_trial} has no evidence")

                full_report = signed_int(row["trueReport"])
                self.trials.append(
                    TrialRecord(
                        source_csv=csv_path.name,
                        csv_trial=int(csv_trial),
                        global_trial=int(global_trial),
                        trial_in_block=float(row.get("trialInBlock", np.nan)),
                        sigma=float(row.get("sigma", np.nan)),
                        true_hazard=float(row["trueHazard"]),
                        true_predict=signed_int(row["truePredict"]),
                        full_true_report=full_report,
                        evidence=[float(x) for x in evidence],
                        states=[float(x) for x in states],
                    )
                )
                global_trial += 1

        if not self.trials:
            raise ValueError("No trials were loaded.")

        lengths = {len(trial.evidence) for trial in self.trials}
        self.evidence_lengths = sorted(lengths)

    def __len__(self) -> int:
        return len(self.trials)

    def validate_prefix_lengths(self, prefix_lengths: list[int]) -> None:
        min_len = min(self.evidence_lengths)
        too_short = [length for length in prefix_lengths if length < 1]
        too_long = [length for length in prefix_lengths if length > min_len]
        if too_short:
            raise ValueError(f"Prefix lengths must be >= 1, got {too_short}")
        if too_long:
            raise ValueError(
                f"Prefix lengths {too_long} exceed the shortest trial length {min_len}"
            )


class PrefixDataset(Dataset):
    def __init__(
        self,
        trial_table: HelicopterTrialTable,
        prefix_length: int,
        n_input: int,
        n_null_timesteps: int,
    ):
        self.trial_table = trial_table
        self.prefix_length = int(prefix_length)
        self.n_input = int(n_input)
        self.n_null_timesteps = int(n_null_timesteps)

    def __len__(self) -> int:
        return len(self.trial_table)

    def __getitem__(self, idx: int):
        trial = self.trial_table.trials[idx]
        evidence = trial.evidence[: self.prefix_length]
        x = encode_evidence_sequence(evidence, self.n_input, self.n_null_timesteps)
        return x, idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained helicopter-task models on evidence prefixes and "
            "compare prefix hidden states against full-length PCA trajectories."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config with model_root and variant_root. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_PREFIX_LENGTHS,
        help="Evidence prefix lengths to evaluate. Default: 1 2 3 5 10 15 20",
    )
    parser.add_argument(
        "--seeds",
        default="all",
        help='Seeds for behavioral evaluation: "all", a comma list, or a space-free range like 0-9.',
    )
    parser.add_argument(
        "--pca-seed",
        type=int,
        default=1,
        help="Seed used for PCA trajectory plots. Default: 1",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint filename to load. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--model-class",
        default=DEFAULT_MODEL_CLASS,
        help=f"Model class in rnn_models.py. Default: {DEFAULT_MODEL_CLASS}",
    )
    parser.add_argument(
        "--variant-split",
        choices=["train", "val", "test"],
        default="test",
        help="Variant split to evaluate. Default: test",
    )
    parser.add_argument(
        "--max-test-csvs",
        type=int,
        default=None,
        help="Optional cap on split CSVs to load, useful for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size. Default: 256",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of PCA components. Default: 3",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=60000,
        help="Reservoir sample size for PCA endpoint scatter plots. Default: 60000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Only run behavioral prefix metrics.",
    )
    return parser.parse_args()


def parse_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return value
    return ast.literal_eval(str(value))


def signed_int(value: Any) -> int:
    return 1 if float(value) > 0 else -1


def target01(signed_value: int | float) -> int:
    return 1 if float(signed_value) > 0 else 0


def parse_seed_spec(seed_spec: str, model_dir: Path, checkpoint_name: str) -> list[int]:
    if seed_spec.strip().lower() == "all":
        seeds = []
        for child in model_dir.iterdir():
            match = SEED_RE.fullmatch(child.name)
            if match and child.is_dir() and (child / checkpoint_name).exists():
                seeds.append(int(match.group(1)))
        if not seeds:
            raise FileNotFoundError(
                f"No seed_* directories with {checkpoint_name} found in {model_dir}"
            )
        return sorted(seeds)

    if "-" in seed_spec and "," not in seed_spec:
        start_text, end_text = seed_spec.split("-", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(f"Invalid seed range: {seed_spec}")
        return list(range(start, end + 1))

    return sorted({int(part.strip()) for part in seed_spec.split(",") if part.strip()})


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    model_subdir = cfg.get("model_subdir") or infer_model_label(cfg["model_root"])
    variant_subdir = cfg.get("variant_subdir") or cfg.get("sigma") or cfg["model_root"].name
    cfg.update(
        {
            "model_subdir": model_subdir,
            "variant_subdir": variant_subdir,
            "model_dir": cfg["model_root"],
            "variant_dir": cfg["variant_root"] / variant_subdir,
            "variant_split": args.variant_split,
            "max_test_csvs": args.max_test_csvs,
            "checkpoint_name": args.checkpoint_name,
            "model_class": args.model_class,
            "prefix_lengths": sorted(dict.fromkeys(args.prefix_lengths)),
            "seeds": parse_seed_spec(args.seeds, cfg["model_root"], args.checkpoint_name),
            "pca_seed": int(args.pca_seed),
            "batch_size": int(args.batch_size),
            "n_components": int(args.n_components),
            "max_plot_points": int(args.max_plot_points),
            "output_dir": args.output_dir.expanduser().resolve(),
            "skip_pca": bool(args.skip_pca),
        }
    )
    return cfg


def list_eval_csvs(cfg: dict[str, Any]) -> list[Path]:
    pattern = f"{cfg['variant_split']}Config_*.csv"
    csvs = sorted(cfg["variant_dir"].glob(pattern), key=natural_key)
    if cfg["max_test_csvs"] is not None:
        csvs = csvs[: int(cfg["max_test_csvs"])]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for {cfg['variant_dir'] / pattern}")
    return csvs


def collate_prefix_batch(batch):
    xs, idxs = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(idxs, dtype=torch.long)


def load_model(model_cls, seed_dir: Path, checkpoint_name: str, hp: dict[str, Any], device: torch.device):
    model = model_cls(hp).to(device)
    state = torch.load(seed_dir / checkpoint_name, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_prefix_loader(
    trial_table: HelicopterTrialTable,
    prefix_length: int,
    hp: dict[str, Any],
    batch_size: int,
) -> DataLoader:
    dataset = PrefixDataset(
        trial_table,
        prefix_length,
        int(hp["n_input"]),
        int(hp.get("n_null_timesteps", 0)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prefix_batch,
    )


@torch.inference_mode()
def evaluate_model_prefixes(
    model,
    trial_table: HelicopterTrialTable,
    seed: int,
    prefix_lengths: list[int],
    hp: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    eps = 1e-7

    for prefix_length in prefix_lengths:
        loader = make_prefix_loader(trial_table, prefix_length, hp, batch_size)
        for x, trial_indices in loader:
            x = x.to(device)
            loc_logits, haz_logits = model(x)
            report_prob_pos = torch.sigmoid(loc_logits[:, -1, 0]).detach().cpu().numpy()
            hazard_prob_high = torch.sigmoid(haz_logits.reshape(-1)).detach().cpu().numpy()

            for batch_pos, trial_idx_tensor in enumerate(trial_indices):
                trial_idx = int(trial_idx_tensor)
                trial = trial_table.trials[trial_idx]
                true_report = signed_int(trial.states[prefix_length - 1])
                true_predict = trial.true_predict
                report_target = target01(true_report)
                hazard_target = target01(true_predict)

                report_prob = float(report_prob_pos[batch_pos])
                hazard_prob = float(hazard_prob_high[batch_pos])
                report_pred01 = int(report_prob >= 0.5)
                hazard_pred01 = int(hazard_prob >= 0.5)
                report_p_correct = report_prob if report_target else 1.0 - report_prob
                hazard_p_correct = hazard_prob if hazard_target else 1.0 - hazard_prob

                rows.append(
                    {
                        "seed": int(seed),
                        "prefix_length": int(prefix_length),
                        "source_csv": trial.source_csv,
                        "csv_trial": trial.csv_trial,
                        "global_trial": trial.global_trial,
                        "trial_in_block": trial.trial_in_block,
                        "sigma": trial.sigma,
                        "true_hazard": trial.true_hazard,
                        "true_predict": true_predict,
                        "true_report_prefix": true_report,
                        "true_report_full": trial.full_true_report,
                        "prefix_state_matches_full": int(true_report == trial.full_true_report),
                        "report_prob_pos": report_prob,
                        "hazard_prob_high": hazard_prob,
                        "report_pred": 1 if report_pred01 else -1,
                        "hazard_pred": 1 if hazard_pred01 else -1,
                        "report_correct": int(report_pred01 == report_target),
                        "hazard_correct": int(hazard_pred01 == hazard_target),
                        "report_p_correct": max(eps, min(1.0 - eps, report_p_correct)),
                        "hazard_p_correct": max(eps, min(1.0 - eps, hazard_p_correct)),
                        "report_pred_matches_full": int(
                            (1 if report_pred01 else -1) == trial.full_true_report
                        ),
                    }
                )

    return pd.DataFrame(rows)


def sem(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def summarize_behavior(pred_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = pred_df.copy()
    pred_df["hazard_bin"] = pd.cut(
        pred_df["true_hazard"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        include_lowest=True,
    )

    seed_summary = (
        pred_df.groupby(["seed", "prefix_length"], as_index=False)
        .agg(
            n_trials=("global_trial", "count"),
            report_accuracy=("report_correct", "mean"),
            hazard_accuracy=("hazard_correct", "mean"),
            report_p_correct=("report_p_correct", "mean"),
            hazard_p_correct=("hazard_p_correct", "mean"),
            report_prob_pos=("report_prob_pos", "mean"),
            hazard_prob_high=("hazard_prob_high", "mean"),
            prefix_state_matches_full=("prefix_state_matches_full", "mean"),
            report_pred_matches_full=("report_pred_matches_full", "mean"),
        )
        .sort_values(["seed", "prefix_length"])
    )

    metric_cols = [
        "report_accuracy",
        "hazard_accuracy",
        "report_p_correct",
        "hazard_p_correct",
        "report_prob_pos",
        "hazard_prob_high",
        "prefix_state_matches_full",
        "report_pred_matches_full",
    ]
    rows = []
    for prefix_length, group in seed_summary.groupby("prefix_length"):
        row = {
            "prefix_length": int(prefix_length),
            "n_seeds": int(group["seed"].nunique()),
            "n_trials_per_seed": float(group["n_trials"].mean()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_sem"] = sem(group[col])
        rows.append(row)
    prefix_summary = pd.DataFrame(rows).sort_values("prefix_length")

    seed_summary.to_csv(output_dir / "prefix_behavior_by_seed.csv", index=False)
    prefix_summary.to_csv(output_dir / "prefix_behavior_summary.csv", index=False)

    hazard_seed_summary = (
        pred_df.groupby(["seed", "prefix_length", "hazard_bin"], observed=True, as_index=False)
        .agg(
            n_trials=("global_trial", "count"),
            report_accuracy=("report_correct", "mean"),
            hazard_accuracy=("hazard_correct", "mean"),
            hazard_prob_high=("hazard_prob_high", "mean"),
        )
        .sort_values(["seed", "prefix_length", "hazard_bin"])
    )
    hazard_seed_summary.to_csv(output_dir / "prefix_behavior_by_hazard_bin_seed.csv", index=False)

    hazard_rows = []
    for (prefix_length, hazard_bin), group in hazard_seed_summary.groupby(
        ["prefix_length", "hazard_bin"], observed=True
    ):
        row = {
            "prefix_length": int(prefix_length),
            "hazard_bin": str(hazard_bin),
            "n_seeds": int(group["seed"].nunique()),
            "n_trials_per_seed": float(group["n_trials"].mean()),
        }
        for col in ["report_accuracy", "hazard_accuracy", "hazard_prob_high"]:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_sem"] = sem(group[col])
        hazard_rows.append(row)
    hazard_summary = pd.DataFrame(hazard_rows).sort_values(["prefix_length", "hazard_bin"])
    hazard_summary.to_csv(output_dir / "prefix_behavior_by_hazard_bin.csv", index=False)
    return prefix_summary, hazard_summary


def plot_metric_with_sem(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    mean_col: str,
    sem_col: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    x = summary[x_col].to_numpy(dtype=float)
    y = summary[mean_col].to_numpy(dtype=float)
    yerr = summary[sem_col].to_numpy(dtype=float)
    ax.plot(x, y, marker=marker, linewidth=2.0, markersize=5, label=label, color=color)
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, color=color, linewidth=0)


def plot_behavior_summary(prefix_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    ax_acc, ax_prob, ax_stability, ax_bias = axes.ravel()

    plot_metric_with_sem(
        ax_acc,
        prefix_summary,
        "prefix_length",
        "report_accuracy_mean",
        "report_accuracy_sem",
        "report accuracy",
        "tab:blue",
        "o",
    )
    plot_metric_with_sem(
        ax_acc,
        prefix_summary,
        "prefix_length",
        "hazard_accuracy_mean",
        "hazard_accuracy_sem",
        "hazard accuracy",
        "tab:orange",
        "s",
    )
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.set_title("Prefix-labeled accuracy")
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(frameon=False)

    plot_metric_with_sem(
        ax_prob,
        prefix_summary,
        "prefix_length",
        "report_p_correct_mean",
        "report_p_correct_sem",
        "report p(correct)",
        "tab:blue",
        "o",
    )
    plot_metric_with_sem(
        ax_prob,
        prefix_summary,
        "prefix_length",
        "hazard_p_correct_mean",
        "hazard_p_correct_sem",
        "hazard p(correct)",
        "tab:orange",
        "s",
    )
    ax_prob.set_ylabel("Mean probability")
    ax_prob.set_ylim(0.0, 1.02)
    ax_prob.set_title("Confidence assigned to the correct answer")
    ax_prob.grid(True, alpha=0.25)
    ax_prob.legend(frameon=False)

    plot_metric_with_sem(
        ax_stability,
        prefix_summary,
        "prefix_length",
        "prefix_state_matches_full_mean",
        "prefix_state_matches_full_sem",
        "prefix state equals full final state",
        "tab:green",
        "o",
    )
    plot_metric_with_sem(
        ax_stability,
        prefix_summary,
        "prefix_length",
        "report_pred_matches_full_mean",
        "report_pred_matches_full_sem",
        "model report equals full final state",
        "tab:red",
        "s",
    )
    ax_stability.set_xlabel("Evidence prefix length")
    ax_stability.set_ylabel("Fraction")
    ax_stability.set_ylim(0.0, 1.02)
    ax_stability.set_title("Why recomputing trueReport matters")
    ax_stability.grid(True, alpha=0.25)
    ax_stability.legend(frameon=False)

    plot_metric_with_sem(
        ax_bias,
        prefix_summary,
        "prefix_length",
        "report_prob_pos_mean",
        "report_prob_pos_sem",
        "mean P(report=+1)",
        "tab:purple",
        "o",
    )
    plot_metric_with_sem(
        ax_bias,
        prefix_summary,
        "prefix_length",
        "hazard_prob_high_mean",
        "hazard_prob_high_sem",
        "mean P(hazard high)",
        "tab:brown",
        "s",
    )
    ax_bias.set_xlabel("Evidence prefix length")
    ax_bias.set_ylabel("Mean output probability")
    ax_bias.set_ylim(0.0, 1.02)
    ax_bias.set_title("Output bias over sequence length")
    ax_bias.grid(True, alpha=0.25)
    ax_bias.legend(frameon=False)

    for ax in axes.ravel():
        ax.set_xticks(prefix_summary["prefix_length"].to_numpy(dtype=int))

    fig.suptitle("Prefix-length behavioral effects", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "prefix_behavior_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_hazard_heatmaps(hazard_summary: pd.DataFrame, output_dir: Path) -> None:
    prefix_lengths = sorted(hazard_summary["prefix_length"].unique())
    hazard_bins = list(hazard_summary["hazard_bin"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    for ax, metric, title, cmap in [
        (axes[0], "report_accuracy_mean", "Report accuracy", "Blues"),
        (axes[1], "hazard_accuracy_mean", "Hazard accuracy", "Oranges"),
    ]:
        pivot = (
            hazard_summary.pivot(index="prefix_length", columns="hazard_bin", values=metric)
            .reindex(index=prefix_lengths, columns=hazard_bins)
        )
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
        ax.set_title(f"{title} by true hazard")
        ax.set_xlabel("True hazard bin")
        ax.set_xticks(np.arange(len(hazard_bins)))
        ax.set_xticklabels(hazard_bins, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(prefix_lengths)))
        ax.set_yticklabels(prefix_lengths)
        for row_idx, prefix_length in enumerate(prefix_lengths):
            for col_idx, hazard_bin in enumerate(hazard_bins):
                value = pivot.loc[prefix_length, hazard_bin]
                if not pd.isna(value):
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Evidence prefix length")
    fig.tight_layout()
    fig.savefig(output_dir / "prefix_accuracy_by_hazard_bin.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_behavior_plots(prefix_summary: pd.DataFrame, hazard_summary: pd.DataFrame, output_dir: Path) -> None:
    plot_behavior_summary(prefix_summary, output_dir)
    plot_hazard_heatmaps(hazard_summary, output_dir)


def event_timestep(prefix_length: int, hp: dict[str, Any]) -> int:
    if int(hp["n_input"]) == 1:
        return prefix_length - 1
    stride = int(hp.get("n_null_timesteps", 0)) + 1
    return (prefix_length - 1) * stride


def add_stat(stats: dict[tuple[Any, ...], dict[str, Any]], key: tuple[Any, ...], values: np.ndarray) -> None:
    if values.size == 0:
        return
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    if key not in stats:
        stats[key] = {
            "count": 0,
            "sum": np.zeros(values.shape[1], dtype=np.float64),
        }
    stats[key]["count"] += values.shape[0]
    stats[key]["sum"] += values.sum(axis=0)


def stats_to_frame(
    stats: dict[tuple[Any, ...], dict[str, Any]],
    key_columns: list[str],
    n_components: int,
) -> pd.DataFrame:
    rows = []
    for key, value in stats.items():
        count = int(value["count"])
        mean = value["sum"] / max(count, 1)
        row = {col: key[idx] for idx, col in enumerate(key_columns)}
        row["n_points"] = count
        for pc_idx in range(n_components):
            row[f"pc{pc_idx + 1}"] = float(mean[pc_idx])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key_columns).reset_index(drop=True)


def update_endpoint_sample(
    sample_rows: list[dict[str, Any]],
    row: dict[str, Any],
    rows_seen: int,
    max_rows: int,
    rng: np.random.Generator,
) -> None:
    if max_rows <= 0:
        return
    if len(sample_rows) < max_rows:
        sample_rows.append(row)
        return
    replace_idx = int(rng.integers(0, rows_seen + 1))
    if replace_idx < max_rows:
        sample_rows[replace_idx] = row


@torch.inference_mode()
def fit_regular_pca(
    model,
    trial_table: HelicopterTrialTable,
    full_length: int,
    hp: dict[str, Any],
    batch_size: int,
    n_components: int,
    device: torch.device,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=n_components)
    loader = make_prefix_loader(trial_table, full_length, hp, batch_size)
    for x, _ in loader:
        hidden = model.rnn(x.to(device)).detach().cpu().numpy()
        pca.partial_fit(hidden.reshape(-1, hidden.shape[-1]))
    return pca


def save_explained_variance(pca: IncrementalPCA, output_dir: Path) -> None:
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(explained)
    rows = [
        {
            "component": f"pc{i + 1}",
            "explained_variance_ratio": float(explained[i]),
            "cumulative_explained_variance_ratio": float(cumulative[i]),
        }
        for i in range(len(explained))
    ]
    pd.DataFrame(rows).to_csv(output_dir / "prefix_pca_explained_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(1, len(explained) + 1)
    ax.plot(x, explained, marker="o", linewidth=2.0, label="per PC")
    ax.plot(x, cumulative, marker="o", linewidth=2.0, label="cumulative")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(x)
    ax.set_ylim(0.0, min(1.05, max(1.0, float(cumulative[-1]) + 0.05)))
    ax.set_title("PCA fit on regular full-length hidden trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "prefix_pca_explained_variance.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def collect_pca_outputs(
    model,
    pca: IncrementalPCA,
    trial_table: HelicopterTrialTable,
    prefix_lengths: list[int],
    hp: dict[str, Any],
    batch_size: int,
    max_plot_points: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_components = len(pca.explained_variance_ratio_)
    full_length = max(prefix_lengths)
    trajectory_stats: dict[tuple[Any, ...], dict[str, Any]] = {}
    regular_event_pcs = {
        prefix_length: np.full((len(trial_table), n_components), np.nan, dtype=np.float32)
        for prefix_length in prefix_lengths
    }

    full_loader = make_prefix_loader(trial_table, full_length, hp, batch_size)
    for x, trial_indices in full_loader:
        hidden = model.rnn(x.to(device)).detach().cpu().numpy()
        transformed = pca.transform(hidden.reshape(-1, hidden.shape[-1])).reshape(
            hidden.shape[0], hidden.shape[1], -1
        )
        for true_predict in [-1, 1]:
            batch_mask = np.array(
                [trial_table.trials[int(idx)].true_predict == true_predict for idx in trial_indices],
                dtype=bool,
            )
            if not np.any(batch_mask):
                continue
            for timestep in range(transformed.shape[1]):
                add_stat(
                    trajectory_stats,
                    ("regular_full", full_length, true_predict, timestep),
                    transformed[batch_mask, timestep, :],
                )

        for prefix_length in prefix_lengths:
            timestep = event_timestep(prefix_length, hp)
            for batch_pos, trial_idx_tensor in enumerate(trial_indices):
                regular_event_pcs[prefix_length][int(trial_idx_tensor), :] = transformed[
                    batch_pos, timestep, :
                ]

    endpoint_rows = []
    endpoint_sample_rows: list[dict[str, Any]] = []
    rows_seen = 0
    rng = np.random.default_rng(0)

    for prefix_length in prefix_lengths:
        loader = make_prefix_loader(trial_table, prefix_length, hp, batch_size)
        final_timestep = event_timestep(prefix_length, hp)

        for x, trial_indices in loader:
            hidden = model.rnn(x.to(device)).detach().cpu().numpy()
            transformed = pca.transform(hidden.reshape(-1, hidden.shape[-1])).reshape(
                hidden.shape[0], hidden.shape[1], -1
            )

            for true_predict in [-1, 1]:
                batch_mask = np.array(
                    [trial_table.trials[int(idx)].true_predict == true_predict for idx in trial_indices],
                    dtype=bool,
                )
                if not np.any(batch_mask):
                    continue
                for timestep in range(transformed.shape[1]):
                    add_stat(
                        trajectory_stats,
                        ("prefix", prefix_length, true_predict, timestep),
                        transformed[batch_mask, timestep, :],
                    )

            endpoint = transformed[:, -1, :]
            for batch_pos, trial_idx_tensor in enumerate(trial_indices):
                trial_idx = int(trial_idx_tensor)
                trial = trial_table.trials[trial_idx]
                regular_same = regular_event_pcs[prefix_length][trial_idx, :]
                regular_final = regular_event_pcs[full_length][trial_idx, :]
                distance_to_same = float(np.linalg.norm(endpoint[batch_pos, :] - regular_same))
                distance_to_final = float(np.linalg.norm(endpoint[batch_pos, :] - regular_final))

                row = {
                    "prefix_length": int(prefix_length),
                    "global_trial": trial.global_trial,
                    "source_csv": trial.source_csv,
                    "csv_trial": trial.csv_trial,
                    "true_hazard": trial.true_hazard,
                    "true_predict": trial.true_predict,
                    "true_report_prefix": signed_int(trial.states[prefix_length - 1]),
                    "true_report_full": trial.full_true_report,
                    "regular_event_timestep": int(final_timestep),
                    "distance_to_regular_same_event": distance_to_same,
                    "distance_to_regular_final": distance_to_final,
                }
                for pc_idx in range(n_components):
                    row[f"pc{pc_idx + 1}"] = float(endpoint[batch_pos, pc_idx])
                    row[f"regular_same_event_pc{pc_idx + 1}"] = float(regular_same[pc_idx])
                    row[f"regular_final_pc{pc_idx + 1}"] = float(regular_final[pc_idx])

                endpoint_rows.append(row)
                update_endpoint_sample(
                    endpoint_sample_rows,
                    row.copy(),
                    rows_seen,
                    max_plot_points,
                    rng,
                )
                rows_seen += 1

    trajectory_df = stats_to_frame(
        trajectory_stats,
        ["trajectory_type", "prefix_length", "true_predict", "timestep"],
        n_components,
    )
    endpoint_df = pd.DataFrame(endpoint_rows)
    endpoint_sample_df = pd.DataFrame(endpoint_sample_rows)
    return trajectory_df, endpoint_df, endpoint_sample_df


def plot_regular_trajectory_with_prefix_endpoints(
    trajectory_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
    prefix_lengths: list[int],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(prefix_lengths), max(prefix_lengths))

    for ax, true_predict in zip(axes, [-1, 1]):
        regular = trajectory_df[
            (trajectory_df["trajectory_type"] == "regular_full")
            & (trajectory_df["true_predict"] == true_predict)
        ].sort_values("timestep")
        ax.plot(
            regular["pc1"],
            regular["pc2"],
            color="0.25",
            linewidth=2.0,
            label="regular full trajectory",
        )

        endpoint_means = (
            endpoint_df[endpoint_df["true_predict"] == true_predict]
            .groupby("prefix_length", as_index=False)[["pc1", "pc2"]]
            .mean()
            .sort_values("prefix_length")
        )
        colors = [cmap(norm(length)) for length in endpoint_means["prefix_length"]]
        ax.scatter(
            endpoint_means["pc1"],
            endpoint_means["pc2"],
            c=colors,
            s=65,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.set_title(f"true_predict={true_predict}")
        ax.set_xlabel("PC1")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("PC2")
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
    colorbar.set_label("Prefix length")
    colorbar.set_ticks(prefix_lengths)
    fig.suptitle("Prefix endpoints on the regular full-length PCA trajectory")
    fig.savefig(
        output_dir / "prefix_pca_regular_trajectory_with_prefix_endpoints.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_endpoint_scatter(endpoint_sample_df: pd.DataFrame, prefix_lengths: list[int], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    scatter = ax.scatter(
        endpoint_sample_df["pc1"],
        endpoint_sample_df["pc2"],
        c=endpoint_sample_df["prefix_length"],
        cmap="viridis",
        s=8,
        alpha=0.28,
        linewidths=0,
        vmin=min(prefix_lengths),
        vmax=max(prefix_lengths),
    )
    full = endpoint_sample_df[endpoint_sample_df["prefix_length"] == max(prefix_lengths)]
    if not full.empty:
        ax.scatter(
            full["pc1"],
            full["pc2"],
            c="none",
            edgecolor="black",
            s=20,
            linewidth=0.45,
            alpha=0.45,
            label=f"regular length {max(prefix_lengths)}",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Final hidden states produced by each prefix length")
    ax.grid(True, alpha=0.25)
    if not full.empty:
        ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Prefix length")
    fig.tight_layout()
    fig.savefig(output_dir / "prefix_pca_endpoint_scatter.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_distances(endpoint_df: pd.DataFrame, output_dir: Path) -> None:
    seedless = (
        endpoint_df.groupby(["prefix_length", "true_predict"], as_index=False)
        .agg(
            distance_to_regular_final_mean=("distance_to_regular_final", "mean"),
            distance_to_regular_final_sem=("distance_to_regular_final", sem),
            distance_to_regular_same_event_mean=("distance_to_regular_same_event", "mean"),
            distance_to_regular_same_event_sem=("distance_to_regular_same_event", sem),
        )
        .sort_values(["true_predict", "prefix_length"])
    )
    seedless.to_csv(output_dir / "prefix_pca_endpoint_distance_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharex=True)
    for true_predict, color, marker in [(-1, "tab:blue", "o"), (1, "tab:orange", "s")]:
        group = seedless[seedless["true_predict"] == true_predict]
        x = group["prefix_length"].to_numpy(dtype=float)

        y_final = group["distance_to_regular_final_mean"].to_numpy(dtype=float)
        e_final = group["distance_to_regular_final_sem"].to_numpy(dtype=float)
        axes[0].plot(x, y_final, color=color, marker=marker, linewidth=2.0, label=f"true_predict={true_predict}")
        axes[0].fill_between(x, y_final - e_final, y_final + e_final, color=color, alpha=0.18, linewidth=0)

        y_same = group["distance_to_regular_same_event_mean"].to_numpy(dtype=float)
        e_same = group["distance_to_regular_same_event_sem"].to_numpy(dtype=float)
        axes[1].plot(x, y_same, color=color, marker=marker, linewidth=2.0, label=f"true_predict={true_predict}")
        axes[1].fill_between(x, y_same - e_same, y_same + e_same, color=color, alpha=0.18, linewidth=0)

    axes[0].set_title("Distance from regular final state")
    axes[0].set_ylabel("Euclidean distance in PCA space")
    axes[1].set_title("Distance from regular same-event state")
    axes[1].set_ylabel("Euclidean distance in PCA space")
    for ax in axes:
        ax.set_xlabel("Evidence prefix length")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "prefix_pca_endpoint_distances.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_pca_plots(
    trajectory_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
    endpoint_sample_df: pd.DataFrame,
    prefix_lengths: list[int],
    output_dir: Path,
) -> None:
    plot_regular_trajectory_with_prefix_endpoints(trajectory_df, endpoint_df, prefix_lengths, output_dir)
    plot_endpoint_scatter(endpoint_sample_df, prefix_lengths, output_dir)
    plot_endpoint_distances(endpoint_df, output_dir)


def write_run_config(cfg: dict[str, Any], csvs: list[Path], output_dir: Path) -> None:
    serializable = {
        "model_root": str(cfg["model_root"]),
        "model_subdir": cfg["model_subdir"],
        "variant_root": str(cfg["variant_root"]),
        "variant_subdir": cfg["variant_subdir"],
        "variant_split": cfg["variant_split"],
        "n_variant_csvs": len(csvs),
        "checkpoint_name": cfg["checkpoint_name"],
        "model_class": cfg["model_class"],
        "prefix_lengths": cfg["prefix_lengths"],
        "seeds": cfg["seeds"],
        "pca_seed": cfg["pca_seed"],
        "batch_size": cfg["batch_size"],
        "n_components": cfg["n_components"],
        "max_plot_points": cfg["max_plot_points"],
        "skip_pca": cfg["skip_pca"],
    }
    with (output_dir / "prefix_length_run_config.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main() -> None:
    args = parse_args()
    cfg = build_run_config(args)
    output_dir = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    csvs = list_eval_csvs(cfg)
    trial_table = HelicopterTrialTable(csvs)
    trial_table.validate_prefix_lengths(cfg["prefix_lengths"])
    write_run_config(cfg, csvs, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    print(f"Using device: {device}")
    print(f"Loaded {len(trial_table)} trials from {len(csvs)} {cfg['variant_split']} CSVs")
    print(f"Evidence lengths present: {trial_table.evidence_lengths}")
    print(f"Evaluating prefix lengths: {cfg['prefix_lengths']}")
    print(f"Behavior seeds: {cfg['seeds']}")

    all_predictions = []
    first_hp: dict[str, Any] | None = None
    for seed in cfg["seeds"]:
        seed_dir = get_seed_dir(cfg["model_dir"], int(seed), cfg["checkpoint_name"])
        hp = load_hp(seed_dir)
        first_hp = hp if first_hp is None else first_hp
        print(f"Evaluating behavior for {seed_dir.name}/{cfg['checkpoint_name']}")
        model = load_model(model_cls, seed_dir, cfg["checkpoint_name"], hp, device)
        pred_df = evaluate_model_prefixes(
            model,
            trial_table,
            int(seed),
            cfg["prefix_lengths"],
            hp,
            cfg["batch_size"],
            device,
        )
        all_predictions.append(pred_df)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output_dir / "prefix_trial_predictions.csv.gz", index=False)
    prefix_summary, hazard_summary = summarize_behavior(predictions, output_dir)
    save_behavior_plots(prefix_summary, hazard_summary, output_dir)

    if not cfg["skip_pca"]:
        pca_seed_dir = get_seed_dir(cfg["model_dir"], int(cfg["pca_seed"]), cfg["checkpoint_name"])
        pca_hp = load_hp(pca_seed_dir)
        print(f"Fitting regular-length PCA for {pca_seed_dir.name}/{cfg['checkpoint_name']}")
        pca_model = load_model(model_cls, pca_seed_dir, cfg["checkpoint_name"], pca_hp, device)
        pca = fit_regular_pca(
            pca_model,
            trial_table,
            max(cfg["prefix_lengths"]),
            pca_hp,
            cfg["batch_size"],
            cfg["n_components"],
            device,
        )
        save_explained_variance(pca, output_dir)
        trajectory_df, endpoint_df, endpoint_sample_df = collect_pca_outputs(
            pca_model,
            pca,
            trial_table,
            cfg["prefix_lengths"],
            pca_hp,
            cfg["batch_size"],
            cfg["max_plot_points"],
            device,
        )
        trajectory_df.to_csv(output_dir / "prefix_pca_trajectory_summary.csv", index=False)
        endpoint_df.to_csv(output_dir / "prefix_pca_endpoint_distances.csv", index=False)
        endpoint_sample_df.to_csv(output_dir / "prefix_pca_endpoint_sample.csv", index=False)
        save_pca_plots(trajectory_df, endpoint_df, endpoint_sample_df, cfg["prefix_lengths"], output_dir)
        del pca_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Saved prefix-length analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()

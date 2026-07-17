#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "behavioral_timing_outputs"
DEFAULT_FIGURE_DIR = DEFAULT_INPUT_DIR / "figures"


ROLE_LABELS = {
    "true_report_training": "true-report trained n5",
    "last_evidence_training": "last-evidence trained n5",
}

ROLE_COLORS = {
    "true_report_training": "#2f6f9f",
    "last_evidence_training": "#a34747",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot checkpoint-wise behavioral timing outputs from test75."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def checkpoint_ticks(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    ticks = (
        df[["checkpoint_order", "checkpoint_label"]]
        .drop_duplicates()
        .sort_values("checkpoint_order")
    )
    return ticks["checkpoint_order"].tolist(), ticks["checkpoint_label"].tolist()


def metric_col(metric: str, suffix: str) -> str:
    return f"{metric}_{suffix}"


def plot_mean_sem(
    ax,
    df: pd.DataFrame,
    metric: str,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str = "o",
    alpha_fill: float = 0.16,
) -> None:
    x = df["checkpoint_order"].to_numpy(dtype=float)
    y = df[metric_col(metric, "mean")].to_numpy(dtype=float)
    sem_col = metric_col(metric, "sem")
    sem = df[sem_col].to_numpy(dtype=float) if sem_col in df else np.full_like(y, np.nan)
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2.0, markersize=4)
    if np.isfinite(sem).any():
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=alpha_fill, linewidth=0)


def style_accuracy_axis(ax, title: str, ylabel: str = "accuracy / agreement") -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.8)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_accuracy_over_training(agg: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
    for ax, role in zip(axes, ["true_report_training", "last_evidence_training"]):
        role_df = agg[agg["model_role"] == role].sort_values("checkpoint_order")
        plot_mean_sem(ax, role_df, "report_accuracy", "report vs trueReport", "#2364aa")
        plot_mean_sem(ax, role_df, "predict_accuracy", "predict vs truePredict", "#2a9d8f")
        plot_mean_sem(
            ax,
            role_df,
            "report_last_evidence_agreement",
            "report agreement with last evidence",
            "#c44536",
            linestyle="--",
        )
        plot_mean_sem(
            ax,
            role_df,
            "heuristic_report_accuracy",
            "last-evidence heuristic accuracy",
            "#555555",
            linestyle=":",
            marker="s",
            alpha_fill=0.0,
        )
        style_accuracy_axis(ax, ROLE_LABELS.get(role, role))
        ax.legend(loc="lower right", ncols=2, fontsize=9)

    ticks, labels = checkpoint_ticks(agg)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    axes[-1].set_xlabel("checkpoint")
    save_fig(fig, figure_dir / "accuracy_and_heuristic_agreement_over_training.png")


def plot_conflict_rescue(agg: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for role in ["true_report_training", "last_evidence_training"]:
        role_df = agg[agg["model_role"] == role].sort_values("checkpoint_order")
        color = ROLE_COLORS.get(role, None)
        plot_mean_sem(
            ax,
            role_df,
            "diagnostic_report_accuracy",
            f"{ROLE_LABELS.get(role, role)}: rescue trueReport",
            color,
        )
        plot_mean_sem(
            ax,
            role_df,
            "diagnostic_last_evidence_agreement",
            f"{ROLE_LABELS.get(role, role)}: follows last evidence",
            color,
            linestyle="--",
            marker="s",
            alpha_fill=0.08,
        )
    style_accuracy_axis(
        ax,
        "Diagnostic conflict trials: final evidence sign disagrees with trueReport",
        ylabel="subset accuracy / agreement",
    )
    ticks, labels = checkpoint_ticks(agg)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("checkpoint")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    save_fig(fig, figure_dir / "diagnostic_conflict_rescue_over_training.png")


def plot_predict_report_coupling(subsets: pd.DataFrame, agg: pd.DataFrame, figure_dir: Path) -> None:
    true_subsets = subsets[subsets["model_role"] == "true_report_training"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)

    for subset_name, label, color in [
        ("predict_correct_and_conflict", "conflict trials, predict correct", "#2a9d8f"),
        ("predict_wrong_and_conflict", "conflict trials, predict wrong", "#c44536"),
    ]:
        df = true_subsets[true_subsets["subset_name"] == subset_name].sort_values("checkpoint_order")
        plot_mean_sem(axes[0], df, "report_accuracy", label, color)

    style_accuracy_axis(
        axes[0],
        "Does report rescue improve specifically when predict is correct?",
        ylabel="report accuracy on conflict trials",
    )
    axes[0].legend(loc="lower right")

    true_agg = agg[agg["model_role"] == "true_report_training"].sort_values("checkpoint_order")
    plot_mean_sem(axes[1], true_agg, "predict_accuracy", "predict accuracy", "#2a9d8f")
    plot_mean_sem(axes[1], true_agg, "diagnostic_report_accuracy", "conflict rescue accuracy", "#2364aa")
    plot_mean_sem(
        axes[1],
        true_agg,
        "report_last_evidence_agreement",
        "overall report-last-evidence agreement",
        "#c44536",
        linestyle="--",
    )
    style_accuracy_axis(axes[1], "Timing of predict learning vs report moving beyond last evidence")
    axes[1].legend(loc="lower right")

    ticks, labels = checkpoint_ticks(agg)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    axes[-1].set_xlabel("checkpoint")
    save_fig(fig, figure_dir / "predict_report_coupling.png")


def plot_scatter_predict_vs_rescue(agg: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for role in ["true_report_training", "last_evidence_training"]:
        df = agg[agg["model_role"] == role].sort_values("checkpoint_order")
        x = df["predict_accuracy_mean"]
        y = df["diagnostic_report_accuracy_mean"]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.6,
            color=ROLE_COLORS.get(role, None),
            label=ROLE_LABELS.get(role, role),
        )
        for _, row in df.iterrows():
            ax.annotate(
                str(row["checkpoint_label"]),
                (row["predict_accuracy_mean"], row["diagnostic_report_accuracy_mean"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("predict accuracy")
    ax.set_ylabel("conflict rescue accuracy")
    ax.set_title("Checkpoint trajectory through predict accuracy and report rescue")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.8)
    ax.legend(loc="lower right")
    save_fig(fig, figure_dir / "predict_accuracy_vs_conflict_rescue.png")


def plot_training_loss(history: pd.DataFrame, figure_dir: Path) -> None:
    if history.empty:
        return
    grouped = (
        history.groupby(["model_role", "split", "epoch"], as_index=False)["loss"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / grouped["count"].apply(lambda n: math.sqrt(n) if n else np.nan)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for role in ["true_report_training", "last_evidence_training"]:
        for split, linestyle in [("train", "-"), ("val", "--")]:
            df = grouped[(grouped["model_role"] == role) & (grouped["split"] == split)].sort_values("epoch")
            if df.empty:
                continue
            color = ROLE_COLORS.get(role, None)
            x = df["epoch"].to_numpy(dtype=float)
            y = df["mean"].to_numpy(dtype=float)
            sem = df["sem"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                label=f"{ROLE_LABELS.get(role, role)} {split}",
                color=color,
                linestyle=linestyle,
                marker="o",
                linewidth=2.0,
            )
            if np.isfinite(sem).any():
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.10, linewidth=0)

    ax.set_title("Training and validation loss histories")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=9)
    save_fig(fig, figure_dir / "training_loss_histories.png")


def plot_metric_heatmap(agg: pd.DataFrame, figure_dir: Path) -> None:
    true_df = agg[agg["model_role"] == "true_report_training"].sort_values("checkpoint_order")
    metric_labels = [
        ("report_accuracy_mean", "report acc"),
        ("predict_accuracy_mean", "predict acc"),
        ("report_last_evidence_agreement_mean", "report-last agree"),
        ("diagnostic_report_accuracy_mean", "conflict rescue"),
        ("diagnostic_last_evidence_agreement_mean", "conflict follows last"),
        ("weak_conflict_report_accuracy_mean", "weak conflict rescue"),
        ("predict_correct_conflict_report_accuracy_mean", "conflict rescue | predict correct"),
        ("predict_wrong_conflict_report_accuracy_mean", "conflict rescue | predict wrong"),
    ]
    values = np.vstack([true_df[col].to_numpy(dtype=float) for col, _ in metric_labels])
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels([label for _, label in metric_labels])
    ax.set_xticks(np.arange(len(true_df)))
    ax.set_xticklabels(true_df["checkpoint_label"].tolist(), rotation=35, ha="right")
    ax.set_title("True-report trained n5: timing summary heatmap")
    fig.colorbar(image, ax=ax, label="mean metric")
    save_fig(fig, figure_dir / "true_report_training_metric_heatmap.png")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()

    agg = load_csv(input_dir / "aggregate_checkpoint_metrics.csv")
    subsets = load_csv(input_dir / "aggregate_subset_metrics.csv")
    history = load_csv(input_dir / "training_history.csv")

    plot_accuracy_over_training(agg, figure_dir)
    plot_conflict_rescue(agg, figure_dir)
    plot_predict_report_coupling(subsets, agg, figure_dir)
    plot_scatter_predict_vs_rescue(agg, figure_dir)
    plot_training_loss(history, figure_dir)
    plot_metric_heatmap(agg, figure_dir)


if __name__ == "__main__":
    main()

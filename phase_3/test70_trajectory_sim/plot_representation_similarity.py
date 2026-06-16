#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "representation_similarity_outputs"
DEFAULT_OUTPUT_DIRNAME = "figures"
PROCRUSTES_RE = re.compile(r"pca_procrustes_rms_pc(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot visual summaries from representation_similarity_ep010.py CSV outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing representation similarity CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write figures. Default: <input-dir>/figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Figure DPI. Default: 250",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not draw numeric labels inside heatmap cells.",
    )
    return parser.parse_args()


def read_optional_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def seed_labels_from_pairs(df: pd.DataFrame) -> list[int]:
    seeds = sorted(set(df["seed_a"].dropna().astype(int)) | set(df["seed_b"].dropna().astype(int)))
    if not seeds:
        raise ValueError("No seed_a/seed_b values found.")
    return seeds


def pairwise_matrix(
    df: pd.DataFrame,
    metric: str,
    diagonal: float,
) -> tuple[np.ndarray, list[int]]:
    seeds = seed_labels_from_pairs(df)
    seed_to_idx = {seed: i for i, seed in enumerate(seeds)}
    matrix = np.full((len(seeds), len(seeds)), np.nan, dtype=float)
    np.fill_diagonal(matrix, diagonal)

    for _, row in df.iterrows():
        if metric not in row or pd.isna(row[metric]):
            continue
        i = seed_to_idx[int(row["seed_a"])]
        j = seed_to_idx[int(row["seed_b"])]
        matrix[i, j] = float(row[metric])
        matrix[j, i] = float(row[metric])

    return matrix, seeds


def contrast_text_color(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value) or vmax <= vmin:
        return "black"
    midpoint = vmin + 0.55 * (vmax - vmin)
    return "white" if value > midpoint else "black"


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
    dpi: int,
    *,
    diagonal: float,
    cmap: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
) -> None:
    if df.empty or metric not in df:
        return

    matrix, seeds = pairwise_matrix(df, metric, diagonal=diagonal)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    if vmin is None:
        vmin = float(np.min(finite))
    if vmax is None:
        vmax = float(np.max(finite))
    if vmax == vmin:
        pad = 1e-6 if vmax == 0.0 else abs(vmax) * 0.01
        vmin -= pad
        vmax += pad

    fig_size = max(6.0, 0.52 * len(seeds) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("seed")
    ax.set_ylabel("seed")
    ax.set_xticks(np.arange(len(seeds)), labels=seeds)
    ax.set_yticks(np.arange(len(seeds)), labels=seeds)
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_xticks(np.arange(-0.5, len(seeds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(seeds), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate and len(seeds) <= 12:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not np.isfinite(value):
                    continue
                text = f"{value:.3f}" if abs(value) >= 0.1 else f"{value:.4f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=contrast_text_color(value, vmin, vmax),
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    save_figure(fig, out_path, dpi)


def metric_columns(combined_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    procrustes_cols = []
    for col in combined_df.columns:
        match = PROCRUSTES_RE.fullmatch(col)
        if match:
            procrustes_cols.append(col)
    procrustes_cols.sort(key=lambda col: int(PROCRUSTES_RE.fullmatch(col).group(1)))

    similarity_cols = [
        col
        for col in ["rsa_rdm_correlation", "linear_cka", "svcca_mean_corr", "svcca_median_corr"]
        if col in combined_df.columns
    ]
    return procrustes_cols, similarity_cols


def friendly_metric_name(metric: str) -> str:
    if metric.startswith("pca_procrustes_rms_pc"):
        n_components = metric.removeprefix("pca_procrustes_rms_pc")
        return f"Procrustes RMS, PC {n_components}"
    return {
        "rsa_rdm_correlation": "RSA/RDM correlation",
        "linear_cka": "Linear CKA",
        "svcca_mean_corr": "SVCCA mean corr",
        "svcca_median_corr": "SVCCA median corr",
    }.get(metric, metric)


def boxplot_with_labels(ax: plt.Axes, data: list[np.ndarray], labels: list[str]) -> None:
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)


def plot_all_heatmaps(
    combined_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    annotate: bool,
) -> None:
    procrustes_cols, similarity_cols = metric_columns(combined_df)

    for metric in procrustes_cols:
        n_components = PROCRUSTES_RE.fullmatch(metric).group(1)
        plot_heatmap(
            combined_df,
            metric,
            title=f"PCA trajectory shape distance, top {n_components} PCs",
            out_path=out_dir / f"heatmap_{metric}.png",
            dpi=dpi,
            diagonal=0.0,
            cmap="magma_r",
            label="Procrustes RMS distance",
            vmin=0.0,
            annotate=annotate,
        )

    for metric in similarity_cols:
        plot_heatmap(
            combined_df,
            metric,
            title=friendly_metric_name(metric),
            out_path=out_dir / f"heatmap_{metric}.png",
            dpi=dpi,
            diagonal=1.0,
            cmap="viridis",
            label=friendly_metric_name(metric),
            vmin=max(0.0, float(combined_df[metric].min()) - 0.03),
            vmax=1.0,
            annotate=annotate,
        )


def plot_procrustes_by_pc(procrustes_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if procrustes_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (seed_a, seed_b), pair_df in procrustes_df.groupby(["seed_a", "seed_b"]):
        pair_df = pair_df.sort_values("n_components")
        ax.plot(
            pair_df["n_components"],
            pair_df["procrustes_rms"],
            color="#9aa0a6",
            alpha=0.35,
            linewidth=1.0,
        )

    summary = (
        procrustes_df.groupby("n_components", as_index=False)["procrustes_rms"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax.errorbar(
        summary["n_components"],
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        linewidth=2.5,
        capsize=4,
        color="#1f77b4",
        label="mean +/- SD",
    )
    ax.set_xlabel("top PCs")
    ax.set_ylabel("Procrustes RMS distance")
    ax.set_title("Trajectory shape distance across PCA dimensionality")
    ax.set_xticks(sorted(procrustes_df["n_components"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, out_dir / "procrustes_distance_by_pc.png", dpi)


def plot_metric_distributions(
    combined_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    if combined_df.empty:
        return

    procrustes_cols, similarity_cols = metric_columns(combined_df)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    if procrustes_cols:
        data = [combined_df[col].dropna().to_numpy(dtype=float) for col in procrustes_cols]
        labels = [col.removeprefix("pca_procrustes_rms_pc") for col in procrustes_cols]
        boxplot_with_labels(axes[0], data, labels)
        axes[0].set_title("PCA trajectory distances")
        axes[0].set_xlabel("top PCs")
        axes[0].set_ylabel("Procrustes RMS distance")
        axes[0].grid(True, axis="y", alpha=0.3)
    else:
        axes[0].axis("off")

    if similarity_cols:
        data = [combined_df[col].dropna().to_numpy(dtype=float) for col in similarity_cols]
        labels = [friendly_metric_name(col) for col in similarity_cols]
        boxplot_with_labels(axes[1], data, labels)
        axes[1].set_title("Representation similarity")
        axes[1].set_ylabel("similarity")
        axes[1].set_ylim(0.0, 1.03)
        axes[1].tick_params(axis="x", labelrotation=25)
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].axis("off")

    save_figure(fig, out_dir / "metric_distributions.png", dpi)


def plot_metric_summary(summary_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df["label"] = summary_df["metric"].map(friendly_metric_name)
    distance_df = summary_df[summary_df["metric"].str.startswith("pca_procrustes")]
    similarity_df = summary_df[~summary_df["metric"].str.startswith("pca_procrustes")]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for ax, sub_df, title, ylabel, ylim in [
        (axes[0], distance_df, "Trajectory distance summary", "mean Procrustes RMS", None),
        (axes[1], similarity_df, "Similarity summary", "mean similarity", (0.0, 1.03)),
    ]:
        if sub_df.empty:
            ax.axis("off")
            continue
        x = np.arange(len(sub_df))
        yerr = sub_df["std"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, sub_df["mean"], yerr=yerr, capsize=4, color="#4c78a8", alpha=0.9)
        ax.set_xticks(x, labels=sub_df["label"], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.3)

    save_figure(fig, out_dir / "metric_summary.png", dpi)


def plot_metric_relationships(combined_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if combined_df.empty:
        return

    candidate_pairs = [
        ("pca_procrustes_rms_pc10", "rsa_rdm_correlation"),
        ("pca_procrustes_rms_pc10", "linear_cka"),
        ("rsa_rdm_correlation", "linear_cka"),
        ("linear_cka", "svcca_mean_corr"),
    ]
    pairs = [(x, y) for x, y in candidate_pairs if x in combined_df and y in combined_df]
    if not pairs:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 4.8 * n_rows), squeeze=False)

    for ax, (x_col, y_col) in zip(axes.ravel(), pairs):
        x = combined_df[x_col].to_numpy(dtype=float)
        y = combined_df[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, s=38, alpha=0.75, color="#2f6f9f", edgecolors="white", linewidths=0.5)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3:
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.text(
                0.04,
                0.94,
                f"r = {r:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
        ax.set_xlabel(friendly_metric_name(x_col))
        ax.set_ylabel(friendly_metric_name(y_col))
        ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[len(pairs) :]:
        ax.axis("off")

    save_figure(fig, out_dir / "metric_relationships.png", dpi)


def plot_seed_pca_variance(variance_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if variance_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for seed, seed_df in variance_df.groupby("seed"):
        seed_df = seed_df.sort_values("component")
        axes[0].plot(
            seed_df["component"],
            seed_df["explained_variance_ratio"],
            color="#9aa0a6",
            alpha=0.45,
            linewidth=1.0,
        )
        axes[1].plot(
            seed_df["component"],
            seed_df["cumulative_explained_variance_ratio"],
            color="#9aa0a6",
            alpha=0.45,
            linewidth=1.0,
        )

    mean_df = (
        variance_df.groupby("component", as_index=False)[
            ["explained_variance_ratio", "cumulative_explained_variance_ratio"]
        ]
        .mean()
        .sort_values("component")
    )
    axes[0].plot(
        mean_df["component"],
        mean_df["explained_variance_ratio"],
        marker="o",
        color="#d62728",
        linewidth=2.2,
        label="mean",
    )
    axes[1].plot(
        mean_df["component"],
        mean_df["cumulative_explained_variance_ratio"],
        marker="o",
        color="#d62728",
        linewidth=2.2,
        label="mean",
    )

    axes[0].set_title("Per-PC explained variance")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("explained variance ratio")
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("cumulative ratio")
    axes[1].set_ylim(0.0, 1.03)
    for ax in axes:
        ax.set_xticks(sorted(variance_df["component"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    save_figure(fig, out_dir / "seed_pca_explained_variance.png", dpi)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / DEFAULT_OUTPUT_DIRNAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_df = read_optional_csv(input_dir, "combined_pairwise_metrics.csv")
    procrustes_df = read_optional_csv(input_dir, "procrustes_pca_trajectory_distance.csv")
    summary_df = read_optional_csv(input_dir, "metric_summary.csv")
    variance_df = read_optional_csv(input_dir, "seed_pca_explained_variance.csv")

    if combined_df.empty and procrustes_df.empty and summary_df.empty and variance_df.empty:
        raise FileNotFoundError(f"No representation metric CSVs found in {input_dir}")

    annotate = not args.no_annotate
    plot_all_heatmaps(combined_df, output_dir, args.dpi, annotate)
    plot_procrustes_by_pc(procrustes_df, output_dir, args.dpi)
    plot_metric_distributions(combined_df, output_dir, args.dpi)
    plot_metric_summary(summary_df, output_dir, args.dpi)
    plot_metric_relationships(combined_df, output_dir, args.dpi)
    plot_seed_pca_variance(variance_df, output_dir, args.dpi)

    print(f"Done. Figures are in {output_dir}")


if __name__ == "__main__":
    main()

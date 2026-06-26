#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REFINED_PCA_SCRIPT = BASE_DIR.parent / "test72_FP_refinement" / "pca_checkpoint_ep010.py"
pca2d = None
plt = None
torch = None
DataLoader = None


def load_runtime_dependencies() -> None:
    global DataLoader, pca2d, plt, torch

    import matplotlib.pyplot as plt_module
    import torch as torch_module
    from torch.utils.data import DataLoader as data_loader_cls

    if not REFINED_PCA_SCRIPT.exists():
        raise FileNotFoundError(f"Missing refined PCA helper: {REFINED_PCA_SCRIPT}")

    spec = importlib.util.spec_from_file_location("refined_pca_checkpoint_ep010", REFINED_PCA_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import refined PCA helper from {REFINED_PCA_SCRIPT}")

    pca2d_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pca2d_module)
    pca2d_module.DEFAULT_CONFIG = BASE_DIR / "config.json"

    pca2d = pca2d_module
    plt = plt_module
    torch = torch_module
    DataLoader = data_loader_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA on checkpoint_ep010 hidden states and save 3D PC1/PC2/PC3 "
            "plots plus 2D PC1/PC2, PC1/PC3, and PC2/PC3 projections."
        )
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of PCA components. Must be at least 3. Default: 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to fit and transform. Default: 0",
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
        default=BASE_DIR / "pca_3d_outputs",
        help="Directory for PCA CSVs, 3D plots, and 2D projections. Default: ./pca_3d_outputs",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=24.0,
        help="3D camera elevation in degrees. Default: 24",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D camera azimuth in degrees. Default: -58",
    )
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.n_components < 3:
        raise ValueError("--n-components must be at least 3 for PC1/PC2/PC3 plots")

    cfg = pca2d.build_run_config(args)
    cfg["plot_elev"] = float(args.elev)
    cfg["plot_azim"] = float(args.azim)
    return cfg


def configure_3d_axes(ax, title: str, cfg: dict[str, Any]) -> None:
    ax.set_xlabel("PC1", labelpad=8)
    ax.set_ylabel("PC2", labelpad=8)
    ax.set_zlabel("PC3", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=cfg["plot_elev"], azim=cfg["plot_azim"])
    ax.grid(True, alpha=0.25)


def scatter_pc1_pc2_pc3(
    df: pd.DataFrame,
    color_col: str,
    title: str,
    out_path: Path,
    cfg: dict[str, Any],
    cmap: str = "viridis",
) -> None:
    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df["pc1"],
        df["pc2"],
        df["pc3"],
        c=df[color_col],
        s=5,
        alpha=0.38,
        linewidths=0,
        cmap=cmap,
        depthshade=False,
    )
    configure_3d_axes(ax, title, cfg)
    fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08, label=color_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def scatter_pc_pair(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df[color_col],
        s=4,
        alpha=0.35,
        linewidths=0,
        cmap=cmap,
    )
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label=color_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mean_trajectory_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label, group_df in sorted(df.groupby("true_predict")):
        trajectory = group_df.groupby("timestep", as_index=False)[[x_col, y_col]].mean()
        ax.plot(
            trajectory[x_col],
            trajectory[y_col],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            label=f"true_predict={label}",
        )

    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.set_title(f"Mean hidden-state trajectory by true_predict ({x_col.upper()}/{y_col.upper()})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mean_trajectory_3d(df: pd.DataFrame, out_path: Path, cfg: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    for label, group_df in sorted(df.groupby("true_predict")):
        trajectory = group_df.groupby("timestep", as_index=False)[["pc1", "pc2", "pc3"]].mean()
        ax.plot(
            trajectory["pc1"],
            trajectory["pc2"],
            trajectory["pc3"],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            label=f"true_predict={label}",
        )

    configure_3d_axes(ax, "Mean hidden-state trajectory by true_predict", cfg)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_dir(cfg: dict[str, Any]) -> Path:
    out_dir = cfg["output_dir"] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_type_dir(cfg: dict[str, Any], plot_type: str, correct_only: bool = False) -> Path:
    suffix = "_correct_only" if correct_only else ""
    out_dir = plot_dir(cfg) / f"{plot_type}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def final_timestep_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "timestep" not in df.columns:
        raise ValueError("Plot data is missing required column: timestep")
    final_timestep = df["timestep"].max()
    return df[df["timestep"] == final_timestep].copy()


def sample_final_timestep_rows(csv_path: Path, max_points: int) -> pd.DataFrame:
    final_timestep = None
    for chunk in pca2d.pd.read_csv(csv_path, usecols=["timestep"], chunksize=200000):
        chunk_max = chunk["timestep"].max()
        if final_timestep is None or chunk_max > final_timestep:
            final_timestep = chunk_max

    if final_timestep is None:
        return pca2d.pd.DataFrame()

    rng = pca2d.np.random.default_rng(1)
    sampled_rows = []
    rows_seen = 0
    for chunk in pca2d.pd.read_csv(csv_path, chunksize=200000):
        final_chunk = chunk[chunk["timestep"] == final_timestep]
        for row in final_chunk.to_dict("records"):
            rows_seen += 1
            if len(sampled_rows) < max_points:
                sampled_rows.append(row)
            else:
                replace_idx = rng.integers(0, rows_seen)
                if replace_idx < max_points:
                    sampled_rows[int(replace_idx)] = row

    return pca2d.pd.DataFrame(sampled_rows)


def save_2d_projection_plots(
    final_df: pd.DataFrame,
    all_timesteps_df: pd.DataFrame,
    cfg: dict[str, Any],
    correct_only: bool = False,
) -> None:
    prefix = "pca_ep010_correct_only" if correct_only else "pca_ep010"
    title_suffix = ", correct only" if correct_only else ""
    pc_pairs = [("pc1", "pc2"), ("pc1", "pc3"), ("pc2", "pc3")]
    final_color_specs = [
        ("true_hazard", "viridis"),
        ("true_report", "coolwarm"),
        ("true_predict", "coolwarm"),
    ]

    for x_col, y_col in pc_pairs:
        pair_name = f"{x_col}_{y_col}"
        pair_label = f"{x_col.upper()}/{y_col.upper()}"
        for color_col, cmap in final_color_specs:
            out_dir = plot_type_dir(cfg, f"by_{color_col}", correct_only)
            scatter_pc_pair(
                final_df,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                title=(
                    f"{cfg['model_subdir']} {cfg['checkpoint_name']} "
                    f"PCA {pair_label} by {color_col}, final timestep{title_suffix}"
                ),
                out_path=out_dir / f"{prefix}_{pair_name}_by_{color_col}.png",
                cmap=cmap,
            )
        out_dir = plot_type_dir(cfg, "by_timestep", correct_only)
        scatter_pc_pair(
            all_timesteps_df,
            x_col=x_col,
            y_col=y_col,
            color_col="timestep",
            title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA {pair_label} by timestep{title_suffix}",
            out_path=out_dir / f"{prefix}_{pair_name}_by_timestep.png",
            cmap="plasma",
        )
        out_dir = plot_type_dir(cfg, "mean_trajectory_by_true_predict", correct_only)
        plot_mean_trajectory_2d(
            all_timesteps_df,
            x_col=x_col,
            y_col=y_col,
            out_path=out_dir / f"{prefix}_mean_trajectory_{pair_name}_by_true_predict.png",
        )


def save_plots(final_df: pd.DataFrame, all_timesteps_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    if final_df.empty and all_timesteps_df.empty:
        print("No sampled rows available for plotting.")
        return

    missing = {"pc1", "pc2", "pc3"} - set(final_df.columns)
    if missing:
        raise ValueError(f"Final-timestep plot data is missing required PCA columns: {sorted(missing)}")
    missing = {"pc1", "pc2", "pc3"} - set(all_timesteps_df.columns)
    if missing:
        raise ValueError(f"All-timestep plot data is missing required PCA columns: {sorted(missing)}")

    out_dir = plot_type_dir(cfg, "by_true_hazard")
    scatter_pc1_pc2_pc3(
        final_df,
        color_col="true_hazard",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_hazard, final timestep",
        out_path=out_dir / "pca_ep010_pc1_pc2_pc3_by_true_hazard.png",
        cfg=cfg,
    )
    out_dir = plot_type_dir(cfg, "by_timestep")
    scatter_pc1_pc2_pc3(
        all_timesteps_df,
        color_col="timestep",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by timestep",
        out_path=out_dir / "pca_ep010_pc1_pc2_pc3_by_timestep.png",
        cfg=cfg,
        cmap="plasma",
    )
    out_dir = plot_type_dir(cfg, "by_true_report")
    scatter_pc1_pc2_pc3(
        final_df,
        color_col="true_report",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_report, final timestep",
        out_path=out_dir / "pca_ep010_pc1_pc2_pc3_by_true_report.png",
        cfg=cfg,
        cmap="coolwarm",
    )
    out_dir = plot_type_dir(cfg, "by_true_predict")
    scatter_pc1_pc2_pc3(
        final_df,
        color_col="true_predict",
        title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_predict, final timestep",
        out_path=out_dir / "pca_ep010_pc1_pc2_pc3_by_true_predict.png",
        cfg=cfg,
        cmap="coolwarm",
    )
    out_dir = plot_type_dir(cfg, "mean_trajectory_by_true_predict")
    plot_mean_trajectory_3d(
        all_timesteps_df,
        out_path=out_dir / "pca_ep010_mean_trajectory_3d_by_true_predict.png",
        cfg=cfg,
    )
    save_2d_projection_plots(final_df, all_timesteps_df, cfg)

    correct_final_df = final_df[final_df["combined_correct"] == 1].copy()
    correct_all_timesteps_df = all_timesteps_df[all_timesteps_df["combined_correct"] == 1].copy()
    if correct_final_df.empty and correct_all_timesteps_df.empty:
        print("No combined-correct sampled rows available for correct-only plots.")
        return

    if not correct_final_df.empty:
        out_dir = plot_type_dir(cfg, "by_true_hazard", correct_only=True)
        scatter_pc1_pc2_pc3(
            correct_final_df,
            color_col="true_hazard",
            title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_hazard, final timestep, correct only",
            out_path=out_dir / "pca_ep010_correct_only_pc1_pc2_pc3_by_true_hazard.png",
            cfg=cfg,
        )
        out_dir = plot_type_dir(cfg, "by_true_report", correct_only=True)
        scatter_pc1_pc2_pc3(
            correct_final_df,
            color_col="true_report",
            title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_report, final timestep, correct only",
            out_path=out_dir / "pca_ep010_correct_only_pc1_pc2_pc3_by_true_report.png",
            cfg=cfg,
            cmap="coolwarm",
        )
        out_dir = plot_type_dir(cfg, "by_true_predict", correct_only=True)
        scatter_pc1_pc2_pc3(
            correct_final_df,
            color_col="true_predict",
            title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by true_predict, final timestep, correct only",
            out_path=out_dir / "pca_ep010_correct_only_pc1_pc2_pc3_by_true_predict.png",
            cfg=cfg,
            cmap="coolwarm",
        )

    if not correct_all_timesteps_df.empty:
        out_dir = plot_type_dir(cfg, "by_timestep", correct_only=True)
        scatter_pc1_pc2_pc3(
            correct_all_timesteps_df,
            color_col="timestep",
            title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} PCA by timestep, correct only",
            out_path=out_dir / "pca_ep010_correct_only_pc1_pc2_pc3_by_timestep.png",
            cfg=cfg,
            cmap="plasma",
        )
        out_dir = plot_type_dir(cfg, "mean_trajectory_by_true_predict", correct_only=True)
        plot_mean_trajectory_3d(
            correct_all_timesteps_df,
            out_path=out_dir / "pca_ep010_correct_only_mean_trajectory_3d_by_true_predict.png",
            cfg=cfg,
        )

    if not correct_final_df.empty and not correct_all_timesteps_df.empty:
        save_2d_projection_plots(correct_final_df, correct_all_timesteps_df, cfg, correct_only=True)


def main() -> None:
    args = parse_args()
    if args.n_components < 3:
        raise ValueError("--n-components must be at least 3 for PC1/PC2/PC3 plots")

    load_runtime_dependencies()
    cfg = build_run_config(args)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = pca2d.import_model_class(cfg["model_root"], cfg["model_class"])
    seed_dir = pca2d.get_seed_dir(cfg["model_dir"], int(cfg["seed"]), cfg["checkpoint_name"])
    hp = pca2d.load_hp(seed_dir)
    batch_size = int(hp.get("batch_size", 256))
    csvs = pca2d.list_eval_csvs(cfg)
    dataset = pca2d.HelicopterPCADataset(
        csvs,
        int(hp["n_input"]),
        int(hp["n_null_timesteps"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pca2d.collate_batch,
    )

    print(f"Using device: {device}")
    print(f"Loaded {len(dataset)} trials from {len(csvs)} {cfg['variant_split']} CSVs")
    print(f"Using {seed_dir.name}/{cfg['checkpoint_name']}")
    print(
        f"Prepared model inputs with n_input={hp['n_input']}, "
        f"n_null_timesteps={hp['n_null_timesteps']}, batch_size={batch_size}"
    )
    print(f"Saving 3D plots with elev={cfg['plot_elev']} and azim={cfg['plot_azim']}")

    pca = pca2d.fit_seed_pca(model_cls, seed_dir, dataloader, cfg, device)
    variance_path = cfg["output_dir"] / "pca_ep010_explained_variance.csv"
    pca2d.save_explained_variance(pca, variance_path)
    variance_plot_path = plot_dir(cfg) / "pca_ep010_explained_variance.png"
    pca2d.plot_explained_variance(pca, variance_plot_path)

    transformed_path = cfg["output_dir"] / "pca_ep010_hidden_states.csv"
    plot_df = pca2d.write_transformed_csv(
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
    final_plot_df = sample_final_timestep_rows(transformed_path, int(cfg["max_plot_points"]))
    final_plot_sample_path = cfg["output_dir"] / "pca_ep010_final_timestep_plot_sample.csv"
    final_plot_df.to_csv(final_plot_sample_path, index=False)
    save_plots(final_plot_df, plot_df, cfg)

    print(f"Saved transformed PCA rows to {transformed_path}")
    print(f"Saved plot sample to {plot_sample_path}")
    print(f"Saved final-timestep plot sample to {final_plot_sample_path}")
    print(f"Saved explained variance to {variance_path}")
    print(f"Saved explained variance plot to {variance_plot_path}")
    print(f"Saved plots to {plot_dir(cfg)}")


if __name__ == "__main__":
    main()

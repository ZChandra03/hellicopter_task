#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "pca_outputs" / "pca_ep010_hidden_states.csv"
DEFAULT_OUTPUT = BASE_DIR / "pca_outputs" / "pca_ep010_pc1_pc2_by_timestep.gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate PC1/PC2 hidden-state points over timesteps, colored by true hazard."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"PCA hidden-state CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output animation path, usually .gif or .mp4. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second. Default: 2",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI. Default: 160",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=None,
        help="Optional deterministic sample cap per timestep.",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Only animate rows where combined_correct == 1.",
    )
    return parser.parse_args()


def load_pca_rows(path: Path, correct_only: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PCA CSV: {path}\n"
            "Run pca_checkpoint_ep010.py first to generate pca_ep010_hidden_states.csv."
        )

    df = pd.read_csv(path)
    required = {"pc1", "pc2", "timestep", "true_hazard"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if correct_only:
        if "combined_correct" not in df.columns:
            raise ValueError("--correct-only requires a combined_correct column in the PCA CSV.")
        df = df[df["combined_correct"] == 1].copy()

    if df.empty:
        raise ValueError("No rows available to animate after filtering.")

    df["timestep"] = pd.to_numeric(df["timestep"], errors="raise").astype(int)
    df["pc1"] = pd.to_numeric(df["pc1"], errors="raise")
    df["pc2"] = pd.to_numeric(df["pc2"], errors="raise")
    df["true_hazard"] = pd.to_numeric(df["true_hazard"], errors="raise")
    return df


def sample_frame(frame_df: pd.DataFrame, max_points: int | None, timestep: int) -> pd.DataFrame:
    if max_points is None or len(frame_df) <= max_points:
        return frame_df
    return frame_df.sample(n=max_points, random_state=timestep)


def padded_limits(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    pad = 0.05 * max(high - low, 1e-6)
    return low - pad, high + pad


def save_animation(df: pd.DataFrame, output_path: Path, fps: int, dpi: int, max_points: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timesteps = sorted(df["timestep"].unique())
    xlim = padded_limits(df["pc1"])
    ylim = padded_limits(df["pc2"])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    first_frame = sample_frame(df[df["timestep"] == timesteps[0]], max_points, timesteps[0])
    scatter = ax.scatter(
        first_frame["pc1"],
        first_frame["pc2"],
        c=first_frame["true_hazard"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=8,
        alpha=0.45,
        linewidths=0,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("true_hazard")

    def update(timestep: int):
        frame_df = sample_frame(df[df["timestep"] == timestep], max_points, timestep)
        scatter.set_offsets(frame_df[["pc1", "pc2"]].to_numpy())
        scatter.set_array(frame_df["true_hazard"].to_numpy())
        ax.set_title(f"PC1/PC2 over timesteps | timestep {timestep}")
        return (scatter,)

    anim = FuncAnimation(fig, update, frames=timesteps, interval=1000 / fps, blit=False)

    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)

    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_pca_rows(args.input_csv.resolve(), args.correct_only)
    save_animation(
        df=df,
        output_path=args.output.resolve(),
        fps=args.fps,
        dpi=args.dpi,
        max_points=args.max_points_per_frame,
    )
    print(f"Saved animation to {args.output.resolve()}")


if __name__ == "__main__":
    main()

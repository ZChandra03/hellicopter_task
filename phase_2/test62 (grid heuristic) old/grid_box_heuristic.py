#!/usr/bin/env python3
"""Plot a grid-box heuristic for one helicopter-task evidence trial.

The evidence values are plotted as x positions. Time/sample index is plotted
on y, with the first evidence point at y=0 and each following point separated
by ``--y-step``. The heuristic count is the number of unique grid boxes touched
by the polyline connecting those evidence points.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "variants" / "data_var1.csv"
DEFAULT_OUT = BASE_DIR / "grid_plots"
EPS = 1e-10


Point = tuple[float, float]
Cell = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count and plot grid boxes intersected by a trial evidence polyline."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Variant CSV to read. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="1-based row number in the CSV. Default: 1.",
    )
    parser.add_argument(
        "--box-width",
        type=float,
        default=0.01,
        help="Grid box width along the evidence/x axis. Default: 0.25.",
    )
    parser.add_argument(
        "--box-height",
        type=float,
        default=0.01,
        help="Grid box height along the time/y axis. Default: 0.05.",
    )
    parser.add_argument(
        "--y-step",
        type=float,
        default=0.05,
        help="Vertical distance between consecutive evidence points. Default: 0.05.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to grid_plots/<csv>_trial_<n>_w<h>.png.",
    )
    return parser.parse_args()


def load_trial(csv_path: Path, trial_row: int) -> dict[str, str]:
    if trial_row < 1:
        raise ValueError("--trial is 1-based and must be >= 1")

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if trial_row > len(rows):
        raise ValueError(f"{csv_path} only has {len(rows)} rows; got trial {trial_row}")

    return rows[trial_row - 1]


def parse_evidence(value: str) -> list[float]:
    evidence = ast.literal_eval(value)
    if not isinstance(evidence, list):
        raise ValueError("Evidence field did not parse to a list")
    return [float(v) for v in evidence]


def evidence_points(evidence: Sequence[float], y_step: float) -> list[Point]:
    return [(float(x), i * y_step) for i, x in enumerate(evidence)]


def floor_to_grid(value: float, step: float) -> int:
    return math.floor((value + EPS) / step)


def ceil_to_grid(value: float, step: float) -> int:
    return math.ceil((value - EPS) / step)


def grid_bounds(points: Sequence[Point], width: float, height: float) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ix_min = min(0, math.floor(min(xs) / width))
    ix_max = max(0, math.ceil(max(xs) / width))
    iy_min = min(0, math.floor(min(ys) / height))
    iy_max = max(0, math.ceil(max(ys) / height))

    if ix_min == ix_max:
        ix_max += 1
    if iy_min == iy_max:
        iy_max += 1

    return ix_min, ix_max, iy_min, iy_max


def point_in_rect(point: Point, left: float, right: float, bottom: float, top: float) -> bool:
    x, y = point
    return left - EPS <= x <= right + EPS and bottom - EPS <= y <= top + EPS


def orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) <= EPS:
        return 0
    return 1 if value > 0 else 2


def on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) - EPS <= b[0] <= max(a[0], c[0]) + EPS
        and min(a[1], c[1]) - EPS <= b[1] <= max(a[1], c[1]) + EPS
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(a, c, b):
        return True
    if o2 == 0 and on_segment(a, d, b):
        return True
    if o3 == 0 and on_segment(c, a, d):
        return True
    if o4 == 0 and on_segment(c, b, d):
        return True
    return False


def segment_intersects_cell(a: Point, b: Point, cell: Cell, width: float, height: float) -> bool:
    ix, iy = cell
    left = ix * width
    right = (ix + 1) * width
    bottom = iy * height
    top = (iy + 1) * height

    if point_in_rect(a, left, right, bottom, top):
        return True
    if point_in_rect(b, left, right, bottom, top):
        return True

    corners = [
        (left, bottom),
        (right, bottom),
        (right, top),
        (left, top),
    ]
    edges = zip(corners, corners[1:] + corners[:1])
    return any(segments_intersect(a, b, edge_start, edge_end) for edge_start, edge_end in edges)


def cells_for_segment(a: Point, b: Point, width: float, height: float) -> set[Cell]:
    x_min = min(a[0], b[0])
    x_max = max(a[0], b[0])
    y_min = min(a[1], b[1])
    y_max = max(a[1], b[1])

    ix_start = floor_to_grid(x_min, width) - 1
    ix_end = ceil_to_grid(x_max, width) + 1
    iy_start = floor_to_grid(y_min, height) - 1
    iy_end = ceil_to_grid(y_max, height) + 1

    cells: set[Cell] = set()
    for ix in range(ix_start, ix_end + 1):
        for iy in range(iy_start, iy_end + 1):
            cell = (ix, iy)
            if segment_intersects_cell(a, b, cell, width, height):
                cells.add(cell)
    return cells


def intersected_cells(points: Sequence[Point], width: float, height: float) -> set[Cell]:
    cells: set[Cell] = set()
    for a, b in zip(points, points[1:]):
        cells.update(cells_for_segment(a, b, width, height))
    ix_min, ix_max, iy_min, iy_max = grid_bounds(points, width, height)
    return {
        (ix, iy)
        for ix, iy in cells
        if ix_min <= ix < ix_max and iy_min <= iy < iy_max
    }


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    width_label = f"{args.box_width:g}".replace(".", "p")
    height_label = f"{args.box_height:g}".replace(".", "p")
    return DEFAULT_OUT / f"{args.csv.stem}_trial_{args.trial}_w{width_label}_h{height_label}.png"


def plot_cells(
    points: Sequence[Point],
    cells: Iterable[Cell],
    trial: dict[str, str],
    width: float,
    height: float,
    path: Path,
) -> None:
    ix_min, ix_max, iy_min, iy_max = grid_bounds(points, width, height)
    touched = set(cells)

    fig, ax = plt.subplots(figsize=(9, 7))

    for ix, iy in touched:
        ax.add_patch(
            Rectangle(
                (ix * width, iy * height),
                width,
                height,
                facecolor="#61dafb",
                edgecolor="none",
                alpha=0.45,
                zorder=1,
            )
        )

    for ix in range(ix_min, ix_max + 1):
        x = ix * width
        ax.axvline(x, color="#d0d0d0", linewidth=0.7, zorder=0)
    for iy in range(iy_min, iy_max + 1):
        y = iy * height
        ax.axhline(y, color="#d0d0d0", linewidth=0.7, zorder=0)

    ax.axvline(0, color="black", linewidth=1.2, zorder=2)
    ax.axhline(0, color="black", linewidth=1.2, zorder=2)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color="#ba1f33", linewidth=2.0, marker="o", markersize=4, zorder=3)

    title = (
        f"{Path(trial.get('source_csv', 'variant')).name} row {trial['source_row']} | "
        f"boxes touched = {len(touched)}"
    )
    subtitle = (
        f"block={trial['blockDifficulty']}, sigma={trial['sigma']}, "
        f"hazard={trial['trueHazard']}, trueRep={trial['trueVal_Rep']}, "
        f"truePred={trial['trueVal_Pred']}, w={width:g}, h={height:g}"
    )
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("Evidence value")
    ax.set_ylabel("Evidence sample y position")
    ax.set_xlim(ix_min * width, ix_max * width)
    ax.set_ylim(iy_min * height, iy_max * height)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.98,
        f"count: {len(touched)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#444"},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.box_width <= 0:
        raise ValueError("--box-width must be positive")
    if args.box_height <= 0:
        raise ValueError("--box-height must be positive")
    if args.y_step <= 0:
        raise ValueError("--y-step must be positive")

    trial = load_trial(args.csv, args.trial)
    trial["source_csv"] = str(args.csv)
    trial["source_row"] = str(args.trial)
    evidence = parse_evidence(trial["evidence"])
    points = evidence_points(evidence, args.y_step)
    cells = intersected_cells(points, args.box_width, args.box_height)

    path = output_path(args)
    plot_cells(points, cells, trial, args.box_width, args.box_height, path)

    print(f"csv: {args.csv}")
    print(f"trial row: {args.trial}")
    print(f"box_width: {args.box_width:g}")
    print(f"box_height: {args.box_height:g}")
    print(f"y_step: {args.y_step:g}")
    print(f"intersected_box_count: {len(cells)}")
    print(f"plot: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit the grid-box heuristic to subject prediction choices.

For each subject, this script aligns the subject response CSV with the matching
``variants/data_varN.csv`` file named in ``subject_XX_vN_Y.csv``. It then fits:

    switch if box_count >= threshold else stay

where ``box_count`` is the number of unique grid boxes touched by the evidence
polyline, ``w`` is the evidence-axis box width, and ``l`` is the vertical/time
box length.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from grid_box_heuristic import evidence_points, grid_bounds


BASE_DIR = Path(__file__).resolve().parent
SUBJECT_DIR = BASE_DIR / "sub_data"
VARIANT_DIR = BASE_DIR / "variants"
RESULTS_DIR = BASE_DIR / "grid_fit_results"
SUBJECT_RE = re.compile(r"subject_(?P<subject>\d+)_v(?P<variant>\d+)_(?P<run>\d+)\.csv$")
MATCH_FIELDS = (
    "blockDifficulty",
    "sigma",
    "trialInBlock",
    "trueHazard",
    "trueVal_Rep",
    "trueVal_Pred",
)
EPS = 1e-10


Point = tuple[float, float]
Cell = tuple[int, int]
RowId = tuple[int, int]


@dataclass
class TrialChoice:
    row_id: RowId
    choice: int


@dataclass
class SubjectData:
    file: str
    subject: int
    variant: int
    run: int
    choices: list[TrialChoice]
    main_rows: int
    aligned_rows: int
    skipped_blank_choices: int
    missing_variant_rows: int


@dataclass
class FitResult:
    l: float
    w: float
    threshold: int
    accuracy: float
    n_correct: int
    n_trials: int
    switch_rate: float


def parse_float_grid(spec: str) -> list[float]:
    """Parse comma-separated floats or start:stop:step ranges."""
    values: list[float] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            pieces = [float(x) for x in item.split(":")]
            if len(pieces) != 3:
                raise ValueError(f"Range {item!r} must be start:stop:step")
            start, stop, step = pieces
            if step <= 0:
                raise ValueError("Grid step must be positive")
            n = int(round((stop - start) / step))
            values.extend(round(start + i * step, 10) for i in range(n + 1))
        else:
            values.append(float(item))
    if not values:
        raise ValueError("Grid spec produced no values")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit grid-box heuristic parameters.")
    parser.add_argument("--subject-dir", type=Path, default=SUBJECT_DIR)
    parser.add_argument("--variant-dir", type=Path, default=VARIANT_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--widths",
        default="0.01,0.05:1.0:0.05",
        help="Candidate evidence-axis widths. Use comma list or start:stop:step.",
    )
    parser.add_argument(
        "--lengths",
        default="0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3",
        help="Candidate vertical/time box lengths. Use comma list or start:stop:step.",
    )
    parser.add_argument(
        "--y-step",
        type=float,
        default=0.05,
        help="Vertical distance between consecutive evidence points.",
    )
    parser.add_argument(
        "--target-column",
        default="key_resp_pred_joint.keys",
        help="Subject response column to fit. Default fits switch/stay predictions.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_evidence(value: str) -> list[float]:
    return [float(v) for v in ast.literal_eval(value)]


def same_trial(subject_row: dict[str, str], variant_row: dict[str, str]) -> bool:
    for field in MATCH_FIELDS:
        if subject_row[field] != variant_row[field]:
            return False
    return True


def parse_choice(value: str) -> int | None:
    value = str(value).strip()
    if not value:
        return None
    numeric = float(value)
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return None


def align_subject(
    subject_path: Path,
    variant_rows_by_num: dict[int, list[dict[str, str]]],
    target_column: str,
) -> SubjectData:
    match = SUBJECT_RE.match(subject_path.name)
    if match is None:
        raise ValueError(f"Unexpected subject filename: {subject_path.name}")

    subject_num = int(match.group("subject"))
    variant_num = int(match.group("variant"))
    run_num = int(match.group("run"))
    variant_rows = variant_rows_by_num[variant_num]
    subject_rows = [r for r in read_csv(subject_path) if r["blockDifficulty"] != "preTest"]

    choices: list[TrialChoice] = []
    missing_variant_rows = 0
    skipped_blank_choices = 0
    i = 0
    j = 0

    while i < len(subject_rows) and j < len(variant_rows):
        subject_row = subject_rows[i]
        variant_row = variant_rows[j]

        if same_trial(subject_row, variant_row):
            choice = parse_choice(subject_row[target_column])
            if choice is None:
                skipped_blank_choices += 1
            else:
                choices.append(TrialChoice((variant_num, j), choice))
            i += 1
            j += 1
            continue

        # Most irregularities in this dataset are dropped trials. If the next
        # generated row matches the current subject row, skip one variant row.
        if j + 1 < len(variant_rows) and same_trial(subject_row, variant_rows[j + 1]):
            missing_variant_rows += 1
            j += 1
            continue

        raise ValueError(
            f"Could not align {subject_path.name} at subject main row {i + 1}, "
            f"variant row {j + 1}"
        )

    missing_variant_rows += len(variant_rows) - j

    return SubjectData(
        file=subject_path.name,
        subject=subject_num,
        variant=variant_num,
        run=run_num,
        choices=choices,
        main_rows=len(subject_rows),
        aligned_rows=len(choices),
        skipped_blank_choices=skipped_blank_choices,
        missing_variant_rows=missing_variant_rows,
    )


def is_grid_boundary(value: float, step: float) -> bool:
    return abs(value / step - round(value / step)) <= 1e-9


def cells_at_point(x: float, y: float, w: float, l: float) -> set[Cell]:
    ix0 = math.floor((x + EPS) / w)
    iy0 = math.floor((y + EPS) / l)
    xs = [ix0]
    ys = [iy0]
    if is_grid_boundary(x, w):
        xs = [ix0 - 1, ix0]
    if is_grid_boundary(y, l):
        ys = [iy0 - 1, iy0]
    return {(ix, iy) for ix in xs for iy in ys}


def segment_cells(a: Point, b: Point, w: float, l: float) -> set[Cell]:
    x0, y0 = a
    x1, y1 = b
    dx = x1 - x0
    dy = y1 - y0
    ts = [0.0, 1.0]

    if abs(dx) > EPS:
        lo = min(x0, x1)
        hi = max(x0, x1)
        k0 = math.ceil(lo / w - EPS)
        k1 = math.floor(hi / w + EPS)
        for k in range(k0, k1 + 1):
            t = (k * w - x0) / dx
            if -EPS <= t <= 1 + EPS:
                ts.append(min(1.0, max(0.0, t)))

    if abs(dy) > EPS:
        lo = min(y0, y1)
        hi = max(y0, y1)
        k0 = math.ceil(lo / l - EPS)
        k1 = math.floor(hi / l + EPS)
        for k in range(k0, k1 + 1):
            t = (k * l - y0) / dy
            if -EPS <= t <= 1 + EPS:
                ts.append(min(1.0, max(0.0, t)))

    ts = sorted(ts)
    unique_ts: list[float] = []
    for t in ts:
        if not unique_ts or abs(t - unique_ts[-1]) > 1e-9:
            unique_ts.append(t)

    cells: set[Cell] = set()
    for t in unique_ts:
        cells.update(cells_at_point(x0 + dx * t, y0 + dy * t, w, l))
    for t0, t1 in zip(unique_ts, unique_ts[1:]):
        if t1 - t0 > 1e-9:
            mid = (t0 + t1) / 2
            cells.update(cells_at_point(x0 + dx * mid, y0 + dy * mid, w, l))
    return cells


def box_count(points: Sequence[Point], w: float, l: float) -> int:
    cells: set[Cell] = set()
    for a, b in zip(points, points[1:]):
        cells.update(segment_cells(a, b, w, l))

    ix_min, ix_max, iy_min, iy_max = grid_bounds(points, w, l)
    return sum(
        1
        for ix, iy in cells
        if ix_min <= ix < ix_max and iy_min <= iy < iy_max
    )


def fit_threshold(counts: Sequence[int], choices: Sequence[int]) -> FitResult:
    if len(counts) != len(choices):
        raise ValueError("counts and choices must have the same length")
    if not counts:
        raise ValueError("cannot fit with zero trials")

    by_count: Counter[int] = Counter()
    switch_by_count: Counter[int] = Counter()
    n_switch = 0
    for count, choice in zip(counts, choices):
        by_count[count] += 1
        if choice > 0:
            switch_by_count[count] += 1
            n_switch += 1

    n_trials = len(counts)
    n_stay = n_trials - n_switch
    best_threshold = max(by_count) + 1
    best_correct = n_stay
    current_correct = n_stay

    for threshold in sorted(by_count, reverse=True):
        total_at_count = by_count[threshold]
        switch_at_count = switch_by_count[threshold]
        stay_at_count = total_at_count - switch_at_count
        current_correct += switch_at_count - stay_at_count
        if current_correct > best_correct:
            best_correct = current_correct
            best_threshold = threshold

    return FitResult(
        l=0.0,
        w=0.0,
        threshold=best_threshold,
        accuracy=best_correct / n_trials,
        n_correct=best_correct,
        n_trials=n_trials,
        switch_rate=n_switch / n_trials,
    )


def better_fit(candidate: FitResult, incumbent: FitResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.n_correct != incumbent.n_correct:
        return candidate.n_correct > incumbent.n_correct
    if candidate.threshold != incumbent.threshold:
        return candidate.threshold > incumbent.threshold
    if candidate.w != incumbent.w:
        return candidate.w > incumbent.w
    return candidate.l > incumbent.l


def fit_all(
    subjects: list[SubjectData],
    points_by_row: dict[RowId, list[Point]],
    widths: Sequence[float],
    lengths: Sequence[float],
) -> tuple[dict[str, FitResult], FitResult, list[dict[str, str]]]:
    subject_best: dict[str, FitResult] = {subject.file: None for subject in subjects}  # type: ignore[assignment]
    global_best: FitResult | None = None
    global_scores: list[dict[str, str]] = []
    all_choices = [choice for subject in subjects for choice in subject.choices]

    combos = [(l, w) for l in lengths for w in widths]
    unique_row_ids = sorted({choice.row_id for choice in all_choices})

    for combo_idx, (l, w) in enumerate(combos, 1):
        count_by_row = {
            row_id: box_count(points_by_row[row_id], w, l)
            for row_id in unique_row_ids
        }

        global_counts = [count_by_row[choice.row_id] for choice in all_choices]
        global_choice_values = [choice.choice for choice in all_choices]
        pooled = fit_threshold(global_counts, global_choice_values)
        pooled.l = l
        pooled.w = w
        global_scores.append(
            {
                "l": f"{l:g}",
                "w": f"{w:g}",
                "threshold": str(pooled.threshold),
                "accuracy": f"{pooled.accuracy:.6f}",
                "n_correct": str(pooled.n_correct),
                "n_trials": str(pooled.n_trials),
            }
        )
        if better_fit(pooled, global_best):
            global_best = pooled

        for subject in subjects:
            counts = [count_by_row[choice.row_id] for choice in subject.choices]
            choices = [choice.choice for choice in subject.choices]
            fitted = fit_threshold(counts, choices)
            fitted.l = l
            fitted.w = w
            if better_fit(fitted, subject_best[subject.file]):
                subject_best[subject.file] = fitted

        if combo_idx == 1 or combo_idx == len(combos) or combo_idx % 20 == 0:
            print(f"[fit] {combo_idx}/{len(combos)} parameter pairs evaluated")

    if global_best is None:
        raise RuntimeError("No global fit was produced")
    return subject_best, global_best, global_scores


def accuracy_under_params(
    subject: SubjectData,
    points_by_row: dict[RowId, list[Point]],
    w: float,
    l: float,
    threshold: int,
) -> float:
    correct = 0
    for choice in subject.choices:
        count = box_count(points_by_row[choice.row_id], w, l)
        prediction = 1 if count >= threshold else -1
        correct += int(prediction == choice.choice)
    return correct / len(subject.choices)


def write_subject_results(
    path: Path,
    subjects: list[SubjectData],
    subject_best: dict[str, FitResult],
    global_best: FitResult,
    points_by_row: dict[RowId, list[Point]],
) -> None:
    fields = [
        "subject_file",
        "subject",
        "variant",
        "run",
        "n_trials_fit",
        "main_rows",
        "missing_variant_rows",
        "blank_prediction_responses",
        "switch_rate",
        "best_l",
        "best_w",
        "best_threshold",
        "best_accuracy",
        "best_n_correct",
        "global_param_accuracy",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for subject in sorted(subjects, key=lambda s: s.subject):
            fit = subject_best[subject.file]
            global_acc = accuracy_under_params(
                subject,
                points_by_row,
                global_best.w,
                global_best.l,
                global_best.threshold,
            )
            writer.writerow(
                {
                    "subject_file": subject.file,
                    "subject": subject.subject,
                    "variant": subject.variant,
                    "run": subject.run,
                    "n_trials_fit": fit.n_trials,
                    "main_rows": subject.main_rows,
                    "missing_variant_rows": subject.missing_variant_rows,
                    "blank_prediction_responses": subject.skipped_blank_choices,
                    "switch_rate": f"{fit.switch_rate:.6f}",
                    "best_l": f"{fit.l:g}",
                    "best_w": f"{fit.w:g}",
                    "best_threshold": fit.threshold,
                    "best_accuracy": f"{fit.accuracy:.6f}",
                    "best_n_correct": fit.n_correct,
                    "global_param_accuracy": f"{global_acc:.6f}",
                }
            )


def write_global_result(path: Path, global_best: FitResult, subjects: list[SubjectData]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "best_l",
                "best_w",
                "best_threshold",
                "pooled_accuracy",
                "pooled_n_correct",
                "pooled_n_trials",
                "pooled_switch_rate",
                "n_subjects",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "best_l": f"{global_best.l:g}",
                "best_w": f"{global_best.w:g}",
                "best_threshold": global_best.threshold,
                "pooled_accuracy": f"{global_best.accuracy:.6f}",
                "pooled_n_correct": global_best.n_correct,
                "pooled_n_trials": global_best.n_trials,
                "pooled_switch_rate": f"{global_best.switch_rate:.6f}",
                "n_subjects": len(subjects),
            }
        )


def write_global_scores(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["l", "w", "threshold", "accuracy", "n_correct", "n_trials"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, subjects: list[SubjectData], subject_best: dict[str, FitResult], global_best: FitResult) -> None:
    fits = [subject_best[s.file] for s in sorted(subjects, key=lambda s: s.subject)]
    labels = [s.subject for s in sorted(subjects, key=lambda s: s.subject)]
    accuracies = [fit.accuracy for fit in fits]
    thresholds = [fit.threshold for fit in fits]
    widths = [fit.w for fit in fits]
    lengths = [fit.l for fit in fits]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    scatter = ax.scatter(widths, lengths, c=accuracies, cmap="viridis", s=70, edgecolor="black")
    ax.scatter([global_best.w], [global_best.l], marker="*", s=220, color="#d62728", edgecolor="black", label="pooled")
    ax.set_xlabel("w")
    ax.set_ylabel("l")
    ax.set_title("Individual best grid sizes")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="accuracy")

    ax = axes[1]
    ax.bar([str(label) for label in labels], thresholds, color="#70a6d8")
    ax.axhline(global_best.threshold, color="#d62728", linewidth=2, label="pooled threshold")
    ax.set_xlabel("subject")
    ax.set_ylabel("threshold")
    ax.set_title("Best thresholds")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend()

    ax = axes[2]
    ax.bar([str(label) for label in labels], accuracies, color="#88c999")
    ax.axhline(global_best.accuracy, color="#d62728", linewidth=2, label="pooled accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlabel("subject")
    ax.set_ylabel("accuracy")
    ax.set_title("Individual fit accuracy")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    widths = parse_float_grid(args.widths)
    lengths = parse_float_grid(args.lengths)
    if args.y_step <= 0:
        raise ValueError("--y-step must be positive")

    variant_rows_by_num: dict[int, list[dict[str, str]]] = {}
    for path in sorted(args.variant_dir.glob("data_var*.csv")):
        match = re.search(r"data_var(\d+)\.csv$", path.name)
        if match is not None:
            variant_rows_by_num[int(match.group(1))] = read_csv(path)

    subjects = [
        align_subject(path, variant_rows_by_num, args.target_column)
        for path in sorted(args.subject_dir.glob("subject_*.csv"))
    ]
    if not subjects:
        raise FileNotFoundError(f"No subject CSVs found in {args.subject_dir}")

    used_row_ids = sorted(
        {choice.row_id for subject in subjects for choice in subject.choices}
    )
    points_by_row: dict[RowId, list[Point]] = {}
    for variant_num, row_idx in used_row_ids:
        evidence = parse_evidence(variant_rows_by_num[variant_num][row_idx]["evidence"])
        points_by_row[(variant_num, row_idx)] = evidence_points(evidence, args.y_step)

    print(f"[data] subjects: {len(subjects)}")
    print(f"[data] fitted prediction choices: {sum(len(s.choices) for s in subjects)}")
    print(f"[data] unique generated trials used: {len(used_row_ids)}")
    print(f"[grid] l candidates: {lengths}")
    print(f"[grid] w candidates: {widths}")

    subject_best, global_best, global_scores = fit_all(subjects, points_by_row, widths, lengths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject_path = args.out_dir / "subject_fit_results.csv"
    global_path = args.out_dir / "global_fit_result.csv"
    scores_path = args.out_dir / "global_grid_search_scores.csv"
    write_subject_results(subject_path, subjects, subject_best, global_best, points_by_row)
    write_global_result(global_path, global_best, subjects)
    write_global_scores(scores_path, global_scores)

    if not args.no_plots:
        plot_summary(args.out_dir / "fit_summary.png", subjects, subject_best, global_best)

    print("[result] pooled best fit")
    print(f"  l={global_best.l:g}, w={global_best.w:g}, threshold={global_best.threshold}")
    print(
        f"  accuracy={global_best.accuracy:.4f} "
        f"({global_best.n_correct}/{global_best.n_trials})"
    )
    print(f"[saved] {subject_path}")
    print(f"[saved] {global_path}")
    print(f"[saved] {scores_path}")
    if not args.no_plots:
        print(f"[saved] {args.out_dir / 'fit_summary.png'}")


if __name__ == "__main__":
    main()

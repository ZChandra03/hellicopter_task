#!/usr/bin/env python3
"""Fit a normative Bayesian observer to subject prediction choices.

For each subject, this script aligns the subject response CSV with the matching
``variants/data_varN.csv`` file named in ``subject_XX_vN_Y.csv``. It then fits
the subject's switch/stay prediction choices with ``NormativeModel.BayesianObserver``.

By default the observer uses each trial's generated ``sigma`` value and fits a
prior ``bias`` over the hazard-rate distribution. Fixed belief-sigma candidates
can also be supplied with ``--belief-sigmas`` to fit mismatched observer models.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist


BASE_DIR = Path(__file__).resolve().parent
SUBJECT_DIR = BASE_DIR / "sub_data"
VARIANT_DIR = BASE_DIR / "variants"
RESULTS_DIR = BASE_DIR / "bayes_fit_results"
SUBJECT_RE = re.compile(r"subject_(?P<subject>\d+)_v(?P<variant>\d+)_(?P<run>\d+)\.csv$")
MATCH_FIELDS = (
    "blockDifficulty",
    "sigma",
    "trialInBlock",
    "trueHazard",
    "trueVal_Rep",
    "trueVal_Pred",
)


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
class PreparedTrial:
    row_id: RowId
    evidence: np.ndarray
    sigma: float


@dataclass
class FitResult:
    belief_sigma: float | None
    bias: float
    accuracy: float
    n_correct: int
    n_trials: int
    switch_rate: float


def format_sigma(value: float | None) -> str:
    return "matched" if value is None else f"{value:g}"


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
            if n < 0:
                raise ValueError(f"Range {item!r} must have stop >= start")
            values.extend(round(start + i * step, 10) for i in range(n + 1))
        else:
            values.append(float(item))
    if not values:
        raise ValueError("Grid spec produced no values")
    return sorted(set(values))


def parse_belief_sigmas(spec: str) -> list[float | None]:
    values: list[float | None] = []
    numeric_parts: list[str] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower() == "matched":
            values.append(None)
        else:
            numeric_parts.append(item)
    if numeric_parts:
        values.extend(parse_float_grid(",".join(numeric_parts)))
    if not values:
        raise ValueError("--belief-sigmas produced no values")
    unique: list[float | None] = []
    seen: set[str] = set()
    for value in values:
        key = format_sigma(value)
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit normative Bayesian observer parameters.")
    parser.add_argument("--subject-dir", type=Path, default=SUBJECT_DIR)
    parser.add_argument("--variant-dir", type=Path, default=VARIANT_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--belief-sigmas",
        default="matched",
        help=(
            "Candidate observer sigmas. Use 'matched' for each trial's sigma, "
            "or a comma list / start:stop:step grid for fixed mismatched sigmas."
        ),
    )
    parser.add_argument(
        "--biases",
        default="-2.0:2.0:0.5",
        help="Candidate hazard-prior biases. Use comma list or start:stop:step.",
    )
    parser.add_argument(
        "--hs",
        default="0.0:0.95:0.05",
        help="Hazard-rate support for the Bayesian observer.",
    )
    parser.add_argument("--mu1", type=float, default=-1.0)
    parser.add_argument("--mu2", type=float, default=1.0)
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


def normal_pdf(values: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((values - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def normative_prediction(
    trial: PreparedTrial,
    belief_sigma: float | None,
    bias: float,
    hs: np.ndarray,
    mu1: float,
    mu2: float,
) -> int:
    evidence = trial.evidence
    sigma = trial.sigma if belief_sigma is None else belief_sigma

    alpha = 1.0 + max(bias, 0.0)
    beta_param = 1.0 + max(-bias, 0.0)
    beta_prior = beta_dist.pdf(hs, alpha, beta_param)

    marg = 2 * len(hs)
    state_1 = beta_prior / marg
    state_2 = state_1.copy()
    p_diff = 1 - state_1.sum() - state_2.sum()
    p_marg = p_diff / (2 * len(hs))
    state_1 = state_1 + p_marg
    state_2 = state_2 + p_marg

    if sigma == 0:
        prob_state_1 = np.array([1.0 if int(value) == mu1 else 0.0 for value in evidence])
        prob_state_2 = np.array([1.0 if int(value) == mu2 else 0.0 for value in evidence])
    else:
        prob_state_1 = normal_pdf(evidence, mu1, sigma)
        prob_state_2 = normal_pdf(evidence, mu2, sigma)

    for p_s1, p_s2 in zip(prob_state_1, prob_state_2):
        next_state_1 = p_s1 * ((1 - hs) * state_1 + hs * state_2)
        next_state_2 = p_s2 * ((1 - hs) * state_2 + hs * state_1)
        total = next_state_1.sum() + next_state_2.sum()
        if total > 0:
            state_1 = next_state_1 / total
            state_2 = next_state_2 / total
        else:
            state_1 = next_state_1
            state_2 = next_state_2

    l_haz = state_1 + state_2
    p_switch = np.sum(hs * l_haz)
    p_stay = np.sum((1 - hs) * l_haz)
    return 1 if p_switch >= p_stay else -1


def score_predictions(predictions: Sequence[int], choices: Sequence[int], belief_sigma: float | None, bias: float) -> FitResult:
    if len(predictions) != len(choices):
        raise ValueError("predictions and choices must have the same length")
    if not predictions:
        raise ValueError("cannot fit with zero trials")

    n_correct = sum(int(prediction == choice) for prediction, choice in zip(predictions, choices))
    n_switch = sum(int(choice > 0) for choice in choices)
    n_trials = len(choices)
    return FitResult(
        belief_sigma=belief_sigma,
        bias=bias,
        accuracy=n_correct / n_trials,
        n_correct=n_correct,
        n_trials=n_trials,
        switch_rate=n_switch / n_trials,
    )


def better_fit(candidate: FitResult, incumbent: FitResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.n_correct != incumbent.n_correct:
        return candidate.n_correct > incumbent.n_correct
    if abs(candidate.bias) != abs(incumbent.bias):
        return abs(candidate.bias) < abs(incumbent.bias)
    if candidate.bias != incumbent.bias:
        return candidate.bias > incumbent.bias
    if candidate.belief_sigma is None and incumbent.belief_sigma is not None:
        return True
    if candidate.belief_sigma is not None and incumbent.belief_sigma is None:
        return False
    if candidate.belief_sigma != incumbent.belief_sigma:
        return (candidate.belief_sigma or 0.0) < (incumbent.belief_sigma or 0.0)
    return False


def fit_all(
    subjects: list[SubjectData],
    trials_by_id: dict[RowId, PreparedTrial],
    belief_sigmas: Sequence[float | None],
    biases: Sequence[float],
    hs: np.ndarray,
    mu1: float,
    mu2: float,
) -> tuple[dict[str, FitResult], FitResult, list[dict[str, str]]]:
    subject_best: dict[str, FitResult] = {subject.file: None for subject in subjects}  # type: ignore[assignment]
    global_best: FitResult | None = None
    global_scores: list[dict[str, str]] = []
    all_choices = [choice for subject in subjects for choice in subject.choices]
    global_choice_values = [choice.choice for choice in all_choices]

    combos = [(belief_sigma, bias) for belief_sigma in belief_sigmas for bias in biases]
    unique_row_ids = sorted({choice.row_id for choice in all_choices})

    for combo_idx, (belief_sigma, bias) in enumerate(combos, 1):
        prediction_by_row = {
            row_id: normative_prediction(trials_by_id[row_id], belief_sigma, bias, hs, mu1, mu2)
            for row_id in unique_row_ids
        }

        global_predictions = [prediction_by_row[choice.row_id] for choice in all_choices]
        pooled = score_predictions(global_predictions, global_choice_values, belief_sigma, bias)
        global_scores.append(
            {
                "belief_sigma": format_sigma(belief_sigma),
                "bias": f"{bias:g}",
                "accuracy": f"{pooled.accuracy:.6f}",
                "n_correct": str(pooled.n_correct),
                "n_trials": str(pooled.n_trials),
            }
        )
        if better_fit(pooled, global_best):
            global_best = pooled

        for subject in subjects:
            predictions = [prediction_by_row[choice.row_id] for choice in subject.choices]
            choices = [choice.choice for choice in subject.choices]
            fitted = score_predictions(predictions, choices, belief_sigma, bias)
            if better_fit(fitted, subject_best[subject.file]):
                subject_best[subject.file] = fitted

        if combo_idx == 1 or combo_idx == len(combos) or combo_idx % 10 == 0:
            print(f"[fit] {combo_idx}/{len(combos)} normative parameter pairs evaluated")

    if global_best is None:
        raise RuntimeError("No global fit was produced")
    return subject_best, global_best, global_scores


def accuracy_under_params(
    subject: SubjectData,
    trials_by_id: dict[RowId, PreparedTrial],
    belief_sigma: float | None,
    bias: float,
    hs: np.ndarray,
    mu1: float,
    mu2: float,
) -> float:
    correct = 0
    for choice in subject.choices:
        prediction = normative_prediction(trials_by_id[choice.row_id], belief_sigma, bias, hs, mu1, mu2)
        correct += int(prediction == choice.choice)
    return correct / len(subject.choices)


def write_subject_results(
    path: Path,
    subjects: list[SubjectData],
    subject_best: dict[str, FitResult],
    global_best: FitResult,
    trials_by_id: dict[RowId, PreparedTrial],
    hs: np.ndarray,
    mu1: float,
    mu2: float,
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
        "best_belief_sigma",
        "best_bias",
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
                trials_by_id,
                global_best.belief_sigma,
                global_best.bias,
                hs,
                mu1,
                mu2,
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
                    "best_belief_sigma": format_sigma(fit.belief_sigma),
                    "best_bias": f"{fit.bias:g}",
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
                "best_belief_sigma",
                "best_bias",
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
                "best_belief_sigma": format_sigma(global_best.belief_sigma),
                "best_bias": f"{global_best.bias:g}",
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
            fieldnames=["belief_sigma", "bias", "accuracy", "n_correct", "n_trials"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    path: Path,
    subjects: list[SubjectData],
    subject_best: dict[str, FitResult],
    global_best: FitResult,
    global_scores: list[dict[str, str]],
) -> None:
    ordered_subjects = sorted(subjects, key=lambda s: s.subject)
    fits = [subject_best[s.file] for s in ordered_subjects]
    labels = [s.subject for s in ordered_subjects]
    accuracies = [fit.accuracy for fit in fits]
    biases = [fit.bias for fit in fits]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar([str(label) for label in labels], biases, color="#70a6d8")
    ax.axhline(global_best.bias, color="#d62728", linewidth=2, label="pooled bias")
    ax.set_xlabel("subject")
    ax.set_ylabel("bias")
    ax.set_title("Best Bayesian prior bias")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend()

    ax = axes[1]
    ax.bar([str(label) for label in labels], accuracies, color="#88c999")
    ax.axhline(global_best.accuracy, color="#d62728", linewidth=2, label="pooled accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlabel("subject")
    ax.set_ylabel("accuracy")
    ax.set_title("Individual fit accuracy")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend()

    ax = axes[2]
    sigma_labels = list(dict.fromkeys(row["belief_sigma"] for row in global_scores))
    bias_values = sorted({float(row["bias"]) for row in global_scores})
    heatmap = np.full((len(bias_values), len(sigma_labels)), np.nan)
    sigma_index = {label: idx for idx, label in enumerate(sigma_labels)}
    bias_index = {bias: idx for idx, bias in enumerate(bias_values)}
    for row in global_scores:
        heatmap[bias_index[float(row["bias"])], sigma_index[row["belief_sigma"]]] = float(row["accuracy"])

    image = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(sigma_labels)))
    ax.set_xticklabels(sigma_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(bias_values)))
    ax.set_yticklabels([f"{bias:g}" for bias in bias_values])
    ax.set_xlabel("belief sigma")
    ax.set_ylabel("bias")
    ax.set_title("Pooled normative fit")
    fig.colorbar(image, ax=ax, label="accuracy")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_variant_rows(variant_dir: Path) -> dict[int, list[dict[str, str]]]:
    variant_rows_by_num: dict[int, list[dict[str, str]]] = {}
    for path in sorted(variant_dir.glob("data_var*.csv")):
        match = re.search(r"data_var(\d+)\.csv$", path.name)
        if match is not None:
            variant_rows_by_num[int(match.group(1))] = read_csv(path)
    if not variant_rows_by_num:
        raise FileNotFoundError(f"No variant CSVs found in {variant_dir}")
    return variant_rows_by_num


def main() -> None:
    args = parse_args()
    belief_sigmas = parse_belief_sigmas(args.belief_sigmas)
    biases = parse_float_grid(args.biases)
    hs = np.array(parse_float_grid(args.hs), dtype=float)
    if len(hs) == 0:
        raise ValueError("--hs must contain at least one hazard value")

    variant_rows_by_num = load_variant_rows(args.variant_dir)
    subjects = [
        align_subject(path, variant_rows_by_num, args.target_column)
        for path in sorted(args.subject_dir.glob("subject_*.csv"))
    ]
    if not subjects:
        raise FileNotFoundError(f"No subject CSVs found in {args.subject_dir}")

    used_row_ids = sorted({choice.row_id for subject in subjects for choice in subject.choices})
    trials_by_id = {}
    for variant_num, row_idx in used_row_ids:
        row = variant_rows_by_num[variant_num][row_idx]
        row_id = (variant_num, row_idx)
        trials_by_id[row_id] = PreparedTrial(
            row_id=row_id,
            evidence=np.array(parse_evidence(row["evidence"]), dtype=float),
            sigma=float(row["sigma"]),
        )

    print(f"[data] subjects: {len(subjects)}")
    print(f"[data] fitted prediction choices: {sum(len(s.choices) for s in subjects)}")
    print(f"[data] unique generated trials used: {len(used_row_ids)}")
    print(f"[bayes] belief sigma candidates: {[format_sigma(value) for value in belief_sigmas]}")
    print(f"[bayes] bias candidates: {biases}")
    print(f"[bayes] hs support: {hs.tolist()}")

    subject_best, global_best, global_scores = fit_all(
        subjects,
        trials_by_id,
        belief_sigmas,
        biases,
        hs,
        args.mu1,
        args.mu2,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject_path = args.out_dir / "subject_fit_results.csv"
    global_path = args.out_dir / "global_fit_result.csv"
    scores_path = args.out_dir / "global_model_search_scores.csv"
    write_subject_results(subject_path, subjects, subject_best, global_best, trials_by_id, hs, args.mu1, args.mu2)
    write_global_result(global_path, global_best, subjects)
    write_global_scores(scores_path, global_scores)

    if not args.no_plots:
        plot_summary(args.out_dir / "fit_summary.png", subjects, subject_best, global_best, global_scores)

    print("[result] pooled best normative fit")
    print(f"  belief_sigma={format_sigma(global_best.belief_sigma)}, bias={global_best.bias:g}")
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

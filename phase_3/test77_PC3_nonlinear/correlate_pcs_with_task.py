#!/usr/bin/env python3
"""Match PCA states to task variables and compare linear/quadratic decoders."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_CONFIG = BASE_DIR / "config.json"
DEFAULT_NORMATIVE_MODEL = REPO_ROOT / "utils" / "NormativeModel.py"
PC_RE = re.compile(r"^pc(\d+)$", re.IGNORECASE)
DEFAULT_DECODE_TARGETS = [
    "true_hazard",
    "true_report",
    "true_predict",
    "report_pred",
    "predict_pred",
    "task_evidence_value",
    "task_cumulative_evidence",
    "task_mean_evidence",
    "task_empirical_hazard",
    "task_current_state_run_length",
    "norm_intermediate_hazard_mean",
    "norm_intermediate_hazard_map",
    "norm_intermediate_hazard_sd",
    "norm_intermediate_hazard_entropy",
    "norm_intermediate_hazard_max_probability",
    "norm_intermediate_state_signed_belief",
    "norm_intermediate_state_confidence",
    "norm_intermediate_state_entropy",
    "norm_intermediate_state2_log_odds",
    "norm_intermediate_resp_report",
    "norm_intermediate_resp_predict",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match leading hidden-state PCs to task/normative variables, then "
            "compare cross-validated linear and quadratic decoders."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "PCA hidden-state CSV from pca_checkpoint_ep010.py. By default, "
            "auto-detect it in ./pca_outputs."
        ),
    )
    parser.add_argument(
        "--variant-dir",
        type=Path,
        default=None,
        help="Folder containing the source testConfig_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "pc_task_nonlinear",
        help="Output folder. Default: ./pc_task_nonlinear",
    )
    parser.add_argument(
        "--normative-model",
        type=Path,
        default=DEFAULT_NORMATIVE_MODEL,
        help=f"NormativeModel.py path. Default: {DEFAULT_NORMATIVE_MODEL}",
    )
    parser.add_argument(
        "--n-null-timesteps",
        type=int,
        default=None,
        help="Null steps inserted between evidence samples. Default: infer from hp.json.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Absolute evidence-distribution mean. Default: read TaskConfig.csv.",
    )
    parser.add_argument(
        "--hazard-step",
        type=float,
        default=0.05,
        help="Spacing of the BayesianObserver hazard grid. Default: 0.05",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Bias passed to BayesianObserver. Default: 0",
    )
    parser.add_argument(
        "--top-n-pcs",
        type=int,
        default=None,
        help="Analyze only this many leading PCs. Default: every PC in the input.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=3,
        help="Minimum finite pairs required for a correlation. Default: 3",
    )
    parser.add_argument(
        "--skip-by-timestep",
        action="store_true",
        help="Do not write the separate correlation-at-each-timestep table.",
    )
    parser.add_argument(
        "--no-posterior-bins",
        action="store_true",
        help="Omit individual L_haz hazard-grid bins from matched data/correlations.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tables only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used only for exact ties in BayesianObserver. Default: 0",
    )
    parser.add_argument(
        "--decode-targets",
        type=str,
        default=",".join(DEFAULT_DECODE_TARGETS),
        help=(
            "Comma-separated variables to decode. Defaults to key task, posterior "
            "mean/width/entropy, state-belief, and response variables."
        ),
    )
    parser.add_argument(
        "--decode-scopes",
        type=str,
        default="evidence_timesteps,final_timestep",
        help=(
            "Comma-separated scopes chosen from all_timesteps, "
            "evidence_timesteps, final_timestep. "
            "Default: evidence_timesteps,final_timestep"
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of trial-grouped cross-validation folds. Default: 5",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 penalty for continuous-target Ridge decoders. Default: 1",
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
        help="Inverse L2 penalty for binary LogisticRegression. Default: 1",
    )
    parser.add_argument(
        "--skip-nonlinear-decoders",
        action="store_true",
        help="Run only the original correlation analysis.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    config = load_json(DEFAULT_CONFIG)
    model_root = Path(config["model_root"]).expanduser().resolve()

    if args.variant_dir is None:
        variant_root = Path(config["variant_root"]).expanduser().resolve()
        variant_subdir = config.get("variant_subdir") or config.get("sigma") or model_root.name
        variant_dir = variant_root / variant_subdir
    else:
        variant_dir = args.variant_dir.expanduser().resolve()

    if args.input is None:
        candidates = [
            BASE_DIR / "pca_outputs" / "pca_ep010_hidden_states.csv",
            BASE_DIR / "pca_outputs" / "pca_ep010_final_timestep_hidden_states.csv",
        ]
        input_path = next((path for path in candidates if path.exists()), candidates[0])
    else:
        input_path = args.input.expanduser().resolve()

    output_dir = args.output_dir.expanduser().resolve()
    return input_path, variant_dir.resolve(), output_dir, {"model_root": model_root}


def infer_null_timesteps(
    requested: int | None,
    model_root: Path,
    pca_df: pd.DataFrame,
) -> int:
    if requested is not None:
        if requested < 0:
            raise ValueError("--n-null-timesteps cannot be negative")
        return requested

    if "seed" not in pca_df or pca_df.empty:
        raise ValueError("Cannot infer n_null_timesteps: PCA data has no seed column")
    seed = int(pca_df["seed"].iloc[0])
    hp_path = model_root / f"seed_{seed}" / "hp.json"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"Cannot infer null timesteps because {hp_path} is missing. "
            "Pass --n-null-timesteps explicitly."
        )
    return int(load_json(hp_path).get("n_null_timesteps", 0))


def load_mu(variant_dir: Path, requested: float | None) -> float:
    if requested is not None:
        return float(requested)

    task_config = variant_dir / "TaskConfig.csv"
    if task_config.exists():
        config = pd.read_csv(task_config, index_col=0)
        if "Mu" in config.index:
            return float(config.loc["Mu"].iloc[0])
    return 1.0


def import_bayesian_observer(path: Path) -> Callable[..., tuple[Any, ...]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Normative model does not exist: {path}")
    spec = importlib.util.spec_from_file_location("pc_analysis_normative_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = getattr(module, "BayesianObserver", None)
    if observer is None:
        raise AttributeError(f"{path} does not define BayesianObserver")
    return observer


def parse_float_list(value: Any, field: str) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse {field}: {value!r}") from exc
    result = np.asarray(parsed, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional sequence")
    return result


def pc_columns(df: pd.DataFrame, top_n: int | None) -> list[str]:
    found = []
    for column in df.columns:
        match = PC_RE.fullmatch(str(column))
        if match:
            found.append((int(match.group(1)), str(column)))
    found.sort()
    if not found:
        raise ValueError("Input has no PC columns named pc1, pc2, ...")
    columns = [column for _, column in found]
    if top_n is not None:
        if top_n <= 0:
            raise ValueError("--top-n-pcs must be positive")
        columns = columns[:top_n]
    return columns


def hazard_label(hazard: float) -> str:
    return f"h_{hazard:.6g}".replace("-", "m").replace(".", "p")


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def log_odds(probability: float, epsilon: float = 1e-12) -> float:
    clipped = float(np.clip(probability, epsilon, 1.0 - epsilon))
    return float(np.log(clipped / (1.0 - clipped)))


def run_lengths(states: np.ndarray) -> np.ndarray:
    lengths = np.ones(len(states), dtype=float)
    for index in range(1, len(states)):
        if states[index] == states[index - 1]:
            lengths[index] = lengths[index - 1] + 1.0
    return lengths


def load_trial_row(
    variant_dir: Path,
    source_csv: str,
    csv_trial: int,
    cache: dict[str, pd.DataFrame],
) -> pd.Series:
    if source_csv not in cache:
        path = variant_dir / source_csv
        if not path.exists():
            raise FileNotFoundError(f"PCA row refers to missing source CSV: {path}")
        cache[source_csv] = pd.read_csv(path)
    source = cache[source_csv]
    if csv_trial < 0 or csv_trial >= len(source):
        raise IndexError(f"{source_csv} has no row {csv_trial}")
    return source.iloc[csv_trial]


def trial_feature_frame(
    trial_pca: pd.DataFrame,
    source_row: pd.Series,
    observer: Callable[..., tuple[Any, ...]],
    hs: np.ndarray,
    mu: float,
    bias: float,
    n_null_timesteps: int,
    include_posterior_bins: bool,
) -> pd.DataFrame:
    evidence = parse_float_list(source_row["evidence"], "evidence")
    states = parse_float_list(source_row["states"], "states")
    if len(states) != len(evidence):
        raise ValueError("states and evidence have different lengths")

    sigma = float(source_row["sigma"])
    normative_return = observer(
        evidence.tolist(),
        mu1=-mu,
        mu2=mu,
        sigma=sigma,
        hs=hs,
        bias=bias,
    )
    if len(normative_return) != 4:
        raise ValueError(
            "BayesianObserver must return (L_haz, L_state, resp_Rep, resp_Pred)"
        )
    l_haz, l_state, resp_report, resp_predict = normative_return
    l_haz = np.asarray(l_haz, dtype=float)
    l_state = np.asarray(l_state, dtype=float)
    expected_haz_shape = (len(hs), len(evidence) + 1)
    expected_state_shape = (2, len(evidence) + 1)
    if l_haz.shape != expected_haz_shape:
        raise ValueError(f"L_haz shape {l_haz.shape}, expected {expected_haz_shape}")
    if l_state.shape != expected_state_shape:
        raise ValueError(f"L_state shape {l_state.shape}, expected {expected_state_shape}")

    timesteps = trial_pca["timestep"].to_numpy(dtype=int)
    stride = n_null_timesteps + 1
    evidence_index = timesteps // stride
    if np.any(evidence_index < 0) or np.any(evidence_index >= len(evidence)):
        expected_last = (len(evidence) - 1) * stride
        raise ValueError(
            f"Hidden timestep is incompatible with {len(evidence)} evidence items and "
            f"{n_null_timesteps} null steps; final evidence should occur at {expected_last}"
        )
    normative_index = evidence_index + 1
    phase = timesteps % stride

    observed_hazard = l_haz[:, normative_index].T
    observed_state = l_state[:, normative_index].T
    hazard_mean = observed_hazard @ hs
    hazard_variance = np.sum(
        observed_hazard * np.square(hs[None, :] - hazard_mean[:, None]), axis=1
    )
    hazard_map = hs[np.argmax(observed_hazard, axis=1)]
    hazard_entropy = np.asarray([entropy(row) for row in observed_hazard])
    state_entropy = np.asarray([entropy(row) for row in observed_state])

    switches = np.zeros(len(states), dtype=float)
    switches[1:] = states[1:] != states[:-1]
    cumulative_switches = np.cumsum(switches)
    transition_counts = evidence_index.astype(float)
    empirical_hazard = np.divide(
        cumulative_switches[evidence_index],
        transition_counts,
        out=np.full(len(evidence_index), np.nan),
        where=transition_counts > 0,
    )
    cumulative_evidence = np.cumsum(evidence)
    evidence_counts = evidence_index + 1
    report_interim = np.sign(observed_state[:, 1] - observed_state[:, 0])
    predict_interim = np.sign(hazard_mean - 0.5)

    result = pd.DataFrame(index=trial_pca.index)
    result["task_sigma"] = sigma
    result["task_block_num"] = pd.to_numeric(source_row.get("blockNum"), errors="coerce")
    result["task_trial_in_block"] = pd.to_numeric(
        source_row.get("trialInBlock"), errors="coerce"
    )
    result["task_evidence_index"] = evidence_index
    result["task_evidence_number"] = evidence_counts
    result["task_is_evidence_timestep"] = (phase == 0).astype(int)
    result["task_null_steps_since_evidence"] = phase
    result["task_evidence_value"] = evidence[evidence_index]
    result["task_abs_evidence"] = np.abs(evidence[evidence_index])
    result["task_evidence_sign"] = np.sign(evidence[evidence_index])
    result["task_latent_state"] = states[evidence_index]
    result["task_state_switch"] = switches[evidence_index]
    result["task_cumulative_state_switches"] = cumulative_switches[evidence_index]
    result["task_empirical_hazard"] = empirical_hazard
    result["task_current_state_run_length"] = run_lengths(states)[evidence_index]
    result["task_cumulative_evidence"] = cumulative_evidence[evidence_index]
    result["task_mean_evidence"] = (
        cumulative_evidence[evidence_index] / evidence_counts
    )

    # L_state and L_haz are direct, time-matched values from BayesianObserver's returns.
    result["norm_return_l_state_s1"] = observed_state[:, 0]
    result["norm_return_l_state_s2"] = observed_state[:, 1]
    result["norm_return_resp_report"] = float(resp_report)
    result["norm_return_resp_predict"] = float(resp_predict)
    if include_posterior_bins:
        for hazard_index, hazard in enumerate(hs):
            result[f"norm_return_l_haz_{hazard_label(float(hazard))}"] = observed_hazard[
                :, hazard_index
            ]

    # These match named intermediate calculations in NormativeModel.py at every prefix.
    result["norm_intermediate_hazard_mean"] = hazard_mean
    result["norm_intermediate_hazard_map"] = hazard_map
    result["norm_intermediate_hazard_sd"] = np.sqrt(np.maximum(hazard_variance, 0.0))
    result["norm_intermediate_hazard_entropy"] = hazard_entropy
    result["norm_intermediate_hazard_entropy_normalized"] = hazard_entropy / np.log(
        len(hs)
    )
    result["norm_intermediate_hazard_max_probability"] = np.max(
        observed_hazard, axis=1
    )
    result["norm_intermediate_p_switch"] = hazard_mean
    result["norm_intermediate_p_stay"] = 1.0 - hazard_mean
    result["norm_intermediate_switch_log_odds"] = [
        log_odds(value) for value in hazard_mean
    ]
    result["norm_intermediate_state_signed_belief"] = (
        observed_state[:, 1] - observed_state[:, 0]
    )
    result["norm_intermediate_state_confidence"] = np.max(observed_state, axis=1)
    result["norm_intermediate_state_entropy"] = state_entropy
    result["norm_intermediate_state2_log_odds"] = [
        log_odds(value) for value in observed_state[:, 1]
    ]
    result["norm_intermediate_resp_report"] = report_interim
    result["norm_intermediate_resp_predict"] = predict_interim
    result["norm_return_report_matches_target"] = (
        float(resp_report) == float(source_row["trueReport"])
    )
    result["norm_return_predict_matches_target"] = (
        float(resp_predict) == float(source_row["truePredict"])
    )
    return result


def build_matched_data(
    pca_df: pd.DataFrame,
    variant_dir: Path,
    observer: Callable[..., tuple[Any, ...]],
    hs: np.ndarray,
    mu: float,
    bias: float,
    n_null_timesteps: int,
    include_posterior_bins: bool,
) -> pd.DataFrame:
    required = {"source_csv", "csv_trial", "global_trial", "timestep"}
    missing = sorted(required - set(pca_df.columns))
    if missing:
        raise ValueError(f"PCA input is missing columns: {missing}")

    cache: dict[str, pd.DataFrame] = {}
    feature_chunks = []
    pending_frames = []
    grouped = pca_df.groupby("global_trial", sort=False)
    total = grouped.ngroups
    for group_number, (_, trial_pca) in enumerate(grouped, start=1):
        source_csv = str(trial_pca["source_csv"].iloc[0])
        csv_trial = int(trial_pca["csv_trial"].iloc[0])
        source_row = load_trial_row(variant_dir, source_csv, csv_trial, cache)
        pending_frames.append(
            trial_feature_frame(
                trial_pca=trial_pca,
                source_row=source_row,
                observer=observer,
                hs=hs,
                mu=mu,
                bias=bias,
                n_null_timesteps=n_null_timesteps,
                include_posterior_bins=include_posterior_bins,
            )
        )
        if len(pending_frames) == 250:
            feature_chunks.append(pd.concat(pending_frames))
            pending_frames.clear()
        if group_number % 500 == 0 or group_number == total:
            print(f"Matched normative trajectories: {group_number}/{total} trials")

    if pending_frames:
        feature_chunks.append(pd.concat(pending_frames))
    features = pd.concat(feature_chunks).sort_index()
    return pd.concat([pca_df, features], axis=1)


def parameter_columns(df: pd.DataFrame) -> list[str]:
    preferred_existing = [
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
    ]
    parameters = [column for column in preferred_existing if column in df]
    parameters.extend(
        column
        for column in df.columns
        if column.startswith("task_") or column.startswith("norm_")
    )
    return list(dict.fromkeys(parameters))


def parameter_family(parameter: str) -> str:
    if parameter.startswith("norm_return"):
        return "normative_return"
    if parameter.startswith("norm_intermediate"):
        return "normative_intermediate"
    if parameter.startswith("task_"):
        return "task_derived"
    if parameter in {"report_pred", "predict_pred"}:
        return "model_response"
    if parameter.endswith("_correct") or parameter == "combined_correct":
        return "model_accuracy"
    return "task_recorded"


def correlation_rows(
    df: pd.DataFrame,
    pc_cols: list[str],
    parameters: list[str],
    scope: str,
    min_pairs: int,
    timestep: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pc in pc_cols:
        x_all = pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=float)
        for parameter in parameters:
            y_all = pd.to_numeric(df[parameter], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x_all) & np.isfinite(y_all)
            x = x_all[valid]
            y = y_all[valid]
            n = len(x)
            x_unique = np.unique(x).size
            y_unique = np.unique(y).size
            if n < min_pairs or x_unique < 2 or y_unique < 2:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pearson_r, pearson_p = pearsonr(x, y)
                    spearman_r, spearman_p = spearmanr(x, y)
            row = {
                "scope": scope,
                "pc": pc,
                "parameter": parameter,
                "parameter_family": parameter_family(parameter),
                "n": n,
                "parameter_unique_values": y_unique,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_r,
                "spearman_p": spearman_p,
            }
            if timestep is not None:
                row["timestep"] = timestep
            rows.append(row)
    return rows


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.dropna()
    if finite.empty:
        return result
    ordered = finite.sort_values()
    count = len(ordered)
    ranks = np.arange(1, count + 1, dtype=float)
    adjusted = ordered.to_numpy(dtype=float) * count / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def add_fdr(correlations: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    correlations = correlations.copy()
    correlations["pearson_q"] = np.nan
    correlations["spearman_q"] = np.nan
    for _, indices in correlations.groupby(group_columns, dropna=False).groups.items():
        correlations.loc[indices, "pearson_q"] = benjamini_hochberg(
            correlations.loc[indices, "pearson_p"]
        )
        correlations.loc[indices, "spearman_q"] = benjamini_hochberg(
            correlations.loc[indices, "spearman_p"]
        )
    return correlations


def final_timestep_rows(df: pd.DataFrame) -> pd.DataFrame:
    indices = df.groupby("global_trial")["timestep"].idxmax()
    return df.loc[indices].copy()


def compute_correlations(
    matched: pd.DataFrame,
    pc_cols: list[str],
    parameters: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    scopes = {
        "all_timesteps": matched,
        "evidence_timesteps": matched[
            matched["task_is_evidence_timestep"].astype(bool)
        ],
        "final_timestep": final_timestep_rows(matched),
    }
    rows = []
    for scope, frame in scopes.items():
        rows.extend(correlation_rows(frame, pc_cols, parameters, scope, min_pairs))
    return add_fdr(pd.DataFrame(rows), ["scope"])


def compute_correlations_by_timestep(
    matched: pd.DataFrame,
    pc_cols: list[str],
    parameters: list[str],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    for timestep, frame in matched.groupby("timestep", sort=True):
        rows.extend(
            correlation_rows(
                frame,
                pc_cols,
                parameters,
                scope="single_timestep",
                min_pairs=min_pairs,
                timestep=int(timestep),
            )
        )
    return add_fdr(pd.DataFrame(rows), ["timestep"])


def plot_correlation_heatmap(
    correlations: pd.DataFrame,
    pc_cols: list[str],
    out_path: Path,
    max_parameters: int = 30,
) -> None:
    usable = correlations[
        (correlations["scope"] == "evidence_timesteps")
        & correlations["pearson_r"].notna()
    ].copy()
    if usable.empty:
        return
    strongest = (
        usable.assign(abs_r=usable["pearson_r"].abs())
        .groupby("parameter")["abs_r"]
        .max()
        .nlargest(max_parameters)
        .index
    )
    matrix = (
        usable[usable["parameter"].isin(strongest)]
        .pivot(index="parameter", columns="pc", values="pearson_r")
        .reindex(columns=pc_cols)
    )
    matrix = matrix.reindex(
        matrix.abs().max(axis=1).sort_values(ascending=False).index
    )

    fig_height = max(6.0, 0.28 * len(matrix))
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(pc_cols), fig_height))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_title("PC correlations at evidence timesteps")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_strongest_correlations(
    correlations: pd.DataFrame,
    out_path: Path,
    count: int = 20,
) -> None:
    usable = correlations[
        (correlations["scope"] == "final_timestep")
        & correlations["pearson_r"].notna()
    ].copy()
    if usable.empty:
        return
    usable["label"] = usable["pc"] + " × " + usable["parameter"]
    usable["abs_r"] = usable["pearson_r"].abs()
    strongest = usable.nlargest(count, "abs_r").sort_values("pearson_r")

    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.34 * len(strongest))))
    colors = np.where(strongest["pearson_r"] >= 0, "#b54a5a", "#4169a1")
    ax.barh(strongest["label"], strongest["pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson r")
    ax.set_title("Strongest final-timestep PC correlations")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_comma_list(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def decoder_scope_frames(
    matched: pd.DataFrame,
    requested_scopes: list[str],
) -> dict[str, pd.DataFrame]:
    valid_scopes = {
        "all_timesteps",
        "evidence_timesteps",
        "final_timestep",
    }
    unknown = sorted(set(requested_scopes) - valid_scopes)
    if unknown:
        raise ValueError(
            f"Unknown decoder scopes {unknown}; choose from {sorted(valid_scopes)}"
        )

    frames = {}
    for scope in requested_scopes:
        if scope == "all_timesteps":
            frames[scope] = matched
        elif scope == "evidence_timesteps":
            frames[scope] = matched[
                matched["task_is_evidence_timestep"].astype(bool)
            ]
        else:
            frames[scope] = final_timestep_rows(matched)
    return frames


def decoder_feature_matrices(
    frame: pd.DataFrame,
    pc_cols: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    linear = frame[pc_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    quadratic = polynomial.fit_transform(linear)
    feature_names = {
        "linear": list(pc_cols),
        "quadratic": polynomial.get_feature_names_out(pc_cols).tolist(),
    }
    return {"linear": linear, "quadratic": quadratic}, feature_names


def target_type(values: np.ndarray) -> str:
    unique = np.unique(values[np.isfinite(values)])
    if len(unique) == 2:
        return "binary_classification"
    return "continuous_regression"


def finite_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pearsonr(x, y).statistic)


def evaluate_decoder_fold(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    ridge_alpha: float,
    logistic_c: float,
) -> dict[str, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    metrics = {
        "r2": np.nan,
        "mae": np.nan,
        "prediction_pearson_r": np.nan,
        "roc_auc": np.nan,
        "balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "log_loss": np.nan,
    }
    if kind == "continuous_regression":
        model = Ridge(alpha=ridge_alpha)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        metrics.update(
            {
                "r2": float(r2_score(y_test, prediction)),
                "mae": float(mean_absolute_error(y_test, prediction)),
                "prediction_pearson_r": finite_pearson(y_test, prediction),
            }
        )
        metrics["primary_metric"] = metrics["r2"]
        metrics["primary_metric_name"] = "r2"
        return metrics

    classes = np.sort(np.unique(y_train))
    class_to_int = {value: index for index, value in enumerate(classes)}
    y_train_binary = np.asarray([class_to_int[value] for value in y_train], dtype=int)
    if any(value not in class_to_int for value in np.unique(y_test)):
        raise ValueError("A test fold contains a class absent from its training fold")
    y_test_binary = np.asarray([class_to_int[value] for value in y_test], dtype=int)
    model = LogisticRegression(
        C=logistic_c,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(x_train, y_train_binary)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics.update(
        {
            "roc_auc": float(roc_auc_score(y_test_binary, probability)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test_binary, prediction)
            ),
            "accuracy": float(accuracy_score(y_test_binary, prediction)),
            "log_loss": float(
                log_loss(y_test_binary, probability, labels=[0, 1])
            ),
        }
    )
    metrics["primary_metric"] = metrics["roc_auc"]
    metrics["primary_metric_name"] = "roc_auc"
    return metrics


def decoder_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "primary_metric",
        "r2",
        "mae",
        "prediction_pearson_r",
        "roc_auc",
        "balanced_accuracy",
        "accuracy",
        "log_loss",
    ]
    group_columns = [
        "scope",
        "target",
        "target_type",
        "primary_metric_name",
        "representation",
    ]
    rows = []
    for keys, group in fold_metrics.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row["n_folds"] = len(group)
        row["n_test_total"] = int(group["n_test"].sum())
        for metric in metric_columns:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def decoder_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "scope",
        "target",
        "target_type",
        "primary_metric_name",
        "fold",
    ]
    paired = fold_metrics.pivot(
        index=index_columns,
        columns="representation",
        values="primary_metric",
    ).reset_index()
    paired = paired.dropna(subset=["linear", "quadratic"]).copy()
    paired["quadratic_gain"] = paired["quadratic"] - paired["linear"]

    rows = []
    group_columns = index_columns[:-1]
    for keys, group in paired.groupby(group_columns, sort=False):
        gains = group["quadratic_gain"]
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_folds": len(group),
                "linear_mean": float(group["linear"].mean()),
                "quadratic_mean": float(group["quadratic"].mean()),
                "quadratic_gain_mean": float(gains.mean()),
                "quadratic_gain_std": (
                    float(gains.std(ddof=1)) if len(gains) > 1 else np.nan
                ),
                "quadratic_gain_sem": (
                    float(gains.std(ddof=1) / np.sqrt(len(gains)))
                    if len(gains) > 1
                    else np.nan
                ),
                "quadratic_better_folds": int((gains > 0).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_nonlinear_decoders(
    matched: pd.DataFrame,
    pc_cols: list[str],
    targets: list[str],
    scopes: list[str],
    cv_folds: int,
    ridge_alpha: float,
    logistic_c: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    if cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if ridge_alpha < 0:
        raise ValueError("--ridge-alpha cannot be negative")
    if logistic_c <= 0:
        raise ValueError("--logistic-c must be positive")

    available_targets = [target for target in targets if target in matched.columns]
    missing_targets = sorted(set(targets) - set(available_targets))
    if missing_targets:
        print(f"[warn] Skipping missing decoder targets: {missing_targets}")
    if not available_targets:
        raise ValueError("None of the requested decoder targets exist in matched data")

    scope_frames = decoder_scope_frames(matched, scopes)
    all_rows = []
    feature_names: dict[str, list[str]] | None = None
    for scope, frame in scope_frames.items():
        matrices, scope_feature_names = decoder_feature_matrices(frame, pc_cols)
        if feature_names is None:
            feature_names = scope_feature_names
        groups_all = pd.to_numeric(
            frame["global_trial"], errors="coerce"
        ).to_numpy(dtype=float)
        pc_finite = np.isfinite(matrices["quadratic"]).all(axis=1)

        for target in available_targets:
            y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
            valid = pc_finite & np.isfinite(groups_all) & np.isfinite(y_all)
            y = y_all[valid]
            groups = groups_all[valid].astype(np.int64)
            kind = target_type(y)
            unique_groups = np.unique(groups)
            if len(unique_groups) < cv_folds or np.unique(y).size < 2:
                print(
                    f"[warn] Skipping {scope}/{target}: insufficient groups or "
                    "target variation"
                )
                continue

            splitter = GroupKFold(n_splits=cv_folds)
            valid_matrices = {
                representation: matrix[valid]
                for representation, matrix in matrices.items()
            }
            split_indices = list(
                splitter.split(valid_matrices["linear"], y, groups=groups)
            )
            print(
                f"Decoding {scope}/{target} ({kind}, n={len(y):,}, "
                f"trials={len(unique_groups):,})"
            )
            for fold, (train_index, test_index) in enumerate(split_indices, start=1):
                y_train = y[train_index]
                y_test = y[test_index]
                if kind == "binary_classification" and (
                    np.unique(y_train).size != 2 or np.unique(y_test).size != 2
                ):
                    print(
                        f"[warn] Skipping fold {fold} for {scope}/{target}: "
                        "both classes are not present"
                    )
                    continue

                for representation, x in valid_matrices.items():
                    metrics = evaluate_decoder_fold(
                        x_train=x[train_index],
                        x_test=x[test_index],
                        y_train=y_train,
                        y_test=y_test,
                        kind=kind,
                        ridge_alpha=ridge_alpha,
                        logistic_c=logistic_c,
                    )
                    all_rows.append(
                        {
                            "scope": scope,
                            "target": target,
                            "target_type": kind,
                            "fold": fold,
                            "representation": representation,
                            "n_features": x.shape[1],
                            "n_train": len(train_index),
                            "n_test": len(test_index),
                            "n_train_trials": np.unique(groups[train_index]).size,
                            "n_test_trials": np.unique(groups[test_index]).size,
                            **metrics,
                        }
                    )

    if not all_rows or feature_names is None:
        raise ValueError("No decoder folds were successfully evaluated")
    fold_metrics = pd.DataFrame(all_rows)
    return (
        fold_metrics,
        decoder_summary(fold_metrics),
        decoder_comparison(fold_metrics),
        feature_names,
    )


def plot_decoder_gains(comparison: pd.DataFrame, output_dir: Path) -> None:
    for scope, frame in comparison.groupby("scope", sort=False):
        frame = frame.sort_values("quadratic_gain_mean")
        colors = np.where(
            frame["target_type"].eq("binary_classification"),
            "#b55f36",
            "#3f6fa5",
        )
        fig, ax = plt.subplots(figsize=(10, max(6.0, 0.38 * len(frame))))
        ax.barh(
            frame["target"],
            frame["quadratic_gain_mean"],
            xerr=frame["quadratic_gain_sem"].fillna(0.0),
            color=colors,
            alpha=0.9,
        )
        ax.axvline(0.0, color="black", linewidth=0.9)
        ax.set_xlabel("Quadratic - linear CV score (R2 or ROC-AUC)")
        ax.set_title(f"Nonlinear decoding gain: {scope}")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"quadratic_decoder_gain_{scope}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_decoder_outputs(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_names: dict[str, list[str]],
    output_dir: Path,
    make_plots: bool,
) -> None:
    fold_metrics.to_csv(output_dir / "decoder_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "decoder_summary.csv", index=False)
    comparison.to_csv(output_dir / "linear_vs_quadratic_decoders.csv", index=False)
    with (output_dir / "decoder_feature_sets.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(feature_names, handle, indent=2)
    if make_plots:
        plot_decoder_gains(comparison, output_dir)


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    input_path: Path,
    variant_dir: Path,
    pc_cols: list[str],
    n_null_timesteps: int,
    mu: float,
    hs: np.ndarray,
    matched: pd.DataFrame,
) -> None:
    config = {
        "input": str(input_path),
        "variant_dir": str(variant_dir),
        "normative_model": str(args.normative_model.expanduser().resolve()),
        "pc_columns": pc_cols,
        "n_null_timesteps": n_null_timesteps,
        "mu": mu,
        "hazard_grid": hs.tolist(),
        "bias": args.bias,
        "min_pairs": args.min_pairs,
        "posterior_bins_included": not args.no_posterior_bins,
        "decoder_targets": parse_comma_list(args.decode_targets),
        "decoder_scopes": parse_comma_list(args.decode_scopes),
        "cv_folds": args.cv_folds,
        "ridge_alpha": args.ridge_alpha,
        "logistic_c": args.logistic_c,
        "nonlinear_decoders_run": not args.skip_nonlinear_decoders,
        "n_rows": len(matched),
        "n_trials": int(matched["global_trial"].nunique()),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    input_path, variant_dir, output_dir, config = resolve_paths(args)
    if not input_path.exists():
        raise FileNotFoundError(
            f"PCA input does not exist: {input_path}\n"
            "Run pca_checkpoint_ep010.py first or pass --input."
        )
    if not variant_dir.exists():
        raise FileNotFoundError(f"Variant directory does not exist: {variant_dir}")
    if args.hazard_step <= 0 or args.hazard_step >= 1:
        raise ValueError("--hazard-step must be between 0 and 1")
    if args.min_pairs < 3:
        raise ValueError("--min-pairs must be at least 3")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    pca_df = pd.read_csv(input_path)
    pcs = pc_columns(pca_df, args.top_n_pcs)
    n_null_timesteps = infer_null_timesteps(
        args.n_null_timesteps, config["model_root"], pca_df
    )
    mu = load_mu(variant_dir, args.mu)
    hs = np.arange(0.0, 1.0, args.hazard_step)
    observer = import_bayesian_observer(args.normative_model)

    print(f"Loaded {len(pca_df):,} PCA rows from {input_path}")
    print(f"Analyzing PCs: {', '.join(pcs)}")
    print(
        f"Normative matching: mu={mu:g}, hazard_step={args.hazard_step:g}, "
        f"null_timesteps={n_null_timesteps}"
    )
    matched = build_matched_data(
        pca_df=pca_df,
        variant_dir=variant_dir,
        observer=observer,
        hs=hs,
        mu=mu,
        bias=args.bias,
        n_null_timesteps=n_null_timesteps,
        include_posterior_bins=not args.no_posterior_bins,
    )

    matched_path = output_dir / "pc_task_normative_matched.csv.gz"
    matched.to_csv(matched_path, index=False, compression="gzip")
    parameters = parameter_columns(matched)
    correlations = compute_correlations(matched, pcs, parameters, args.min_pairs)
    correlation_path = output_dir / "pc_parameter_correlations.csv"
    correlations.to_csv(correlation_path, index=False)

    if not args.skip_by_timestep:
        by_timestep = compute_correlations_by_timestep(
            matched, pcs, parameters, args.min_pairs
        )
        by_timestep.to_csv(
            output_dir / "pc_parameter_correlations_by_timestep.csv", index=False
        )

    if not args.no_plots:
        plot_correlation_heatmap(
            correlations, pcs, output_dir / "pc_parameter_correlation_heatmap.png"
        )
        plot_strongest_correlations(
            correlations, output_dir / "strongest_final_pc_correlations.png"
        )

    if not args.skip_nonlinear_decoders:
        fold_metrics, decoder_metrics, decoder_comparisons, feature_names = (
            run_nonlinear_decoders(
                matched=matched,
                pc_cols=pcs,
                targets=parse_comma_list(args.decode_targets),
                scopes=parse_comma_list(args.decode_scopes),
                cv_folds=args.cv_folds,
                ridge_alpha=args.ridge_alpha,
                logistic_c=args.logistic_c,
            )
        )
        save_decoder_outputs(
            fold_metrics=fold_metrics,
            summary=decoder_metrics,
            comparison=decoder_comparisons,
            feature_names=feature_names,
            output_dir=output_dir,
            make_plots=not args.no_plots,
        )
        print(
            "Wrote trial-grouped linear/quadratic decoder comparisons to "
            f"{output_dir}"
        )

    write_run_config(
        output_dir / "run_config.json",
        args,
        input_path,
        variant_dir,
        pcs,
        n_null_timesteps,
        mu,
        hs,
        matched,
    )
    print(f"Wrote matched PC/task/normative rows to {matched_path}")
    print(f"Wrote correlations to {correlation_path}")
    print(f"Wrote run metadata and plots to {output_dir}")


if __name__ == "__main__":
    main()

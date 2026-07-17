#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "behavioral_timing_outputs"
DEFAULT_MODEL_CLASS = "GRUModel"
DEFAULT_VARIANT_SPLIT = "test"
CHECKPOINT_EPOCH_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")
SEED_RE = re.compile(r"seed_(\d+)$")


@dataclass(frozen=True)
class ModelSpec:
    role: str
    label: str
    root: Path


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    label: str
    order: int
    epoch: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether n5 report behavior tracks the last-evidence heuristic "
            "before/while the predict head is learned."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant-split", default=DEFAULT_VARIANT_SPLIT)
    parser.add_argument("--max-variant-csvs", type=int, default=None)
    parser.add_argument(
        "--seeds",
        default="all",
        help='Comma-separated seed ids, e.g. "0,1,2", or "all". Default: all',
    )
    parser.add_argument(
        "--checkpoints",
        default="training",
        help=(
            '"training" uses checkpoint_init.pt, checkpoint_ep*.pt, final.pt. '
            'Use "all" to include checkpoint_best.pt too, or pass comma-separated names.'
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="'cuda', 'cpu', 'auto', or a torch device string. Default: cuda",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--model-class", default=DEFAULT_MODEL_CLASS)
    parser.add_argument(
        "--weak-evidence-threshold",
        type=float,
        default=0.5,
        help="Absolute final-evidence cutoff for the weak-final-evidence subset.",
    )
    parser.add_argument(
        "--zero-report",
        type=int,
        choices=[-1, 1],
        default=1,
        help="Sign used when the final evidence sample is exactly zero.",
    )
    parser.add_argument(
        "--save-trial-predictions",
        action="store_true",
        help="Write one row per trial/model/checkpoint. This can be large.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["model_root", "model_comparison_root", "variant_root"]
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    for key in required:
        cfg[key] = Path(cfg[key]).expanduser().resolve()
    return cfg


def safe_model_label(root: Path) -> str:
    parts = root.parts[-3:] if len(root.parts) >= 3 else root.parts
    return "_".join(parts).replace(" ", "_")


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def parse_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    parsed = ast.literal_eval(str(value))
    return [float(item) for item in parsed]


def sign_with_zero(value: float, zero_report: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return int(zero_report)


def count_sign_switches(values: list[float], zero_report: int) -> int:
    signs = [sign_with_zero(float(value), zero_report) for value in values]
    return int(sum(1 for prev, curr in zip(signs, signs[1:]) if prev != curr))


def encode_evidence_sequence(
    evidence: list[float],
    n_input: int,
    n_null_timesteps: int,
) -> torch.Tensor:
    if not evidence:
        raise ValueError("Evidence sequence cannot be empty")

    if n_input == 1:
        return torch.tensor(evidence, dtype=torch.float32).unsqueeze(-1)

    if n_input == 2:
        steps: list[list[float]] = []
        null_step = [0.0, 0.0]
        for i, evidence_t in enumerate(evidence):
            steps.append([float(evidence_t), 1.0])
            if i < len(evidence) - 1:
                steps.extend([null_step.copy() for _ in range(n_null_timesteps)])
        return torch.tensor(steps, dtype=torch.float32)

    raise ValueError(f"Unsupported n_input={n_input}; expected 1 or 2")


def list_variant_csvs(
    variant_dir: Path,
    split: str,
    max_variant_csvs: int | None,
) -> list[Path]:
    csvs = sorted(variant_dir.glob(f"{split}Config_*.csv"), key=natural_key)
    if max_variant_csvs is not None:
        csvs = csvs[:max_variant_csvs]
    if not csvs:
        raise FileNotFoundError(f"No {split}Config_*.csv files found in {variant_dir}")
    return csvs


def load_metadata_and_inputs(
    csvs: list[Path],
    n_input: int,
    n_null_timesteps: int,
    zero_report: int,
) -> tuple[pd.DataFrame, torch.Tensor]:
    rows: list[dict[str, Any]] = []
    xs: list[torch.Tensor] = []
    global_trial = 0

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        for csv_trial, row in df.reset_index(drop=True).iterrows():
            evidence = parse_list(row["evidence"])
            states = parse_list(row["states"]) if "states" in row else []
            last_evidence = float(evidence[-1])
            last_sign = sign_with_zero(last_evidence, zero_report)
            true_report = int(float(row["trueReport"]))
            true_predict = int(float(row["truePredict"]))
            true_hazard = float(row["trueHazard"])
            latent_switch_count = count_sign_switches(states, zero_report) if states else np.nan

            xs.append(encode_evidence_sequence(evidence, n_input, n_null_timesteps))
            rows.append(
                {
                    "source_csv": csv_path.name,
                    "csv_trial": int(csv_trial),
                    "global_trial": int(global_trial),
                    "trial_in_block": row.get("trialInBlock", np.nan),
                    "true_hazard": true_hazard,
                    "true_report": true_report,
                    "true_predict": true_predict,
                    "last_evidence": last_evidence,
                    "abs_last_evidence": abs(last_evidence),
                    "last_evidence_sign": last_sign,
                    "last_matches_true_report": int(last_sign == true_report),
                    "last_conflicts_true_report": int(last_sign != true_report),
                    "apparent_evidence_switch_count": count_sign_switches(evidence, zero_report),
                    "latent_switch_count": latent_switch_count,
                }
            )
            global_trial += 1

    if not xs:
        raise ValueError("No trials were loaded.")

    return pd.DataFrame(rows), torch.stack(xs, dim=0)


def find_model_code_root(model_dir: Path) -> Path:
    for path in (model_dir, *model_dir.parents):
        if (path / "rnn_models.py").exists():
            return path
    raise FileNotFoundError(f"Could not find rnn_models.py at or above {model_dir}")


def import_model_class(model_root: Path, class_name: str):
    module_path = find_model_code_root(model_root) / "rnn_models.py"
    module_id = abs(hash(str(module_path.resolve())))
    spec = importlib.util.spec_from_file_location(f"test75_rnn_models_{module_id}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} does not define {class_name}") from exc


def load_hp(seed_dir: Path) -> dict[str, Any]:
    hp_path = seed_dir / "hp.json"
    if not hp_path.exists():
        raise FileNotFoundError(f"Missing hyperparameter file: {hp_path}")
    with hp_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(
    model_cls,
    hp: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
):
    model = model_cls(hp).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def discover_seed_dirs(root: Path, seeds: str) -> list[Path]:
    if seeds.strip().lower() == "all":
        seed_dirs = []
        for path in root.iterdir():
            match = SEED_RE.fullmatch(path.name)
            if path.is_dir() and match and (path / "hp.json").exists():
                seed_dirs.append(path)
        seed_dirs.sort(key=lambda path: int(SEED_RE.fullmatch(path.name).group(1)))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed_* directories found in {root}")
        return seed_dirs

    seed_dirs = []
    for part in seeds.split(","):
        if not part.strip():
            continue
        seed_dir = root / f"seed_{int(part.strip())}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
        seed_dirs.append(seed_dir)
    if not seed_dirs:
        raise ValueError("--seeds did not specify any seed ids.")
    return seed_dirs


def checkpoint_spec_from_path(path: Path, max_epoch: int | None = None) -> CheckpointSpec:
    if path.name == "checkpoint_init.pt":
        return CheckpointSpec(path.name, "init", 0, 0.0)
    match = CHECKPOINT_EPOCH_RE.fullmatch(path.name)
    if match:
        epoch = int(match.group(1))
        return CheckpointSpec(path.name, f"ep{epoch:03d}", epoch, float(epoch))
    if path.name == "final.pt":
        order = (max_epoch + 1) if max_epoch is not None else 10_001
        epoch = float(order)
        return CheckpointSpec(path.name, "final", order, epoch)
    if path.name == "checkpoint_best.pt":
        order = (max_epoch + 2) if max_epoch is not None else 10_002
        epoch = float("nan")
        return CheckpointSpec(path.name, "best", order, epoch)
    return CheckpointSpec(path.name, path.stem, 20_000, float("nan"))


def discover_checkpoints(seed_dir: Path, value: str) -> list[CheckpointSpec]:
    epoch_paths = sorted(seed_dir.glob("checkpoint_ep*.pt"), key=lambda p: int(CHECKPOINT_EPOCH_RE.fullmatch(p.name).group(1)))
    max_epoch = max((int(CHECKPOINT_EPOCH_RE.fullmatch(p.name).group(1)) for p in epoch_paths), default=None)

    if value.strip().lower() in {"training", "default"}:
        paths: list[Path] = []
        init_path = seed_dir / "checkpoint_init.pt"
        if init_path.exists():
            paths.append(init_path)
        paths.extend(epoch_paths)
        final_path = seed_dir / "final.pt"
        if final_path.exists():
            paths.append(final_path)
        if not paths:
            raise FileNotFoundError(f"No training checkpoints found in {seed_dir}")
        return [checkpoint_spec_from_path(path, max_epoch) for path in paths]

    if value.strip().lower() == "all":
        paths = sorted(
            [path for path in seed_dir.glob("*.pt") if path.is_file()],
            key=lambda p: checkpoint_spec_from_path(p, max_epoch).order,
        )
        return [checkpoint_spec_from_path(path, max_epoch) for path in paths]

    specs = []
    for raw_name in value.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if not name.endswith(".pt"):
            if name == "init":
                name = "checkpoint_init.pt"
            elif name == "final":
                name = "final.pt"
            elif name == "best":
                name = "checkpoint_best.pt"
            elif name.isdigit():
                name = f"checkpoint_ep{int(name):03d}.pt"
            elif re.fullmatch(r"ep\d+", name):
                name = f"checkpoint_{name}.pt"
        path = seed_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        specs.append(checkpoint_spec_from_path(path, max_epoch))
    if not specs:
        raise ValueError("--checkpoints did not specify any checkpoints.")
    return sorted(specs, key=lambda spec: spec.order)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def predict_model(
    model,
    x_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(x_tensor, torch.arange(x_tensor.shape[0])),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
    )
    report_logits = np.empty(x_tensor.shape[0], dtype=np.float32)
    predict_logits = np.empty(x_tensor.shape[0], dtype=np.float32)

    with torch.inference_mode():
        for xb, idx in loader:
            xb = xb.to(device, non_blocking=use_cuda)
            loc_logits, haz_logits = model(xb)
            report_logits[idx.numpy()] = loc_logits[:, -1, 0].detach().cpu().numpy()
            predict_logits[idx.numpy()] = haz_logits[:, 0].detach().cpu().numpy()

    report_prob = 1.0 / (1.0 + np.exp(-report_logits))
    predict_prob = 1.0 / (1.0 + np.exp(-predict_logits))
    return {
        "report_logit": report_logits,
        "predict_logit": predict_logits,
        "report_prob": report_prob,
        "predict_prob": predict_prob,
        "report_pred": np.where(report_prob > 0.5, 1, -1).astype(np.int16),
        "predict_pred": np.where(predict_prob > 0.5, 1, -1).astype(np.int16),
    }


def mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def summarize_mask(
    mask: np.ndarray,
    report_pred: np.ndarray,
    predict_pred: np.ndarray,
    report_prob: np.ndarray,
    predict_prob: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, float]:
    idx = np.asarray(mask, dtype=bool)
    n = int(idx.sum())
    true_report = meta["true_report"].to_numpy()
    true_predict = meta["true_predict"].to_numpy()
    last_sign = meta["last_evidence_sign"].to_numpy()

    if n == 0:
        return {
            "n_trials": 0,
            "report_accuracy": float("nan"),
            "predict_accuracy": float("nan"),
            "combined_accuracy": float("nan"),
            "report_last_evidence_agreement": float("nan"),
            "heuristic_report_accuracy": float("nan"),
            "report_advantage_over_heuristic": float("nan"),
            "report_confidence": float("nan"),
            "predict_confidence": float("nan"),
        }

    report_accuracy = mean_or_nan(report_pred[idx] == true_report[idx])
    predict_accuracy = mean_or_nan(predict_pred[idx] == true_predict[idx])
    heuristic_report_accuracy = mean_or_nan(last_sign[idx] == true_report[idx])
    return {
        "n_trials": n,
        "report_accuracy": report_accuracy,
        "predict_accuracy": predict_accuracy,
        "combined_accuracy": 0.5 * (report_accuracy + predict_accuracy),
        "report_last_evidence_agreement": mean_or_nan(report_pred[idx] == last_sign[idx]),
        "heuristic_report_accuracy": heuristic_report_accuracy,
        "report_advantage_over_heuristic": report_accuracy - heuristic_report_accuracy,
        "report_confidence": mean_or_nan(np.abs(report_prob[idx] - 0.5) * 2.0),
        "predict_confidence": mean_or_nan(np.abs(predict_prob[idx] - 0.5) * 2.0),
    }


def subset_masks(
    meta: pd.DataFrame,
    report_pred: np.ndarray,
    predict_pred: np.ndarray,
    weak_evidence_threshold: float,
) -> dict[str, np.ndarray]:
    true_report = meta["true_report"].to_numpy()
    true_predict = meta["true_predict"].to_numpy()
    last_sign = meta["last_evidence_sign"].to_numpy()
    true_hazard = meta["true_hazard"].to_numpy()
    abs_last = meta["abs_last_evidence"].to_numpy()

    all_mask = np.ones(len(meta), dtype=bool)
    conflict = last_sign != true_report
    aligned = last_sign == true_report
    weak = abs_last <= weak_evidence_threshold
    predict_correct = predict_pred == true_predict
    report_correct = report_pred == true_report
    high_hazard = true_hazard >= 0.5

    return {
        "all": all_mask,
        "last_matches_true_report": aligned,
        "last_conflicts_true_report": conflict,
        "weak_final_evidence": weak,
        "weak_final_evidence_and_conflict": weak & conflict,
        "predict_correct": predict_correct,
        "predict_wrong": ~predict_correct,
        "predict_correct_and_conflict": predict_correct & conflict,
        "predict_wrong_and_conflict": (~predict_correct) & conflict,
        "report_correct": report_correct,
        "report_wrong": ~report_correct,
        "low_true_hazard": ~high_hazard,
        "high_true_hazard": high_hazard,
        "low_true_hazard_and_conflict": (~high_hazard) & conflict,
        "high_true_hazard_and_conflict": high_hazard & conflict,
    }


def flatten_prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def make_metric_rows(
    base: dict[str, Any],
    preds: dict[str, np.ndarray],
    meta: pd.DataFrame,
    weak_evidence_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    masks = subset_masks(meta, preds["report_pred"], preds["predict_pred"], weak_evidence_threshold)
    subset_rows = []
    wide_row = dict(base)

    for subset_name, mask in masks.items():
        metrics = summarize_mask(
            mask=mask,
            report_pred=preds["report_pred"],
            predict_pred=preds["predict_pred"],
            report_prob=preds["report_prob"],
            predict_prob=preds["predict_prob"],
            meta=meta,
        )
        subset_rows.append({**base, "subset_name": subset_name, **metrics})
        if subset_name == "all":
            wide_row.update(metrics)
        elif subset_name in {
            "last_conflicts_true_report",
            "last_matches_true_report",
            "weak_final_evidence",
            "weak_final_evidence_and_conflict",
            "predict_correct_and_conflict",
            "predict_wrong_and_conflict",
            "low_true_hazard_and_conflict",
            "high_true_hazard_and_conflict",
        }:
            prefix = {
                "last_conflicts_true_report": "conflict",
                "last_matches_true_report": "aligned",
                "weak_final_evidence": "weak",
                "weak_final_evidence_and_conflict": "weak_conflict",
                "predict_correct_and_conflict": "predict_correct_conflict",
                "predict_wrong_and_conflict": "predict_wrong_conflict",
                "low_true_hazard_and_conflict": "low_hazard_conflict",
                "high_true_hazard_and_conflict": "high_hazard_conflict",
            }[subset_name]
            wide_row.update(flatten_prefixed(prefix, metrics))

    wide_row["diagnostic_n_trials"] = wide_row.get("conflict_n_trials", 0)
    wide_row["diagnostic_report_accuracy"] = wide_row.get("conflict_report_accuracy", float("nan"))
    wide_row["diagnostic_last_evidence_agreement"] = wide_row.get(
        "conflict_report_last_evidence_agreement", float("nan")
    )
    wide_row["diagnostic_report_advantage_over_heuristic"] = wide_row.get(
        "conflict_report_advantage_over_heuristic", float("nan")
    )
    return wide_row, subset_rows


def trial_prediction_rows(
    base: dict[str, Any],
    preds: dict[str, np.ndarray],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    df = meta[
        [
            "source_csv",
            "csv_trial",
            "global_trial",
            "trial_in_block",
            "true_hazard",
            "true_report",
            "true_predict",
            "last_evidence",
            "abs_last_evidence",
            "last_evidence_sign",
            "last_matches_true_report",
            "last_conflicts_true_report",
            "apparent_evidence_switch_count",
            "latent_switch_count",
        ]
    ].copy()
    for key, value in base.items():
        df[key] = value
    for key in ["report_logit", "predict_logit", "report_prob", "predict_prob", "report_pred", "predict_pred"]:
        df[key] = preds[key]
    df["report_correct"] = (df["report_pred"] == df["true_report"]).astype(int)
    df["predict_correct"] = (df["predict_pred"] == df["true_predict"]).astype(int)
    df["report_last_evidence_agreement"] = (df["report_pred"] == df["last_evidence_sign"]).astype(int)
    return df


def aggregate_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"seed", "checkpoint_order", "checkpoint_epoch"}
    ]
    rows = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}
        row["n_seed_rows"] = int(len(group_df))
        for col in numeric_cols:
            values = pd.to_numeric(group_df[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else float("nan")
            row[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else float("nan")
            row[f"{col}_sem"] = (
                float(values.std(ddof=1) / math.sqrt(values.notna().sum()))
                if values.notna().sum() > 1
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_training_history(model_specs: list[ModelSpec], seeds_by_role: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    for spec in model_specs:
        for seed_dir in seeds_by_role[spec.role]:
            seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
            for split, filename in [("train", "loss_history.json"), ("val", "val_loss_history.json")]:
                path = seed_dir / filename
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as f:
                    values = json.load(f)
                for i, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "model_role": spec.role,
                            "model_label": spec.label,
                            "seed": seed,
                            "split": split,
                            "epoch": i,
                            "loss": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def write_timing_correlations(checkpoint_metrics: pd.DataFrame, out_path: Path) -> None:
    metric_pairs = [
        ("predict_accuracy", "diagnostic_report_accuracy"),
        ("predict_accuracy", "diagnostic_last_evidence_agreement"),
        ("checkpoint_order", "diagnostic_report_accuracy"),
        ("checkpoint_order", "report_last_evidence_agreement"),
        ("checkpoint_order", "predict_accuracy"),
    ]
    rows = []
    for role, role_df in checkpoint_metrics.groupby("model_role"):
        base_mean_df = (
            role_df.groupby(["checkpoint_order", "checkpoint_label"], as_index=False)
            .mean(numeric_only=True)
            .sort_values("checkpoint_order")
        )
        filters = {
            "all_checkpoints": np.ones(len(base_mean_df), dtype=bool),
            "epoch_checkpoints_only": base_mean_df["checkpoint_label"].astype(str).str.startswith("ep").to_numpy(),
            "post_init": (base_mean_df["checkpoint_label"].astype(str) != "init").to_numpy(),
        }
        for filter_name, filter_mask in filters.items():
            mean_df = base_mean_df[filter_mask].copy()
            for x_col, y_col in metric_pairs:
                if x_col not in mean_df or y_col not in mean_df:
                    continue
                valid = mean_df[[x_col, y_col]].dropna()
                corr = float(valid[x_col].corr(valid[y_col])) if len(valid) >= 2 else float("nan")
                rows.append(
                    {
                        "model_role": role,
                        "checkpoint_filter": filter_name,
                        "x_metric": x_col,
                        "y_metric": y_col,
                        "pearson_r_across_checkpoint_means": corr,
                        "n_checkpoints": int(len(valid)),
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    model_specs = [
        ModelSpec(
            role="true_report_training",
            label=safe_model_label(cfg["model_root"]),
            root=cfg["model_root"],
        ),
        ModelSpec(
            role="last_evidence_training",
            label=safe_model_label(cfg["model_comparison_root"]),
            root=cfg["model_comparison_root"],
        ),
    ]

    first_hp = load_hp(discover_seed_dirs(model_specs[0].root, args.seeds)[0])
    variant_subdir = cfg.get("variant_subdir") or cfg["model_root"].name
    variant_dir = cfg["variant_root"] / variant_subdir
    csvs = list_variant_csvs(variant_dir, args.variant_split, args.max_variant_csvs)
    meta, x_tensor = load_metadata_and_inputs(
        csvs=csvs,
        n_input=int(first_hp["n_input"]),
        n_null_timesteps=int(first_hp.get("n_null_timesteps", 0)),
        zero_report=int(args.zero_report),
    )

    print(f"Loaded {len(meta)} trials from {len(csvs)} {args.variant_split} CSVs.")
    print(f"Using device: {device}")

    seeds_by_role = {spec.role: discover_seed_dirs(spec.root, args.seeds) for spec in model_specs}
    model_classes = {spec.role: import_model_class(spec.root, args.model_class) for spec in model_specs}
    checkpoint_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []

    for spec in model_specs:
        for seed_dir in seeds_by_role[spec.role]:
            seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
            hp = load_hp(seed_dir)
            checkpoint_specs = discover_checkpoints(seed_dir, args.checkpoints)
            for ckpt in checkpoint_specs:
                checkpoint_path = seed_dir / ckpt.name
                print(f"Evaluating {spec.role} seed_{seed} {ckpt.label}...")
                model = load_model(model_classes[spec.role], hp, checkpoint_path, device)
                preds = predict_model(model, x_tensor, int(args.batch_size), device)
                base = {
                    "model_role": spec.role,
                    "model_label": spec.label,
                    "model_root": str(spec.root),
                    "seed": seed,
                    "checkpoint": ckpt.name,
                    "checkpoint_label": ckpt.label,
                    "checkpoint_order": ckpt.order,
                    "checkpoint_epoch": ckpt.epoch,
                    "report_target": hp.get("report_target", "true_report"),
                    "train_heads": hp.get("train_heads", ""),
                    "loss_type": hp.get("loss_type", ""),
                }
                wide_row, rows = make_metric_rows(base, preds, meta, float(args.weak_evidence_threshold))
                checkpoint_rows.append(wide_row)
                subset_rows.extend(rows)
                if args.save_trial_predictions:
                    trial_frames.append(trial_prediction_rows(base, preds, meta))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(
        ["model_role", "seed", "checkpoint_order", "checkpoint"]
    )
    subset_df = pd.DataFrame(subset_rows).sort_values(
        ["model_role", "seed", "checkpoint_order", "subset_name"]
    )
    aggregate_checkpoint_df = aggregate_metrics(
        checkpoint_df,
        ["model_role", "model_label", "checkpoint", "checkpoint_label", "checkpoint_order"],
    ).sort_values(["model_role", "checkpoint_order", "checkpoint"])
    aggregate_subset_df = aggregate_metrics(
        subset_df,
        ["model_role", "model_label", "checkpoint", "checkpoint_label", "checkpoint_order", "subset_name"],
    ).sort_values(["model_role", "subset_name", "checkpoint_order", "checkpoint"])
    history_df = load_training_history(model_specs, seeds_by_role)

    checkpoint_path = output_dir / "checkpoint_metrics.csv"
    subset_path = output_dir / "subset_metrics.csv"
    aggregate_checkpoint_path = output_dir / "aggregate_checkpoint_metrics.csv"
    aggregate_subset_path = output_dir / "aggregate_subset_metrics.csv"
    history_path = output_dir / "training_history.csv"
    correlations_path = output_dir / "timing_correlations.csv"

    checkpoint_df.to_csv(checkpoint_path, index=False)
    subset_df.to_csv(subset_path, index=False)
    aggregate_checkpoint_df.to_csv(aggregate_checkpoint_path, index=False)
    aggregate_subset_df.to_csv(aggregate_subset_path, index=False)
    history_df.to_csv(history_path, index=False)
    write_timing_correlations(checkpoint_df, correlations_path)

    if args.save_trial_predictions:
        pd.concat(trial_frames, ignore_index=True).to_csv(output_dir / "trial_predictions.csv", index=False)

    run_config = {
        "config": str(args.config.expanduser().resolve()),
        "output_dir": str(output_dir),
        "variant_dir": str(variant_dir),
        "variant_split": args.variant_split,
        "variant_csvs": [str(path) for path in csvs],
        "n_trials": int(len(meta)),
        "model_specs": [spec.__dict__ | {"root": str(spec.root)} for spec in model_specs],
        "seeds": args.seeds,
        "checkpoints": args.checkpoints,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "weak_evidence_threshold": float(args.weak_evidence_threshold),
        "zero_report": int(args.zero_report),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"Saved checkpoint metrics to {checkpoint_path}")
    print(f"Saved subset metrics to {subset_path}")
    print(f"Saved aggregate metrics to {aggregate_checkpoint_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()

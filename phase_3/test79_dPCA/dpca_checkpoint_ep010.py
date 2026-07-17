#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config.json"
DEFAULT_VARIANT_SPLIT = "test"
DEFAULT_MAX_VARIANT_CSVS = None
DEFAULT_MODEL_CLASS = "GRUModel"
SEED_RE = re.compile(r"seed_(\d+)$")
CHECKPOINT_NAME = "checkpoint_ep010.pt"
ANALYSIS_SPECS: dict[str, dict[str, Any]] = {
    "hazard_continuous": {
        "factor_names": ("hazard_bin", "true_report", "timestep"),
        "default_marginals": (
            "time",
            "hazard",
            "hazard_time",
            "report",
            "report_time",
            "hazard_report",
            "hazard_report_time",
            "task_interaction",
        ),
        "marginal_groups": {
            "time": (("timestep",),),
            "hazard": (("hazard_bin",),),
            "hazard_time": (("hazard_bin", "timestep"),),
            "report": (("true_report",),),
            "report_time": (("true_report", "timestep"),),
            "hazard_report": (("hazard_bin", "true_report"),),
            "hazard_report_time": (("hazard_bin", "true_report", "timestep"),),
            "task_interaction": (
                ("hazard_bin", "true_report"),
                ("hazard_bin", "true_report", "timestep"),
            ),
        },
        "dropped_from_demixing": ["true_predict"],
        "note": (
            "true_predict is a binary thresholded/readout version of continuous "
            "trueHazard. It is retained as metadata/readout only, not as an "
            "independent dPCA factor."
        ),
    },
    "legacy_predict": {
        "factor_names": ("hazard_bin", "true_report", "true_predict", "timestep"),
        "default_marginals": (
            "time",
            "hazard",
            "hazard_time",
            "report",
            "report_time",
            "predict",
            "predict_time",
            "task_interaction",
        ),
        "marginal_groups": {
            "time": (("timestep",),),
            "hazard": (("hazard_bin",),),
            "report": (("true_report",),),
            "predict": (("true_predict",),),
            "hazard_time": (("hazard_bin", "timestep"),),
            "report_time": (("true_report", "timestep"),),
            "predict_time": (("true_predict", "timestep"),),
            "hazard_report": (("hazard_bin", "true_report"),),
            "hazard_predict": (("hazard_bin", "true_predict"),),
            "report_predict": (("true_report", "true_predict"),),
            "hazard_report_time": (("hazard_bin", "true_report", "timestep"),),
            "hazard_predict_time": (("hazard_bin", "true_predict", "timestep"),),
            "report_predict_time": (("true_report", "true_predict", "timestep"),),
            "task": (("hazard_bin", "true_report", "true_predict"),),
            "condition": (("hazard_bin", "true_report", "true_predict", "timestep"),),
            "task_interaction": (
                ("hazard_bin", "true_report"),
                ("hazard_bin", "true_predict"),
                ("true_report", "true_predict"),
                ("hazard_bin", "true_report", "true_predict"),
                ("hazard_bin", "true_report", "timestep"),
                ("hazard_bin", "true_predict", "timestep"),
                ("true_report", "true_predict", "timestep"),
                ("hazard_bin", "true_report", "true_predict", "timestep"),
            ),
        },
        "dropped_from_demixing": [],
        "note": "Legacy mode preserves the previous true_predict demixing factor.",
    },
}


@dataclass
class DPCAFit:
    analysis_mode: str
    factor_names: tuple[str, ...]
    marginal_names: list[str]
    marginal_groups: dict[str, tuple[tuple[str, ...], ...]]
    components: dict[str, np.ndarray]
    singular_values: dict[str, np.ndarray]
    explained_variance_ratio: dict[str, np.ndarray]
    within_marginal_explained_variance_ratio: dict[str, np.ndarray]
    marginal_variance: dict[str, float]
    total_variance: float
    grand_mean: np.ndarray
    factor_levels: dict[str, list[Any]]
    condition_counts: np.ndarray
    hazard_bin_edges: list[float]
    hazard_bin_labels: dict[int, str]


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


def compute_hazard_bins(
    values: np.ndarray,
    n_bins: int,
    strategy: str,
) -> tuple[np.ndarray, list[float], dict[int, str]]:
    if n_bins < 1:
        raise ValueError("--hazard-bins must be at least 1")

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bin an empty hazard vector")

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Hazard values are all NaN or infinite")

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    unique_values = np.unique(finite_values)
    effective_bins = min(int(n_bins), int(unique_values.size))

    if effective_bins <= 1 or np.isclose(min_value, max_value):
        edges = [min_value, max_value]
        indices = np.zeros(values.shape, dtype=int)
    else:
        if strategy == "quantile":
            raw_edges = np.quantile(finite_values, np.linspace(0.0, 1.0, effective_bins + 1))
        elif strategy == "equal":
            raw_edges = np.linspace(min_value, max_value, effective_bins + 1)
        else:
            raise ValueError(f"Unsupported hazard bin strategy: {strategy}")

        edges_array = np.unique(raw_edges.astype(float))
        if edges_array.size <= 1:
            edges = [min_value, max_value]
            indices = np.zeros(values.shape, dtype=int)
        else:
            edges_array[0] = min_value
            edges_array[-1] = max_value
            edges = [float(edge) for edge in edges_array]
            indices = np.searchsorted(edges_array[1:-1], values, side="right").astype(int)
            indices = np.clip(indices, 0, len(edges) - 2)

    interval_count = max(1, len(edges) - 1)
    base_labels = ["low", "mid", "high"] if interval_count == 3 else [
        f"bin_{i + 1}" for i in range(interval_count)
    ]
    labels = {
        i: f"{base_labels[i]} [{edges[i]:.3f}, {edges[i + 1]:.3f}]"
        for i in range(interval_count)
    }
    return indices, edges, labels


class HelicopterDPCADataset(Dataset):
    def __init__(
        self,
        csv_paths: list[Path],
        n_input: int,
        n_null_timesteps: int,
        hazard_bins: int,
        hazard_bin_strategy: str,
    ):
        self.x = []
        self.trial_meta = []

        global_trial = 0
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            for csv_trial, row in df.reset_index(drop=True).iterrows():
                evidence = row["evidence"]
                if not isinstance(evidence, list):
                    evidence = ast.literal_eval(str(evidence))

                self.x.append(encode_evidence_sequence(evidence, n_input, n_null_timesteps))
                true_report = int(float(row["trueReport"]))
                true_predict = int(float(row["truePredict"]))
                self.trial_meta.append(
                    {
                        "source_csv": csv_path.name,
                        "csv_trial": int(csv_trial),
                        "global_trial": int(global_trial),
                        "trial_in_block": row.get("trialInBlock", np.nan),
                        "true_hazard": float(row["trueHazard"]),
                        "true_report": true_report,
                        "true_predict": true_predict,
                    }
                )
                global_trial += 1

        if not self.x:
            raise ValueError("No dPCA trials were loaded.")

        hazards = np.asarray([meta["true_hazard"] for meta in self.trial_meta], dtype=float)
        bin_indices, edges, labels = compute_hazard_bins(
            hazards,
            int(hazard_bins),
            hazard_bin_strategy,
        )
        self.hazard_bin_edges = edges
        self.hazard_bin_labels = labels
        for meta, bin_index in zip(self.trial_meta, bin_indices, strict=True):
            hazard_bin = int(bin_index)
            meta["hazard_bin"] = hazard_bin
            meta["hazard_bin_label"] = labels[hazard_bin]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dPCA on seed checkpoint_ep010 hidden states."
    )
    parser.add_argument(
        "--analysis-mode",
        choices=sorted(ANALYSIS_SPECS),
        default="hazard_continuous",
        help=(
            "Analysis mode. Default: hazard_continuous. Use legacy_predict "
            "to preserve the previous true_predict dPCA factor."
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of dPCA components per marginal. Default: 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed to fit and transform. Default: 1",
    )
    parser.add_argument(
        "--hazard-bins",
        type=int,
        default=3,
        help="Number of bins used to demix continuous true_hazard. Default: 3",
    )
    parser.add_argument(
        "--hazard-bin-strategy",
        choices=["quantile", "equal"],
        default="quantile",
        help="How to build true_hazard bins. Default: quantile",
    )
    parser.add_argument(
        "--marginals",
        default=None,
        help=(
            "Comma-separated dPCA marginals. Use 'all' for every supported "
            "marginal. Default: analysis-mode-specific defaults."
        ),
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge alpha for continuous true_hazard probes. Default: 10.0",
    )
    parser.add_argument(
        "--probe-test-size",
        type=float,
        default=0.25,
        help="Held-out fraction for continuous true_hazard probes. Default: 0.25",
    )
    parser.add_argument(
        "--probe-random-state",
        type=int,
        default=0,
        help="Random seed for probe train/test splits. Default: 0",
    )
    parser.add_argument(
        "--dry-run-check",
        action="store_true",
        help="Validate factors/marginals and print the dPCA tensor shape without fitting.",
    )
    parser.add_argument(
        "--max-variant-csvs",
        type=int,
        default=DEFAULT_MAX_VARIANT_CSVS,
        help="Maximum number of variant CSVs to load. Default: all",
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
        default=BASE_DIR / "dpca_outputs",
        help="Directory for dPCA CSVs and plots. Default: ./dpca_outputs",
    )
    parser.add_argument(
        "--timestep-mode",
        choices=["all", "final"],
        default="all",
        help="Which hidden-state timesteps to save and plot. Default: all",
    )
    return parser.parse_args()


def analysis_spec(analysis_mode: str) -> dict[str, Any]:
    try:
        return ANALYSIS_SPECS[analysis_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}") from exc


def parse_marginals(value: str | None, analysis_mode: str) -> list[str]:
    spec = analysis_spec(analysis_mode)
    marginal_groups = spec["marginal_groups"]
    if value is None:
        return list(spec["default_marginals"])

    if value.strip().lower() == "all":
        return list(marginal_groups)

    marginal_names = [part.strip() for part in value.split(",") if part.strip()]
    if not marginal_names:
        raise ValueError("--marginals must name at least one marginal")

    if analysis_mode == "hazard_continuous":
        predict_marginals = [name for name in marginal_names if "predict" in name]
        if predict_marginals:
            raise ValueError(
                "hazard_continuous mode drops true_predict from dPCA demixing; "
                f"predict marginal(s) are not allowed: {predict_marginals}"
            )

    unknown = [name for name in marginal_names if name not in marginal_groups]
    if unknown:
        known = ", ".join(sorted(marginal_groups))
        raise ValueError(f"Unknown dPCA marginal(s): {unknown}. Supported: {known}")

    return marginal_names


def validate_analysis_config(cfg: dict[str, Any]) -> None:
    factor_names = tuple(cfg["factor_names"])
    marginal_groups = cfg["marginal_groups"]
    unknown_factors = {
        factor
        for marginal_name in cfg["marginals"]
        for subset in marginal_groups[marginal_name]
        for factor in subset
        if factor not in factor_names
    }
    if unknown_factors:
        raise ValueError(
            "Marginals reference factor(s) not present in this analysis mode: "
            f"{sorted(unknown_factors)}"
        )

    if cfg["analysis_mode"] == "hazard_continuous":
        predict_marginals = [name for name in cfg["marginals"] if "predict" in name]
        predict_subsets = [
            subset
            for marginal_name in cfg["marginals"]
            for subset in marginal_groups[marginal_name]
            if "true_predict" in subset
        ]
        if predict_marginals or predict_subsets:
            raise ValueError(
                "hazard_continuous mode must not include predict marginals or "
                f"true_predict subsets. marginals={predict_marginals}, subsets={predict_subsets}"
            )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["model_root", "variant_root"]
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    cfg["model_root"] = Path(cfg["model_root"]).expanduser().resolve()
    cfg["variant_root"] = Path(cfg["variant_root"]).expanduser().resolve()
    return cfg


def infer_model_label(model_root: Path) -> str:
    if model_root.parent.name:
        return f"{model_root.parent.name}/{model_root.name}"
    return model_root.name


def find_model_code_root(model_dir: Path) -> Path:
    for path in (model_dir, *model_dir.parents):
        if (path / "rnn_models.py").exists():
            return path
    raise FileNotFoundError(f"Could not find rnn_models.py at or above {model_dir}")


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(DEFAULT_CONFIG)
    spec = analysis_spec(args.analysis_mode)
    model_subdir = cfg.get("model_subdir") or infer_model_label(cfg["model_root"])
    variant_subdir = cfg.get("variant_subdir") or cfg.get("sigma") or cfg["model_root"].name
    cfg.update(
        {
            "analysis_mode": args.analysis_mode,
            "factor_names": tuple(spec["factor_names"]),
            "marginal_groups": spec["marginal_groups"],
            "dropped_from_demixing": list(spec["dropped_from_demixing"]),
            "analysis_note": spec["note"],
            "model_subdir": model_subdir,
            "variant_subdir": variant_subdir,
            "variant_split": DEFAULT_VARIANT_SPLIT,
            "max_variant_csvs": args.max_variant_csvs,
            "model_class": DEFAULT_MODEL_CLASS,
            "n_components": args.n_components,
            "seed": args.seed,
            "hazard_bins": args.hazard_bins,
            "hazard_bin_strategy": args.hazard_bin_strategy,
            "marginals": parse_marginals(args.marginals, args.analysis_mode),
            "ridge_alpha": args.ridge_alpha,
            "probe_test_size": args.probe_test_size,
            "probe_random_state": args.probe_random_state,
            "dry_run_check": args.dry_run_check,
            "max_plot_points": args.max_plot_points,
            "output_dir": args.output_dir.expanduser().resolve(),
            "checkpoint_name": CHECKPOINT_NAME,
            "timestep_mode": args.timestep_mode,
        }
    )
    validate_analysis_config(cfg)
    cfg["model_dir"] = cfg["model_root"]
    cfg["variant_dir"] = cfg["variant_root"] / cfg["variant_subdir"]
    return cfg


def output_prefix(cfg: dict[str, Any]) -> str:
    return f"dpca_ep010_{cfg['analysis_mode']}"


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def list_eval_csvs(cfg: dict[str, Any]) -> list[Path]:
    pattern = f"{cfg['variant_split']}Config_*.csv"
    csvs = sorted(cfg["variant_dir"].glob(pattern), key=natural_key)
    if cfg["max_variant_csvs"] is not None:
        csvs = csvs[: int(cfg["max_variant_csvs"])]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for {cfg['variant_dir'] / pattern}")
    return csvs


def import_model_class(model_root: Path, class_name: str):
    module_path = find_model_code_root(model_root) / "rnn_models.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find {module_path}")

    spec = importlib.util.spec_from_file_location("ots_rnn_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} does not define {class_name}") from exc


def get_seed_dir(model_dir: Path, seed: int, checkpoint_name: str) -> Path:
    seed_dir = model_dir / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
    if not (seed_dir / checkpoint_name).exists():
        raise FileNotFoundError(f"Missing checkpoint: {seed_dir / checkpoint_name}")
    return seed_dir


def load_hp(seed_dir: Path) -> dict[str, Any]:
    hp_path = seed_dir / "hp.json"
    if hp_path.exists():
        with hp_path.open("r", encoding="utf-8") as f:
            hp = json.load(f)
    else:
        hp = {}

    hp.setdefault("n_input", 1)
    hp.setdefault("n_rnn", 128)
    hp.setdefault("batch_size", 25)
    hp.setdefault("train_heads", "both")
    hp.setdefault("n_null_timesteps", 4)
    return hp


def collate_batch(batch):
    xs, idxs = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(idxs, dtype=torch.long)


def load_model(model_cls, seed_dir: Path, checkpoint_name: str, device: torch.device):
    hp = load_hp(seed_dir)
    model = model_cls(hp).to(device)
    state = torch.load(seed_dir / checkpoint_name, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def iter_hidden_batches(model, dataloader: DataLoader, device: torch.device):
    for x, trial_idx in dataloader:
        x = x.to(device)
        hidden = model.rnn(x)
        loc_logits = model.loc_head(hidden)
        predict_logits = model.haz_head(hidden[:, -1])
        report_pred = (torch.sigmoid(loc_logits[:, -1, :]) > 0.5).cpu().numpy().astype(int)
        predict_pred = (torch.sigmoid(predict_logits) > 0.5).cpu().numpy().astype(int)
        yield hidden.detach().cpu().numpy(), trial_idx.numpy(), report_pred, predict_pred


def factor_levels_from_dataset(
    dataset: HelicopterDPCADataset,
    n_time: int,
    factor_names: tuple[str, ...],
) -> dict[str, list[Any]]:
    levels: dict[str, list[Any]] = {}
    for factor in factor_names:
        if factor == "hazard_bin":
            levels[factor] = sorted({int(meta["hazard_bin"]) for meta in dataset.trial_meta})
        elif factor == "true_report":
            levels[factor] = sorted({int(meta["true_report"]) for meta in dataset.trial_meta})
        elif factor == "true_predict":
            levels[factor] = sorted({int(meta["true_predict"]) for meta in dataset.trial_meta})
        elif factor == "timestep":
            levels[factor] = list(range(int(n_time)))
        else:
            raise ValueError(f"Unsupported dPCA factor: {factor}")
    return levels


def factor_index_maps(factor_levels: dict[str, list[Any]]) -> dict[str, dict[Any, int]]:
    return {
        factor: {level: i for i, level in enumerate(levels)}
        for factor, levels in factor_levels.items()
    }


def compute_anova_effects(
    condition_means: np.ndarray,
    factor_names: tuple[str, ...],
) -> dict[tuple[str, ...], np.ndarray]:
    condition_axes = tuple(range(len(factor_names)))
    centered = condition_means - condition_means.mean(axis=condition_axes, keepdims=True)
    effects: dict[tuple[str, ...], np.ndarray] = {}

    for size in range(1, len(factor_names) + 1):
        for subset in itertools.combinations(factor_names, size):
            keep_axes = {factor_names.index(factor) for factor in subset}
            mean_axes = tuple(axis for axis in condition_axes if axis not in keep_axes)
            effect = centered.mean(axis=mean_axes, keepdims=True)
            for lower_size in range(1, size):
                for lower_subset in itertools.combinations(subset, lower_size):
                    effect = effect - effects[lower_subset]
            effects[subset] = effect

    return effects


def marginal_tensor(
    effects: dict[tuple[str, ...], np.ndarray],
    marginal_name: str,
    marginal_groups: dict[str, tuple[tuple[str, ...], ...]],
) -> np.ndarray:
    tensors = [effects[subset] for subset in marginal_groups[marginal_name]]
    total = np.zeros_like(tensors[0])
    for tensor in tensors:
        total = total + tensor
    return total


def fit_components_for_marginal(
    matrix: np.ndarray,
    n_components: int,
    total_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n_units = matrix.shape[1]
    n_keep = min(int(n_components), n_units)
    marginal_variance = float(np.sum(matrix * matrix))

    if n_keep < 1:
        raise ValueError("--n-components must be at least 1")

    if marginal_variance <= 1e-12 or total_variance <= 1e-12:
        components = np.zeros((n_keep, n_units), dtype=float)
        singular_values = np.zeros(n_keep, dtype=float)
    else:
        _, singular_values_all, vt = np.linalg.svd(matrix, full_matrices=False)
        keep = min(n_keep, vt.shape[0])
        components = np.zeros((n_keep, n_units), dtype=float)
        singular_values = np.zeros(n_keep, dtype=float)
        components[:keep] = vt[:keep]
        singular_values[:keep] = singular_values_all[:keep]

    explained = (singular_values * singular_values) / max(total_variance, 1e-12)
    within = (singular_values * singular_values) / max(marginal_variance, 1e-12)
    return components, singular_values, explained, within, marginal_variance


def fill_missing_condition_means(
    condition_sums: np.ndarray,
    condition_counts: np.ndarray,
    factor_names: tuple[str, ...],
) -> np.ndarray:
    means = np.zeros_like(condition_sums, dtype=float)
    np.divide(
        condition_sums,
        condition_counts[..., None],
        out=means,
        where=condition_counts[..., None] > 0,
    )

    missing = condition_counts == 0
    if not np.any(missing):
        return means

    if "timestep" in factor_names:
        time_axis = factor_names.index("timestep")
        sum_axes = tuple(axis for axis in range(condition_counts.ndim) if axis != time_axis)
    else:
        time_axis = None
        sum_axes = tuple(range(condition_counts.ndim))

    time_sums = condition_sums.sum(axis=sum_axes)
    time_counts = condition_counts.sum(axis=sum_axes)
    time_means = np.zeros_like(time_sums, dtype=float)
    if time_axis is None:
        np.divide(time_sums, time_counts, out=time_means, where=time_counts > 0)
    else:
        np.divide(
            time_sums,
            time_counts[:, None],
            out=time_means,
            where=time_counts[:, None] > 0,
        )
    global_count = float(condition_counts.sum())
    if global_count > 0:
        global_mean = condition_sums.sum(axis=tuple(range(condition_counts.ndim))) / global_count
    else:
        global_mean = np.zeros(condition_sums.shape[-1], dtype=float)

    for missing_index in np.argwhere(missing):
        if time_axis is None:
            fill_value = global_mean
        else:
            t_idx = int(missing_index[time_axis])
            fill_value = time_means[t_idx]
            if time_counts[t_idx] <= 0:
                fill_value = global_mean
        means[tuple(missing_index)] = fill_value

    print(f"Filled {int(missing.sum())} missing condition/time cells before dPCA.")
    return means


def fit_seed_dpca(
    model_cls,
    seed_dir: Path,
    dataset: HelicopterDPCADataset,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> DPCAFit:
    factor_names = tuple(cfg["factor_names"])
    marginal_groups = cfg["marginal_groups"]
    condition_sums: np.ndarray | None = None
    condition_counts: np.ndarray | None = None
    factor_levels: dict[str, list[Any]] | None = None
    index_maps: dict[str, dict[Any, int]] | None = None

    print(f"Fitting dPCA from {seed_dir.name}/{cfg['checkpoint_name']}")
    model = load_model(model_cls, seed_dir, cfg["checkpoint_name"], device)
    for hidden, trial_indices, _, _ in iter_hidden_batches(model, dataloader, device):
        _, n_time, n_units = hidden.shape
        if condition_sums is None:
            factor_levels = factor_levels_from_dataset(dataset, n_time, factor_names)
            index_maps = factor_index_maps(factor_levels)
            condition_shape = tuple(len(factor_levels[factor]) for factor in factor_names)
            condition_sums = np.zeros((*condition_shape, n_units), dtype=np.float64)
            condition_counts = np.zeros(condition_shape, dtype=np.float64)

        assert condition_sums is not None
        assert condition_counts is not None
        assert factor_levels is not None
        assert index_maps is not None

        for batch_pos, trial_idx in enumerate(trial_indices):
            meta = dataset.trial_meta[int(trial_idx)]
            condition_index = []
            for factor in factor_names:
                if factor == "timestep":
                    condition_index.append(slice(None))
                else:
                    condition_index.append(index_maps[factor][int(meta[factor])])

            condition_sums[tuple(condition_index) + (slice(None),)] += hidden[batch_pos].astype(
                np.float64
            )
            condition_counts[tuple(condition_index)] += 1.0

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if condition_sums is None or condition_counts is None or factor_levels is None:
        raise ValueError("No hidden states were collected for dPCA fitting.")

    condition_means = fill_missing_condition_means(condition_sums, condition_counts, factor_names)
    condition_axes = tuple(range(len(factor_names)))
    grand_mean = condition_means.mean(axis=condition_axes)
    centered_condition_means = condition_means - grand_mean
    total_variance = float(np.sum(centered_condition_means * centered_condition_means))
    if total_variance <= 1e-12:
        print("Warning: condition-averaged hidden states have near-zero variance.")

    effects = compute_anova_effects(condition_means, factor_names)
    components: dict[str, np.ndarray] = {}
    singular_values: dict[str, np.ndarray] = {}
    explained: dict[str, np.ndarray] = {}
    within: dict[str, np.ndarray] = {}
    marginal_variance: dict[str, float] = {}

    for marginal_name in cfg["marginals"]:
        tensor = marginal_tensor(effects, marginal_name, marginal_groups)
        matrix = tensor.reshape(-1, tensor.shape[-1])
        (
            components[marginal_name],
            singular_values[marginal_name],
            explained[marginal_name],
            within[marginal_name],
            marginal_variance[marginal_name],
        ) = fit_components_for_marginal(matrix, int(cfg["n_components"]), total_variance)

    return DPCAFit(
        analysis_mode=str(cfg["analysis_mode"]),
        factor_names=factor_names,
        marginal_names=list(cfg["marginals"]),
        marginal_groups=marginal_groups,
        components=components,
        singular_values=singular_values,
        explained_variance_ratio=explained,
        within_marginal_explained_variance_ratio=within,
        marginal_variance=marginal_variance,
        total_variance=total_variance,
        grand_mean=grand_mean,
        factor_levels=factor_levels,
        condition_counts=condition_counts,
        hazard_bin_edges=dataset.hazard_bin_edges,
        hazard_bin_labels=dataset.hazard_bin_labels,
    )


def component_col(marginal_name: str, component_idx: int) -> str:
    return f"{marginal_name}_dc{component_idx + 1}"


def component_cols(fit: DPCAFit) -> list[str]:
    cols: list[str] = []
    for marginal_name in fit.marginal_names:
        for component_idx in range(fit.components[marginal_name].shape[0]):
            cols.append(component_col(marginal_name, component_idx))
    return cols


def project_hidden(hidden: np.ndarray, fit: DPCAFit) -> dict[str, np.ndarray]:
    n_batch, n_time, n_units = hidden.shape
    flat_centered = hidden.reshape(-1, n_units).astype(np.float64) - fit.grand_mean
    projections: dict[str, np.ndarray] = {}
    for marginal_name in fit.marginal_names:
        scores = flat_centered @ fit.components[marginal_name].T
        projections[marginal_name] = scores.reshape(n_batch, n_time, -1)
    return projections


def write_transformed_csv(
    model_cls,
    fit: DPCAFit,
    seed_dir: Path,
    dataset: HelicopterDPCADataset,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    out_path: Path,
) -> pd.DataFrame:
    fieldnames = [
        "model",
        "seed",
        "checkpoint",
        "source_csv",
        "csv_trial",
        "global_trial",
        "timestep",
        "trial_in_block",
        "true_hazard",
        "hazard_bin",
        "hazard_bin_label",
        "true_report",
        "true_predict",
        "report_pred",
        "predict_pred",
        "report_correct",
        "predict_correct",
        "combined_correct",
        *component_cols(fit),
    ]

    rng = np.random.default_rng(0)
    plot_rows = []
    max_plot_points = int(cfg["max_plot_points"])
    rows_seen = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        match = SEED_RE.fullmatch(seed_dir.name)
        seed = int(match.group(1)) if match else int(cfg["seed"])
        print(f"Transforming {seed_dir.name}/{cfg['checkpoint_name']}")
        model = load_model(model_cls, seed_dir, cfg["checkpoint_name"], device)

        for hidden, trial_indices, report_pred, predict_pred in iter_hidden_batches(model, dataloader, device):
            projections = project_hidden(hidden, fit)
            _, n_time, _ = hidden.shape
            timesteps = [n_time - 1] if cfg["timestep_mode"] == "final" else range(n_time)

            for batch_pos, trial_idx in enumerate(trial_indices):
                meta = dataset.trial_meta[int(trial_idx)]
                true_report01 = 1 if meta["true_report"] > 0 else 0
                true_predict01 = 1 if meta["true_predict"] > 0 else 0
                batch_report_pred = int(report_pred[batch_pos, 0])
                batch_predict_pred = int(predict_pred[batch_pos, 0])
                report_correct = int(batch_report_pred == true_report01)
                predict_correct = int(batch_predict_pred == true_predict01)
                combined_correct = int(report_correct == 1 and predict_correct == 1)

                for timestep in timesteps:
                    row = {
                        "model": seed_dir.name,
                        "seed": seed,
                        "checkpoint": cfg["checkpoint_name"],
                        "source_csv": meta["source_csv"],
                        "csv_trial": meta["csv_trial"],
                        "global_trial": meta["global_trial"],
                        "timestep": timestep,
                        "trial_in_block": meta["trial_in_block"],
                        "true_hazard": meta["true_hazard"],
                        "hazard_bin": meta["hazard_bin"],
                        "hazard_bin_label": meta["hazard_bin_label"],
                        "true_report": meta["true_report"],
                        "true_predict": meta["true_predict"],
                        "report_pred": 1 if batch_report_pred == 1 else -1,
                        "predict_pred": 1 if batch_predict_pred == 1 else -1,
                        "report_correct": report_correct,
                        "predict_correct": predict_correct,
                        "combined_correct": combined_correct,
                    }
                    for marginal_name in fit.marginal_names:
                        marginal_scores = projections[marginal_name][batch_pos, timestep]
                        for component_idx, value in enumerate(marginal_scores):
                            row[component_col(marginal_name, component_idx)] = float(value)

                    writer.writerow(row)
                    rows_seen += 1

                    if len(plot_rows) < max_plot_points:
                        plot_rows.append(row.copy())
                    else:
                        replace_idx = rng.integers(0, rows_seen)
                        if replace_idx < max_plot_points:
                            plot_rows[int(replace_idx)] = row.copy()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(plot_rows)


def save_explained_variance(fit: DPCAFit, out_path: Path) -> pd.DataFrame:
    rows = []
    for marginal_name in fit.marginal_names:
        cumulative_within = 0.0
        for component_idx, ratio in enumerate(
            fit.explained_variance_ratio[marginal_name],
            start=1,
        ):
            within_ratio = float(
                fit.within_marginal_explained_variance_ratio[marginal_name][component_idx - 1]
            )
            cumulative_within += within_ratio
            rows.append(
                {
                    "marginal": marginal_name,
                    "component": f"dc{component_idx}",
                    "singular_value": float(fit.singular_values[marginal_name][component_idx - 1]),
                    "marginal_variance": float(fit.marginal_variance[marginal_name]),
                    "total_variance": float(fit.total_variance),
                    "explained_variance_ratio": float(ratio),
                    "within_marginal_explained_variance_ratio": within_ratio,
                    "cumulative_within_marginal_explained_variance_ratio": cumulative_within,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def save_component_weights(fit: DPCAFit, out_path: Path) -> None:
    rows = []
    for marginal_name in fit.marginal_names:
        components = fit.components[marginal_name]
        for component_idx in range(components.shape[0]):
            for unit_idx, weight in enumerate(components[component_idx]):
                rows.append(
                    {
                        "marginal": marginal_name,
                        "component": f"dc{component_idx + 1}",
                        "unit": unit_idx,
                        "weight": float(weight),
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_condition_counts(fit: DPCAFit, out_path: Path) -> None:
    rows = []
    for index in np.ndindex(fit.condition_counts.shape):
        row: dict[str, Any] = {}
        for axis, factor in enumerate(fit.factor_names):
            value = fit.factor_levels[factor][index[axis]]
            row[factor] = value
            if factor == "hazard_bin":
                row["hazard_bin_label"] = fit.hazard_bin_labels[int(value)]
        row["count"] = int(fit.condition_counts[index])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_fit_summary(fit: DPCAFit, cfg: dict[str, Any], out_path: Path) -> None:
    empty_cells = int((fit.condition_counts == 0).sum())
    condition_cells = int(fit.condition_counts.size)
    summary = {
        "analysis_mode": cfg["analysis_mode"],
        "checkpoint_name": cfg["checkpoint_name"],
        "model_subdir": cfg["model_subdir"],
        "variant_subdir": cfg["variant_subdir"],
        "seed": cfg["seed"],
        "n_components": cfg["n_components"],
        "factor_names": list(fit.factor_names),
        "marginals": fit.marginal_names,
        "marginal_groups": {
            name: [list(subset) for subset in fit.marginal_groups[name]]
            for name in fit.marginal_names
        },
        "dropped_from_demixing": cfg["dropped_from_demixing"],
        "note": cfg["analysis_note"],
        "factor_levels": fit.factor_levels,
        "hazard_bin_edges": fit.hazard_bin_edges,
        "hazard_bin_labels": fit.hazard_bin_labels,
        "total_condition_variance": fit.total_variance,
        "condition_shape": list(fit.condition_counts.shape),
        "condition_cells": condition_cells,
        "empty_condition_cells": empty_cells,
        "empty_condition_cell_percent": 100.0 * empty_cells / max(condition_cells, 1),
        "timestep_mode": cfg["timestep_mode"],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_explained_variance(variance_df: pd.DataFrame, out_path: Path) -> None:
    if variance_df.empty:
        print("No dPCA variance rows available for plotting.")
        return

    plot_df = variance_df.copy()
    plot_df["label"] = plot_df["marginal"] + ":" + plot_df["component"]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(plot_df)), 5.2))
    ax.bar(plot_df["label"], plot_df["explained_variance_ratio"], color="#4c78a8")
    ax.set_ylabel("Explained condition variance ratio")
    ax.set_title("dPCA component variance")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def output_stem(cfg: dict[str, Any], base: str) -> str:
    if cfg["timestep_mode"] == "final":
        return f"{base}_final_timestep"
    return base


def scatter_component_pair(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    if x_col not in df or y_col not in df or color_col not in df:
        return

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
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label=color_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

def plot_mean_trajectory(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str | None,
    title: str,
    out_path: Path,
) -> None:
    if x_col not in df or y_col not in df or "timestep" not in df:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6))
    if group_col is None:
        trajectory = df.groupby("timestep", as_index=False)[[x_col, y_col]].mean()
        ax.plot(
            trajectory[x_col],
            trajectory[y_col],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            label="mean",
        )
    else:
        for label, group_df in sorted(df.groupby(group_col), key=lambda item: item[0]):
            trajectory = group_df.groupby("timestep", as_index=False)[[x_col, y_col]].mean()
            ax.plot(
                trajectory[x_col],
                trajectory[y_col],
                marker="o",
                linewidth=2.0,
                markersize=3.5,
                label=f"{group_col}={label}",
            )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

def color_column_for_marginal(marginal_name: str) -> tuple[str, str]:
    if "hazard" in marginal_name:
        return "hazard_bin", "viridis"
    if "report" in marginal_name:
        return "true_report", "coolwarm"
    if "predict" in marginal_name:
        return "true_predict", "coolwarm"
    if "time" in marginal_name or marginal_name == "condition":
        return "timestep", "plasma"
    return "true_hazard", "viridis"


def trajectory_group_for_marginal(marginal_name: str) -> str | None:
    if marginal_name == "time":
        return None
    if "hazard" in marginal_name:
        return "hazard_bin"
    if "report" in marginal_name:
        return "true_report"
    if "predict" in marginal_name:
        return "true_predict"
    return None


def save_plots(plot_df: pd.DataFrame, fit: DPCAFit, cfg: dict[str, Any]) -> None:
    if plot_df.empty:
        print("No sampled rows available for plotting.")
        return

    stem = output_stem(cfg, output_prefix(cfg))
    for marginal_name in fit.marginal_names:
        x_col = component_col(marginal_name, 0)
        y_col = component_col(marginal_name, 1)
        if x_col not in plot_df or y_col not in plot_df:
            continue

        color_col, cmap = color_column_for_marginal(marginal_name)
        scatter_component_pair(
            plot_df,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            title=f"{cfg['model_subdir']} {cfg['checkpoint_name']} dPCA {marginal_name}",
            out_path=cfg["output_dir"] / f"{stem}_{marginal_name}_dc1_dc2_by_{color_col}.png",
            cmap=cmap,
        )

        if "time" in marginal_name or marginal_name == "time":
            plot_mean_trajectory(
                plot_df,
                x_col=x_col,
                y_col=y_col,
                group_col=trajectory_group_for_marginal(marginal_name),
                title=f"Mean {marginal_name} dPCA trajectory",
                out_path=cfg["output_dir"] / f"{stem}_{marginal_name}_mean_trajectory.png",
            )

    correct_df = plot_df[plot_df["combined_correct"] == 1].copy()
    if correct_df.empty:
        print("No combined-correct sampled rows available for correct-only plots.")
        return

    for marginal_name in fit.marginal_names:
        x_col = component_col(marginal_name, 0)
        y_col = component_col(marginal_name, 1)
        if x_col not in correct_df or y_col not in correct_df:
            continue
        color_col, cmap = color_column_for_marginal(marginal_name)
        scatter_component_pair(
            correct_df,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            title=(
                f"{cfg['model_subdir']} {cfg['checkpoint_name']} "
                f"dPCA {marginal_name}, correct only"
            ),
            out_path=(
                cfg["output_dir"]
                / f"{stem}_correct_only_{marginal_name}_dc1_dc2_by_{color_col}.png"
            ),
            cmap=cmap,
        )


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    return safe_pearsonr(x_rank, y_rank)


def ridge_probe_metrics(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    test_size: float,
    random_state: int,
) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite_mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[finite_mask]
    y = y[finite_mask]
    n_samples = int(y.shape[0])
    if n_samples < 4:
        return {
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "mse": float("nan"),
            "n_train": 0,
            "n_test": 0,
        }

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    n_test = int(np.ceil(n_samples * float(test_size)))
    n_test = min(max(n_test, 2), n_samples - 2)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std <= 1e-12] = 1.0
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    model = Ridge(alpha=float(alpha))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    residual = y_test - y_pred
    mse = float(np.mean(residual * residual))
    total = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float(1.0 - np.sum(residual * residual) / total) if total > 1e-12 else float("nan")
    return {
        "r2": r2,
        "pearson_r": safe_pearsonr(y_test, y_pred),
        "spearman_r": safe_spearmanr(y_test, y_pred),
        "mse": mse,
        "n_train": int(train_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
    }


def collect_hidden_for_probes(
    model_cls,
    seed_dir: Path,
    dataset: HelicopterDPCADataset,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_batches = []
    trial_index_batches = []

    print("Collecting hidden states for continuous true_hazard probes")
    model = load_model(model_cls, seed_dir, cfg["checkpoint_name"], device)
    for hidden, trial_indices, _, _ in iter_hidden_batches(model, dataloader, device):
        hidden_batches.append(hidden.astype(np.float32, copy=False))
        trial_index_batches.append(trial_indices.astype(int, copy=False))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hidden_all = np.concatenate(hidden_batches, axis=0)
    trial_indices_all = np.concatenate(trial_index_batches, axis=0)
    hazards = np.asarray(
        [dataset.trial_meta[int(idx)]["true_hazard"] for idx in trial_indices_all],
        dtype=np.float64,
    )
    true_predict = np.asarray(
        [dataset.trial_meta[int(idx)]["true_predict"] for idx in trial_indices_all],
        dtype=int,
    )
    return hidden_all, hazards, true_predict


def continuous_hazard_probe_rows(
    hidden: np.ndarray,
    hazards: np.ndarray,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    alpha = float(cfg["ridge_alpha"])
    test_size = float(cfg["probe_test_size"])
    random_state = int(cfg["probe_random_state"])
    for timestep in range(hidden.shape[1]):
        metrics = ridge_probe_metrics(
            hidden[:, timestep, :],
            hazards,
            alpha=alpha,
            test_size=test_size,
            random_state=random_state + timestep,
        )
        rows.append({"timestep": timestep, **metrics, "alpha": alpha})
    return rows


def hazard_within_predict_probe_rows(
    hidden: np.ndarray,
    hazards: np.ndarray,
    true_predict: np.ndarray,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    alpha = float(cfg["ridge_alpha"])
    test_size = float(cfg["probe_test_size"])
    random_state = int(cfg["probe_random_state"])
    for predict_value in sorted(np.unique(true_predict)):
        mask = true_predict == predict_value
        for timestep in range(hidden.shape[1]):
            metrics = ridge_probe_metrics(
                hidden[mask, timestep, :],
                hazards[mask],
                alpha=alpha,
                test_size=test_size,
                random_state=random_state + 10000 + timestep + int(predict_value > 0) * 1000,
            )
            rows.append(
                {
                    "true_predict": int(predict_value),
                    "timestep": timestep,
                    **metrics,
                    "alpha": alpha,
                }
            )
    return rows


def plot_probe_metric(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    group_col: str | None = None,
) -> None:
    if df.empty or metric not in df:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if group_col and group_col in df:
        for label, group_df in sorted(df.groupby(group_col), key=lambda item: item[0]):
            ax.plot(
                group_df["timestep"],
                group_df[metric],
                linewidth=2.0,
                label=f"{group_col}={label}",
            )
        ax.legend(frameon=False)
    else:
        ax.plot(df["timestep"], df[metric], linewidth=2.0)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_continuous_hazard_probes(
    model_cls,
    seed_dir: Path,
    dataset: HelicopterDPCADataset,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[Path, Path]:
    hidden, hazards, true_predict = collect_hidden_for_probes(
        model_cls,
        seed_dir,
        dataset,
        dataloader,
        cfg,
        device,
    )

    probe_df = pd.DataFrame(continuous_hazard_probe_rows(hidden, hazards, cfg))
    probe_path = cfg["output_dir"] / "dpca_ep010_continuous_hazard_probe.csv"
    probe_df.to_csv(probe_path, index=False)
    plot_probe_metric(
        probe_df,
        metric="r2",
        out_path=cfg["output_dir"] / "dpca_ep010_continuous_hazard_probe_r2.png",
        title="Held-out ridge decoding of continuous true_hazard",
    )
    plot_probe_metric(
        probe_df,
        metric="pearson_r",
        out_path=cfg["output_dir"] / "dpca_ep010_continuous_hazard_probe_correlations.png",
        title="Held-out continuous true_hazard decoding correlation",
    )

    within_df = pd.DataFrame(hazard_within_predict_probe_rows(hidden, hazards, true_predict, cfg))
    within_path = cfg["output_dir"] / "dpca_ep010_hazard_within_predict_probe.csv"
    within_df.to_csv(within_path, index=False)
    plot_probe_metric(
        within_df,
        metric="r2",
        out_path=cfg["output_dir"] / "dpca_ep010_hazard_within_predict_probe_r2.png",
        title="Held-out true_hazard decoding within true_predict class",
        group_col="true_predict",
    )
    plot_probe_metric(
        within_df,
        metric="pearson_r",
        out_path=cfg["output_dir"] / "dpca_ep010_hazard_within_predict_probe_correlations.png",
        title="Within-true_predict true_hazard decoding correlation",
        group_col="true_predict",
    )
    return probe_path, within_path


def main() -> None:
    args = parse_args()
    try:
        cfg = build_run_config(args)
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    seed_dir = get_seed_dir(cfg["model_dir"], int(cfg["seed"]), cfg["checkpoint_name"])
    hp = load_hp(seed_dir)
    batch_size = int(hp.get("batch_size", 256))
    csvs = list_eval_csvs(cfg)
    dataset = HelicopterDPCADataset(
        csvs,
        int(hp["n_input"]),
        int(hp["n_null_timesteps"]),
        int(cfg["hazard_bins"]),
        str(cfg["hazard_bin_strategy"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(f"Using device: {device}")
    print(f"Analysis mode: {cfg['analysis_mode']}")
    print(f"dPCA factor names: {', '.join(cfg['factor_names'])}")
    print(f"dPCA marginals: {', '.join(cfg['marginals'])}")
    if cfg["analysis_mode"] == "hazard_continuous":
        print("Sanity check: predict marginals are absent in hazard_continuous mode.")
    if cfg["dropped_from_demixing"]:
        print(f"Dropped from dPCA demixing: {', '.join(cfg['dropped_from_demixing'])}")
    print(f"Loaded {len(dataset)} trials from {len(csvs)} {cfg['variant_split']} CSVs")
    print(f"Using {seed_dir.name}/{cfg['checkpoint_name']}")
    print(
        f"Prepared model inputs with n_input={hp['n_input']}, "
        f"n_null_timesteps={hp['n_null_timesteps']}, batch_size={batch_size}"
    )
    print(
        f"Binned true_hazard into {len(dataset.hazard_bin_labels)} "
        f"{cfg['hazard_bin_strategy']} bins"
    )
    print(f"Saving timestep mode: {cfg['timestep_mode']}")

    first_x, _ = dataset[0]
    dry_factor_levels = factor_levels_from_dataset(
        dataset,
        int(first_x.shape[0]),
        tuple(cfg["factor_names"]),
    )
    dry_condition_shape = tuple(len(dry_factor_levels[factor]) for factor in cfg["factor_names"])
    print(f"dPCA condition tensor shape before units: {dry_condition_shape}")
    if cfg["dry_run_check"]:
        print("Dry-run check complete; no model fitting performed.")
        return

    fit = fit_seed_dpca(model_cls, seed_dir, dataset, dataloader, cfg, device)
    empty_cells = int((fit.condition_counts == 0).sum())
    total_cells = int(fit.condition_counts.size)
    print(
        f"Empty dPCA condition cells: {empty_cells}/{total_cells} "
        f"({100.0 * empty_cells / max(total_cells, 1):.2f}%)"
    )

    prefix = output_prefix(cfg)
    variance_path = cfg["output_dir"] / f"{prefix}_explained_variance.csv"
    variance_df = save_explained_variance(fit, variance_path)
    variance_plot_path = cfg["output_dir"] / f"{prefix}_explained_variance.png"
    plot_explained_variance(variance_df, variance_plot_path)

    weights_path = cfg["output_dir"] / f"{prefix}_component_weights.csv"
    save_component_weights(fit, weights_path)
    counts_path = cfg["output_dir"] / f"{prefix}_condition_counts.csv"
    save_condition_counts(fit, counts_path)
    summary_path = cfg["output_dir"] / f"{prefix}_fit_summary.json"
    save_fit_summary(fit, cfg, summary_path)

    stem = output_stem(cfg, prefix)
    transformed_path = cfg["output_dir"] / f"{stem}_hidden_states.csv"
    plot_df = write_transformed_csv(
        model_cls,
        fit,
        seed_dir,
        dataset,
        dataloader,
        cfg,
        device,
        transformed_path,
    )
    plot_sample_path = cfg["output_dir"] / f"{stem}_plot_sample.csv"
    plot_df.to_csv(plot_sample_path, index=False)
    save_plots(plot_df, fit, cfg)

    probe_path = None
    within_probe_path = None
    if cfg["analysis_mode"] == "hazard_continuous":
        probe_path, within_probe_path = save_continuous_hazard_probes(
            model_cls,
            seed_dir,
            dataset,
            dataloader,
            cfg,
            device,
        )

    print(f"Saved transformed dPCA rows to {transformed_path}")
    print(f"Saved plot sample to {plot_sample_path}")
    print(f"Saved explained variance to {variance_path}")
    print(f"Saved explained variance plot to {variance_plot_path}")
    print(f"Saved component weights to {weights_path}")
    print(f"Saved condition counts to {counts_path}")
    print(f"Saved fit summary to {summary_path}")
    if probe_path is not None:
        print(f"Saved continuous hazard probe to {probe_path}")
    if within_probe_path is not None:
        print(f"Saved within-predict hazard probe to {within_probe_path}")


if __name__ == "__main__":
    main()

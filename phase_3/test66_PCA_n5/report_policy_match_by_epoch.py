#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
NORMATIVE_MODEL_PATH = REPO_ROOT / "utils" / "NormativeModel.py"


def import_bayesian_observer():
    if not NORMATIVE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find {NORMATIVE_MODEL_PATH}")

    spec = importlib.util.spec_from_file_location("ots_normative_model", NORMATIVE_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {NORMATIVE_MODEL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.BayesianObserver
    except AttributeError as exc:
        raise AttributeError(
            f"{NORMATIVE_MODEL_PATH} does not define BayesianObserver"
        ) from exc


BayesianObserver = import_bayesian_observer()


DEFAULT_CONFIG = BASE_DIR / "accuracy_by_checkpoint_config.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "report_policy_match"
DEFAULT_HEURISTIC_CSV = REPO_ROOT / "variants" / "last_evidence_report_heuristic_test_accuracy.csv"
DEFAULT_NORMATIVE_CSV = REPO_ROOT / "variants" / "normative_model_test_accuracy.csv"

CHECKPOINT_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")
SEED_RE = re.compile(r"seed_(\d+)$")
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare report-head policies at selected epochs against true report, "
            "last-evidence heuristic report, and normative Bayesian report."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-subdir", default="bce_both/sigma_1")
    parser.add_argument(
        "--variant-subdir",
        default=None,
        help="Variant folder under variant_root. Defaults to the model-subdir leaf.",
    )
    parser.add_argument("--variant-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-variant-csvs", type=int, default=None)
    parser.add_argument("--model-class", default="GRUModel")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epochs to evaluate. Default: all checkpoint_ep*.pt files found in the seed folders.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heuristic-accuracy-csv", type=Path, default=DEFAULT_HEURISTIC_CSV)
    parser.add_argument("--normative-accuracy-csv", type=Path, default=DEFAULT_NORMATIVE_CSV)
    parser.add_argument("--final-evidence-window", type=float, default=0.2)
    parser.add_argument("--hazard-step", type=float, default=0.05)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--zero-report", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--seed", type=int, default=0, help="Seed for rare normative ties.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


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


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config.expanduser().resolve())
    variant_subdir = args.variant_subdir or Path(args.model_subdir).name
    cfg.update(
        {
            "model_subdir": args.model_subdir,
            "variant_subdir": variant_subdir,
            "variant_split": args.variant_split,
            "max_variant_csvs": args.max_variant_csvs,
            "model_class": args.model_class,
            "epochs": None if args.epochs is None else sorted(dict.fromkeys(args.epochs)),
            "batch_size": int(args.batch_size),
            "output_dir": args.output_dir.expanduser().resolve(),
            "heuristic_accuracy_csv": args.heuristic_accuracy_csv.expanduser().resolve(),
            "normative_accuracy_csv": args.normative_accuracy_csv.expanduser().resolve(),
            "final_evidence_window": float(args.final_evidence_window),
            "hazard_step": float(args.hazard_step),
            "bias": float(args.bias),
            "zero_report": float(args.zero_report),
            "seed": int(args.seed),
            "make_plots": not args.no_plots,
        }
    )
    cfg["model_dir"] = cfg["model_root"] / cfg["model_subdir"]
    cfg["variant_dir"] = cfg["variant_root"] / cfg["variant_subdir"]
    cfg["variant_folder"] = f"{cfg['variant_root'].name}/{Path(cfg['variant_subdir']).as_posix()}"
    return cfg


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
    module_path = model_root / "rnn_models.py"
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


def list_seed_dirs(model_dir: Path) -> list[Path]:
    seed_dirs = [p for p in model_dir.iterdir() if p.is_dir() and SEED_RE.fullmatch(p.name)]
    seed_dirs.sort(key=lambda p: int(SEED_RE.fullmatch(p.name).group(1)))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {model_dir}")
    return seed_dirs


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


def parse_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    parsed = ast.literal_eval(str(value))
    return [float(item) for item in parsed]


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
        for i, e_k in enumerate(evidence):
            steps.append([float(e_k), 1.0])
            if i < len(evidence) - 1:
                steps.extend([null_step.copy() for _ in range(n_null_timesteps)])
        return torch.tensor(steps, dtype=torch.float32)

    raise ValueError(f"Unsupported n_input={n_input}; expected 1 or 2")


def sign_to_label01(value: int | float) -> int:
    return 1 if float(value) > 0 else 0


def label01_to_sign(value: int | float) -> int:
    return 1 if int(value) == 1 else -1


def heuristic_report_sign(evidence: list[float], zero_report: float) -> int:
    if evidence[-1] > 0:
        return 1
    if evidence[-1] < 0:
        return -1
    return int(zero_report)


def load_mu(variant_dir: Path, default: float = 1.0) -> float:
    config_path = variant_dir / "TaskConfig.csv"
    if not config_path.exists():
        return default

    config = pd.read_csv(config_path, index_col=0)
    if "Mu" in config.index:
        return float(config.loc["Mu"].iloc[0])
    return default


def normative_report_sign(
    evidence: list[float],
    sigma: float,
    hs: np.ndarray,
    mu: float,
    bias: float,
) -> int:
    _, _, resp_report, _ = BayesianObserver(
        evidence,
        mu1=-mu,
        mu2=mu,
        sigma=sigma,
        hs=hs,
        bias=bias,
    )
    return int(resp_report)


def load_trials(cfg: dict[str, Any]) -> pd.DataFrame:
    csvs = list_eval_csvs(cfg)
    hs = np.arange(0, 1, cfg["hazard_step"])
    mu = load_mu(cfg["variant_dir"])
    rows = []
    global_idx = 0

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        for row_idx, row in df.iterrows():
            evidence = parse_list(row["evidence"])
            true_report = int(float(row["trueReport"]))
            final_evidence = float(evidence[-1])
            heuristic_report = heuristic_report_sign(evidence, cfg["zero_report"])
            normative_report = normative_report_sign(
                evidence=evidence,
                sigma=float(row["sigma"]),
                hs=hs,
                mu=mu,
                bias=cfg["bias"],
            )
            rows.append(
                {
                    "trial_index": global_idx,
                    "source_file": csv_path.name,
                    "source_row": int(row_idx),
                    "sigma": float(row["sigma"]),
                    "true_report": true_report,
                    "true_report01": sign_to_label01(true_report),
                    "final_evidence": final_evidence,
                    "final_evidence_abs": abs(final_evidence),
                    "final_evidence_wrong_side": bool(final_evidence * true_report < 0),
                    "heuristic_report": heuristic_report,
                    "heuristic_report01": sign_to_label01(heuristic_report),
                    "normative_report": normative_report,
                    "normative_report01": sign_to_label01(normative_report),
                    "_evidence": evidence,
                }
            )
            global_idx += 1

    if not rows:
        raise ValueError("No evaluation trials were loaded.")
    return pd.DataFrame(rows)


class ReportPolicyDataset(Dataset):
    def __init__(self, trials: pd.DataFrame, n_input: int, n_null_timesteps: int):
        self.x = [
            encode_evidence_sequence(evidence, n_input, n_null_timesteps)
            for evidence in trials["_evidence"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def collate_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, 0)


def checkpoint_path(seed_dir: Path, epoch: int) -> Path:
    path = seed_dir / f"checkpoint_ep{epoch:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for epoch {epoch}: {path}")
    return path


def discover_checkpoint_epochs(seed_dirs: list[Path]) -> list[int]:
    epoch_sets: list[set[int]] = []
    for seed_dir in seed_dirs:
        epochs = set()
        for path in seed_dir.glob("checkpoint_ep*.pt"):
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match:
                epochs.add(int(match.group(1)))
        if not epochs:
            raise FileNotFoundError(f"No checkpoint_ep*.pt files found in {seed_dir}")
        epoch_sets.append(epochs)

    shared_epochs = set.intersection(*epoch_sets)
    if not shared_epochs:
        raise FileNotFoundError("No checkpoint epochs are shared across all seed folders.")
    return sorted(shared_epochs)


@torch.inference_mode()
def model_report_probabilities(
    model_cls,
    checkpoint: Path,
    hp: dict[str, Any],
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model = model_cls(hp).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    probs = []
    for x in dataloader:
        x = x.to(device)
        loc_logits, _ = model(x)
        probs.append(torch.sigmoid(loc_logits[:, -1, 0]).cpu().numpy())
    return np.concatenate(probs, axis=0)


def binary_cross_entropy(prob: np.ndarray, target01: np.ndarray) -> float:
    prob = np.clip(prob, EPS, 1.0 - EPS)
    return float(-(target01 * np.log(prob) + (1 - target01) * np.log(1 - prob)).mean())


def subset_masks(trials: pd.DataFrame, final_evidence_window: float) -> dict[str, np.ndarray]:
    return {
        "all_test_trials": np.ones(len(trials), dtype=bool),
        "final_evidence_within_0.2_center": (
            trials["final_evidence_abs"].to_numpy(dtype=float) <= final_evidence_window
        ),
        "final_evidence_wrong_side_vs_true_report": trials[
            "final_evidence_wrong_side"
        ].to_numpy(dtype=bool),
    }


def score_subset(
    probs: np.ndarray,
    pred01: np.ndarray,
    target01: np.ndarray,
    mask: np.ndarray,
    metric_name: str,
) -> dict[str, float | int | str]:
    n_trials = int(mask.sum())
    row: dict[str, float | int | str] = {
        "metric": metric_name,
        "n_trials": n_trials,
    }
    if n_trials == 0:
        row.update({"match_accuracy": np.nan, "bce": np.nan})
        return row

    row.update(
        {
            "match_accuracy": float((pred01[mask] == target01[mask]).mean()),
            "bce": binary_cross_entropy(probs[mask], target01[mask]),
        }
    )
    return row


def evaluate_models(cfg: dict[str, Any], trials: pd.DataFrame) -> pd.DataFrame:
    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    seed_dirs = list_seed_dirs(cfg["model_dir"])
    epochs = cfg["epochs"] if cfg["epochs"] is not None else discover_checkpoint_epochs(seed_dirs)
    masks = subset_masks(trials, cfg["final_evidence_window"])

    true01 = trials["true_report01"].to_numpy(dtype=int)
    heuristic01 = trials["heuristic_report01"].to_numpy(dtype=int)
    normative01 = trials["normative_report01"].to_numpy(dtype=int)
    target_specs = [
        ("true_report", true01),
        ("last_evidence_heuristic_report", heuristic01),
        ("normative_report", normative01),
    ]

    dataloaders: dict[tuple[int, int], DataLoader] = {}
    rows = []
    print(f"Using device: {device}")
    print(f"Loaded {len(trials)} trials from {cfg['variant_dir']}")
    print(f"Evaluating epochs: {', '.join(str(epoch) for epoch in epochs)}")

    for seed_dir in seed_dirs:
        seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
        hp = load_hp(seed_dir)
        dataset_key = (int(hp["n_input"]), int(hp["n_null_timesteps"]))
        if dataset_key not in dataloaders:
            dataset = ReportPolicyDataset(trials, *dataset_key)
            dataloaders[dataset_key] = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                collate_fn=collate_batch,
            )
            print(
                f"Prepared model inputs with n_input={dataset_key[0]}, "
                f"n_null_timesteps={dataset_key[1]}"
            )

        for epoch in epochs:
            ckpt = checkpoint_path(seed_dir, epoch)
            probs = model_report_probabilities(
                model_cls=model_cls,
                checkpoint=ckpt,
                hp=hp,
                dataloader=dataloaders[dataset_key],
                device=device,
            )
            pred01 = (probs >= 0.5).astype(int)
            pred_sign = np.array([label01_to_sign(value) for value in pred01])

            for subset_name, mask in masks.items():
                for metric_name, target01 in target_specs:
                    row = score_subset(probs, pred01, target01, mask, metric_name)
                    row.update(
                        {
                            "model_subdir": cfg["model_subdir"],
                            "variant_folder": cfg["variant_folder"],
                            "variant_split": cfg["variant_split"],
                            "seed": seed,
                            "model": seed_dir.name,
                            "epoch": int(epoch),
                            "checkpoint": ckpt.name,
                            "subset": subset_name,
                            "mean_p_report_plus": float(probs[mask].mean()) if mask.any() else np.nan,
                            "model_plus_rate": float((pred_sign[mask] == 1).mean())
                            if mask.any()
                            else np.nan,
                        }
                    )
                    rows.append(row)

            print(f"{seed_dir.name} epoch {epoch}: evaluated {ckpt.name}")

    return pd.DataFrame(rows)


def summarize_model_results(model_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["match_accuracy", "bce", "mean_p_report_plus", "model_plus_rate"]
    summary = (
        model_results.groupby(
            ["model_subdir", "variant_folder", "variant_split", "epoch", "subset", "metric"]
        )[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def load_reference_accuracy(path: Path, variant_folder: str, metric_column: str) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing reference CSV: {path}")
    df = pd.read_csv(path)
    if "variant_folder" not in df.columns:
        raise ValueError(f"{path} does not contain 'variant_folder'")
    if metric_column not in df.columns:
        raise ValueError(f"{path} does not contain {metric_column!r}")
    match = df[df["variant_folder"] == variant_folder]
    if match.empty:
        raise ValueError(f"No row found for {variant_folder!r} in {path}")
    return float(match.iloc[0][metric_column])


def reference_policy_summary(cfg: dict[str, Any], trials: pd.DataFrame) -> pd.DataFrame:
    masks = subset_masks(trials, cfg["final_evidence_window"])
    true01 = trials["true_report01"].to_numpy(dtype=int)
    heuristic01 = trials["heuristic_report01"].to_numpy(dtype=int)
    normative01 = trials["normative_report01"].to_numpy(dtype=int)

    try:
        heuristic_all_accuracy = load_reference_accuracy(
            cfg["heuristic_accuracy_csv"], cfg["variant_folder"], "report_accuracy"
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[warn] {exc}")
        heuristic_all_accuracy = np.nan

    try:
        normative_all_accuracy = load_reference_accuracy(
            cfg["normative_accuracy_csv"], cfg["variant_folder"], "report_accuracy"
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[warn] {exc}")
        normative_all_accuracy = np.nan

    refs = []
    specs = [
        ("last_evidence_heuristic_report", heuristic01, heuristic_all_accuracy),
        ("normative_report", normative01, normative_all_accuracy),
    ]
    for subset_name, mask in masks.items():
        for policy_name, policy01, source_all_accuracy in specs:
            n_trials = int(mask.sum())
            row = {
                "variant_folder": cfg["variant_folder"],
                "variant_split": cfg["variant_split"],
                "subset": subset_name,
                "policy": policy_name,
                "n_trials": n_trials,
                "report_accuracy_vs_true": float((policy01[mask] == true01[mask]).mean())
                if n_trials
                else np.nan,
                "source_all_trials_report_accuracy": source_all_accuracy
                if subset_name == "all_test_trials"
                else np.nan,
            }
            refs.append(row)

        refs.append(
            {
                "variant_folder": cfg["variant_folder"],
                "variant_split": cfg["variant_split"],
                "subset": subset_name,
                "policy": "heuristic_matches_normative",
                "n_trials": n_trials,
                "report_accuracy_vs_true": float((heuristic01[mask] == normative01[mask]).mean())
                if n_trials
                else np.nan,
                "source_all_trials_report_accuracy": np.nan,
            }
        )

    return pd.DataFrame(refs)


def write_subset_trial_counts(cfg: dict[str, Any], trials: pd.DataFrame) -> pd.DataFrame:
    masks = subset_masks(trials, cfg["final_evidence_window"])
    rows = []
    for subset_name, mask in masks.items():
        rows.append(
            {
                "variant_folder": cfg["variant_folder"],
                "variant_split": cfg["variant_split"],
                "subset": subset_name,
                "n_trials": int(mask.sum()),
                "fraction_of_trials": float(mask.mean()),
                "final_evidence_window": cfg["final_evidence_window"],
            }
        )
    return pd.DataFrame(rows)


METRIC_LABELS = {
    "true_report": "true report",
    "last_evidence_heuristic_report": "last evidence",
    "normative_report": "normative",
}
SUBSET_LABELS = {
    "all_test_trials": "All test trials",
    "final_evidence_within_0.2_center": "Final evidence within 0.2 of center",
    "final_evidence_wrong_side_vs_true_report": "Final evidence on wrong side",
}
POLICY_LABELS = {
    "last_evidence_heuristic_report": "last evidence",
    "normative_report": "normative",
    "heuristic_matches_normative": "heuristic vs normative",
}
METRIC_COLORS = {
    "true_report": "#2f5d8c",
    "last_evidence_heuristic_report": "#228b73",
    "normative_report": "#b3475f",
}


def plot_metric_by_epoch(
    model_summary: pd.DataFrame,
    metric_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    out_path: Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    subsets = list(model_summary["subset"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(subsets),
        figsize=(5.1 * len(subsets), 4.8),
        sharey=True,
        constrained_layout=False,
    )
    if len(subsets) == 1:
        axes = [axes]

    for ax, subset in zip(axes, subsets):
        subset_df = model_summary[model_summary["subset"] == subset]
        for metric_name, metric_df in subset_df.groupby("metric", sort=False):
            metric_df = metric_df.sort_values("epoch")
            x = metric_df["epoch"].to_numpy(dtype=float)
            y = metric_df[metric_column].to_numpy(dtype=float)
            yerr = metric_df[std_column].to_numpy(dtype=float)
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=METRIC_COLORS.get(metric_name, "#555555"),
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=METRIC_LABELS.get(metric_name, metric_name),
            )

        ax.set_title(SUBSET_LABELS.get(subset, subset), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_xticks(sorted(subset_df["epoch"].drop_duplicates()))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if y_limits is not None:
            ax.set_ylim(*y_limits)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(top=0.74, bottom=0.16, wspace=0.08)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference_policies(reference_summary: pd.DataFrame, out_path: Path) -> None:
    subsets = list(reference_summary["subset"].drop_duplicates())
    policies = list(reference_summary["policy"].drop_duplicates())
    colors = ["#228b73", "#b3475f", "#6b7280"]
    fig, ax = plt.subplots(figsize=(max(7.0, 2.8 * len(subsets)), 4.6))

    x = np.arange(len(subsets))
    width = 0.22
    offsets = np.linspace(-width, width, len(policies))
    for offset, policy, color in zip(offsets, policies, colors):
        policy_df = (
            reference_summary[reference_summary["policy"] == policy]
            .set_index("subset")
            .reindex(subsets)
        )
        ax.bar(
            x + offset,
            policy_df["report_accuracy_vs_true"],
            width,
            label=POLICY_LABELS.get(policy, policy),
            color=color,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(subset, subset) for subset in subsets], rotation=18, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Accuracy / agreement")
    ax.set_title("Reference report policies by trial subset")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=min(3, len(policies)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_subset_counts(subset_counts: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(subset_counts))
    bars = ax.bar(
        x,
        subset_counts["n_trials"],
        color=["#2f5d8c", "#228b73", "#b3475f"][: len(subset_counts)],
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SUBSET_LABELS.get(subset, subset) for subset in subset_counts["subset"]],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Trials")
    ax.set_title("Trial counts by subset")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, frac in zip(bars, subset_counts["fraction_of_trials"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{frac:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    cfg: dict[str, Any],
    model_summary: pd.DataFrame,
    reference_summary_df: pd.DataFrame,
    subset_counts: pd.DataFrame,
) -> list[Path]:
    plot_paths = [
        cfg["output_dir"] / "report_policy_match_accuracy_by_epoch.png",
        cfg["output_dir"] / "report_policy_match_bce_by_epoch.png",
        cfg["output_dir"] / "report_policy_model_plus_rate_by_epoch.png",
        cfg["output_dir"] / "reference_report_policy_accuracy_by_subset.png",
        cfg["output_dir"] / "report_policy_match_subset_counts.png",
    ]
    plot_metric_by_epoch(
        model_summary,
        metric_column="match_accuracy_mean",
        std_column="match_accuracy_std",
        ylabel="Policy match accuracy",
        title=f"{cfg['model_subdir']} report policy match",
        out_path=plot_paths[0],
        y_limits=(0, 1.04),
    )
    plot_metric_by_epoch(
        model_summary,
        metric_column="bce_mean",
        std_column="bce_std",
        ylabel="Binary cross entropy",
        title=f"{cfg['model_subdir']} report policy BCE",
        out_path=plot_paths[1],
    )
    plot_metric_by_epoch(
        model_summary,
        metric_column="model_plus_rate_mean",
        std_column="model_plus_rate_std",
        ylabel="Model + report rate",
        title=f"{cfg['model_subdir']} report sign bias",
        out_path=plot_paths[2],
        y_limits=(0, 1.04),
    )
    plot_reference_policies(reference_summary_df, plot_paths[3])
    plot_subset_counts(subset_counts, plot_paths[4])
    return plot_paths


def main() -> None:
    args = parse_args()
    cfg = build_run_config(args)
    if cfg["hazard_step"] <= 0:
        raise ValueError("--hazard-step must be positive")
    np.random.seed(cfg["seed"])
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    trials = load_trials(cfg)
    model_results = evaluate_models(cfg, trials)
    model_summary = summarize_model_results(model_results)
    reference_summary = reference_policy_summary(cfg, trials)
    subset_counts = write_subset_trial_counts(cfg, trials)

    by_seed_path = cfg["output_dir"] / "report_policy_match_by_seed.csv"
    summary_path = cfg["output_dir"] / "report_policy_match_summary.csv"
    reference_path = cfg["output_dir"] / "reference_report_policy_summary.csv"
    subset_counts_path = cfg["output_dir"] / "report_policy_match_subset_counts.csv"

    model_results.to_csv(by_seed_path, index=False)
    model_summary.to_csv(summary_path, index=False)
    reference_summary.to_csv(reference_path, index=False)
    subset_counts.to_csv(subset_counts_path, index=False)
    plot_paths = make_plots(cfg, model_summary, reference_summary, subset_counts) if cfg["make_plots"] else []

    print()
    print("Subset counts")
    print(subset_counts.to_string(index=False))
    print()
    print("Model policy match summary")
    print(model_summary.to_string(index=False))
    print()
    print("Reference report policy summary")
    print(reference_summary.to_string(index=False))
    print()
    print(f"Saved per-seed results to {by_seed_path}")
    print(f"Saved model summary to {summary_path}")
    print(f"Saved reference summary to {reference_path}")
    print(f"Saved subset counts to {subset_counts_path}")
    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

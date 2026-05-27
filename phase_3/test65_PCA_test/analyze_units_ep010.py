#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "accuracy_by_checkpoint_config.json"
CHECKPOINT_NAME = "checkpoint_ep010.pt"


class HiddenStateDataset(Dataset):
    def __init__(self, csv_paths: list[Path]):
        xs = []
        states = []
        true_hazard = []
        true_predict = []
        true_report = []
        source_csv = []
        csv_trial = []

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            for idx, row in df.reset_index(drop=True).iterrows():
                evidence = row["evidence"]
                if not isinstance(evidence, list):
                    evidence = ast.literal_eval(str(evidence))

                state_seq = row["states"]
                if not isinstance(state_seq, list):
                    state_seq = ast.literal_eval(str(state_seq))

                xs.append(torch.tensor(evidence, dtype=torch.float32).unsqueeze(-1))
                states.append(np.asarray(state_seq, dtype=np.float32))
                true_hazard.append(float(row["trueHazard"]))
                true_predict.append(int(row["truePredict"]))
                true_report.append(int(row["trueReport"]))
                source_csv.append(csv_path.name)
                csv_trial.append(int(idx))

        if not xs:
            raise ValueError("No trials were loaded.")

        self.x = xs
        self.states = np.stack(states, axis=0)
        self.true_hazard = np.asarray(true_hazard, dtype=np.float32)
        self.true_predict = np.asarray(true_predict, dtype=np.int64)
        self.true_report = np.asarray(true_report, dtype=np.int64)
        self.source_csv = source_csv
        self.csv_trial = csv_trial

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze hazard/state unit selectivity for one seed at checkpoint_ep010.pt."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-subdir", default="bce_both/sigma_1")
    parser.add_argument("--variant-subdir", default=None)
    parser.add_argument("--variant-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-variant-csvs", type=int, default=None)
    parser.add_argument("--model-class", default="GRUModel")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "unit_analysis_outputs")
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
    cfg = load_config(args.config.resolve())
    variant_subdir = args.variant_subdir or Path(args.model_subdir).name
    cfg.update(
        {
            "model_subdir": args.model_subdir,
            "variant_subdir": variant_subdir,
            "variant_split": args.variant_split,
            "max_variant_csvs": args.max_variant_csvs,
            "model_class": args.model_class,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "n_pcs": args.n_pcs,
            "cv_folds": args.cv_folds,
            "output_dir": args.output_dir.expanduser().resolve(),
            "checkpoint_name": CHECKPOINT_NAME,
        }
    )
    cfg["model_dir"] = cfg["model_root"] / cfg["model_subdir"]
    cfg["variant_dir"] = cfg["variant_root"] / cfg["variant_subdir"]
    cfg["seed_dir"] = cfg["model_dir"] / f"seed_{cfg['seed']}"
    cfg["checkpoint_path"] = cfg["seed_dir"] / cfg["checkpoint_name"]
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
    return hp


def collate_batch(batch):
    xs, idxs = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(idxs, dtype=torch.long)


def load_model(model_cls, cfg: dict[str, Any], device: torch.device):
    if not cfg["checkpoint_path"].exists():
        raise FileNotFoundError(f"Missing checkpoint: {cfg['checkpoint_path']}")

    hp = load_hp(cfg["seed_dir"])
    model = model_cls(hp).to(device)
    state = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def collect_hidden_states(model, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    chunks = []
    for x, _ in dataloader:
        hidden = model.rnn(x.to(device))
        chunks.append(hidden.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def corr_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = np.sum(X_centered * y_centered[:, None], axis=0)
    denominator = np.sqrt(np.sum(X_centered**2, axis=0) * np.sum(y_centered**2))
    out = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return out


def auc_strength_columns(X: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    scores = np.zeros(X.shape[1], dtype=np.float64)
    if len(np.unique(y_bin)) < 2:
        return scores
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.std(col) == 0:
            scores[i] = 0.0
            continue
        auc = roc_auc_score(y_bin, col)
        scores[i] = abs(float(auc) - 0.5)
    return scores


def bce_logits_np(logits: np.ndarray, y: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits_t, y_t).item()


def hidden_ablation_delta(H: np.ndarray, w: np.ndarray, y01: np.ndarray) -> np.ndarray:
    y = y01.astype(np.float32)[:, None]
    base_logits = H @ w[:, None]
    base_loss = bce_logits_np(base_logits, y)
    deltas = np.zeros(H.shape[1], dtype=np.float64)

    for i in range(H.shape[1]):
        H_abl = H.copy()
        H_abl[:, i] = H[:, i].mean()
        logits_abl = H_abl @ w[:, None]
        deltas[i] = bce_logits_np(logits_abl, y) - base_loss

    return deltas


def fit_probe(X: np.ndarray, y_bin: np.ndarray, cv_folds: int) -> tuple[np.ndarray, float, float]:
    if len(np.unique(y_bin)) < 2:
        return np.zeros(X.shape[1], dtype=np.float64), float("nan"), float("nan")

    min_class_count = int(np.min(np.bincount(y_bin.astype(int))))
    n_splits = max(2, min(cv_folds, min_class_count))
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(probe, X, y_bin, cv=cv)
    probe.fit(X, y_bin)
    weights = probe.named_steps["logisticregression"].coef_.squeeze()
    return weights, float(scores.mean()), float(scores.std())


def cosine_abs(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(abs(np.dot(a, b)) / denom)


def pca_alignments(H_flat: np.ndarray, w_haz: np.ndarray, w_loc: np.ndarray, n_pcs: int) -> pd.DataFrame:
    n_components = min(n_pcs, H_flat.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(H_flat)

    rows = []
    for idx, pc in enumerate(pca.components_, start=1):
        rows.append(
            {
                "pc": idx,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[idx - 1]),
                "haz_readout_alignment": cosine_abs(w_haz, pc),
                "loc_readout_alignment": cosine_abs(w_loc, pc),
            }
        )
    return pd.DataFrame(rows)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.ones_like(values, dtype=np.float64)
    return ranks / (len(values) - 1)


def build_unit_summary(
    H: np.ndarray,
    dataset: HiddenStateDataset,
    model,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, float]]:
    H_final = H[:, -1, :]
    H_flat = H.reshape(-1, H.shape[-1])
    state_flat = dataset.states.reshape(-1)
    hazard_cont = dataset.true_hazard
    hazard_bin = (dataset.true_predict > 0).astype(int)
    state_bin = (state_flat > 0).astype(int)

    w_haz = model.haz_head.weight.detach().cpu().numpy().squeeze()
    w_loc = model.loc_head.weight.detach().cpu().numpy().squeeze()

    haz_readout = np.abs(w_haz) * H_final.std(axis=0)
    state_readout = np.abs(w_loc) * H_flat.std(axis=0)

    haz_corr = corr_columns(H_final, hazard_cont)
    haz_auc_strength = auc_strength_columns(H_final, hazard_bin)
    state_corr = corr_columns(H_flat, state_flat)

    state_corr_by_time = np.stack(
        [corr_columns(H[:, t, :], dataset.states[:, t]) for t in range(H.shape[1])],
        axis=0,
    )
    haz_corr_by_time = np.stack(
        [corr_columns(H[:, t, :], hazard_cont) for t in range(H.shape[1])],
        axis=0,
    )

    haz_probe_w, haz_probe_acc, haz_probe_acc_std = fit_probe(
        H_final, hazard_bin, int(cfg["cv_folds"])
    )
    state_probe_w, state_probe_acc, state_probe_acc_std = fit_probe(
        H_flat, state_bin, int(cfg["cv_folds"])
    )

    haz_ablation = hidden_ablation_delta(H_final, w_haz, hazard_bin.astype(np.float32))
    state_ablation = hidden_ablation_delta(H_flat, w_loc, state_bin.astype(np.float32))

    haz_score = np.mean(
        np.vstack(
            [
                rank_percentile(haz_readout),
                rank_percentile(np.abs(haz_corr)),
                rank_percentile(np.abs(haz_probe_w)),
                rank_percentile(haz_ablation),
            ]
        ),
        axis=0,
    )
    state_score = np.mean(
        np.vstack(
            [
                rank_percentile(state_readout),
                rank_percentile(np.abs(state_corr)),
                rank_percentile(np.abs(state_probe_w)),
                rank_percentile(state_ablation),
            ]
        ),
        axis=0,
    )

    rows = []
    for unit in range(H.shape[-1]):
        rows.append(
            {
                "unit": unit,
                "haz_readout": haz_readout[unit],
                "haz_corr": haz_corr[unit],
                "haz_auc_strength": haz_auc_strength[unit],
                "haz_probe_weight": haz_probe_w[unit],
                "haz_ablation_delta": haz_ablation[unit],
                "haz_score": haz_score[unit],
                "state_readout": state_readout[unit],
                "state_corr": state_corr[unit],
                "state_probe_weight": state_probe_w[unit],
                "state_ablation_delta": state_ablation[unit],
                "state_score": state_score[unit],
            }
        )

    metrics = {
        "haz_probe_acc_mean": haz_probe_acc,
        "haz_probe_acc_std": haz_probe_acc_std,
        "state_probe_acc_mean": state_probe_acc,
        "state_probe_acc_std": state_probe_acc_std,
        "n_trials": float(H.shape[0]),
        "n_timesteps": float(H.shape[1]),
        "n_units": float(H.shape[2]),
    }

    return pd.DataFrame(rows), state_corr_by_time, haz_corr_by_time, metrics


def save_top_units(unit_df: pd.DataFrame, out_path: Path, top_k: int) -> None:
    ranking_specs = [
        ("haz_readout", False),
        ("haz_corr", True),
        ("haz_auc_strength", False),
        ("haz_probe_weight", True),
        ("haz_ablation_delta", False),
        ("haz_score", False),
        ("state_readout", False),
        ("state_corr", True),
        ("state_probe_weight", True),
        ("state_ablation_delta", False),
        ("state_score", False),
    ]
    rows = []
    for metric, absolute in ranking_specs:
        values = unit_df[metric].abs() if absolute else unit_df[metric]
        top = unit_df.assign(rank_value=values).sort_values("rank_value", ascending=False).head(top_k)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "unit": int(row["unit"]),
                    "value": float(row[metric]),
                    "rank_value": float(row["rank_value"]),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_hazard_state_scatter(unit_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(
        unit_df["haz_score"],
        unit_df["state_score"],
        c=unit_df["haz_ablation_delta"] - unit_df["state_ablation_delta"],
        cmap="coolwarm",
        s=36,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )
    ax.set_xlabel("Hazard score")
    ax.set_ylabel("State score")
    ax.set_title("Unit hazard vs state score")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="haz ablation - state ablation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_time_heatmap(matrix: np.ndarray, out_path: Path, title: str, top_units: np.ndarray) -> None:
    ordered = matrix[:, top_units].T
    vmax = float(np.nanmax(np.abs(ordered)))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(ordered, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Unit")
    ax.set_yticks(np.arange(len(top_units)))
    ax.set_yticklabels(top_units)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_pca_alignment(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["pc"], df["haz_readout_alignment"], marker="o", label="haz readout")
    ax1.plot(df["pc"], df["loc_readout_alignment"], marker="o", label="state readout")
    ax1.set_xlabel("PC")
    ax1.set_ylabel("Absolute cosine alignment")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(df["pc"], df["explained_variance_ratio"], alpha=0.22, color="gray", label="explained var")
    ax2.set_ylabel("Explained variance ratio")

    ax1.set_title("Readout direction alignment with PCs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = build_run_config(args)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Missing model directory: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Missing variant directory: {cfg['variant_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    model = load_model(model_cls, cfg, device)
    csvs = list_eval_csvs(cfg)
    dataset = HiddenStateDataset(csvs)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(f"Using device: {device}")
    print(f"Analyzing {cfg['checkpoint_path']}")
    print(f"Loaded {len(dataset)} trials from {len(csvs)} {cfg['variant_split']} CSVs")

    H = collect_hidden_states(model, dataloader, device)
    unit_df, state_time, haz_time, probe_metrics = build_unit_summary(H, dataset, model, cfg)

    prefix = f"seed_{cfg['seed']}_ep010"
    unit_summary_path = cfg["output_dir"] / f"{prefix}_unit_summary.csv"
    top_units_path = cfg["output_dir"] / f"{prefix}_top_units.csv"
    probe_metrics_path = cfg["output_dir"] / f"{prefix}_probe_metrics.json"
    pca_alignment_path = cfg["output_dir"] / f"{prefix}_pca_alignment.csv"

    unit_df.to_csv(unit_summary_path, index=False)
    save_top_units(unit_df, top_units_path, int(cfg["top_k"]))
    with probe_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(probe_metrics, f, indent=2)

    H_flat = H.reshape(-1, H.shape[-1])
    pca_df = pca_alignments(
        H_flat,
        model.haz_head.weight.detach().cpu().numpy().squeeze(),
        model.loc_head.weight.detach().cpu().numpy().squeeze(),
        int(cfg["n_pcs"]),
    )
    pca_df.to_csv(pca_alignment_path, index=False)

    plot_hazard_state_scatter(
        unit_df,
        cfg["output_dir"] / f"{prefix}_hazard_vs_state_unit_scores.png",
    )
    plot_pca_alignment(
        pca_df,
        cfg["output_dir"] / f"{prefix}_pca_readout_alignment.png",
    )

    top_state_units = (
        unit_df.assign(rank_value=unit_df["state_score"])
        .sort_values("rank_value", ascending=False)
        .head(int(cfg["top_k"]))["unit"]
        .to_numpy(dtype=int)
    )
    top_haz_units = (
        unit_df.assign(rank_value=unit_df["haz_score"])
        .sort_values("rank_value", ascending=False)
        .head(int(cfg["top_k"]))["unit"]
        .to_numpy(dtype=int)
    )
    plot_time_heatmap(
        state_time,
        cfg["output_dir"] / f"{prefix}_state_corr_by_time_heatmap.png",
        "State correlation by timestep",
        top_state_units,
    )
    plot_time_heatmap(
        haz_time,
        cfg["output_dir"] / f"{prefix}_hazard_corr_by_time_heatmap.png",
        "Hazard correlation by timestep",
        top_haz_units,
    )

    print(f"Saved unit summary to {unit_summary_path}")
    print(f"Saved top unit rankings to {top_units_path}")
    print(f"Saved probe metrics to {probe_metrics_path}")
    print(f"Saved PCA alignment to {pca_alignment_path}")


if __name__ == "__main__":
    main()

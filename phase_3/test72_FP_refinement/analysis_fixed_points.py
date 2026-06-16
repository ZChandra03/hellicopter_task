#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_CONFIG = BASE_DIR / "config.json"
DEFAULT_MODEL_CLASS = "GRUModel"

# Keep config.json path-only. Analysis behavior lives here and can still be
# overridden from the CLI when needed.
RUN_DEFAULTS = {
    "checkpoint": "final.pt",
    "train_heads": "both",
    "loss_type": "bce",
    "seed": 0,
    "n_inits": 500,
    "max_val_trials": 1000,
    "opt_steps": 3000,
    "lr": 0.01,
    "device": "cuda",
    "noise_scale": 0.0,
    "cluster_eps": 1e-3,
    "fixed_tol": 1e-6,
    "slow_tol": 1e-3,
    "eig_tol": 1e-2,
    "dtype": "float32",
    "opt_batch_size": 128,
    "patience": 400,
    "rel_improve_tol": 1e-6,
    "input_mode": "constant",
    "cycle_null_steps": 4,
    "rollout_steps": 0,
    "lbfgs_steps": 0,
    "max_plot_real_points": 50000,
    "model_class": DEFAULT_MODEL_CLASS,
}

FIXED_INPUTS = {
    "neg_-1_1": [-1.0, 1.0],
    "pos_1_1": [1.0, 1.0],
}


@dataclass
class Candidate:
    input_name: str
    init_index: int
    q: float
    step_norm: float
    h: np.ndarray
    cluster_count: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find GRU fixed/slow points and linearize recurrent dynamics."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory, e.g. bce_both/sigma_1/seed_0. "
            "If omitted, config.json model_root plus --seed is used."
        ),
    )
    parser.add_argument("--checkpoint", default=RUN_DEFAULTS["checkpoint"], help="Checkpoint filename.")
    parser.add_argument("--group_key", default=None, help="Variant group, e.g. sigma_1.")
    parser.add_argument("--train_heads", default=RUN_DEFAULTS["train_heads"], choices=["rep", "haz", "both"])
    parser.add_argument("--loss_type", default=RUN_DEFAULTS["loss_type"], choices=["reinforce", "bce"])
    parser.add_argument("--seed", type=int, default=RUN_DEFAULTS["seed"], help="Random seed and default seed_N dir.")
    parser.add_argument("--n_inits", type=int, default=RUN_DEFAULTS["n_inits"])
    parser.add_argument("--max_val_trials", type=int, default=RUN_DEFAULTS["max_val_trials"])
    parser.add_argument("--opt_steps", type=int, default=RUN_DEFAULTS["opt_steps"])
    parser.add_argument("--lr", type=float, default=RUN_DEFAULTS["lr"])
    parser.add_argument("--device", default=RUN_DEFAULTS["device"], help="'cuda', 'cpu', or a torch device string.")
    parser.add_argument("--noise_scale", type=float, default=RUN_DEFAULTS["noise_scale"])
    parser.add_argument("--cluster_eps", type=float, default=RUN_DEFAULTS["cluster_eps"])
    parser.add_argument("--fixed_tol", type=float, default=RUN_DEFAULTS["fixed_tol"])
    parser.add_argument("--slow_tol", type=float, default=RUN_DEFAULTS["slow_tol"])
    parser.add_argument("--eig_tol", type=float, default=RUN_DEFAULTS["eig_tol"])
    parser.add_argument("--dtype", choices=["float32", "float64"], default=RUN_DEFAULTS["dtype"])
    parser.add_argument("--batch_size", type=int, default=None, help="Validation batch size.")
    parser.add_argument(
        "--opt_batch_size",
        type=int,
        default=RUN_DEFAULTS["opt_batch_size"],
        help="How many initial states to optimize together. Default: 128",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=RUN_DEFAULTS["patience"],
        help="Early-stop patience in optimization steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--rel_improve_tol",
        type=float,
        default=RUN_DEFAULTS["rel_improve_tol"],
        help="Relative improvement threshold for early stopping.",
    )
    parser.add_argument(
        "--input_mode",
        choices=["constant", "null_then_evidence"],
        default=RUN_DEFAULTS["input_mode"],
        help=(
            "Input map to optimize: constant applies the evidence input once; "
            "null_then_evidence applies N null steps followed by the evidence input."
        ),
    )
    parser.add_argument(
        "--cycle_null_steps",
        type=int,
        default=RUN_DEFAULTS["cycle_null_steps"],
        help="Number of [0, 0] steps before evidence in null_then_evidence mode.",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=RUN_DEFAULTS["rollout_steps"],
        help="Input-map rollout steps applied to matched initial states before optimization.",
    )
    parser.add_argument(
        "--lbfgs_steps",
        type=int,
        default=RUN_DEFAULTS["lbfgs_steps"],
        help="Optional LBFGS polish iterations after Adam. Use 0 to disable.",
    )
    parser.add_argument(
        "--max_points_per_input",
        type=int,
        default=None,
        help="Optional cap on deduplicated points to linearize per fixed input.",
    )
    parser.add_argument(
        "--max_plot_real_points",
        type=int,
        default=RUN_DEFAULTS["max_plot_real_points"],
        help="Maximum real hidden states shown in PCA scatter plots.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to fixed_point_analysis/<model>/<checkpoint>/",
    )
    parser.add_argument(
        "--variant_root",
        type=Path,
        default=None,
        help="Root containing group valConfig_*.csv directories.",
    )
    parser.add_argument("--model_class", default=RUN_DEFAULTS["model_class"])
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("model_root", "variant_root"):
        if cfg.get(key):
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    return cfg


def get_default_hp(loss_type: str = "bce", train_heads: str = "both") -> dict[str, Any]:
    return {
        "n_input": 2,
        "n_rnn": 128,
        "batch_size": 25,
        "learning_rate": 3e-4,
        "target_loss": 1e-3,
        "max_epochs": 10,
        "max_csv": 20,
        "n_null_timesteps": 4,
        "loss_type": loss_type,
        "train_heads": train_heads,
    }


def resolve_model_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.model_dir is None:
        if not cfg.get("model_root"):
            raise ValueError("Provide --model_dir or define model_root in config.json.")
        model_dir = Path(cfg["model_root"]) / f"seed_{args.seed}"
        return model_dir.resolve()

    raw = args.model_dir.expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                BASE_DIR / raw,
                REPO_ROOT / raw,
                REPO_ROOT / "models" / "models_n5" / raw,
            ]
        )
        if cfg.get("model_root"):
            candidates.append(Path(cfg["model_root"]) / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def infer_group_key(model_dir: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    if args.group_key:
        return args.group_key
    if model_dir.name.startswith("seed_") and model_dir.parent.name:
        return model_dir.parent.name
    if model_dir.name.startswith("sigma_"):
        return model_dir.name
    if cfg.get("sigma"):
        return str(cfg["sigma"])
    if cfg.get("model_root"):
        return Path(cfg["model_root"]).name
    raise ValueError("Could not infer group_key; pass --group_key.")


def resolve_variant_root(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.variant_root is not None:
        return args.variant_root.expanduser().resolve()
    if cfg.get("variant_root"):
        return Path(cfg["variant_root"]).expanduser().resolve()
    return (REPO_ROOT / "variants" / "variants_current").resolve()


def find_model_code_root(model_dir: Path) -> Path:
    for path in (model_dir, *model_dir.parents):
        if (path / "rnn_models.py").exists():
            return path
    fallback = REPO_ROOT / "models" / "models_n5"
    if (fallback / "rnn_models.py").exists():
        return fallback
    raise FileNotFoundError(f"Could not find rnn_models.py at or above {model_dir}")


def import_model_class(model_dir: Path, class_name: str):
    module_path = find_model_code_root(model_dir) / "rnn_models.py"
    spec = importlib.util.spec_from_file_location("fixed_point_rnn_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} does not define {class_name}") from exc


def load_hp(model_dir: Path, loss_type: str, train_heads: str) -> dict[str, Any]:
    hp_path = model_dir / "hp.json"
    if hp_path.exists():
        with hp_path.open("r", encoding="utf-8") as f:
            hp = json.load(f)
    else:
        hp = get_default_hp(loss_type, train_heads)

    defaults = get_default_hp(loss_type, train_heads)
    for key, value in defaults.items():
        hp.setdefault(key, value)
    return hp


def load_model(
    model_dir: Path,
    checkpoint: str,
    model_class: str,
    loss_type: str,
    train_heads: str,
    device: torch.device,
):
    checkpoint_path = model_dir / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    hp = load_hp(model_dir, loss_type, train_heads)
    model_cls = import_model_class(model_dir, model_class)
    model = model_cls(hp).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, hp, checkpoint_path


def encode_evidence_sequence(
    evidence: list[float],
    n_input: int,
    n_null_timesteps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(evidence) == 0:
        raise ValueError("Evidence sequence cannot be empty")
    if n_input != 2:
        raise ValueError(f"Fixed-point analysis currently expects n_input=2, got {n_input}")

    steps: list[list[float]] = []
    null_step = [0.0, 0.0]
    for i, evidence_t in enumerate(evidence):
        steps.append([float(evidence_t), 1.0])
        if i < len(evidence) - 1:
            steps.extend([null_step.copy() for _ in range(n_null_timesteps)])
    return torch.tensor(steps, dtype=dtype)


class HelicopterFixedPointDataset(Dataset):
    def __init__(self, df: pd.DataFrame, hp: dict[str, Any], dtype: torch.dtype):
        self.x: list[torch.Tensor] = []
        self.trial_meta: list[dict[str, Any]] = []
        n_input = int(hp["n_input"])
        n_null = int(hp.get("n_null_timesteps", 4))

        for trial_idx, row in df.reset_index(drop=True).iterrows():
            evidence = row["evidence"]
            if not isinstance(evidence, list):
                evidence = ast.literal_eval(str(evidence))
            x = encode_evidence_sequence(evidence, n_input, n_null, dtype)
            self.x.append(x)
            self.trial_meta.append(
                {
                    "trial_index": int(trial_idx),
                    "trial_in_block": row.get("trialInBlock", np.nan),
                    "true_hazard": float(row.get("trueHazard", np.nan)),
                    "true_report": int(float(row["trueReport"])),
                    "true_predict": int(float(row["truePredict"])),
                }
            )
        if not self.x:
            raise ValueError("No validation trials were loaded.")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], idx


def collate_batch(batch):
    xs, idxs = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(idxs, dtype=torch.long)


def load_validation_dataframe(
    variant_root: Path,
    group_key: str,
    max_val_trials: int | None,
) -> tuple[pd.DataFrame, list[Path]]:
    variant_dir = variant_root / group_key
    paths = [variant_dir / f"valConfig_{k}.csv" for k in range(5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing validation CSVs: {missing}")

    dfs = [pd.read_csv(path) for path in paths]
    df = pd.concat(dfs, ignore_index=True)
    if max_val_trials is not None and max_val_trials > 0:
        df = df.iloc[:max_val_trials].copy()
    return df, paths


@torch.inference_mode()
def collect_hidden_states(
    model,
    val_df: pd.DataFrame,
    hp: dict[str, Any],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dataset = HelicopterFixedPointDataset(val_df, hp, dtype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    hidden_batches = []
    input_batches = []
    meta_rows: list[dict[str, Any]] = []

    for x, trial_idxs in dataloader:
        x = x.to(device)
        hidden = model.rnn(x)
        hidden_np = hidden.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        hidden_batches.append(hidden_np)
        input_batches.append(x_np)

        for batch_i, trial_idx in enumerate(trial_idxs.tolist()):
            trial_meta = dataset.trial_meta[trial_idx]
            for timestep in range(x_np.shape[1]):
                evidence_value = float(x_np[batch_i, timestep, 0])
                evidence_present = float(x_np[batch_i, timestep, 1])
                meta_rows.append(
                    {
                        **trial_meta,
                        "timestep": int(timestep),
                        "input_type": "evidence" if evidence_present != 0.0 else "null",
                        "evidence_value": evidence_value if evidence_present != 0.0 else 0.0,
                    }
                )

    h_all = np.concatenate(hidden_batches, axis=0)
    x_all = np.concatenate(input_batches, axis=0)
    h_real = h_all.reshape(-1, h_all.shape[-1])
    x_flat = x_all.reshape(-1, x_all.shape[-1])
    metadata = pd.DataFrame(meta_rows)
    return h_real, x_flat, metadata


def select_matched_init_indices(
    metadata: pd.DataFrame,
    x_values: list[float],
    n_inits: int,
    rng: np.random.Generator,
    match_window: float = 0.5,
) -> np.ndarray:
    evidence_value = float(x_values[0])
    evidence_present = float(x_values[1])

    input_type = metadata["input_type"].fillna("null")

    if evidence_present == 0.0:
        pool = np.flatnonzero(input_type != "evidence")
    else:
        ev = metadata["evidence_value"].to_numpy(dtype=float)

        near = (
            (input_type.to_numpy() == "evidence")
            & (np.abs(ev - evidence_value) <= match_window)
        )
        pool = np.flatnonzero(near)

        if len(pool) < max(50, n_inits // 10):
            same_sign = (
                (input_type.to_numpy() == "evidence")
                & (np.sign(ev) == np.sign(evidence_value))
            )
            pool = np.flatnonzero(same_sign)

    if len(pool) == 0:
        pool = np.arange(len(metadata))

    replace = len(pool) < n_inits
    return rng.choice(pool, size=n_inits, replace=replace)


def gru_one_step(model, h: torch.Tensor, x_fixed: torch.Tensor) -> torch.Tensor:
    squeeze = h.ndim == 1
    if squeeze:
        h_batch = h.unsqueeze(0)
    else:
        h_batch = h
    x_batch = x_fixed.to(device=h_batch.device, dtype=h_batch.dtype).view(1, 1, -1)
    x_batch = x_batch.expand(h_batch.shape[0], 1, -1)
    h0 = h_batch.unsqueeze(0)
    _, h_last = model.rnn.gru(x_batch, h0)
    h_next = h_last.squeeze(0)
    return h_next.squeeze(0) if squeeze else h_next


def build_input_sequence(
    x_values: list[float],
    input_mode: str,
    cycle_null_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cycle_null_steps < 0:
        raise ValueError(f"cycle_null_steps must be nonnegative, got {cycle_null_steps}")

    if input_mode == "constant":
        values = [x_values]
    elif input_mode == "null_then_evidence":
        values = [[0.0, 0.0] for _ in range(cycle_null_steps)]
        values.append(x_values)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    return torch.tensor(values, device=device, dtype=dtype)


def apply_input_sequence(model, h: torch.Tensor, x_sequence: torch.Tensor) -> torch.Tensor:
    if x_sequence.ndim == 1:
        return gru_one_step(model, h, x_sequence)

    h_next = h
    for x_t in x_sequence:
        h_next = gru_one_step(model, h_next, x_t)
    return h_next


@torch.no_grad()
def rollout_input_sequence(model, init_states: torch.Tensor, x_sequence: torch.Tensor, steps: int) -> torch.Tensor:
    h = init_states
    for _ in range(steps):
        h = apply_input_sequence(model, h, x_sequence)
    return h


def speed_objective(model, h: torch.Tensor, x_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = apply_input_sequence(model, h, x_sequence) - h
    step_norm = torch.linalg.norm(delta, dim=-1)
    q = 0.5 * torch.sum(delta * delta, dim=-1)
    return q, step_norm


def optimize_slow_points_for_input(
    model,
    init_states: torch.Tensor,
    input_name: str,
    x_sequence: torch.Tensor,
    opt_steps: int,
    lr: float,
    opt_batch_size: int,
    patience: int,
    rel_improve_tol: float,
    lbfgs_steps: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    total = init_states.shape[0]

    for start in range(0, total, opt_batch_size):
        end = min(start + opt_batch_size, total)
        h = init_states[start:end].detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([h], lr=lr)
        best_loss = math.inf
        stale_steps = 0

        for _ in range(opt_steps):
            optimizer.zero_grad(set_to_none=True)
            q, _ = speed_objective(model, h, x_sequence)
            loss = q.mean()
            loss.backward()
            optimizer.step()

            if patience > 0:
                loss_value = float(loss.detach().cpu())
                rel_improve = (best_loss - loss_value) / max(abs(best_loss), 1e-12)
                if loss_value < best_loss and (math.isinf(best_loss) or rel_improve > rel_improve_tol):
                    best_loss = loss_value
                    stale_steps = 0
                else:
                    stale_steps += 1
                if stale_steps >= patience:
                    break

        if lbfgs_steps > 0:
            optimizer = torch.optim.LBFGS(
                [h],
                lr=1.0,
                max_iter=lbfgs_steps,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                q, _ = speed_objective(model, h, x_sequence)
                loss = q.mean()
                loss.backward()
                return loss

            optimizer.step(closure)

        with torch.no_grad():
            q_final, step_final = speed_objective(model, h, x_sequence)
            h_np = h.detach().cpu().numpy()
            q_np = q_final.detach().cpu().numpy()
            step_np = step_final.detach().cpu().numpy()

        for local_i in range(end - start):
            candidates.append(
                Candidate(
                    input_name=input_name,
                    init_index=start + local_i,
                    q=float(q_np[local_i]),
                    step_norm=float(step_np[local_i]),
                    h=h_np[local_i].astype(np.float32, copy=True),
                )
            )

    return candidates


def deduplicate_points(candidates: list[Candidate], cluster_eps: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.q):
        assigned = False
        for point in kept:
            if np.linalg.norm(candidate.h - point.h) <= cluster_eps:
                point.cluster_count += 1
                assigned = True
                break
        if not assigned:
            kept.append(candidate)
    return kept


def classify_point(step_norm: float, fixed_tol: float, slow_tol: float) -> str:
    if step_norm < fixed_tol:
        return "fixed"
    if step_norm < slow_tol:
        return "slow"
    return "not_slow"


def label_from_prob(prob: float, low_label: str, high_label: str, mid_label: str) -> str:
    if prob < 0.25:
        return low_label
    if prob > 0.75:
        return high_label
    return mid_label


def label_point(model, h_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    with torch.no_grad():
        h = torch.tensor(h_np, device=device, dtype=dtype).view(1, -1)
        report_logit = float(model.loc_head(h).view(-1)[0].detach().cpu())
        hazard_logit = float(model.haz_head(h).view(-1)[0].detach().cpu())
        report_prob = float(torch.sigmoid(torch.tensor(report_logit)))
        hazard_prob = float(torch.sigmoid(torch.tensor(hazard_logit)))
    return {
        "report_logit": report_logit,
        "hazard_logit": hazard_logit,
        "report_prob": report_prob,
        "hazard_prob": hazard_prob,
        "report_label": label_from_prob(
            report_prob, "-1 basin", "+1 basin", "report boundary/uncertain"
        ),
        "hazard_label": label_from_prob(
            hazard_prob, "low hazard", "high hazard", "hazard boundary/uncertain"
        ),
    }


def jacobian_at_point(model, h_np: np.ndarray, x_sequence: torch.Tensor, device: torch.device, dtype: torch.dtype):
    h_star = torch.tensor(h_np, device=device, dtype=dtype).detach().clone().requires_grad_(True)

    def f_of_h(h):
        return apply_input_sequence(model, h, x_sequence)

    jac = torch.autograd.functional.jacobian(f_of_h, h_star, vectorize=True)
    return jac.detach()


def classify_stability(eigvals: torch.Tensor, eig_tol: float) -> tuple[str, dict[str, Any]]:
    radii = torch.abs(eigvals)
    n_unstable = int((radii > 1.0 + eig_tol).sum().item())
    n_slow = int((torch.abs(radii - 1.0) <= eig_tol).sum().item())
    n_stable = int((radii < 1.0 - eig_tol).sum().item())

    if n_unstable > 0 and n_slow > 0:
        label = "saddle_with_slow_directions"
    elif n_unstable > 0:
        label = "saddle_or_unstable"
    elif n_slow > 0:
        label = "stable_with_slow_directions"
    else:
        label = "stable"

    return label, {
        "max_abs_eig": float(radii.max().detach().cpu()),
        "n_stable_eigs": n_stable,
        "n_slow_eigs": n_slow,
        "n_unstable_eigs": n_unstable,
    }


def summarize_top_eigs(eigvals: torch.Tensor, n: int = 10) -> dict[str, str]:
    eig_cpu = eigvals.detach().cpu()
    order = torch.argsort(torch.abs(eig_cpu), descending=True)[:n]
    top = eig_cpu[order]
    return {
        "top_10_eigvals_real": json.dumps([float(v.real) for v in top]),
        "top_10_eigvals_imag": json.dumps([float(v.imag) for v in top]),
        "top_10_abs_eigvals": json.dumps([float(torch.abs(v)) for v in top]),
    }


def compute_real_speeds(
    model,
    h_real: np.ndarray,
    x_sequence: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 4096,
) -> np.ndarray:
    speeds = []
    for start in range(0, len(h_real), batch_size):
        h = torch.tensor(h_real[start : start + batch_size], device=device, dtype=dtype)
        with torch.no_grad():
            _, step_norm = speed_objective(model, h, x_sequence)
        speeds.append(step_norm.detach().cpu().numpy())
    return np.concatenate(speeds, axis=0)


def percentile_of_score(distribution: np.ndarray, value: float) -> float:
    if distribution.size == 0:
        return float("nan")
    return float(100.0 * np.mean(distribution <= value))


def nearest_real_distance(h_real: np.ndarray, h: np.ndarray, chunk_size: int = 50000) -> float:
    best = math.inf
    for start in range(0, len(h_real), chunk_size):
        chunk = h_real[start : start + chunk_size]
        distances = np.linalg.norm(chunk - h, axis=1)
        best = min(best, float(distances.min()))
    return best


def safe_output_name(model_dir: Path) -> str:
    if model_dir.name.startswith("seed_") and model_dir.parent.name and model_dir.parent.parent.name:
        parts = [model_dir.parent.parent.name, model_dir.parent.name, model_dir.name]
    else:
        parts = list(model_dir.parts[-3:])
    return "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in parts)


def default_output_dir(model_dir: Path, checkpoint: str) -> Path:
    checkpoint_stem = Path(checkpoint).stem
    return BASE_DIR / "fixed_point_analysis" / safe_output_name(model_dir) / checkpoint_stem


def save_config(out_dir: Path, args: argparse.Namespace, cfg: dict[str, Any], extra: dict[str, Any]) -> None:
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "config_json": cfg,
        **extra,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_npz_outputs(out_dir: Path, rows: list[dict[str, Any]], h_points: list[np.ndarray], eigvals: list[np.ndarray]) -> None:
    input_names = np.array([row["input_name"] for row in rows], dtype=object)
    point_ids = np.array([row["point_id"] for row in rows], dtype=np.int64)
    h_arr = np.stack(h_points, axis=0) if h_points else np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(out_dir / "slow_points_full.npz", h_points=h_arr, input_names=input_names, point_ids=point_ids)

    max_len = max((len(v) for v in eigvals), default=0)
    eig_arr = np.full((len(eigvals), max_len), np.nan + 1j * np.nan, dtype=np.complex64)
    for i, values in enumerate(eigvals):
        eig_arr[i, : len(values)] = values.astype(np.complex64)
    np.savez_compressed(out_dir / "eigvals_by_point.npz", eigvals=eig_arr, input_names=input_names, point_ids=point_ids)


def sample_real_for_plot(h_real: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(h_real) <= max_points:
        return h_real
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(h_real), size=max_points, replace=False)
    return h_real[idx]


def is_plottable_point(row: dict[str, Any]) -> bool:
    return str(row.get("point_type", "")).replace("periodic_", "") in {"fixed", "slow"}


def plot_pca_slow_points(
    out_dir: Path,
    h_real: np.ndarray,
    point_rows: list[dict[str, Any]],
    h_points: list[np.ndarray],
    max_real_points: int,
    seed: int,
) -> None:
    plot_idx = [i for i, row in enumerate(point_rows) if is_plottable_point(row)]
    if len(h_real) < 3 or not plot_idx:
        return

    plot_rows = [point_rows[i] for i in plot_idx]
    plot_h_points = [h_points[i] for i in plot_idx]

    pca = PCA(n_components=3).fit(h_real)
    real_sample = sample_real_for_plot(h_real, max_real_points, seed)
    real_pc = pca.transform(real_sample)
    point_arr = np.stack(plot_h_points, axis=0)
    point_pc = pca.transform(point_arr)

    input_colors = {
        "neg_-1_1": "#e45756",
        "pos_1_1": "#54a24b",
    }
    stability_markers = {
        "stable": "o",
        "stable_with_slow_directions": "s",
        "saddle_or_unstable": "^",
        "saddle_with_slow_directions": "X",
    }

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(real_pc[:, 0], real_pc[:, 1], s=3, c="0.75", alpha=0.18, linewidths=0, label="real states")
    for stability, marker in stability_markers.items():
        idx = [i for i, row in enumerate(plot_rows) if row["stability_label"] == stability]
        if not idx:
            continue
        colors = [input_colors.get(plot_rows[i]["input_name"], "black") for i in idx]
        ax.scatter(
            point_pc[idx, 0],
            point_pc[idx, 1],
            s=70,
            c=colors,
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA real trajectories plus fixed/slow points")
    ax.grid(True, alpha=0.25)
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="0.65",
            markeredgecolor="black",
            markersize=8,
            linestyle="none",
            label=stability,
        )
        for stability, marker in stability_markers.items()
        if any(row["stability_label"] == stability for row in plot_rows)
    ]
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            linestyle="none",
            label=input_name,
        )
        for input_name, color in input_colors.items()
        if any(row["input_name"] == input_name for row in plot_rows)
    ]
    shape_legend = ax.legend(handles=shape_handles, title="stability", frameon=False, fontsize=8, loc="best")
    ax.add_artist(shape_legend)
    ax.legend(handles=color_handles, title="input", frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_slow_points_by_input.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    for field, filename, cmap in [
        ("report_logit", "pca_points_report_logit.png", "coolwarm"),
        ("hazard_logit", "pca_points_hazard_logit.png", "viridis"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        values = np.array([float(row[field]) for row in plot_rows])
        ax.scatter(real_pc[:, 0], real_pc[:, 1], s=3, c="0.75", alpha=0.18, linewidths=0)
        scatter = ax.scatter(
            point_pc[:, 0],
            point_pc[:, 1],
            c=values,
            s=80,
            cmap=cmap,
            edgecolors="black",
            linewidths=0.6,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Fixed/slow points colored by {field}")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label=field)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=250, bbox_inches="tight")
        plt.close(fig)


def plot_eigenspectra(out_dir: Path, point_rows: list[dict[str, Any]], eigvals_np: list[np.ndarray]) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    for input_name in FIXED_INPUTS:
        idx = [
            i for i, row in enumerate(point_rows)
            if row["input_name"] == input_name and is_plottable_point(row)
        ]
        if not idx:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.0, alpha=0.65, label="unit circle")
        for i in idx:
            values = eigvals_np[i]
            ax.scatter(values.real, values.imag, s=12, alpha=0.5, linewidths=0)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(0.0, color="0.7", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Real(lambda)")
        ax.set_ylabel("Imag(lambda)")
        ax.set_title(f"Eigenspectrum: {input_name}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"eigenspectrum_{input_name}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)


def plot_speed_histograms(
    out_dir: Path,
    real_speeds_by_input: dict[str, np.ndarray],
    raw_candidates_by_input: dict[str, list[Candidate]],
) -> None:
    for input_name, real_speeds in real_speeds_by_input.items():
        candidate_speeds = np.array([c.step_norm for c in raw_candidates_by_input[input_name]], dtype=float)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bins = np.logspace(
            math.log10(max(min(real_speeds.min(), candidate_speeds.min(), 1e-12), 1e-12)),
            math.log10(max(real_speeds.max(), candidate_speeds.max(), 1e-8)),
            60,
        )
        ax.hist(real_speeds, bins=bins, alpha=0.55, label="real trajectory speeds", density=True)
        ax.hist(candidate_speeds, bins=bins, alpha=0.55, label="optimized candidate speeds", density=True)
        ax.set_xscale("log")
        ax.set_xlabel("||F(h, x) - h||")
        ax.set_ylabel("Density")
        ax.set_title(f"Speed histogram: {input_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"speed_hist_{input_name}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    cfg = load_config()
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    model_dir = resolve_model_dir(args, cfg)
    group_key = infer_group_key(model_dir, args, cfg)
    variant_root = resolve_variant_root(args, cfg)
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(model_dir, args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, hp, checkpoint_path = load_model(
        model_dir,
        args.checkpoint,
        args.model_class,
        args.loss_type,
        args.train_heads,
        device,
    )
    if dtype == torch.float64:
        model = model.double()
    else:
        model = model.float()

    val_df, val_paths = load_validation_dataframe(variant_root, group_key, args.max_val_trials)
    batch_size = args.batch_size or int(hp.get("batch_size", 25))
    h_real, x_flat, metadata = collect_hidden_states(model, val_df, hp, batch_size, device, dtype)
    metadata.to_csv(out_dir / "real_state_metadata.csv", index=False)

    rng = np.random.default_rng(args.seed)

    print(f"Using device: {device}")
    print(f"Loaded model: {checkpoint_path}")
    print(f"Loaded {len(val_df)} validation trials from {len(val_paths)} CSVs.")
    print(f"Collected {len(h_real)} hidden states; optimizing {args.n_inits} matched initial states per input.")
    print(f"Input mode: {args.input_mode}")

    rows: list[dict[str, Any]] = []
    h_points: list[np.ndarray] = []
    eigvals_np: list[np.ndarray] = []
    raw_candidates_by_input: dict[str, list[Candidate]] = {}
    real_speeds_by_input: dict[str, np.ndarray] = {}
    init_indices_by_input: dict[str, list[int]] = {}
    input_sequences_by_input: dict[str, list[list[float]]] = {}

    for input_name, x_values in FIXED_INPUTS.items():
        print(f"Optimizing candidates for {input_name}...")
        x_sequence = build_input_sequence(
            x_values=x_values,
            input_mode=args.input_mode,
            cycle_null_steps=args.cycle_null_steps,
            device=device,
            dtype=dtype,
        )
        input_sequence = x_sequence.detach().cpu().tolist()
        input_sequences_by_input[input_name] = input_sequence

        init_idx = select_matched_init_indices(
            metadata=metadata,
            x_values=x_values,
            n_inits=args.n_inits,
            rng=rng,
            match_window=0.5,
        )
        init_indices_by_input[input_name] = init_idx.astype(int).tolist()
        init_states = torch.tensor(h_real[init_idx], device=device, dtype=dtype)
        if args.noise_scale > 0.0:
            init_states = init_states + args.noise_scale * torch.randn_like(init_states)
        if args.rollout_steps > 0:
            init_states = rollout_input_sequence(model, init_states, x_sequence, args.rollout_steps)

        real_speeds = compute_real_speeds(model, h_real, x_sequence, device, dtype)
        real_speeds_by_input[input_name] = real_speeds

        raw_candidates = optimize_slow_points_for_input(
            model,
            init_states,
            input_name,
            x_sequence,
            args.opt_steps,
            args.lr,
            args.opt_batch_size,
            args.patience,
            args.rel_improve_tol,
            args.lbfgs_steps,
        )
        raw_candidates_by_input[input_name] = raw_candidates
        kept = deduplicate_points(raw_candidates, args.cluster_eps)
        if args.max_points_per_input is not None:
            kept = kept[: args.max_points_per_input]
        print(f"  kept {len(kept)} deduplicated points from {len(raw_candidates)} candidates")

        for point_id, candidate in enumerate(kept):
            labels = label_point(model, candidate.h, device, dtype)
            jac = jacobian_at_point(model, candidate.h, x_sequence, device, dtype)
            eigvals = torch.linalg.eigvals(jac)
            stability_label, stability_metrics = classify_stability(eigvals, args.eig_tol)
            eig_summary = summarize_top_eigs(eigvals)
            eig_np = eigvals.detach().cpu().numpy()
            point_type = classify_point(candidate.step_norm, args.fixed_tol, args.slow_tol)
            if args.input_mode != "constant":
                point_type = f"periodic_{point_type}"

            row = {
                "input_name": input_name,
                "input_mode": args.input_mode,
                "input_sequence": json.dumps(input_sequence),
                "input_sequence_steps": len(input_sequence),
                "point_id": point_id,
                "q": candidate.q,
                "step_norm": candidate.step_norm,
                "point_type": point_type,
                "cluster_count": candidate.cluster_count,
                **labels,
                **stability_metrics,
                "stability_label": stability_label,
                **eig_summary,
                "distance_to_nearest_real_state": nearest_real_distance(h_real, candidate.h),
                "real_speed_percentile": percentile_of_score(real_speeds, candidate.step_norm),
            }
            rows.append(row)
            h_points.append(candidate.h)
            eigvals_np.append(eig_np)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "slow_points_summary.csv", index=False)
    save_npz_outputs(out_dir, rows, h_points, eigvals_np)
    save_config(
        out_dir,
        args,
        cfg,
        {
            "model_dir": str(model_dir),
            "checkpoint_path": str(checkpoint_path),
            "group_key": group_key,
            "variant_root": str(variant_root),
            "validation_csvs": [str(path) for path in val_paths],
            "hp": hp,
            "n_hidden_states": int(len(h_real)),
            "init_indices_by_input": init_indices_by_input,
            "input_sequences_by_input": input_sequences_by_input,
            "x_flat_shape": list(x_flat.shape),
        },
    )

    plot_pca_slow_points(out_dir, h_real, rows, h_points, args.max_plot_real_points, args.seed)
    plot_eigenspectra(out_dir, rows, eigvals_np)
    plot_speed_histograms(out_dir, real_speeds_by_input, raw_candidates_by_input)

    print(f"Saved summary and plots to {out_dir}")


if __name__ == "__main__":
    main()

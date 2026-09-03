from __future__ import annotations

import ast
import importlib.util
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_CONFIG = BASE_DIR / "config.json"
CHECKPOINT_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_root: Path
    seed: int

    @property
    def seed_dir(self) -> Path:
        return self.model_root / f"seed_{self.seed}"


@dataclass(frozen=True)
class TrialRecord:
    source_csv: str
    csv_trial: int
    global_trial: int
    sigma: float
    true_hazard: float
    true_report: int
    true_predict: int
    evidence: np.ndarray
    states: np.ndarray


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = load_json(path.expanduser().resolve())
    if not cfg.get("models"):
        raise ValueError("config.json must define a non-empty models list")
    if not cfg.get("variant_root") or not cfg.get("variant_subdir"):
        raise ValueError("config.json must define variant_root and variant_subdir")
    return cfg


def model_specs(cfg: dict[str, Any], requested: set[str] | None = None) -> list[ModelSpec]:
    seed = int(cfg.get("seed", 0))
    specs: list[ModelSpec] = []
    for entry in cfg["models"]:
        key = str(entry["key"])
        if requested is not None and key not in requested:
            continue
        spec = ModelSpec(
            key=key,
            label=str(entry.get("label", key)),
            model_root=Path(entry["model_root"]).expanduser().resolve(),
            seed=seed,
        )
        if not spec.seed_dir.is_dir():
            raise FileNotFoundError(f"Missing model directory: {spec.seed_dir}")
        specs.append(spec)
    if not specs:
        raise ValueError(f"No configured models match {sorted(requested or [])}")
    return specs


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_model_filter(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return set(parse_csv_list(value))


def checkpoint_sort_key(path: Path) -> tuple[int, int]:
    if path.name == "checkpoint_init.pt":
        return (0, 0)
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match:
        return (1, int(match.group(1)))
    if path.name == "checkpoint_best.pt":
        return (2, 0)
    if path.name == "final.pt":
        return (3, 0)
    return (4, 0)


def checkpoint_label(path: Path) -> str:
    if path.name == "checkpoint_init.pt":
        return "init"
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match:
        return f"ep{int(match.group(1)):03d}"
    if path.name == "checkpoint_best.pt":
        return "best"
    if path.name == "final.pt":
        return "final"
    return path.stem


def resolve_checkpoints(seed_dir: Path, requested: str) -> list[Path]:
    tokens = parse_csv_list(requested)
    paths: list[Path] = []
    if not tokens or tokens == ["all"]:
        candidates = [seed_dir / "checkpoint_init.pt"]
        candidates.extend(seed_dir.glob("checkpoint_ep*.pt"))
        candidates.append(seed_dir / "final.pt")
        paths = [path for path in candidates if path.exists()]
    else:
        for token in tokens:
            lower = token.lower()
            if lower in {"init", "initial"}:
                name = "checkpoint_init.pt"
            elif lower == "final":
                name = "final.pt"
            elif lower == "best":
                name = "checkpoint_best.pt"
            elif lower.startswith("ep"):
                name = f"checkpoint_ep{int(lower[2:]):03d}.pt"
            else:
                name = f"checkpoint_ep{int(lower):03d}.pt"
            path = seed_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing requested checkpoint: {path}")
            paths.append(path)
    unique = {path.resolve(): path.resolve() for path in paths}
    return sorted(unique.values(), key=checkpoint_sort_key)


def parse_array(value: Any, field: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value.astype(float, copy=False)
    elif isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=float)
    else:
        try:
            result = np.asarray(ast.literal_eval(str(value)), dtype=float)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse {field}: {value!r}") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional sequence")
    return result


def signed(value: float) -> int:
    return 1 if float(value) > 0 else -1


def load_trials(
    cfg: dict[str, Any],
    split: str = "val",
    max_csvs: int | None = None,
    max_trials: int | None = None,
) -> tuple[list[TrialRecord], list[Path]]:
    variant_dir = (
        Path(cfg["variant_root"]).expanduser().resolve() / str(cfg["variant_subdir"])
    )
    paths = sorted(variant_dir.glob(f"{split}Config_*.csv"))
    if max_csvs is not None:
        paths = paths[:max_csvs]
    if not paths:
        raise FileNotFoundError(f"No {split}Config_*.csv files in {variant_dir}")

    records: list[TrialRecord] = []
    global_trial = 0
    for path in paths:
        frame = pd.read_csv(path)
        for csv_trial, row in frame.reset_index(drop=True).iterrows():
            evidence = parse_array(row["evidence"], "evidence")
            states = parse_array(row["states"], "states")
            if evidence.shape != states.shape:
                raise ValueError(f"{path.name} row {csv_trial}: evidence/states mismatch")
            records.append(
                TrialRecord(
                    source_csv=path.name,
                    csv_trial=int(csv_trial),
                    global_trial=global_trial,
                    sigma=float(row.get("sigma", 1.0)),
                    true_hazard=float(row["trueHazard"]),
                    true_report=signed(float(row["trueReport"])),
                    true_predict=signed(float(row["truePredict"])),
                    evidence=evidence,
                    states=states,
                )
            )
            global_trial += 1
            if max_trials is not None and len(records) >= max_trials:
                return records, paths
    return records, paths


def load_hp(seed_dir: Path) -> dict[str, Any]:
    hp = load_json(seed_dir / "hp.json")
    hp.setdefault("n_input", 2)
    hp.setdefault("n_rnn", 128)
    hp.setdefault("n_null_timesteps", 4)
    return hp


def encode_evidence(evidence: Iterable[float], n_null_timesteps: int) -> torch.Tensor:
    evidence_values = [float(value) for value in evidence]
    if not evidence_values:
        raise ValueError("Evidence sequence cannot be empty")
    rows: list[list[float]] = []
    for index, value in enumerate(evidence_values):
        rows.append([value, 1.0])
        if index < len(evidence_values) - 1:
            rows.extend([[0.0, 0.0]] * n_null_timesteps)
    return torch.tensor(rows, dtype=torch.float32)


def build_input_tensor(
    trials: list[TrialRecord], n_null_timesteps: int, mirrored: bool = False
) -> torch.Tensor:
    lengths = {len(trial.evidence) for trial in trials}
    if len(lengths) != 1:
        raise ValueError(f"Trials have unequal evidence lengths: {sorted(lengths)}")
    sign = -1.0 if mirrored else 1.0
    return torch.stack(
        [encode_evidence(sign * trial.evidence, n_null_timesteps) for trial in trials]
    )


def evidence_timestep_indices(n_evidence: int, n_null_timesteps: int) -> np.ndarray:
    return np.arange(n_evidence, dtype=int) * (n_null_timesteps + 1)


def find_model_code_root(model_root: Path) -> Path:
    for path in (model_root, *model_root.parents):
        if (path / "rnn_models.py").exists():
            return path
    raise FileNotFoundError(f"Could not find rnn_models.py at or above {model_root}")


def import_model_class(model_root: Path, class_name: str = "GRUModel"):
    module_path = find_model_code_root(model_root) / "rnn_models.py"
    module_name = f"manifold_rnn_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} has no class {class_name}") from exc


def load_model(
    spec: ModelSpec,
    checkpoint: Path,
    device: torch.device,
    class_name: str = "GRUModel",
):
    hp = load_hp(spec.seed_dir)
    model_cls = import_model_class(spec.model_root, class_name)
    model = model_cls(hp).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, hp


@torch.inference_mode()
def collect_evidence_states(
    model,
    inputs: torch.Tensor,
    evidence_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_batches: list[np.ndarray] = []
    report_batches: list[np.ndarray] = []
    hazard_batches: list[np.ndarray] = []
    index_tensor = torch.as_tensor(evidence_indices, dtype=torch.long, device=device)
    for start in range(0, len(inputs), batch_size):
        x = inputs[start : start + batch_size].to(device)
        hidden = model.rnn(x).index_select(1, index_tensor)
        hidden_batches.append(hidden.cpu().numpy())
        report_batches.append(model.loc_head(hidden).squeeze(-1).cpu().numpy())
        hazard_batches.append(model.haz_head(hidden).squeeze(-1).cpu().numpy())
    return (
        np.concatenate(hidden_batches, axis=0),
        np.concatenate(report_batches, axis=0),
        np.concatenate(hazard_batches, axis=0),
    )


def import_bayesian_observer(path: Path) -> Callable[..., tuple[Any, ...]]:
    module_path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("manifold_normative_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = getattr(module, "BayesianObserver", None)
    if observer is None:
        raise AttributeError(f"{module_path} has no BayesianObserver")
    return observer


def entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def compute_normative_features(
    trials: list[TrialRecord], normative_model: Path, hazard_step: float = 0.05
) -> dict[str, np.ndarray]:
    observer = import_bayesian_observer(normative_model)
    hs = np.arange(0.0, 1.0, hazard_step, dtype=float)
    n_trials = len(trials)
    n_evidence = len(trials[0].evidence)
    result = {
        "bayes_state_belief": np.empty((n_trials, n_evidence), dtype=np.float32),
        "bayes_hazard_mean": np.empty((n_trials, n_evidence), dtype=np.float32),
        "bayes_hazard_sd": np.empty((n_trials, n_evidence), dtype=np.float32),
        "bayes_state_entropy": np.empty((n_trials, n_evidence), dtype=np.float32),
        "last_evidence": np.empty((n_trials, n_evidence), dtype=np.float32),
        "cumulative_evidence": np.empty((n_trials, n_evidence), dtype=np.float32),
        "bayes_report": np.empty(n_trials, dtype=np.int8),
        "bayes_predict": np.empty(n_trials, dtype=np.int8),
    }
    for trial_index, trial in enumerate(trials):
        l_haz, l_state, report, predict = observer(
            trial.evidence.tolist(),
            mu1=-1.0,
            mu2=1.0,
            sigma=float(trial.sigma),
            hs=hs,
            bias=0,
        )
        hazard_posterior = np.asarray(l_haz, dtype=float)[:, 1:].T
        state_posterior = np.asarray(l_state, dtype=float)[:, 1:].T
        hazard_mean = hazard_posterior @ hs
        hazard_var = np.sum(
            hazard_posterior * (hs[None, :] - hazard_mean[:, None]) ** 2, axis=1
        )
        result["bayes_state_belief"][trial_index] = (
            state_posterior[:, 1] - state_posterior[:, 0]
        )
        result["bayes_hazard_mean"][trial_index] = hazard_mean
        result["bayes_hazard_sd"][trial_index] = np.sqrt(np.maximum(hazard_var, 0.0))
        result["bayes_state_entropy"][trial_index] = entropy_rows(state_posterior)
        result["last_evidence"][trial_index] = trial.evidence
        result["cumulative_evidence"][trial_index] = np.cumsum(trial.evidence)
        result["bayes_report"][trial_index] = signed(report)
        result["bayes_predict"][trial_index] = signed(predict)
    return result


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def covariance_eigendecomposition(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(states, axis=0, dtype=np.float64)
    centered = states.astype(np.float64, copy=False) - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    return mean, eigenvalues, eigenvectors


def participation_ratio(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    denominator = float(np.sum(values * values))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(values) ** 2 / denominator)


def variance_dimension(eigenvalues: np.ndarray, threshold: float) -> int:
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(eigenvalues) / total, threshold) + 1)


def twonn_dimension(states: np.ndarray, max_points: int, seed: int) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    if len(states) > max_points:
        indices = rng.choice(len(states), size=max_points, replace=False)
        sample = states[indices]
    else:
        sample = states
    neighbors = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(sample)
    distances, _ = neighbors.kneighbors(sample)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = (r1 > 1e-12) & (r2 > r1)
    ratios = np.sort(r2[valid] / r1[valid])
    if len(ratios) < 20:
        return float("nan"), int(len(ratios))
    # TWO-NN uses -log(1-F(mu)) = d log(mu). Trim the noisiest tail.
    empirical_f = (np.arange(len(ratios), dtype=float) + 0.5) / len(ratios)
    keep = empirical_f <= 0.9
    x = np.log(ratios[keep])
    y = -np.log1p(-empirical_f[keep])
    denominator = float(x @ x)
    estimate = float((x @ y) / denominator) if denominator > 0 else float("nan")
    return estimate, int(np.sum(keep))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x.astype(np.float64, copy=False) - np.mean(x, axis=0)
    y_centered = y.astype(np.float64, copy=False) - np.mean(y, axis=0)
    cross = x_centered.T @ y_centered
    xx = x_centered.T @ x_centered
    yy = y_centered.T @ y_centered
    denominator = np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro")
    return float(np.linalg.norm(cross, "fro") ** 2 / denominator) if denominator else float("nan")


def write_run_config(path: Path, payload: dict[str, Any]) -> None:
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, torch.device):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

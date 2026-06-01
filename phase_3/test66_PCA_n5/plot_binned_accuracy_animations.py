#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "accuracy_by_checkpoint_config.json"
DEFAULT_VARIANT_SPLIT = "test"
DEFAULT_MAX_VARIANT_CSVS = None
DEFAULT_MODEL_CLASS = "GRUModel"

CHECKPOINT_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")
SEED_RE = re.compile(r"seed_(\d+)$")
SIGMA_RE = re.compile(r"sigma_\d+(?:p\d+)?$")

VALID_LOSS_TYPES = ("bce",)
VALID_TRAIN_HEADS = ("rep", "haz", "both")

BIN_WIDTH = 0.05
BIN_EDGES = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
GIF_SLOWDOWN_FACTOR = 1.5


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


class HelicopterBinnedEvalDataset(Dataset):
    def __init__(self, csv_paths: list[Path], n_input: int, n_null_timesteps: int):
        xs = []
        report_targets = []
        predict_targets = []
        hazards = []

        for csv_path in csv_paths:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    evidence = ast.literal_eval(str(row["evidence"]))
                    xs.append(encode_evidence_sequence(evidence, n_input, n_null_timesteps))
                    report_targets.append((float(row["trueReport"]) + 1.0) * 0.5)
                    predict_targets.append((float(row["truePredict"]) + 1.0) * 0.5)
                    hazards.append(float(row["trueHazard"]))

        if not xs:
            raise ValueError("No evaluation trials were loaded.")

        self.x = xs
        self.y_report = torch.tensor(report_targets, dtype=torch.float32).unsqueeze(1)
        self.y_predict = torch.tensor(predict_targets, dtype=torch.float32).unsqueeze(1)
        self.hazards = torch.tensor(hazards, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_report[idx], self.y_predict[idx], self.hazards[idx]


def collate_batch(batch):
    xs, y_report, y_predict, hazards = zip(*batch)
    return torch.stack(xs, 0), torch.stack(y_report, 0), torch.stack(y_predict, 0), torch.stack(hazards, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create report/hazard binned-accuracy animations plus a hazard skew plot."
    )
    parser.add_argument("--loss-types", nargs="+", default=["bce"], help="bce or all")
    parser.add_argument("--train-heads", nargs="+", default=["both"], help="rep, haz, both, or all")
    parser.add_argument("--sigmas", nargs="+", default=["sigma_1"], help="sigma_1 sigma_2 sigma_3, or all")
    parser.add_argument("--heads", nargs="+", default=["report", "hazard"], choices=["report", "hazard"])
    parser.add_argument("--seeds", nargs="+", default=["all"], help="Seed ids, e.g. 0 1 2, or all")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs" / "binned_accuracy_animations")
    parser.add_argument("--include-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-final", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-best", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--skew-split", type=float, default=0.5)
    parser.add_argument("--no-skew-plot", action="store_true")
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


def expand_all(values: list[str], valid: tuple[str, ...]) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(valid)
    return values


def import_model_class(model_root: Path, class_name: str):
    module_path = model_root / "rnn_models.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find {module_path}")

    spec = importlib.util.spec_from_file_location("n5_rnn_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} does not define {class_name}") from exc


def run_key(loss_type: str, train_heads: str) -> str:
    return f"{loss_type}_{train_heads}"


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def list_sigmas(run_root: Path, requested_sigmas: list[str]) -> list[str]:
    if len(requested_sigmas) == 1 and requested_sigmas[0].lower() == "all":
        sigmas = [p.name for p in run_root.iterdir() if p.is_dir() and SIGMA_RE.fullmatch(p.name)]
        return sorted(sigmas, key=natural_key)
    return requested_sigmas


def list_seed_dirs(group_root: Path, requested_seeds: list[str]) -> list[Path]:
    seed_dirs = [p for p in group_root.iterdir() if p.is_dir() and SEED_RE.fullmatch(p.name)]
    seed_dirs.sort(key=lambda p: int(SEED_RE.fullmatch(p.name).group(1)))

    if len(requested_seeds) != 1 or requested_seeds[0].lower() != "all":
        wanted = {int(s) for s in requested_seeds}
        seed_dirs = [p for p in seed_dirs if int(SEED_RE.fullmatch(p.name).group(1)) in wanted]

    if not seed_dirs:
        raise FileNotFoundError(f"No matching seed_* directories found in {group_root}")
    return seed_dirs


def list_eval_csvs(variant_dir: Path, variant_split: str, max_variant_csvs: int | None) -> list[Path]:
    pattern = f"{variant_split}Config_*.csv"
    csvs = sorted(variant_dir.glob(pattern), key=natural_key)
    if max_variant_csvs is not None:
        csvs = csvs[:max_variant_csvs]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for {variant_dir / pattern}")
    return csvs


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    if name == "checkpoint_init.pt":
        return (0, 0, name)
    match = CHECKPOINT_RE.fullmatch(name)
    if match:
        return (1, int(match.group(1)), name)
    if name == "final.pt":
        return (2, 0, name)
    if name == "checkpoint_best.pt":
        return (3, 0, name)
    return (4, 0, name)


def checkpoint_label(path: Path) -> str:
    name = path.name
    if name == "checkpoint_init.pt":
        return "init"
    if name == "final.pt":
        return "final"
    if name == "checkpoint_best.pt":
        return "best"
    match = CHECKPOINT_RE.fullmatch(name)
    if match:
        return f"ep{int(match.group(1)):03d}"
    return path.stem


def list_checkpoints(seed_dir: Path, args: argparse.Namespace) -> list[Path]:
    ckpts = []
    if args.include_init:
        ckpts.append(seed_dir / "checkpoint_init.pt")
    ckpts.extend(seed_dir.glob("checkpoint_ep*.pt"))
    if args.include_final:
        ckpts.append(seed_dir / "final.pt")
    if args.include_best:
        ckpts.append(seed_dir / "checkpoint_best.pt")

    existing = sorted({p for p in ckpts if p.exists()}, key=checkpoint_sort_key)
    if not existing:
        raise FileNotFoundError(f"No checkpoints found in {seed_dir}")
    return existing


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


@torch.inference_mode()
def evaluate_checkpoint(
    model_cls,
    checkpoint_path: Path,
    hp: dict[str, Any],
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model_cls(hp).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    hazard_values = []
    report_correct = []
    predict_correct = []

    for x, y_report, y_predict, hazards in dataloader:
        x = x.to(device)
        y_report = y_report.to(device)
        y_predict = y_predict.to(device)

        loc_logits, predict_logits = model(x)
        report_pred = (torch.sigmoid(loc_logits[:, -1, :]) > 0.5).float()
        predict_pred = (torch.sigmoid(predict_logits) > 0.5).float()

        hazard_values.extend(hazards.cpu().numpy().tolist())
        report_correct.extend((report_pred == y_report).squeeze(1).cpu().numpy().tolist())
        predict_correct.extend((predict_pred == y_predict).squeeze(1).cpu().numpy().tolist())

    return (
        np.array(hazard_values, dtype=float),
        np.array(report_correct, dtype=bool),
        np.array(predict_correct, dtype=bool),
    )


def bin_accuracy(hazards: np.ndarray, correct_mask: np.ndarray) -> np.ndarray:
    idx = np.digitize(hazards, BIN_EDGES) - 1
    idx = np.clip(idx, 0, len(BIN_CENTERS) - 1)

    total = np.zeros(len(BIN_CENTERS), dtype=int)
    good = np.zeros(len(BIN_CENTERS), dtype=int)

    for bin_idx, is_correct in zip(idx, correct_mask):
        total[bin_idx] += 1
        if is_correct:
            good[bin_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, good / total, np.nan)


def make_seed_palette(seed_ids: list[int]) -> dict[int, tuple]:
    cmap_name = "tab20" if len(seed_ids) <= 20 else "hsv"
    cmap = plt.get_cmap(cmap_name, len(seed_ids))
    return {seed: cmap(i) for i, seed in enumerate(seed_ids)}


def build_binned_frames(
    model_cls,
    seed_dirs: list[Path],
    csvs: list[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[str], dict[str, list[tuple[int, np.ndarray]]], dict[str, list[tuple[int, np.ndarray]]]]:
    report_by_checkpoint: dict[str, list[tuple[int, np.ndarray]]] = {}
    hazard_by_checkpoint: dict[str, list[tuple[int, np.ndarray]]] = {}
    dataloaders: dict[tuple[int, int], DataLoader] = {}

    for seed_dir in seed_dirs:
        seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
        hp = load_hp(seed_dir)
        dataset_key = (int(hp["n_input"]), int(hp["n_null_timesteps"]))
        if dataset_key not in dataloaders:
            dataset = HelicopterBinnedEvalDataset(csvs, *dataset_key)
            batch_size = int(hp.get("batch_size", 256))
            dataloaders[dataset_key] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_batch,
            )
            print(
                f"Prepared {len(dataset)} trials with "
                f"n_input={dataset_key[0]}, n_null_timesteps={dataset_key[1]}, "
                f"batch_size={batch_size} from {seed_dir.name}/hp.json"
            )

        for ckpt in list_checkpoints(seed_dir, args):
            hazards, report_ok, predict_ok = evaluate_checkpoint(
                model_cls,
                ckpt,
                hp,
                dataloaders[dataset_key],
                device,
            )
            label = checkpoint_label(ckpt)
            report_by_checkpoint.setdefault(label, []).append((seed, bin_accuracy(hazards, report_ok)))
            hazard_by_checkpoint.setdefault(label, []).append((seed, bin_accuracy(hazards, predict_ok)))
            print(f"{seed_dir.name} {label}: evaluated {ckpt.name}")

    all_checkpoint_paths = list_checkpoints(seed_dirs[0], args)
    labels = [checkpoint_label(path) for path in all_checkpoint_paths]
    labels = [label for label in labels if label in report_by_checkpoint or label in hazard_by_checkpoint]
    return labels, report_by_checkpoint, hazard_by_checkpoint


def plot_hazard_skew(
    checkpoint_labels: list[str],
    hazard_by_checkpoint: dict[str, list[tuple[int, np.ndarray]]],
    seed_ids: list[int],
    seed_color: dict[int, tuple],
    title_prefix: str,
    out_path: Path,
    split: float,
    dpi: int,
) -> None:
    high_mask = BIN_CENTERS >= split
    low_mask = BIN_CENTERS < split

    skew = np.full((len(checkpoint_labels), len(seed_ids)), np.nan)
    seed_index = {seed: i for i, seed in enumerate(seed_ids)}

    for t, label in enumerate(checkpoint_labels):
        for seed, acc in hazard_by_checkpoint.get(label, []):
            hi = np.nanmean(acc[high_mask])
            lo = np.nanmean(acc[low_mask])
            skew[t, seed_index[seed]] = hi - lo

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    xs = np.arange(len(checkpoint_labels))

    for j, seed in enumerate(seed_ids):
        ax.plot(
            xs,
            skew[:, j],
            label=f"seed {seed}",
            alpha=0.85,
            linewidth=1.6,
            color=seed_color[seed],
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Skew = acc(high) - acc(low)")
    ax.set_xlabel("Checkpoint")
    ax.set_xticks(xs)
    ax.set_xticklabels(checkpoint_labels)
    ax.set_title(f"{title_prefix} | hazard-head skew over time")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def animate_binned_accuracy(
    checkpoint_labels: list[str],
    acc_by_checkpoint: dict[str, list[tuple[int, np.ndarray]]],
    seed_ids: list[int],
    seed_color: dict[int, tuple],
    head: str,
    title_prefix: str,
    out_path: Path,
    fps: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlabel("True hazard (bin centers, width = 0.05)")
    ax.set_ylabel(("Report" if head == "report" else "Hazard") + " accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(True, alpha=0.3)

    title = ax.set_title("")
    lines = {}
    legend_handles = []
    for seed in seed_ids:
        (line,) = ax.plot([], [], linewidth=1.6, alpha=0.9, color=seed_color[seed])
        lines[seed] = line
        legend_handles.append(line)

    ax.legend(
        legend_handles,
        [f"seed {seed}" for seed in seed_ids],
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=9,
    )

    def init():
        for seed in seed_ids:
            lines[seed].set_data([], [])
        title.set_text("")
        return list(lines.values()) + [title]

    def update(frame_idx: int):
        label = checkpoint_labels[frame_idx]
        seed_acc_map = {seed: acc for seed, acc in acc_by_checkpoint.get(label, [])}

        for seed in seed_ids:
            if seed in seed_acc_map:
                lines[seed].set_data(BIN_CENTERS, seed_acc_map[seed])
            else:
                lines[seed].set_data([], [])

        title.set_text(f"{title_prefix} | {head} head | {label}")
        return list(lines.values()) + [title]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(checkpoint_labels),
        interval=max(1, int(round((1000 / fps) * GIF_SLOWDOWN_FACTOR))),
        blit=True,
    )
    anim.save(out_path, writer="pillow", dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def process_group(
    model_cls,
    model_root: Path,
    variant_root: Path,
    cfg: dict[str, Any],
    loss_type: str,
    train_heads: str,
    sigma: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    key = run_key(loss_type, train_heads)
    group_root = model_root / key / sigma
    variant_dir = variant_root / sigma
    if not group_root.exists():
        print(f"[skip] missing model directory: {group_root}")
        return
    if not variant_dir.exists():
        print(f"[skip] missing variant directory: {variant_dir}")
        return

    csvs = list_eval_csvs(variant_dir, DEFAULT_VARIANT_SPLIT, DEFAULT_MAX_VARIANT_CSVS)
    seed_dirs = list_seed_dirs(group_root, args.seeds)
    seed_ids = [int(SEED_RE.fullmatch(path.name).group(1)) for path in seed_dirs]

    print(f"Running {key}/{sigma} with {len(seed_dirs)} seeds and {len(csvs)} CSVs on {device}")
    checkpoint_labels, report_by_checkpoint, hazard_by_checkpoint = build_binned_frames(
        model_cls,
        seed_dirs,
        csvs,
        args,
        device,
    )

    out_dir = args.output_dir.expanduser().resolve() / key / sigma
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_color = make_seed_palette(seed_ids)
    title_prefix = f"{key} | {sigma}"

    if "report" in args.heads:
        animate_binned_accuracy(
            checkpoint_labels,
            report_by_checkpoint,
            seed_ids,
            seed_color,
            "report",
            title_prefix,
            out_dir / "report_binned_accuracy_anim.gif",
            args.fps,
            args.dpi,
        )

    if "hazard" in args.heads:
        if not args.no_skew_plot:
            plot_hazard_skew(
                checkpoint_labels,
                hazard_by_checkpoint,
                seed_ids,
                seed_color,
                title_prefix,
                out_dir / "hazard_skew_over_time.png",
                args.skew_split,
                args.dpi,
            )
        animate_binned_accuracy(
            checkpoint_labels,
            hazard_by_checkpoint,
            seed_ids,
            seed_color,
            "hazard",
            title_prefix,
            out_dir / "hazard_binned_accuracy_anim.gif",
            args.fps,
            args.dpi,
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(DEFAULT_CONFIG)
    model_cls = import_model_class(cfg["model_root"], DEFAULT_MODEL_CLASS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_types = expand_all(args.loss_types, VALID_LOSS_TYPES)
    train_heads_list = expand_all(args.train_heads, VALID_TRAIN_HEADS)

    for loss_type in loss_types:
        for train_heads in train_heads_list:
            current_run_key = run_key(loss_type, train_heads)
            run_root = cfg["model_root"] / current_run_key
            if not run_root.exists():
                print(f"[skip] missing run directory: {run_root}")
                continue

            for sigma in list_sigmas(run_root, args.sigmas):
                process_group(
                    model_cls,
                    cfg["model_root"],
                    cfg["variant_root"],
                    cfg,
                    loss_type,
                    train_heads,
                    sigma,
                    args,
                    device,
                )

    print("All requested binned accuracy outputs complete.")


if __name__ == "__main__":
    main()

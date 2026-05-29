#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "accuracy_by_checkpoint_config.json"
CHECKPOINT_RE = re.compile(r"checkpoint_ep(\d+)\.pt$")
SEED_RE = re.compile(r"seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train linear decoders from checkpoint hidden states using the same "
            "model and variant configuration as plot_accuracy_by_checkpoint.py."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to config JSON. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--model-subdir",
        default="models_OTS/bce_both/sigma_1",
        help="Experiment folder under model_root. Default: models_OTS/bce_both/sigma_1",
    )
    parser.add_argument(
        "--variant-subdir",
        default=None,
        help="Variant folder under variant_root. Defaults to the model-subdir leaf, e.g. sigma_1.",
    )
    parser.add_argument(
        "--model-class",
        default="GRUModel",
        help="Model class in model_root/rnn_models.py. Default: GRUModel",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        choices=["train", "val", "test"],
        help="Variant split used to fit the decoder. Default: train",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        choices=["train", "val", "test"],
        help="Variant split used to validate the decoder. Default: val",
    )
    parser.add_argument(
        "--max-train-csvs",
        type=int,
        default=20,
        help="Optional cap on decoder training CSVs. Default: 20",
    )
    parser.add_argument(
        "--max-val-csvs",
        type=int,
        default=5,
        help="Optional cap on decoder validation CSVs. Default: 5",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Hidden-state extraction batch size. Default: 128",
    )
    parser.add_argument(
        "--decoder-epochs",
        type=int,
        default=300,
        help="Linear decoder training epochs per checkpoint. Default: 300",
    )
    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=1e-2,
        help="Linear decoder learning rate. Default: 1e-2",
    )
    parser.add_argument(
        "--decoder-weight-decay",
        type=float,
        default=1e-4,
        help="Linear decoder Adam weight decay. Default: 1e-4",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Decoder progress interval in epochs. Default: 50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "decoder_results",
        help="Directory for decoder CSV and plots. Default: ./decoder_results",
    )
    parser.add_argument(
        "--skip-best",
        action="store_true",
        help="Do not include checkpoint_best.pt.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Do not include final.pt.",
    )
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
            "model_class": args.model_class,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "max_train_csvs": args.max_train_csvs,
            "max_val_csvs": args.max_val_csvs,
            "batch_size": args.batch_size,
            "decoder_epochs": args.decoder_epochs,
            "decoder_lr": args.decoder_lr,
            "decoder_weight_decay": args.decoder_weight_decay,
            "print_every": args.print_every,
            "include_checkpoint_best": not args.skip_best,
            "include_final": not args.skip_final,
            "output_dir": args.output_dir.expanduser().resolve(),
        }
    )
    cfg["model_dir"] = cfg["model_root"] / cfg["model_subdir"]
    cfg["variant_dir"] = cfg["variant_root"] / cfg["variant_subdir"]
    return cfg


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


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


def list_variant_csvs(
    variant_dir: Path,
    split: str,
    max_csvs: int | None,
) -> list[Path]:
    pattern = f"{split}Config_*.csv"
    csvs = sorted(variant_dir.glob(pattern), key=natural_key)
    if max_csvs is not None:
        csvs = csvs[:max_csvs]
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for {variant_dir / pattern}")
    return csvs


def import_model_class(model_root: Path, class_name: str):
    module_path = model_root / "rnn_models.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find {module_path}")

    spec = importlib.util.spec_from_file_location("decoder_rnn_models", module_path)
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
    if name == "checkpoint_best.pt":
        return "best"
    if name == "final.pt":
        return "final"
    match = CHECKPOINT_RE.fullmatch(name)
    if match:
        return f"ep{int(match.group(1)):03d}"
    return path.stem


def checkpoint_epoch(path: Path) -> int | None:
    if path.name == "checkpoint_init.pt":
        return 0
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match:
        return int(match.group(1))
    return None


def list_checkpoints(seed_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    ckpts = [seed_dir / "checkpoint_init.pt"]
    ckpts.extend(seed_dir.glob("checkpoint_ep*.pt"))
    if cfg["include_final"]:
        ckpts.append(seed_dir / "final.pt")
    if cfg["include_checkpoint_best"]:
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


class TrialDataset(Dataset):
    def __init__(
        self,
        csv_paths: list[Path],
        n_input: int,
        n_null_timesteps: int,
    ):
        xs = []
        y_predict = []

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                evidence = parse_list(row["evidence"])
                xs.append(encode_evidence_sequence(evidence, n_input, n_null_timesteps))
                y_predict.append(float((row["truePredict"] + 1) * 0.5))

        if not xs:
            raise ValueError("No trials were loaded.")

        self.xs = xs
        self.y_predict = torch.tensor(y_predict, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int):
        return self.xs[idx], self.y_predict[idx]


def collate_batch(batch):
    xs, y_predict = zip(*batch)
    return torch.stack(xs, 0), torch.stack(y_predict, 0)


def find_first_gru(model: nn.Module) -> nn.GRU:
    for _, module in model.named_modules():
        if isinstance(module, nn.GRU):
            return module
    raise RuntimeError("Could not find an nn.GRU module inside the model")


@torch.inference_mode()
def collect_final_hidden_and_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    gru = find_first_gru(model)
    cache: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, outputs):
        cache["gru_out"] = outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach()

    handle = gru.register_forward_hook(hook)
    x_all = []
    y_all = []

    try:
        for x, y_predict in dataloader:
            x = x.to(device)

            cache.clear()
            _ = model(x)
            if "gru_out" not in cache:
                raise RuntimeError("GRU forward hook did not capture activity")

            h_seq = cache["gru_out"]
            if h_seq.ndim != 3:
                raise RuntimeError(f"Expected GRU activity [B, T, H], got {tuple(h_seq.shape)}")

            x_all.append(h_seq[:, -1, :].cpu().numpy())
            y_all.append(y_predict.numpy())
    finally:
        handle.remove()

    if not x_all:
        raise RuntimeError("No hidden states were collected")

    x_np = np.concatenate(x_all, axis=0)
    y_np = np.concatenate(y_all, axis=0).reshape(-1)
    return x_np, y_np


class LinearDecoder(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


@dataclass
class DecoderResult:
    model: str
    seed: int
    checkpoint: str
    checkpoint_file: str
    checkpoint_order: int
    epoch: int | None
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float
    n_train_examples: int
    n_val_examples: int


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    return float((pred == y).float().mean().item())


def fit_linear_decoder(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    x_train_raw = torch.tensor(x_train_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
    x_val_raw = torch.tensor(x_val_np, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)

    mean = x_train_raw.mean(dim=0, keepdim=True)
    std = x_train_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std

    decoder = LinearDecoder(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(
        decoder.parameters(),
        lr=float(cfg["decoder_lr"]),
        weight_decay=float(cfg["decoder_weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(int(cfg["decoder_epochs"])):
        decoder.train()
        opt.zero_grad()
        train_logits = decoder(x_train)
        train_loss = criterion(train_logits, y_train)
        train_loss.backward()
        opt.step()

        decoder.eval()
        with torch.no_grad():
            val_logits = decoder(x_val)
            val_loss = criterion(val_logits, y_val)

        if float(val_loss.item()) < best_val_loss:
            best_val_loss = float(val_loss.item())
            best_state = copy.deepcopy(decoder.state_dict())

        print_every = int(cfg["print_every"])
        if print_every > 0 and (epoch % print_every == 0 or epoch == int(cfg["decoder_epochs"]) - 1):
            with torch.no_grad():
                train_acc = accuracy_from_logits(train_logits, y_train)
                val_acc = accuracy_from_logits(val_logits, y_val)
            print(
                f"decoder ep {epoch:03d}/{cfg['decoder_epochs']} "
                f"train_loss {float(train_loss.item()):.4f} "
                f"train_acc {train_acc:.4f} "
                f"val_loss {float(val_loss.item()):.4f} "
                f"val_acc {val_acc:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Decoder training did not produce a best state")

    decoder.load_state_dict(best_state)
    decoder.eval()
    with torch.no_grad():
        train_logits = decoder(x_train)
        val_logits = decoder(x_val)
        return {
            "train_acc": accuracy_from_logits(train_logits, y_train),
            "val_acc": accuracy_from_logits(val_logits, y_val),
            "train_loss": float(criterion(train_logits, y_train).item()),
            "val_loss": float(criterion(val_logits, y_val).item()),
        }


def load_model(model_cls, seed_dir: Path, checkpoint_path: Path, hp: dict[str, Any], device: torch.device) -> nn.Module:
    model = model_cls(hp).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_dataloaders(
    cfg: dict[str, Any],
    hp: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    train_csvs = list_variant_csvs(
        cfg["variant_dir"],
        cfg["train_split"],
        cfg["max_train_csvs"],
    )
    val_csvs = list_variant_csvs(
        cfg["variant_dir"],
        cfg["val_split"],
        cfg["max_val_csvs"],
    )
    n_input = int(hp["n_input"])
    n_null_timesteps = int(hp["n_null_timesteps"])

    train_dataset = TrialDataset(train_csvs, n_input, n_null_timesteps)
    val_dataset = TrialDataset(val_csvs, n_input, n_null_timesteps)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )
    print(
        f"Prepared decoder data: train={len(train_dataset)} val={len(val_dataset)} "
        f"n_input={n_input} n_null_timesteps={n_null_timesteps}"
    )
    return train_loader, val_loader


def collect_results(cfg: dict[str, Any]) -> pd.DataFrame:
    if not cfg["model_dir"].exists():
        raise FileNotFoundError(f"Model directory does not exist: {cfg['model_dir']}")
    if not cfg["variant_dir"].exists():
        raise FileNotFoundError(f"Variant directory does not exist: {cfg['variant_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = import_model_class(cfg["model_root"], cfg["model_class"])
    seed_dirs = list_seed_dirs(cfg["model_dir"])

    print(f"Model directory: {cfg['model_dir']}")
    print(f"Variant directory: {cfg['variant_dir']}")
    print(f"Using device: {device}")

    rows: list[DecoderResult] = []
    dataloaders: dict[tuple[int, int], tuple[DataLoader, DataLoader]] = {}

    for seed_dir in seed_dirs:
        seed = int(SEED_RE.fullmatch(seed_dir.name).group(1))
        hp = load_hp(seed_dir)
        dataset_key = (int(hp["n_input"]), int(hp["n_null_timesteps"]))
        if dataset_key not in dataloaders:
            dataloaders[dataset_key] = build_dataloaders(cfg, hp)
        train_loader, val_loader = dataloaders[dataset_key]

        for order, checkpoint_path in enumerate(list_checkpoints(seed_dir, cfg)):
            print()
            print(f"Training decoder for {seed_dir.name}/{checkpoint_path.name}")
            model = load_model(model_cls, seed_dir, checkpoint_path, hp, device)
            train_x, train_y = collect_final_hidden_and_labels(model, train_loader, device)
            val_x, val_y = collect_final_hidden_and_labels(model, val_loader, device)
            stats = fit_linear_decoder(train_x, train_y, val_x, val_y, cfg, device)

            rows.append(
                DecoderResult(
                    model=seed_dir.name,
                    seed=seed,
                    checkpoint=checkpoint_label(checkpoint_path),
                    checkpoint_file=checkpoint_path.name,
                    checkpoint_order=order,
                    epoch=checkpoint_epoch(checkpoint_path),
                    train_acc=stats["train_acc"],
                    val_acc=stats["val_acc"],
                    train_loss=stats["train_loss"],
                    val_loss=stats["val_loss"],
                    n_train_examples=int(train_y.shape[0]),
                    n_val_examples=int(val_y.shape[0]),
                )
            )
            print(
                f"{seed_dir.name} {checkpoint_label(checkpoint_path)}: "
                f"decoder_train={stats['train_acc']:.4f} "
                f"decoder_val={stats['val_acc']:.4f}"
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def plot_results(df: pd.DataFrame, cfg: dict[str, Any], out_path: Path) -> None:
    checkpoint_table = (
        df[["checkpoint", "checkpoint_order"]]
        .drop_duplicates()
        .sort_values("checkpoint_order")
        .reset_index(drop=True)
    )
    x_by_checkpoint = {
        row.checkpoint: i for i, row in enumerate(checkpoint_table.itertuples(index=False))
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (model_name, model_df) in enumerate(df.groupby("model", sort=True)):
        model_df = model_df.sort_values("checkpoint_order")
        x = [x_by_checkpoint[label] for label in model_df["checkpoint"]]
        ax.plot(
            x,
            model_df["val_acc"],
            color=cmap(i % 10),
            alpha=0.42,
            linewidth=1.2,
            marker="o",
            markersize=3.2,
            label=model_name,
        )

    avg = (
        df.groupby(["checkpoint", "checkpoint_order"], as_index=False)["val_acc"]
        .mean()
        .sort_values("checkpoint_order")
    )
    ax.plot(
        [x_by_checkpoint[label] for label in avg["checkpoint"]],
        avg["val_acc"],
        color="black",
        linewidth=3.4,
        marker="o",
        markersize=5,
        label="average",
        zorder=10,
    )

    ax.set_xticks(range(len(checkpoint_table)))
    ax.set_xticklabels(checkpoint_table["checkpoint"], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Decoder validation accuracy")
    ax.set_title(f"{cfg['model_subdir']} truePredict decoder by checkpoint")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = build_run_config(args)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    df = collect_results(cfg)
    if df.empty:
        raise RuntimeError("No decoder results were produced")

    df = df.sort_values(["seed", "checkpoint_order"]).reset_index(drop=True)
    csv_path = cfg["output_dir"] / "decoder_accuracy_by_checkpoint.csv"
    plot_path = cfg["output_dir"] / "decoder_val_accuracy_by_checkpoint.png"
    df.to_csv(csv_path, index=False)
    plot_results(df, cfg, plot_path)

    print()
    print(df[["model", "checkpoint", "train_acc", "val_acc"]].to_string(index=False))
    print(f"Saved decoder metrics to {csv_path}")
    print(f"Saved decoder plot to {plot_path}")


if __name__ == "__main__":
    main()

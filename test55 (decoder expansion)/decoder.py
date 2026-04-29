#!/usr/bin/env python3
"""
decode_stay_switch_from_activity.py
===================================
Train linear decoders to predict the trial-level stay/switch label
(truePredict) from the hidden activity of trained GRU models.

This version:
    - loads each saved GRU checkpoint except checkpoint_best.pt and final.pt
    - extracts only the final GRU hidden state h_T
    - trains one linear decoder per checkpoint to predict truePredict
    - sweeps all 10 seeds, all 3 head settings (rep, haz, both), and all 3 sigma groups
    - plots decoder validation accuracy vs checkpoint
    - saves 3 figures total: one for sigma_1, one for sigma_2, one for sigma_3

Each sigma figure contains 3 subplots:
    - rep
    - haz
    - both

Within each subplot:
    - each seed is a faint line
    - the across-seed mean is a bold line
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rnn_models import GRUModel


# ───────────────────────────── hardcoded settings ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(BASE_DIR, "variants")
MODELS_ROOT = os.path.join(BASE_DIR, "models_OTS")

GROUP_KEYS = ["sigma_1", "sigma_2", "sigma_3"]
TRAIN_HEADS_LIST = ["rep", "haz", "both"]
LOSS_TYPE = "bce"
SEEDS = range(10)

TRAIN_IDXS = range(20)
VAL_IDXS = range(5)

BATCH_SIZE = 128
DECODER_LR = 1e-2
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 300
PRINT_EVERY = 50

SAVE_RESULTS = True
RESULTS_DIR = os.path.join(BASE_DIR, "decoder_results")
RAW_RESULTS_CSV = "decoder_accuracy_by_checkpoint_all_runs.csv"

NORM_PRED_ACC = {
    "sigma_1": 0.8213,
    "sigma_2": 0.6293,
    "sigma_3": 0.5453,
}


# ───────────────────────────── helpers ─────────────────────────────
def parse_seq(value) -> list[float]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    raise TypeError(f"Unsupported sequence type: {type(value)}")


def model_dir_from_settings(group_key: str, loss_type: str, train_heads: str, seed: int) -> str:
    if loss_type == "reinforce" and train_heads == "both":
        root = MODELS_ROOT
    else:
        root = os.path.join(MODELS_ROOT, f"{loss_type}_{train_heads}")
    return os.path.join(root, group_key, f"seed_{seed}")


def load_hp(model_dir: str) -> dict:
    hp_path = os.path.join(model_dir, "hp.json")
    with open(hp_path, "r") as f:
        return json.load(f)


def load_model(model_dir: str, checkpoint_name: str) -> nn.Module:
    hp = load_hp(model_dir)
    model = GRUModel(hp).to(DEVICE)

    ckpt_path = os.path.join(model_dir, checkpoint_name)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def checkpoint_epoch_from_name(checkpoint_name: str) -> int:
    if checkpoint_name == "checkpoint_init.pt":
        return 0

    m = re.fullmatch(r"checkpoint_ep(\d+)\.pt", checkpoint_name)
    if m is None:
        raise ValueError(f"Unrecognized checkpoint name: {checkpoint_name}")
    return int(m.group(1))


def checkpoint_label_from_epoch(epoch: int) -> str:
    if epoch == 0:
        return "init"
    return f"ep{epoch:03d}"


def list_checkpoint_names(model_dir: str) -> list[str]:
    names: list[str] = []

    init_name = "checkpoint_init.pt"
    init_path = os.path.join(model_dir, init_name)
    if os.path.exists(init_path):
        names.append(init_name)

    ep_pat = re.compile(r"checkpoint_ep(\d+)\.pt$")
    ep_names = []
    for name in os.listdir(model_dir):
        m = ep_pat.fullmatch(name)
        if m is not None:
            ep_names.append((int(m.group(1)), name))
    ep_names.sort(key=lambda x: x[0])
    names.extend(name for _, name in ep_names)

    if not names:
        raise FileNotFoundError(
            f"No checkpoint_init.pt or checkpoint_ep*.pt files found in {model_dir}"
        )

    return names


def csv_paths(group_key: str, stem: str, idxs: Iterable[int]) -> list[str]:
    paths = []
    for k in idxs:
        path = os.path.join(VARIANTS_DIR, group_key, f"{stem}_{k}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        paths.append(path)
    return paths


class TrialDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        xs, y_pred = [], []
        for _, row in df.iterrows():
            evidence = parse_seq(row["evidence"])
            x = torch.tensor(evidence, dtype=torch.float32).unsqueeze(-1)
            xs.append(x)

            label01 = 1.0 if int(row["truePredict"]) > 0 else 0.0
            y_pred.append(torch.tensor([label01], dtype=torch.float32))

        self.xs = xs
        self.y_pred = y_pred

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int):
        return self.xs[idx], self.y_pred[idx]


def find_first_gru(model: nn.Module) -> nn.GRU:
    for _, module in model.named_modules():
        if isinstance(module, nn.GRU):
            return module
    raise RuntimeError("Could not find an nn.GRU module inside GRUModel")


@torch.no_grad()
def collect_final_hidden_and_labels(model: nn.Module, df: pd.DataFrame, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = TrialDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    gru = find_first_gru(model)
    cache: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, outputs):
        if isinstance(outputs, tuple):
            cache["gru_out"] = outputs[0].detach()
        else:
            cache["gru_out"] = outputs.detach()

    handle = gru.register_forward_hook(hook)

    x_all = []
    y_all = []

    for x, y_pred in loader:
        x = x.to(DEVICE)
        y_pred = y_pred.to(DEVICE)

        cache.clear()
        _ = model(x)
        if "gru_out" not in cache:
            raise RuntimeError("GRU forward hook did not capture activity")

        h_seq = cache["gru_out"]
        if h_seq.ndim != 3:
            raise RuntimeError(f"Expected GRU activity of shape [B, T, H], got {tuple(h_seq.shape)}")

        x_all.append(h_seq[:, -1, :].cpu().numpy())
        y_all.append(y_pred.cpu().numpy())

    handle.remove()

    if not x_all:
        raise RuntimeError("No data collected")

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0).reshape(-1)
    return x, y


class LinearDecoder(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


@dataclass
class DecoderResult:
    group_key: str
    train_heads: str
    seed: int
    checkpoint_name: str
    checkpoint_epoch: int
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    return float((pred == y).float().mean().item())


def fit_linear_decoder(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_val_np: np.ndarray,
) -> tuple[LinearDecoder, dict]:
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)
    x_val = torch.tensor(x_val_np, dtype=torch.float32, device=DEVICE)
    y_val = torch.tensor(y_val_np, dtype=torch.float32, device=DEVICE)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    decoder = LinearDecoder(x_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(
        decoder.parameters(),
        lr=DECODER_LR,
        weight_decay=DECODER_WEIGHT_DECAY,
    )

    best_state = None
    best_val_loss = float("inf")
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(DECODER_EPOCHS):
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
            best_state = {
                "model": copy.deepcopy(decoder.state_dict()),
                "mean": mean.detach().cpu(),
                "std": std.detach().cpu(),
            }

        if epoch % PRINT_EVERY == 0 or epoch == DECODER_EPOCHS - 1:
            with torch.no_grad():
                train_acc = accuracy_from_logits(train_logits, y_train)
                val_acc = accuracy_from_logits(val_logits, y_val)
            print(
                f"decoder ep {epoch:03d}/{DECODER_EPOCHS} "
                f"train_loss {float(train_loss.item()):.4f} "
                f"train_acc {train_acc:.4f} "
                f"val_loss {float(val_loss.item()):.4f} "
                f"val_acc {val_acc:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Decoder training did not produce a best_state")

    decoder.load_state_dict(best_state["model"])
    decoder.eval()

    with torch.no_grad():
        mean = best_state["mean"].to(DEVICE)
        std = best_state["std"].to(DEVICE)
        x_train_best = (torch.tensor(x_train_np, dtype=torch.float32, device=DEVICE) - mean) / std
        x_val_best = (torch.tensor(x_val_np, dtype=torch.float32, device=DEVICE) - mean) / std
        y_train_t = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE)
        y_val_t = torch.tensor(y_val_np, dtype=torch.float32, device=DEVICE)

        train_logits = decoder(x_train_best)
        val_logits = decoder(x_val_best)
        train_loss = float(criterion(train_logits, y_train_t).item())
        val_loss = float(criterion(val_logits, y_val_t).item())
        train_acc = accuracy_from_logits(train_logits, y_train_t)
        val_acc = accuracy_from_logits(val_logits, y_val_t)

    stats = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    return decoder, stats


def load_group_data(group_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.concat(
        [pd.read_csv(p) for p in csv_paths(group_key, "trainConfig", TRAIN_IDXS)],
        ignore_index=True,
    )
    val_df = pd.concat(
        [pd.read_csv(p) for p in csv_paths(group_key, "valConfig", VAL_IDXS)],
        ignore_index=True,
    )
    return train_df, val_df


def run_one_setting(group_key: str, train_heads: str, seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[DecoderResult]:
    model_dir = model_dir_from_settings(group_key, LOSS_TYPE, train_heads, seed)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    checkpoint_names = list_checkpoint_names(model_dir)
    results: list[DecoderResult] = []

    print()
    print(f"group={group_key} heads={train_heads} seed={seed}")
    print(f"model_dir={model_dir}")

    for checkpoint_name in checkpoint_names:
        print()
        print(f"Training decoder for checkpoint={checkpoint_name} using final time step")

        model = load_model(model_dir, checkpoint_name)
        train_x, train_y = collect_final_hidden_and_labels(model, train_df, BATCH_SIZE)
        val_x, val_y = collect_final_hidden_and_labels(model, val_df, BATCH_SIZE)

        _, stats = fit_linear_decoder(
            train_x,
            train_y,
            val_x,
            val_y,
        )

        results.append(
            DecoderResult(
                group_key=group_key,
                train_heads=train_heads,
                seed=seed,
                checkpoint_name=checkpoint_name,
                checkpoint_epoch=checkpoint_epoch_from_name(checkpoint_name),
                train_acc=stats["train_acc"],
                val_acc=stats["val_acc"],
                train_loss=stats["train_loss"],
                val_loss=stats["val_loss"],
            )
        )

    return results


def plot_one_sigma(df: pd.DataFrame, group_key: str, output_dir: str) -> None:
    sigma_df = df[df["group_key"] == group_key].copy()
    if sigma_df.empty:
        print(f"No results to plot for {group_key}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, train_heads in zip(axes, TRAIN_HEADS_LIST):
        head_df = sigma_df[sigma_df["train_heads"] == train_heads].copy()
        head_df = head_df.sort_values(["seed", "checkpoint_epoch"])

        if head_df.empty:
            ax.set_title(train_heads)
            ax.set_xlabel("Checkpoint epoch (0 = init)")
            ax.grid(True, alpha=0.3)
            continue

        for seed in sorted(head_df["seed"].unique()):
            seed_df = head_df[head_df["seed"] == seed].sort_values("checkpoint_epoch")
            ax.plot(
                seed_df["checkpoint_epoch"],
                seed_df["val_acc"],
                linewidth=1.0,
                alpha=0.22,
            )

        mean_df = (
            head_df.groupby("checkpoint_epoch", as_index=False)["val_acc"]
            .mean()
            .sort_values("checkpoint_epoch")
        )
        ax.plot(
            mean_df["checkpoint_epoch"],
            mean_df["val_acc"],
            linewidth=3.0,
            label="mean",
        )

        if group_key in NORM_PRED_ACC:
            ax.axhline(
                NORM_PRED_ACC[group_key],
                linewidth=2.0,
                linestyle="-",
                label="norm_pred",
            )

        ax.set_title(train_heads)
        ax.set_xlabel("Checkpoint epoch (0 = init)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Decoder val accuracy")
    fig.suptitle(f"{group_key}: decoder val accuracy vs checkpoint")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{group_key}_val_acc_vs_checkpoint.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> None:
    all_results: list[DecoderResult] = []

    for group_key in GROUP_KEYS:
        train_df, val_df = load_group_data(group_key)
        for train_heads in TRAIN_HEADS_LIST:
            for seed in SEEDS:
                run_results = run_one_setting(group_key, train_heads, seed, train_df, val_df)
                all_results.extend(run_results)

    if not all_results:
        raise RuntimeError("No decoder results were produced")

    results_df = pd.DataFrame([r.__dict__ for r in all_results])
    results_df["checkpoint_label"] = results_df["checkpoint_epoch"].map(checkpoint_label_from_epoch)
    results_df = results_df.sort_values(["group_key", "train_heads", "seed", "checkpoint_epoch"])

    print()
    print(results_df[["group_key", "train_heads", "seed", "checkpoint_name", "val_acc"]].to_string(index=False))

    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, RAW_RESULTS_CSV)
        results_df.to_csv(csv_path, index=False)
        print(f"Saved raw results: {csv_path}")

        for group_key in GROUP_KEYS:
            plot_one_sigma(results_df, group_key, RESULTS_DIR)


if __name__ == "__main__":
    main()

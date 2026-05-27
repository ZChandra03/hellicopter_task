#!/usr/bin/env python3
"""
decode_stay_switch_from_activity.py
===================================
Train linear decoders to predict the trial-level stay/switch label
(truePredict) from the hidden activity of a trained GRU model.

Assumption used here:
    truePredict = stay/switch target from the CSV
    (-1 = stay, +1 = switch)

Decoder setup:
    - load a trained GRU checkpoint
    - extract the GRU hidden state h_t at every evidence time step
    - train one linear decoder per time step to predict truePredict
    - fit decoders on trainConfig_0..19
    - evaluate on valConfig_0..4

This is intended for rep-only models that were not explicitly trained on
stay/switch, but it will also work for other trained heads.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from dataclasses import dataclass
from typing import Iterable

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

GROUP_KEY = "sigma_1"
LOSS_TYPE = "bce"
TRAIN_HEADS = "rep"
SEED = 0
CHECKPOINT_NAME = "checkpoint_best.pt"   # or "final.pt"

TRAIN_IDXS = range(20)   # trainConfig_0.csv ... trainConfig_19.csv
VAL_IDXS = range(5)      # valConfig_0.csv ... valConfig_4.csv

BATCH_SIZE = 128
DECODER_LR = 1e-2
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 300
PRINT_EVERY = 50

SAVE_RESULTS = True
RESULTS_DIR = os.path.join(BASE_DIR, "decoder_results")


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
def collect_hidden_and_labels(model: nn.Module, df: pd.DataFrame, batch_size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
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

    x_by_t = None
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

        if x_by_t is None:
            x_by_t = [[] for _ in range(h_seq.shape[1])]

        for t in range(h_seq.shape[1]):
            x_by_t[t].append(h_seq[:, t, :].cpu().numpy())

        y_all.append(y_pred.cpu().numpy())

    handle.remove()

    if x_by_t is None:
        raise RuntimeError("No data collected")

    x_by_t = [np.concatenate(parts, axis=0) for parts in x_by_t]
    y = np.concatenate(y_all, axis=0).reshape(-1)
    y_by_t = [y.copy() for _ in range(len(x_by_t))]
    return x_by_t, y_by_t


class LinearDecoder(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


@dataclass
class DecoderResult:
    time_step: int
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
        "mean": best_state["mean"],
        "std": best_state["std"],
    }
    return decoder, stats


def print_results(results: list[DecoderResult]) -> None:
    print()
    print("Stay/switch linear decoder results")
    print(f"group={GROUP_KEY} seed={SEED} checkpoint={CHECKPOINT_NAME}")
    print()
    print(f"{'t':>3} {'train_acc':>10} {'val_acc':>10} {'train_loss':>12} {'val_loss':>12}")
    print("-" * 52)
    for r in results:
        print(
            f"{r.time_step:>3d} "
            f"{r.train_acc:>10.4f} "
            f"{r.val_acc:>10.4f} "
            f"{r.train_loss:>12.4f} "
            f"{r.val_loss:>12.4f}"
        )

    best = max(results, key=lambda r: r.val_acc)
    print()
    print(
        f"best val_acc at t={best.time_step}: "
        f"train_acc={best.train_acc:.4f}, val_acc={best.val_acc:.4f}"
    )


def main() -> None:
    model_dir = model_dir_from_settings(GROUP_KEY, LOSS_TYPE, TRAIN_HEADS, SEED)
    model = load_model(model_dir, CHECKPOINT_NAME)

    train_df = pd.concat(
        [pd.read_csv(p) for p in csv_paths(GROUP_KEY, "trainConfig", TRAIN_IDXS)],
        ignore_index=True,
    )
    val_df = pd.concat(
        [pd.read_csv(p) for p in csv_paths(GROUP_KEY, "valConfig", VAL_IDXS)],
        ignore_index=True,
    )

    train_x_by_t, train_y_by_t = collect_hidden_and_labels(model, train_df, BATCH_SIZE)
    val_x_by_t, val_y_by_t = collect_hidden_and_labels(model, val_df, BATCH_SIZE)

    if len(train_x_by_t) != len(val_x_by_t):
        raise RuntimeError("Train/val time dimensions do not match")

    results: list[DecoderResult] = []
    saved = {}

    for t in range(len(train_x_by_t)):
        print()
        print(f"Training linear decoder for time step t={t}")
        decoder, stats = fit_linear_decoder(
            train_x_by_t[t],
            train_y_by_t[t],
            val_x_by_t[t],
            val_y_by_t[t],
        )
        results.append(
            DecoderResult(
                time_step=t,
                train_acc=stats["train_acc"],
                val_acc=stats["val_acc"],
                train_loss=stats["train_loss"],
                val_loss=stats["val_loss"],
            )
        )
        saved[t] = {
            "state_dict": copy.deepcopy(decoder.state_dict()),
            "mean": stats["mean"],
            "std": stats["std"],
        }

    print_results(results)

    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_dir = os.path.join(
            RESULTS_DIR,
            f"stay_switch_{GROUP_KEY}_{LOSS_TYPE}_{TRAIN_HEADS}_seed{SEED}",
        )
        os.makedirs(result_dir, exist_ok=True)

        pd.DataFrame([r.__dict__ for r in results]).to_csv(
            os.path.join(result_dir, "decoder_accuracy_by_time.csv"),
            index=False,
        )
        torch.save(saved, os.path.join(result_dir, "linear_decoders_by_time.pt"))


if __name__ == "__main__":
    main()

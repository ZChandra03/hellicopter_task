#!/usr/bin/env python3
"""
train_gen_hazard_OTS_hardcoded.py
=================================
Compares how different uniform hazard groups affect GRU training on the
helicopter task.

This version:
  - keeps the original REINFORCE-style objective as an option
  - adds BCEWithLogits training as an option
  - lets you train on the report head only, hazard head only, or both
  - removes argparse and uses hardcoded settings near the top of the file
  - uses valConfig_{k}.csv files as the validation set
  - expects variants/sigma_1, variants/sigma_2, and variants/sigma_3
  - saves 4 checkpoints per epoch
  - limits training to at most 3 epochs

Default run below is set to:
  LOSS_TYPE = "bce"
  TRAIN_HEADS = "rep"
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os
import random
import re
import time
from typing import List, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rnn_models import GRUModel

# ───────────────────────────── configuration ──────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(BASE_DIR, "variants")

GROUP_KEYS = [
    "sigma_1",
]

VALID_LOSS_TYPES = {"reinforce", "bce"}
VALID_TRAIN_HEADS = {"rep", "haz", "both"}

LOSS_TYPE = "bce"          # "reinforce" or "bce"
TRAIN_HEADS = "both"        # "rep", "haz", or "both"
SEED_START = 0
N_SEEDS = 10
GROUPS_TO_RUN = GROUP_KEYS

MAX_EPOCHS = 3
CHECKPOINTS_PER_EPOCH = 4


def validate_run_config() -> None:
    if LOSS_TYPE not in VALID_LOSS_TYPES:
        raise ValueError(f"LOSS_TYPE must be one of {sorted(VALID_LOSS_TYPES)}, got {LOSS_TYPE!r}")
    if TRAIN_HEADS not in VALID_TRAIN_HEADS:
        raise ValueError(f"TRAIN_HEADS must be one of {sorted(VALID_TRAIN_HEADS)}, got {TRAIN_HEADS!r}")
    if not isinstance(SEED_START, int):
        raise TypeError("SEED_START must be an int")
    if not isinstance(N_SEEDS, int) or N_SEEDS <= 0:
        raise ValueError("N_SEEDS must be a positive int")
    if not isinstance(GROUPS_TO_RUN, (list, tuple)) or len(GROUPS_TO_RUN) == 0:
        raise ValueError("GROUPS_TO_RUN must be a non-empty list or tuple")
    if not isinstance(MAX_EPOCHS, int) or MAX_EPOCHS <= 0:
        raise ValueError("MAX_EPOCHS must be a positive int")
    if not isinstance(CHECKPOINTS_PER_EPOCH, int) or CHECKPOINTS_PER_EPOCH <= 0:
        raise ValueError("CHECKPOINTS_PER_EPOCH must be a positive int")


def get_default_hp(loss_type: str = "reinforce", train_heads: str = "both") -> dict:
    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {sorted(VALID_LOSS_TYPES)}, got {loss_type!r}")
    if train_heads not in VALID_TRAIN_HEADS:
        raise ValueError(f"train_heads must be one of {sorted(VALID_TRAIN_HEADS)}, got {train_heads!r}")

    return {
        "n_input": 1,
        "n_rnn": 128,
        "batch_size": 25,
        "learning_rate": 3e-4,
        "target_loss": 1e-3,
        "max_epochs": MAX_EPOCHS,
        "max_csv": 20,
        "loss_type": loss_type,
        "train_heads": train_heads,
    }


# ───────────────────────────── dataset helper ─────────────────────────────
class HelicopterDataset(Dataset):
    """Convert one CSV of trials into tensors for PyTorch.

    Targets:
      - report → trueReport  (−1/+1 → 0/1)
      - hazard → truePredict (−1/+1 → 0/1)
    """

    def __init__(self, df: pd.DataFrame):
        xs, yr, yh = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))

            x = torch.tensor(evid, dtype=torch.float32).unsqueeze(-1)
            xs.append(x)

            yr.append(torch.tensor([(row["trueReport"] + 1) * 0.5], dtype=torch.float32))
            yh.append(torch.tensor([(row["truePredict"] + 1) * 0.5], dtype=torch.float32))

        self.x, self.y_rep, self.y_haz = xs, yr, yh

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y_rep[i], self.y_haz[i]


# ───────────────────────────── CSV utilities ─────────────────────────────
def _list_train_variants(group_key: str) -> List[str]:
    pat = os.path.join(VARIANTS_DIR, group_key, "trainConfig_*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No files match {pat}")
    return paths


def _get_fixed_val_csvs(group_key: str) -> List[str]:
    paths = []
    for k in range(5):
        path = os.path.join(VARIANTS_DIR, group_key, f"valConfig_{k}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing validation file: {path}")
        paths.append(path)
    return paths


def _extract_cfg_idx(path: str, prefix: str) -> int:
    name = os.path.basename(path)
    m = re.fullmatch(rf"{prefix}_(\d+)\.csv", name)
    if m is None:
        raise ValueError(f"Unexpected filename format: {name}")
    return int(m.group(1))


def _load_concat_df(paths: List[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


# ───────────────────────────── loss helpers ──────────────────────────────
def _ensure_col(v: torch.Tensor) -> torch.Tensor:
    v = v.float()
    if v.ndim == 1:
        v = v.unsqueeze(1)
    return v


def _p_correct_from_logits(logits: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    logits = _ensure_col(logits)
    y01 = _ensure_col(y01)
    p1 = torch.sigmoid(logits)
    p0 = 1.0 - p1
    return torch.where(y01 > 0.5, p1, p0)


def _selected_p_correct(
    o_rep_last: torch.Tensor,
    o_haz: torch.Tensor,
    y_rep: torch.Tensor,
    y_haz: torch.Tensor,
    train_heads: str,
) -> list[torch.Tensor]:
    selected: list[torch.Tensor] = []
    if train_heads in {"rep", "both"}:
        selected.append(_p_correct_from_logits(o_rep_last, y_rep))
    if train_heads in {"haz", "both"}:
        selected.append(_p_correct_from_logits(o_haz, y_haz))
    if not selected:
        raise ValueError(f"Unsupported train_heads: {train_heads!r}")
    return selected


def proxy_reward_loss(
    o_rep_last: torch.Tensor,
    o_haz: torch.Tensor,
    y_rep: torch.Tensor,
    y_haz: torch.Tensor,
    train_heads: str = "both",
) -> torch.Tensor:
    selected = _selected_p_correct(o_rep_last, o_haz, y_rep, y_haz, train_heads)
    reward = torch.stack([p.squeeze(1) for p in selected], dim=0).mean(dim=0)
    return (1.0 - reward).mean()


def batch_accuracy(
    o_rep_last: torch.Tensor,
    o_haz: torch.Tensor,
    y_rep: torch.Tensor,
    y_haz: torch.Tensor,
    train_heads: str = "both",
) -> torch.Tensor:
    selected = _selected_p_correct(o_rep_last, o_haz, y_rep, y_haz, train_heads)
    acc = torch.stack([(p > 0.5).float().squeeze(1) for p in selected], dim=0).mean(dim=0)
    return acc.mean()


class _PGState:
    def __init__(self, baseline: float = 0.0):
        self.baseline = baseline


def reinforce_loss(
    o_rep_last: torch.Tensor,
    o_haz: torch.Tensor,
    y_rep: torch.Tensor,
    y_haz: torch.Tensor,
    state: _PGState,
    train_heads: str = "both",
    entropy_coef: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state, entropy_coef

    z_r = _ensure_col(o_rep_last).squeeze(1)
    z_h = _ensure_col(o_haz).squeeze(1)
    y_r = _ensure_col(y_rep).squeeze(1)
    y_h = _ensure_col(y_haz).squeeze(1)

    rewards = []
    logps = []

    if train_heads in {"rep", "both"}:
        m_r = torch.distributions.Bernoulli(logits=z_r)
        a_r = m_r.sample()
        rewards.append((a_r == y_r).float())
        logps.append(m_r.log_prob(a_r))

    if train_heads in {"haz", "both"}:
        m_h = torch.distributions.Bernoulli(logits=z_h)
        a_h = m_h.sample()
        rewards.append((a_h == y_h).float())
        logps.append(m_h.log_prob(a_h))

    if not rewards:
        raise ValueError(f"Unsupported train_heads: {train_heads!r}")

    R = torch.stack(rewards, dim=0).mean(dim=0)
    logp = torch.stack(logps, dim=0).sum(dim=0)
    loss = -(R.detach() * logp).mean()
    return loss, R.mean().detach()


def bce_loss(
    o_rep_last: torch.Tensor,
    o_haz: torch.Tensor,
    y_rep: torch.Tensor,
    y_haz: torch.Tensor,
    train_heads: str = "both",
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []

    if train_heads in {"rep", "both"}:
        losses.append(nn.functional.binary_cross_entropy_with_logits(
            _ensure_col(o_rep_last), _ensure_col(y_rep)
        ))

    if train_heads in {"haz", "both"}:
        losses.append(nn.functional.binary_cross_entropy_with_logits(
            _ensure_col(o_haz), _ensure_col(y_haz)
        ))

    if not losses:
        raise ValueError(f"Unsupported train_heads: {train_heads!r}")

    loss = sum(losses) / len(losses)
    proxy_reward = 1.0 - proxy_reward_loss(o_rep_last, o_haz, y_rep, y_haz, train_heads)
    return loss, proxy_reward.detach()


def compute_training_loss(
    o_rep_last: torch.Tensor,
    o_haz: torch.Tensor,
    y_rep: torch.Tensor,
    y_haz: torch.Tensor,
    hp: dict,
    pg_state: _PGState,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_type = hp["loss_type"]
    train_heads = hp["train_heads"]

    if loss_type == "reinforce":
        return reinforce_loss(o_rep_last, o_haz, y_rep, y_haz, pg_state, train_heads=train_heads)
    if loss_type == "bce":
        return bce_loss(o_rep_last, o_haz, y_rep, y_haz, train_heads=train_heads)

    raise ValueError(f"Unsupported loss_type: {loss_type!r}")


def compute_validation_metrics(model: nn.Module, val_df: pd.DataFrame, hp: dict) -> tuple[float, float]:
    model.eval()
    val_dl = DataLoader(
        HelicopterDataset(val_df),
        batch_size=hp["batch_size"],
        shuffle=False,
    )

    vtot = 0.0
    vacc = 0.0
    with torch.no_grad():
        for x, y_r, y_h in val_dl:
            x = x.to(DEVICE)
            y_r = y_r.to(DEVICE)
            y_h = y_h.to(DEVICE)
            o_r, o_h = model(x)
            o_r_last = _ensure_col(o_r[:, -1])
            vtot += proxy_reward_loss(
                o_r_last,
                o_h,
                y_r,
                y_h,
                train_heads=hp["train_heads"],
            ).item()
            vacc += batch_accuracy(
                o_r_last,
                o_h,
                y_r,
                y_h,
                train_heads=hp["train_heads"],
            ).item()

    return vtot / len(val_dl), vacc / len(val_dl)


def _count_batches_for_csv(path: str, batch_size: int) -> int:
    df_len = len(pd.read_csv(path))
    return df_len // batch_size


def _build_epoch_save_marks(total_batches_per_epoch: int, checkpoints_per_epoch: int) -> dict[int, int]:
    if total_batches_per_epoch <= 0:
        raise ValueError("total_batches_per_epoch must be positive")

    save_marks = {}
    for part in range(1, checkpoints_per_epoch + 1):
        batch_mark = max(1, math.ceil(part * total_batches_per_epoch / checkpoints_per_epoch))
        save_marks[batch_mark] = part
    return save_marks


# ───────────────────────────── training loop ──────────────────────────────
def train_group(
    model_cls: Type[nn.Module],
    group_key: str,
    seed: int,
    loss_type: str = "reinforce",
    train_heads: str = "both",
) -> None:
    train_csvs_all = _list_train_variants(group_key)
    hp = get_default_hp(loss_type=loss_type, train_heads=train_heads)

    max_csv = int(hp.get("max_csv", len(train_csvs_all)))
    train_csvs = train_csvs_all[:max_csv]
    if not train_csvs:
        raise RuntimeError(f"No training CSVs selected for {group_key}")

    val_df = _load_concat_df(_get_fixed_val_csvs(group_key))
    n_csv = len(train_csvs)
    n_total_epochs = int(hp.get("max_epochs", 1))

    total_batches_per_epoch = sum(_count_batches_for_csv(path, hp["batch_size"]) for path in train_csvs)
    if total_batches_per_epoch <= 0:
        raise RuntimeError(f"No train batches available for {group_key}")

    save_marks = _build_epoch_save_marks(total_batches_per_epoch, CHECKPOINTS_PER_EPOCH)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if loss_type == "reinforce" and train_heads == "both":
        model_root = os.path.join(BASE_DIR, "models_OTS")
    else:
        model_root = os.path.join(BASE_DIR, "models_OTS", f"{loss_type}_{train_heads}")

    model_dir = os.path.join(model_root, group_key, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    ckpt_init = os.path.join(model_dir, "checkpoint_init.pt")
    ckpt_best = os.path.join(model_dir, "checkpoint_best.pt")
    ckpt_final = os.path.join(model_dir, "final.pt")

    model = model_cls(hp).to(DEVICE)
    torch.save(model.state_dict(), ckpt_init)
    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    pg_state = _PGState(baseline=0.0)
    best, loss_hist, val_hist = float("inf"), [], []
    t0 = time.time()
    early_stop = False

    init_loss, init_acc = compute_validation_metrics(model, val_df, hp)
    print(
        f"{group_key}|seed{seed} "
        f"[{loss_type}/{train_heads}] "
        f"init "
        f"val_loss {init_loss:.4e} "
        f"val_acc {init_acc:.4f}"
    )

    if init_loss < best:
        best = init_loss
        torch.save(model.state_dict(), ckpt_best)

    for epoch in range(n_total_epochs):
        model.train()
        running = 0.0
        running_acc = 0.0
        n_train_batches = 0
        batch_reward_value = float("nan")
        epoch_batch_idx = 0
        saved_parts_this_epoch = set()

        csv_order = rng.permutation(n_csv)
        for idx in csv_order:
            df = pd.read_csv(train_csvs[idx])
            dl = DataLoader(
                HelicopterDataset(df),
                batch_size=hp["batch_size"],
                shuffle=True,
                drop_last=True,
            )

            for x, y_r, y_h in dl:
                x = x.to(DEVICE)
                y_r = y_r.to(DEVICE)
                y_h = y_h.to(DEVICE)

                opt.zero_grad()
                o_r, o_h = model(x)
                o_r_last = _ensure_col(o_r[:, -1])

                loss, batch_reward = compute_training_loss(o_r_last, o_h, y_r, y_h, hp, pg_state)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                running += float(loss.detach())
                running_acc += float(batch_accuracy(o_r_last, o_h, y_r, y_h, train_heads=hp["train_heads"]).detach())
                n_train_batches += 1
                batch_reward_value = float(batch_reward)

                epoch_batch_idx += 1
                if epoch_batch_idx in save_marks:
                    part = save_marks[epoch_batch_idx]
                    ckpt_path = os.path.join(
                        model_dir,
                        f"checkpoint_ep{epoch+1:03}_part{part:02}.pt"
                    )
                    torch.save(model.state_dict(), ckpt_path)
                    saved_parts_this_epoch.add(part)

        epoch_loss = running / n_train_batches
        epoch_acc = running_acc / n_train_batches
        loss_hist.append(epoch_loss)

        val_loss, val_acc = compute_validation_metrics(model, val_df, hp)
        val_hist.append(val_loss)

        for part in range(1, CHECKPOINTS_PER_EPOCH + 1):
            if part not in saved_parts_this_epoch:
                ckpt_path = os.path.join(
                    model_dir,
                    f"checkpoint_ep{epoch+1:03}_part{part:02}.pt"
                )
                torch.save(model.state_dict(), ckpt_path)

        print(
            f"{group_key}|seed{seed} "
            f"[{loss_type}/{train_heads}] "
            f"ep {epoch:03}/{n_total_epochs} "
            f"train_loss {epoch_loss:.4e} "
            f"train_acc {epoch_acc:.4f} "
            f"val_loss {val_loss:.4e} "
            f"val_acc {val_acc:.4f} "
            f"reward {batch_reward_value:.4f}"
        )

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), ckpt_best)

        if best < hp["target_loss"]:
            print(
                f"{group_key}|seed{seed} [{loss_type}/{train_heads}] "
                f"early-stop @ ep {epoch} (best val {best:.4e})"
            )
            early_stop = True
            break

        if early_stop:
            break

    torch.save(model.state_dict(), ckpt_final)
    json.dump(loss_hist, open(os.path.join(model_dir, "loss_history.json"), "w"), indent=2)
    json.dump(val_hist, open(os.path.join(model_dir, "val_loss_history.json"), "w"), indent=2)
    json.dump(hp, open(os.path.join(model_dir, "hp.json"), "w"), indent=2)
    print(
        f"{group_key}|seed{seed} [{loss_type}/{train_heads}] "
        f"finished in {time.time() - t0:.1f}s | best val {best:.4e}"
    )


# ───────────────────────────── entry-point ────────────────────────────────
def main() -> None:
    validate_run_config()
    seeds = range(SEED_START, SEED_START + N_SEEDS)

    for seed in seeds:
        for group_key in GROUPS_TO_RUN:
            train_group(
                GRUModel,
                group_key,
                seed,
                loss_type=LOSS_TYPE,
                train_heads=TRAIN_HEADS,
            )


if __name__ == "__main__":
    main()
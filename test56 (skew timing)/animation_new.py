#!/usr/bin/env python3
"""
Animate binned accuracy over training checkpoints for the new OTS layout
=======================================================================

Expected model layout
---------------------
models_OTS/
    bce_rep/
        sigma_1/
            seed_0/
                checkpoint_ep010.pt
                checkpoint_ep020.pt
                ...
                final.pt
                hp.json
        sigma_2/
        sigma_3/
    bce_haz/
    bce_both/
    reinforce_both/   # optional future layout
or, for the special case used in the training script:
    models_OTS/
        sigma_1/
        sigma_2/
        sigma_3/
when loss_type="reinforce" and train_heads="both"

Expected data layout
--------------------
variants/
    sigma_1/
        testConfig_0.csv
        ...
    sigma_2/
    sigma_3/

Output
------
Saves animations under:
    figures/animations/<loss_type>_<train_heads>/<sigma_key>/

Example runs
------------
python animation_new.py --loss-types bce --train-heads all --groups all --head both --n-test 20 --include-final
python animation_new.py --loss-types bce --train-heads rep --groups sigma_1 sigma_2 sigma_3 --head hazard
python animation_new.py --loss-types reinforce --train-heads both --groups all --head both --include-final
"""

from __future__ import annotations
import argparse
import ast
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation

from rnn_models import GRUModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BASE_DIR, "models_OTS")
VARIANTS_ROOT = os.path.join(BASE_DIR, "variants")
OUT_ROOT = os.path.join(BASE_DIR, "figures", "animations")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIN_WIDTH = 0.05
BIN_EDGES = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

SIGMA_RE = re.compile(r"sigma_\d+(?:p\d+)?$")
SEED_RE = re.compile(r"seed_(\d+)$")
CKPT_RE = re.compile(r"checkpoint_ep(\d{3})\.pt$")

VALID_LOSS_TYPES = ("bce", "reinforce")
VALID_TRAIN_HEADS = ("rep", "haz", "both")


class HelicopterEvalDS(Dataset):
    def __init__(self, df: pd.DataFrame):
        xs, rep_targets, hazards = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))

            xs.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))
            rep_targets.append(float(1.0 if row["trueReport"] > 0 else 0.0))
            hazards.append(float(row["trueHazard"]))

        self.x = xs
        self.y_rep = torch.tensor(rep_targets, dtype=torch.float32).unsqueeze(1)
        self.haz = torch.tensor(hazards, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y_rep[i], self.haz[i]


def collate(batch):
    xs, yr, hz = zip(*batch)
    return torch.stack(xs, 0), torch.stack(yr, 0), torch.stack(hz, 0)


@torch.no_grad()
def eval_model_on_csvs(model: torch.nn.Module, csvs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hazards, rep_ok, haz_ok = [], [], []

    for p in csvs:
        df = pd.read_csv(p)
        dl = DataLoader(HelicopterEvalDS(df), batch_size=512, shuffle=False, collate_fn=collate)

        for x, y_rep, hz in dl:
            x = x.to(DEVICE)
            y_rep = y_rep.to(DEVICE)

            loc_logits, haz_logits = model(x)

            rep_pred = (torch.sigmoid(loc_logits[:, -1, :]) > 0.5).float()
            rep_ok.extend((rep_pred.squeeze(1) == y_rep.squeeze(1)).cpu().numpy().tolist())

            haz_pred = (torch.sigmoid(haz_logits) > 0.5).float().squeeze(1)
            haz_true_bin = (hz.to(haz_pred.device) > 0.5).float()
            haz_ok.extend((haz_pred == haz_true_bin).cpu().numpy().tolist())

            hazards.extend(hz.cpu().numpy().tolist())

    return np.array(hazards), np.array(rep_ok, dtype=bool), np.array(haz_ok, dtype=bool)


def bin_accuracy(hazards: np.ndarray, correct_mask: np.ndarray) -> np.ndarray:
    idx = np.digitize(hazards, BIN_EDGES) - 1
    idx = np.clip(idx, 0, len(BIN_EDGES) - 2)

    total = np.zeros(len(BIN_CENTERS), dtype=int)
    good = np.zeros(len(BIN_CENTERS), dtype=int)

    for i, ok in zip(idx, correct_mask):
        total[i] += 1
        if ok:
            good[i] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(total > 0, good / total, np.nan)

    return acc


def make_run_key(loss_type: str, train_heads: str) -> str:
    return f"{loss_type}_{train_heads}"


def get_run_root(loss_type: str, train_heads: str) -> str:
    if loss_type == "reinforce" and train_heads == "both":
        return MODELS_ROOT
    return os.path.join(MODELS_ROOT, f"{loss_type}_{train_heads}")


def list_requested(values: List[str], valid: Tuple[str, ...]) -> List[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(valid)
    return values


def list_groups(run_root: str, groups_cli: List[str]) -> List[str]:
    if len(groups_cli) == 1 and groups_cli[0].lower() == "all":
        if not os.path.isdir(run_root):
            return []
        return sorted(
            d for d in os.listdir(run_root)
            if SIGMA_RE.fullmatch(d) and os.path.isdir(os.path.join(run_root, d))
        )
    return groups_cli


def list_seeds(group_root: str) -> List[int]:
    if not os.path.isdir(group_root):
        return []

    seeds = []
    for d in sorted(os.listdir(group_root)):
        m = SEED_RE.fullmatch(d)
        if m and os.path.isdir(os.path.join(group_root, d)):
            seeds.append(int(m.group(1)))
    return seeds


def load_hp(seed_dir: str) -> Dict:
    with open(os.path.join(seed_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    hp.setdefault("n_input", 1)
    hp.setdefault("n_rnn", 128)
    return hp


def load_model_for_ckpt(seed_dir: str, ckpt_path: str) -> torch.nn.Module:
    hp = load_hp(seed_dir)
    model = GRUModel(hp).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def list_checkpoints(seed_dir: str, include_final: bool) -> List[Tuple[int, str]]:
    ckpts = []

    for p in glob.glob(os.path.join(seed_dir, "checkpoint_ep*.pt")):
        m = CKPT_RE.search(os.path.basename(p))
        if m:
            ckpts.append((int(m.group(1)), p))

    if include_final:
        final_path = os.path.join(seed_dir, "final.pt")
        if os.path.exists(final_path):
            ckpts.append((10**9, final_path))

    ckpts.sort(key=lambda t: t[0])
    return ckpts


def prepare_test_csvs(group_key: str, n_test: int) -> List[str]:
    group_dir = os.path.join(VARIANTS_ROOT, group_key)
    paths = sorted(glob.glob(os.path.join(group_dir, "testConfig_*.csv")))[:n_test]

    if not paths:
        raise FileNotFoundError(f"No testConfig_*.csv found in {group_dir}")
    if len(paths) < n_test:
        print(f"[warn] only {len(paths)} test configs found in {group_dir}")

    return paths


def make_seed_palette(seed_ids: List[int]) -> Dict[int, tuple]:
    if len(seed_ids) <= 20:
        cmap = plt.get_cmap("tab20", len(seed_ids))
    else:
        cmap = plt.get_cmap("hsv", len(seed_ids))
    return {s: cmap(i) for i, s in enumerate(seed_ids)}


def build_frames(run_root: str, group_key: str, head: str, n_test: int, include_final: bool):
    group_root = os.path.join(run_root, group_key)
    seed_ids = list_seeds(group_root)
    if not seed_ids:
        raise RuntimeError(f"No seeds found under {group_root}")

    csvs = prepare_test_csvs(group_key, n_test)
    epoch_to_accs: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    for s in seed_ids:
        seed_dir = os.path.join(group_root, f"seed_{s}")
        ckpts = list_checkpoints(seed_dir, include_final)
        if not ckpts:
            print(f"[skip] {seed_dir}: no checkpoints found")
            continue

        for ep, ckpt in ckpts:
            try:
                model = load_model_for_ckpt(seed_dir, ckpt)
            except Exception as e:
                print(f"[warn] failed to load {ckpt}: {e}")
                continue

            hazards, rep_ok, haz_ok = eval_model_on_csvs(model, csvs)
            acc = bin_accuracy(hazards, rep_ok if head == "report" else haz_ok)
            epoch_to_accs.setdefault(ep, []).append((s, acc))

    if not epoch_to_accs:
        raise RuntimeError(f"No evaluations produced for {group_root} (head={head})")

    epochs_sorted = sorted(epoch_to_accs.keys())
    return epochs_sorted, epoch_to_accs, seed_ids


def plot_hazard_skew(run_key: str,
                     group_key: str,
                     epochs_sorted: List[int],
                     acc_by_epoch: Dict[int, List[Tuple[int, np.ndarray]]],
                     seed_ids: List[int],
                     seed_color: Dict[int, tuple],
                     out_dir: str,
                     split: float = 0.5,
                     dpi: int = 200) -> None:
    high_mask = BIN_CENTERS >= split
    low_mask = BIN_CENTERS < split

    n_epochs = len(epochs_sorted)
    n_seeds = len(seed_ids)
    skew = np.full((n_epochs, n_seeds), np.nan)
    s_index = {s: i for i, s in enumerate(seed_ids)}

    for t, ep in enumerate(epochs_sorted):
        for s, acc in acc_by_epoch[ep]:
            hi = np.nanmean(acc[high_mask])
            lo = np.nanmean(acc[low_mask])
            skew[t, s_index[s]] = hi - lo

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    xs = np.arange(n_epochs)

    for j, s in enumerate(seed_ids):
        ax.plot(xs, skew[:, j], label=f"seed {s}", alpha=0.85, linewidth=1.6, color=seed_color[s])

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Skew = acc(high) − acc(low)")
    ax.set_xlabel("Checkpoint")
    ax.set_xticks(xs)
    ax.set_xticklabels(["final" if ep >= 10**9 else f"ep {ep:03d}" for ep in epochs_sorted])
    ax.set_title(f"{run_key} | {group_key} | hazard-head skew over time")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "hazard_skew_over_time.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def animate_group(run_root: str,
                  run_key: str,
                  group_key: str,
                  head: str,
                  n_test: int,
                  include_final: bool,
                  fps: int,
                  dpi: int,
                  do_skew_plot: bool,
                  skew_split: float) -> None:
    epochs_sorted, acc_by_epoch, seed_ids = build_frames(run_root, group_key, head, n_test, include_final)
    seed_color = make_seed_palette(seed_ids)

    out_dir = os.path.join(OUT_ROOT, run_key, group_key)
    os.makedirs(out_dir, exist_ok=True)

    if head == "hazard" and do_skew_plot:
        plot_hazard_skew(run_key, group_key, epochs_sorted, acc_by_epoch, seed_ids, seed_color, out_dir, split=skew_split, dpi=dpi)

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
    for s in seed_ids:
        line, = ax.plot([], [], linewidth=1.6, alpha=0.9, color=seed_color[s])
        lines[s] = line
        legend_handles.append(line)

    ax.legend(legend_handles, [f"seed {s}" for s in seed_ids], frameon=False, loc="lower left", ncol=2, fontsize=9)

    def init():
        for s in seed_ids:
            lines[s].set_data([], [])
        title.set_text("")
        return list(lines.values()) + [title]

    def update(frame_idx: int):
        ep = epochs_sorted[frame_idx]
        seed_acc_map = {s: acc for s, acc in acc_by_epoch[ep]}

        for s in seed_ids:
            if s in seed_acc_map:
                lines[s].set_data(BIN_CENTERS, seed_acc_map[s])
            else:
                lines[s].set_data([], [])

        ep_text = "final" if ep >= 10**9 else f"ep {ep:03d}"
        title.set_text(f"{run_key} | {group_key} | {head} head | {ep_text}")
        return list(lines.values()) + [title]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(epochs_sorted),
        interval=1000 // fps,
        blit=True,
    )

    mp4_path = os.path.join(out_dir, f"{head}_binned_accuracy_anim.mp4")
    gif_path = os.path.join(out_dir, f"{head}_binned_accuracy_anim.gif")

    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata=dict(artist="animation_new.py"), bitrate=1800)
        anim.save(mp4_path, writer=writer, dpi=dpi)
        print(f"Saved {mp4_path}")
    except Exception as e:
        print(f"[info] ffmpeg unavailable or failed ({e}); saving GIF instead")
        anim.save(gif_path, writer="pillow", dpi=dpi)
        print(f"Saved {gif_path}")
    finally:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Animate binned accuracy for the new models_OTS/sigma_* layout.")
    ap.add_argument("--loss-types", nargs="+", default=["bce"], help="bce, reinforce, or all")
    ap.add_argument("--train-heads", nargs="+", default=["all"], help="rep, haz, both, or all")
    ap.add_argument("--groups", nargs="+", default=["all"], help="sigma_1 sigma_2 sigma_3 or all")
    ap.add_argument("--head", choices=["report", "hazard", "both"], default="both", help="Which output head to evaluate.")
    ap.add_argument("--n-test", type=int, default=20, help="Number of test CSVs per sigma group.")
    ap.add_argument("--include-final", action="store_true", help="Append final.pt as the last frame if present.")
    ap.add_argument("--fps", type=int, default=2, help="Frames per second.")
    ap.add_argument("--dpi", type=int, default=200, help="DPI for saved outputs.")
    ap.add_argument("--no-skew-plot", action="store_true", help="Disable the hazard skew plot.")
    ap.add_argument("--skew-split", type=float, default=0.5, help="Split point for low-vs-high hazard skew.")
    args = ap.parse_args()

    loss_types = list_requested(args.loss_types, VALID_LOSS_TYPES)
    train_heads_list = list_requested(args.train_heads, VALID_TRAIN_HEADS)

    any_run_found = False

    for loss_type in loss_types:
        for train_heads in train_heads_list:
            run_root = get_run_root(loss_type, train_heads)
            run_key = make_run_key(loss_type, train_heads)

            if not os.path.isdir(run_root):
                print(f"[skip] missing run root: {run_root}")
                continue

            groups = list_groups(run_root, args.groups)
            if not groups:
                print(f"[skip] no sigma groups found under {run_root}")
                continue

            any_run_found = True

            for group_key in groups:
                if args.head in ("report", "both"):
                    animate_group(
                        run_root=run_root,
                        run_key=run_key,
                        group_key=group_key,
                        head="report",
                        n_test=args.n_test,
                        include_final=args.include_final,
                        fps=args.fps,
                        dpi=args.dpi,
                        do_skew_plot=False,
                        skew_split=args.skew_split,
                    )

                if args.head in ("hazard", "both"):
                    animate_group(
                        run_root=run_root,
                        run_key=run_key,
                        group_key=group_key,
                        head="hazard",
                        n_test=args.n_test,
                        include_final=args.include_final,
                        fps=args.fps,
                        dpi=args.dpi,
                        do_skew_plot=(not args.no_skew_plot),
                        skew_split=args.skew_split,
                    )

    if not any_run_found:
        raise SystemExit("No matching run roots found under models_OTS")

    print("All requested animations complete.")


if __name__ == "__main__":
    main()
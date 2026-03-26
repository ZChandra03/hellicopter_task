#!/usr/bin/env python3
"""
Animate binned accuracy over training checkpoints (every 20 epochs)
==================================================================
This script mirrors the evaluation logic in *binned_accuracy.py* but, instead of
only testing `final.pt`, it sweeps through **checkpoint_epXXX.pt** files saved by
`train_batch_binary_onehot_reinforce.py` (every 20 epochs by default) and
renders an animation of accuracy-vs-hazard **per Beta** and **per head**.

Output
------
For each present Beta folder under ./models (e.g., beta_1p0), saves up to two
animations in `./figures/animations/<beta_key>/`:
  • `report_binned_accuracy_anim.mp4`  – report head
  • `hazard_binned_accuracy_anim.mp4`  – hazard head

Each frame corresponds to one checkpoint (ep = 20, 40, 60, ...), plotting
faint per-seed curves and a bold per-frame mean across seeds that have that
checkpoint.

Notes
-----
• Fixed test set: by default we test on variants/beta_1p0 (first N_TEST_CFGS),
  matching the earlier binned-accuracy script for comparability.
• We do **not** combine different betas in a single plot; animations are
  produced per-beta.
• If ffmpeg is unavailable, the script falls back to Pillow writer (GIF).

Run examples
------------
python animate_binned_accuracy.py --betas all --head both --n-test 20 --include-final
python animate_binned_accuracy.py --betas beta_1p0 beta_2p0 --head hazard

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

# Uses the same model interface as your existing codebase
from rnn_models import GRUModel  # default; swap if needed

# ───────────────────────────── paths & constants ─────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VAR_DIR     = os.path.join(BASE_DIR, "variants", "beta_1p0")   # fixed test set
OUT_ROOT    = os.path.join(BASE_DIR, "figures", "animations")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIN_WIDTH   = 0.05
BIN_EDGES   = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)  # 21 edges
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])     # 20 bins

# Optional: consistent color per beta (used for lines)
COLOR_BY_BETA = {
    "beta_0p1" : "tab:blue",
    "beta_0p5" : "tab:orange",
    "beta_1p0" : "tab:green",
    "beta_2p0" : "tab:red",
    "beta_10p0": "tab:purple",
}

# ───────────────────────────── data helpers ─────────────────────────────
class HelicopterEvalDS(Dataset):
    def __init__(self, df: pd.DataFrame):
        xs, rep_targets, hazards = [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            xs.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))  # (T,1)
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
    """Return hazards, report-correct mask, hazard-correct mask for all rows."""
    hazards, rep_ok, haz_ok = [], [], []
    for p in csvs:
        df = pd.read_csv(p)
        dl = DataLoader(HelicopterEvalDS(df), batch_size=512, shuffle=False, collate_fn=collate)
        for x, y_rep, hz in dl:
            x = x.to(DEVICE); y_rep = y_rep.to(DEVICE)
            loc_logits, haz_logits = model(x)                # (B,T,1), (B,1)

            # Report accuracy at last step
            rep_pred = (torch.sigmoid(loc_logits[:, -1, :]) > 0.5).float()
            rep_ok.extend((rep_pred.squeeze(1) == y_rep.squeeze(1)).cpu().numpy().tolist())

            # Binary hazard accuracy (threshold 0.5 on both pred and truth)
            haz_pred = (torch.sigmoid(haz_logits) > 0.5).float().squeeze(1)
            haz_true_bin = (hz.to(haz_pred.device) > 0.5).float()
            haz_ok.extend((haz_pred == haz_true_bin).cpu().numpy().tolist())

            hazards.extend(hz.cpu().numpy().tolist())
    return np.array(hazards), np.array(rep_ok, bool), np.array(haz_ok, bool)


def bin_accuracy(hazards: np.ndarray, correct_mask: np.ndarray) -> np.ndarray:
    idx = np.digitize(hazards, BIN_EDGES) - 1
    idx = np.clip(idx, 0, len(BIN_EDGES)-2)
    total = np.zeros(len(BIN_CENTERS), int)
    good  = np.zeros(len(BIN_CENTERS), int)
    for i, ok in zip(idx, correct_mask):
        total[i] += 1
        if ok: good[i] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(total > 0, good / total, np.nan)
    return acc


# ───────────────────────────── model / ckpt helpers ─────────────────────────────
_ckpt_re = re.compile(r"checkpoint_ep(\d{3})\.pt$")

def list_beta_keys(betas_cli: List[str]) -> List[str]:
    if len(betas_cli) == 1 and betas_cli[0].lower() == "all":
        return sorted([d for d in os.listdir(MODELS_DIR) if d.startswith("beta_") and os.path.isdir(os.path.join(MODELS_DIR, d))])
    return betas_cli


def list_seeds(beta_key: str) -> List[int]:
    root = os.path.join(MODELS_DIR, beta_key)
    seeds = []
    for d in sorted(os.listdir(root)):
        m = re.match(r"seed_(\d+)$", d)
        if m and os.path.isdir(os.path.join(root, d)):
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
    # periodic checkpoints
    for p in glob.glob(os.path.join(seed_dir, "checkpoint_ep*.pt")):
        m = _ckpt_re.search(os.path.basename(p))
        if m:
            ckpts.append((int(m.group(1)), p))
    # optional final
    if include_final:
        f = os.path.join(seed_dir, "final.pt")
        if os.path.exists(f):
            # put it at the end as a large epoch number sentinel
            ckpts.append((10**9, f))
    # sort by epoch
    ckpts.sort(key=lambda t: t[0])
    return ckpts


# ───────────────────────────── animation core ─────────────────────────────

def prepare_test_csvs(n_test: int) -> List[str]:
    paths = sorted(glob.glob(os.path.join(VAR_DIR, "testConfig_*.csv")))[:n_test]
    if not paths:
        raise FileNotFoundError(f"No testConfig_*.csv found in {VAR_DIR}")
    if len(paths) < n_test:
        print(f"[warn] only {len(paths)} test configs found in {VAR_DIR}")
    return paths


def build_frames(beta_key: str, head: str, n_test: int, include_final: bool) -> Tuple[List[int], Dict[int, List[np.ndarray]]]:
    """Collect per-frame (per-epoch) binned accuracies across seeds.

    Returns
    -------
    epochs_sorted : list of epoch identifiers (integers; 1e9 for final)
    acc_by_epoch  : {epoch: [acc_seed0, acc_seed1, ...]} where each acc_* is (n_bins,)
    """
    seed_ids = list_seeds(beta_key)
    if not seed_ids:
        raise RuntimeError(f"No seeds found under models/{beta_key}")
    csvs = prepare_test_csvs(n_test)

    # union of all epochs available across seeds
    epoch_to_accs: Dict[int, List[np.ndarray]] = {}

    for s in seed_ids:
        seed_dir = os.path.join(MODELS_DIR, beta_key, f"seed_{s}")
        ckpts = list_checkpoints(seed_dir, include_final)
        if not ckpts:
            print(f"[skip] {beta_key}/seed_{s}: no checkpoints found")
            continue
        for ep, ckpt in ckpts:
            try:
                model = load_model_for_ckpt(seed_dir, ckpt)
            except Exception as e:
                print(f"[warn] failed to load {ckpt}: {e}")
                continue
            hazards, rep_ok, haz_ok = eval_model_on_csvs(model, csvs)
            if head == "report":
                acc = bin_accuracy(hazards, rep_ok)
            else:
                acc = bin_accuracy(hazards, haz_ok)
            epoch_to_accs.setdefault(ep, []).append(acc)

    if not epoch_to_accs:
        raise RuntimeError(f"No evaluations produced for {beta_key} (head={head})")

    epochs_sorted = sorted(epoch_to_accs.keys())
    return epochs_sorted, epoch_to_accs


def animate_beta(beta_key: str, head: str, n_test: int, include_final: bool, fps: int, dpi: int) -> None:
    epochs_sorted, acc_by_epoch = build_frames(beta_key, head, n_test, include_final)

    color = COLOR_BY_BETA.get(beta_key, "tab:gray")
    out_dir = os.path.join(OUT_ROOT, beta_key)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlabel("True hazard (bin centers, width = 0.05)")
    ax.set_ylabel(("Report" if head == "report" else "Hazard") + "-head accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    title = ax.set_title("")
    lines_faint: List[plt.Line2D] = []  # one per seed in *current frame*
    line_mean, = ax.plot([], [], color=color, linewidth=2.6, label=f"{beta_key} mean")
    legend = ax.legend(frameon=False, loc="lower left")

    def init():
        # start empty
        for ln in lines_faint:
            ln.remove()
        lines_faint.clear()
        line_mean.set_data([], [])
        title.set_text("")
        return [line_mean, title, legend]

    def update(frame_idx: int):
        ep = epochs_sorted[frame_idx]
        acc_list = acc_by_epoch[ep]

        # clear previous faint lines
        for ln in lines_faint:
            ln.remove()
        lines_faint.clear()

        # draw faint per-seed lines
        for acc in acc_list:
            ln, = ax.plot(BIN_CENTERS, acc, color=color, alpha=0.28, linewidth=0.9)
            lines_faint.append(ln)

        # bold mean for this frame
        mean_arr = np.nanmean(np.vstack(acc_list), axis=0)
        line_mean.set_data(BIN_CENTERS, mean_arr)

        # nice title text
        ep_text = ("final" if ep >= 10**9 else f"ep {ep}")
        head_txt = "Report" if head == "report" else "Hazard"
        title.set_text(f"{beta_key} — {head_txt} head binned accuracy — {ep_text}")

        return [*lines_faint, line_mean, title, legend]

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(epochs_sorted), interval=1000//fps, blit=True
    )

    # Try ffmpeg → MP4; fall back to Pillow → GIF
    mp4_path = os.path.join(out_dir, f"{head}_binned_accuracy_anim.mp4")
    gif_path = os.path.join(out_dir, f"{head}_binned_accuracy_anim.gif")
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='animate_binned_accuracy.py'), bitrate=1800)
        anim.save(mp4_path, writer=writer, dpi=dpi)
        print(f"Saved {mp4_path}")
    except Exception as e:
        print(f"[info] ffmpeg not available or failed ({e}); saving GIF instead…")
        anim.save(gif_path, writer='pillow', dpi=dpi)
        print(f"Saved {gif_path}")
    finally:
        plt.close(fig)


# ───────────────────────────── CLI ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Animate binned accuracy by sweeping checkpoints.")
    ap.add_argument('--betas', nargs='+', default=['all'], help="List of beta_* keys or 'all' to auto-detect.")
    ap.add_argument('--head', choices=['report', 'hazard', 'both'], default='both', help="Which head to animate.")
    ap.add_argument('--n-test', type=int, default=1, help="Number of test CSVs from variants/beta_1p0 to use.")
    ap.add_argument('--include-final', action='store_true', help="Append final.pt as last frame if present.")
    ap.add_argument('--fps', type=int, default=2, help="Frames per second for the animation.")
    ap.add_argument('--dpi', type=int, default=200, help="DPI for saved animations.")
    args = ap.parse_args()

    betas = list_beta_keys(args.betas)
    if not betas:
        raise SystemExit("No beta folders found under ./models")

    for beta_key in betas:
        if args.head in ('report', 'both'):
            animate_beta(beta_key, head='report', n_test=args.n_test, include_final=args.include_final, fps=args.fps, dpi=args.dpi)
        if args.head in ('hazard', 'both'):
            animate_beta(beta_key, head='hazard', n_test=args.n_test, include_final=args.include_final, fps=args.fps, dpi=args.dpi)

    print("All requested animations complete.")


if __name__ == '__main__':
    main()

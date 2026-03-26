#!/usr/bin/env python3
# binned_accuracy.py — uniform-groups edition
# ===============================================================
# Evaluates report-head & hazard-head accuracy binned by TRUE hazard
# using the *new* uniform hazard group folders produced by
# TaskConfig_Generator_Continuous.py and trained by train_gen_hazard_MTS.py.
#
# Expected layout:
#   variants/
#     hz_0_0p4/
#     hz_0p6_1/
#     hz_edges_0_0p2_0p8_1/
#     hz_0p3_0p7/
#     hz_flat_0_1/    <-- used as a fixed test set in this script
#   models_MTS/
#     <group_key>/seed_<k>/{final.pt, hp.json}
#
# Figures and a CSV of mean curves are written to ./figures/.
#
# Notes:
# - Hazard head is treated as *binary*: prediction = sigmoid(logits) > 0.5
#   and truth = (trueHazard > 0.5).
# - Report-head accuracy uses the LAST timestep's logit vs trueReport (0/1).
#
# If you trained into a different models folder, update MODELS_DIR below.

import os, glob, json, ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from rnn_models import GRUModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# Updated to the new training output dir (with a fallback)
MODELS_DIR_PRIMARY = os.path.join(BASE_DIR, "models_OTS")
MODELS_DIR_FALLBACK = os.path.join(BASE_DIR, "models")
MODELS_DIR = MODELS_DIR_PRIMARY if os.path.isdir(MODELS_DIR_PRIMARY) else MODELS_DIR_FALLBACK

# Use the flat group (0.0–1.0) as a consistent test distribution
VAR_DIR    = os.path.join(BASE_DIR, "variants", "hz_flat_0_1")
OUT_DIR    = os.path.join(BASE_DIR, "figures_OTS")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TEST_CFGS = 50  # number of test CSVs to use from the fixed test set
SEEDS       = range(10)  # default training seeds; skip silently if missing
BIN_WIDTH   = 0.05
BIN_EDGES   = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)  # 21 edges
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])     # 20 bins

# New uniform hazard groups: (folder_slug, pretty_label, color)
GROUP_SPECS = [
    ("hz_0_0p4",             "Uniform 0.0–0.4",           "tab:blue"),
    ("hz_0p6_1",             "Uniform 0.6–1.0",           "tab:orange"),
    ("hz_edges_0_0p2_0p8_1", "Edges 0.0–0.2 ∪ 0.8–1.0",   "tab:green"),
    ("hz_0p3_0p7",           "Uniform 0.3–0.7",           "tab:red"),
    ("hz_flat_0_1",          "Uniform 0.0–1.0 (flat)",    "tab:purple"),
]
LABEL_BY_FOLDER = {k: lbl for k,lbl,_ in GROUP_SPECS}
COLOR_BY_FOLDER = {k: col for k,_,col in GROUP_SPECS}

def test_csvs():
    paths = sorted(glob.glob(os.path.join(VAR_DIR, "testConfig_*.csv")))[:N_TEST_CFGS]
    if len(paths) < N_TEST_CFGS:
        print(f"[warn] only {len(paths)} test configs found in {VAR_DIR}")
    return paths

def load_hp(seed_dir):
    with open(os.path.join(seed_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    hp.setdefault("n_input", 1)
    hp.setdefault("n_rnn", 128)
    return hp

def load_model(group_dir, seed):
    seed_dir = os.path.join(MODELS_DIR, group_dir, f"seed_{seed}")
    ckpt = os.path.join(seed_dir, "final.pt")  # final.pt only
    if not (os.path.exists(ckpt) and os.path.exists(os.path.join(seed_dir, "hp.json"))):
        return None
    hp = load_hp(seed_dir)
    model = GRUModel(hp).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

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

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y_rep[i], self.haz[i]

def collate(batch):
    xs, yr, hz = zip(*batch)
    return torch.stack(xs, 0), torch.stack(yr, 0), torch.stack(hz, 0)

@torch.no_grad()
def eval_model_on_csvs(model, csvs):
    hazards, rep_ok, haz_ok = [], [], []
    for p in csvs:
        df = pd.read_csv(p)
        dl = DataLoader(HelicopterEvalDS(df), batch_size=256, shuffle=False, collate_fn=collate)
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

def bin_accuracy(hazards, correct_mask):
    idx = np.digitize(hazards, BIN_EDGES) - 1
    idx = np.clip(idx, 0, len(BIN_EDGES)-2)
    total = np.zeros(len(BIN_CENTERS), int)
    good  = np.zeros(len(BIN_CENTERS), int)
    for i, ok in zip(idx, correct_mask):
        total[i] += 1
        if ok: good[i] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(total > 0, good / total, np.nan)
    return acc, total, good

def save_fig(fig, filename_base):
    png = os.path.join(OUT_DIR, f"{filename_base}.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")

def main():
    csvs = test_csvs()

    # Collect per-seed accuracy arrays for each group in fixed order
    rep_acc, haz_acc = {}, {}
    for folder, _, _ in GROUP_SPECS:
        model_path = os.path.join(MODELS_DIR, folder)
        if not os.path.isdir(model_path):
            print(f"[warn] missing model folder: {folder} under {MODELS_DIR} — skipping")
            continue
        seeds_dict_rep, seeds_dict_haz = {}, {}
        any_seed = False
        for s in SEEDS:
            model = load_model(folder, s)
            if model is None:
                # silent skip if checkpoint missing for this seed
                continue
            any_seed = True
            hz, rep_ok, haz_ok = eval_model_on_csvs(model, csvs)
            acc_rep, _, _ = bin_accuracy(hz, rep_ok)
            acc_haz, _, _ = bin_accuracy(hz, haz_ok)
            seeds_dict_rep[s] = acc_rep
            seeds_dict_haz[s] = acc_haz
        if any_seed:
            rep_acc[folder] = seeds_dict_rep
            haz_acc[folder] = seeds_dict_haz

    # --- absolute accuracy plots (per-seed faint + bold mean) ---
    # Report head
    fig1, ax1 = plt.subplots(figsize=(9, 5.2))
    for folder, label, color in GROUP_SPECS:
        if folder not in rep_acc: continue
        seeds_dict = rep_acc[folder]
        for arr in seeds_dict.values():
            ax1.plot(BIN_CENTERS, arr, color=color, alpha=0.30, linewidth=0.9)
        mean_arr = np.nanmean(np.vstack(list(seeds_dict.values())), axis=0)
        ax1.plot(BIN_CENTERS, mean_arr, color=color, linewidth=2.6, label=label)
    ax1.set_xlabel("True hazard (bin centers, width = 0.05)")
    ax1.set_ylabel("Report-head accuracy")
    ax1.set_title("Accuracy vs hazard — tested on hz_flat_0_1 (first 20 configs), seeds as available")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, ncol=2)
    fig1.tight_layout()
    save_fig(fig1, "report_head_accuracy_by_hazard_uniform")

    # Hazard head
    fig2, ax2 = plt.subplots(figsize=(9, 5.2))
    for folder, label, color in GROUP_SPECS:
        if folder not in haz_acc: continue
        seeds_dict = haz_acc[folder]
        for arr in seeds_dict.values():
            ax2.plot(BIN_CENTERS, arr, color=color, alpha=0.30, linewidth=0.9)
        mean_arr = np.nanmean(np.vstack(list(seeds_dict.values())), axis=0)
        ax2.plot(BIN_CENTERS, mean_arr, color=color, linewidth=2.6, label=label)
    ax2.set_xlabel("True hazard (bin centers, width = 0.05)")
    ax2.set_ylabel("Hazard-head accuracy (threshold 0.5)")
    ax2.set_title("Hazard-head vs hazard — tested on hz_flat_0_1 (first 20 configs), seeds as available")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, ncol=2)
    fig2.tight_layout()
    save_fig(fig2, "hazard_head_accuracy_by_hazard_uniform")

    # --- delta vs flat baseline (baseline mean) ---
    baseline_key = "hz_flat_0_1"
    if baseline_key not in rep_acc or baseline_key not in haz_acc:
        print("[warn] hz_flat_0_1 not available; skipping delta plots.")
    else:
        base_rep_mean = np.nanmean(np.vstack(list(rep_acc[baseline_key].values())), axis=0)
        base_haz_mean = np.nanmean(np.vstack(list(haz_acc[baseline_key].values())), axis=0)

        # Report deltas
        fig3, ax3 = plt.subplots(figsize=(9, 5.2))
        for folder, label, color in GROUP_SPECS:
            if folder == baseline_key or folder not in rep_acc: continue
            seeds_dict = rep_acc[folder]
            for arr in seeds_dict.values():
                ax3.plot(BIN_CENTERS, arr - base_rep_mean, color=color, alpha=0.30, linewidth=0.9)
            mean_arr = np.nanmean(np.vstack(list(seeds_dict.values())), axis=0)
            ax3.plot(BIN_CENTERS, mean_arr - base_rep_mean, color=color, linewidth=2.6, label=label)
        ax3.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.6)
        ax3.set_xlabel("True hazard (bin centers, width = 0.05)")
        ax3.set_ylabel("Δ Report accuracy vs Uniform 0.0–1.0")
        ax3.set_title("Report-head: accuracy difference relative to Uniform 0.0–1.0")
        ax3.grid(True, alpha=0.3)
        ax3.legend(frameon=False, ncol=2)
        fig3.tight_layout()
        save_fig(fig3, "report_head_accuracy_delta_vs_flat")

        # Hazard deltas
        fig4, ax4 = plt.subplots(figsize=(9, 5.2))
        for folder, label, color in GROUP_SPECS:
            if folder == baseline_key or folder not in haz_acc: continue
            seeds_dict = haz_acc[folder]
            for arr in seeds_dict.values():
                ax4.plot(BIN_CENTERS, arr - base_haz_mean, color=color, alpha=0.30, linewidth=0.9)
            mean_arr = np.nanmean(np.vstack(list(seeds_dict.values())), axis=0)
            ax4.plot(BIN_CENTERS, mean_arr - base_haz_mean, color=color, linewidth=2.6, label=label)
        ax4.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.6)
        ax4.set_xlabel("True hazard (bin centers, width = 0.05)")
        ax4.set_ylabel("Δ Hazard-head accuracy vs Uniform 0.0–1.0")
        ax4.set_title("Hazard-head: accuracy difference relative to Uniform 0.0–1.0")
        ax4.grid(True, alpha=0.3)
        ax4.legend(frameon=False, ncol=2)
        fig4.tight_layout()
        save_fig(fig4, "hazard_head_accuracy_delta_vs_flat")

    # Optional CSV of mean lines
    rows = []
    for folder, label, _ in GROUP_SPECS:
        if folder not in rep_acc: continue
        mean_rep = np.nanmean(np.vstack(list(rep_acc[folder].values())), axis=0)
        mean_haz = np.nanmean(np.vstack(list(haz_acc[folder].values())), axis=0)
        for i, c in enumerate(BIN_CENTERS):
            rows.append({"group_folder": folder, "label": label,
                         "bin_center": float(c),
                         "report_acc_mean": float(mean_rep[i]),
                         "hazard_acc_mean": float(mean_haz[i])})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "binned_acc_uniform_means.csv"), index=False)
    print("Saved means CSV -> figures/binned_acc_uniform_means.csv")

if __name__ == "__main__":
    main()

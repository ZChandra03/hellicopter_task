#!/usr/bin/env python3
"""
hist_errors_models_test.py  –  v4  (July 2025)

Evaluate every checkpoint under ./models_test/ and plot hazard‑error
histograms.  The figure footer lists **separate mean BCE losses** for
the report head, hazard head, and their composite total for *both* the
network and the Bayesian normative observer.

CLI
----
--ckpt <file.pt>   exact checkpoint file in each folder   [default: checkpoint_best.pt]
--max_csvs N       max unsorted test CSVs to load         [default: 20 | "None" → all]
"""
import os, glob, ast, argparse, math
from typing import List, Tuple, Type
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from rnn_models     import GRUModel, LSTMModel, RNNModel
from NormativeModel import BayesianObserver                                   # ← new

# ───────────────────────── CLI ─────────────────────────────────────────────
cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli.add_argument("--ckpt", default="checkpoint_best.pt",
                 help="Checkpoint file name expected in every model folder")
cli.add_argument("--max_csvs", default="20",
                 help='Cap on test CSVs ("None" → use all)')
args = cli.parse_args()
MAX_CSVS = None if args.max_csvs.lower() == "none" else int(args.max_csvs)

# ───────────────── paths & constants ───────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(HERE, "models_test")
CSV_GLOB    = os.path.join(HERE, "variants", "test", "unsorted", "test_unsorted_*.csv")

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS     = 1e-10
HS_GRID = np.arange(0, 1.05, 0.05)
MU1, MU2 = -1, 1                           # as in training / evaluate_3_max

# ───────────────── helper funcs ────────────────────────────────────────────
def walk_models(root: str, ckpt_name: str):
    for dirpath, _, files in os.walk(root):
        if ckpt_name in files:
            tag = os.path.relpath(dirpath, root).replace(os.sep, "_")
            yield dirpath, tag

def guess_cls(tag: str) -> Type[torch.nn.Module]:
    t = tag.lower()
    if "lstm" in t: return LSTMModel
    if "rnn" in t and "lstm" not in t: return RNNModel
    return GRUModel

def list_csvs(max_n: int | None) -> List[str]:
    paths = sorted(glob.glob(CSV_GLOB))
    return paths if max_n is None else paths[:max_n]

def clip01(x: float) -> float:
    return float(np.clip(x, EPS, 1 - EPS))

def bce(p: float, y: float) -> float:
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def default_hp() -> dict:
    return {"n_input": 1, "n_rnn": 128}

# ───────────────── evaluation core ─────────────────────────────────────────
def collect(model: torch.nn.Module):
    err_net, err_bay = [], []

    # loss arrays: [rep, haz, total]
    L_net_rep, L_net_haz, L_net_tot = [], [], []
    L_bay_rep, L_bay_haz, L_bay_tot = [], [], []

    for csv in list_csvs(MAX_CSVS):
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            sigma    = float(row.get("sigma", row.get("noise", 0.0)))
            haz_true = float(row["trueHazard"])
            rep_true = int(row["trueReport"])
            y_rep    = 0.5 * (rep_true + 1)             # −1/+1 → 0/1

            # Bayesian ideal observer --------------------------------------------------
            L_haz, L_state, _rep_norm, _ = BayesianObserver(evid, MU1, MU2, sigma, HS_GRID.copy())
            p_state2_bay = clip01(L_state[1, -1])
            haz_est_bay  = float(np.dot(HS_GRID, L_haz[:, -1]))
            p_haz_bay    = clip01(haz_est_bay)

            loss_rep_bay = bce(p_state2_bay, y_rep)
            loss_haz_bay = bce(p_haz_bay, haz_true)
            loss_tot_bay = loss_rep_bay + 0.5 * loss_haz_bay

            err_bay.append(haz_est_bay - haz_true)
            L_bay_rep.append(loss_rep_bay)
            L_bay_haz.append(loss_haz_bay)
            L_bay_tot.append(loss_tot_bay)

            # Network prediction -------------------------------------------------------
            x = torch.tensor(evid, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                loc_logits, haz_logits = model(x)
            loc_last = loc_logits[0, -1, 0]
            p_state2_net = clip01(torch.sigmoid(loc_last).item())
            p_haz_net    = clip01(torch.sigmoid(haz_logits)[0, 0].item())

            loss_rep_net = bce(p_state2_net, y_rep)
            loss_haz_net = bce(p_haz_net, haz_true)
            loss_tot_net = loss_rep_net + 0.5 * loss_haz_net

            err_net.append(p_haz_net - haz_true)
            L_net_rep.append(loss_rep_net)
            L_net_haz.append(loss_haz_net)
            L_net_tot.append(loss_tot_net)

    return (np.asarray(err_net), np.asarray(err_bay),
            np.asarray(L_net_rep), np.asarray(L_net_haz), np.asarray(L_net_tot),
            np.asarray(L_bay_rep), np.asarray(L_bay_haz), np.asarray(L_bay_tot))

# ───────────────── plotting ───────────────────────────────────────────────
def plot_hist(err_net, err_bay,
              Lnr, Lnh, Lnt, Lbr, Lbh, Lbt,
              tag, ckpt_name, out_dir="."):
    bins = np.arange(-1.0, 1.0001, 0.05)
    plt.figure(figsize=(8, 4.5))
    plt.hist(err_bay, bins=bins, alpha=0.6, label="Bayesian", edgecolor="black")
    plt.hist(err_net, bins=bins, alpha=0.6, label=tag,       edgecolor="black")
    plt.axvline(0, color="k", lw=1)

    stem = os.path.splitext(ckpt_name)[0]
    plt.title(f"{tag} — {stem}  (hazard‑error histogram)")
    plt.xlabel("Prediction error  (pred − true hazard)")
    plt.ylabel("Count")

    foot = (f"mean BCE loss  →  Net  (rep={Lnr.mean():.4f}, haz={Lnh.mean():.4f}, "
            f"tot={Lnt.mean():.4f})   |   "
            f"Bayes (rep={Lbr.mean():.4f}, haz={Lbh.mean():.4f}, tot={Lbt.mean():.4f})")
    plt.figtext(0.99, 0.02, foot, ha="right", fontsize=8)

    plt.legend(); plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hist_{tag}_{stem}.png")
    plt.savefig(out_path, dpi=180)
    print(f"[✓] saved {out_path}")

# ───────────────── driver ────────────────────────────────────────────────
def main():
    for folder, tag in walk_models(MODELS_DIR, args.ckpt):
        model_cls = guess_cls(tag)
        model = model_cls(default_hp()).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(folder, args.ckpt),
                                         map_location=DEVICE))
        model.eval()

        print(f"[→] evaluating {tag} ({args.ckpt}) …")
        e_net, e_bay, LnR, LnH, LnT, LbR, LbH, LbT = collect(model)
        if e_net.size:
            plot_hist(e_net, e_bay, LnR, LnH, LnT, LbR, LbH, LbT,
                      tag, args.ckpt)
        else:
            print(f"[warn] no trials for {tag}")

if __name__ == "__main__":
    main()

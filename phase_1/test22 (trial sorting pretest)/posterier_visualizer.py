#!/usr/bin/env python3
"""
viz_fixed_params.py – quick visualiser for *one* σ and hazard h
================================================================
"""
import os, copy
import numpy as np
import matplotlib.pyplot as plt

import TaskConfig_Generator as TCG          # provides genEvidence + default params :contentReference[oaicite:0]{index=0}
from NormativeModel import BayesianObserver  # normative observer                  :contentReference[oaicite:1]{index=1}

# ─────────────────────────────── globals ──────────────────────────────────────
SIGMA   = 1      # <-- change me (evidence noise σ)
HAZARD  = 0.20      # <-- change me (switch probability h per step)
N_TRIAL = 10        # number of trials plotted

HS_GRID = np.arange(0, 1.05, 0.05)          # must match training conventions
MU1, MU2 = -1, 1                            # latent means used by the observer

# ─────────────────────────────── main ─────────────────────────────────────────
def main():
    # 1) generate fresh trials from the task generator
    base_params = copy.deepcopy(TCG.params)                               # keeps generator state intact
    trials = [
        TCG.genEvidence(HAZARD, SIGMA, base_params)  # returns (evidence, hidden_states)
        for _ in range(N_TRIAL)
    ]                                                                      # :contentReference[oaicite:2]{index=2}

    # 2) build a tall figure – one axis per trial
    fig, axes = plt.subplots(N_TRIAL, 1,
                             figsize=(6, 1.6 * N_TRIAL),
                             sharex=True)

    # 3) loop & visualise
    for k, (evid, _) in enumerate(trials):
        # run normative observer
        L_haz, _, _, _ = BayesianObserver(evid, MU1, MU2, SIGMA, HS_GRID.copy())   # :contentReference[oaicite:3]{index=3}

        L_haz = L_haz[:, 1:]                     # ★ skip prior column – removes the “big box”
        n_step = L_haz.shape[1]

        ax = axes[k]
        im = ax.imshow(
            L_haz,
            aspect="auto",
            origin="lower",
            interpolation="nearest",            # ★ ensures each index is one crisp column
            extent=[-0.5, n_step-0.5, HS_GRID[0], HS_GRID[-1]]  # ★ centres pixels on integers
        )

        # ticks: every evidence index + every hazard slice
        ax.set_xticks(np.arange(n_step))         # ★ every time-step visible
        #ax.set_yticks(HS_GRID)                   # optional – comment out if too dense
        ax.set_ylabel("hazard h")
        ax.set_title(f"Trial {k+1}  (h={HAZARD:.2f}, σ={SIGMA})")

    axes[-1].set_xlabel("evidence index t")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

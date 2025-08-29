
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig_Generator_Continuous.py
==================================
Generate *continuous‑hazard* configs for several symmetric Beta(x, x) priors.

Compared to the legacy *Beta_Sweep* script this version **does not snap**
sampled hazards onto a fixed grid – every trial keeps its raw Beta draw
``trueHazard ∈ (0, 1)``.

For each *x* in ``BETA_LIST`` the script:

1. sets ``params['hazard_beta'] = (x, x)`` (x = 1 → uniform prior);
2. writes ``TRAIN_VARIANTS`` train CSVs ‑‑ and, if requested,
   ``TEST_VARIANTS`` test CSVs ‑‑ into ``variants/beta_<x>/``
   where the dot is replaced by ``'p'`` for filesystem safety.

Run once → entire folder structure ready for training.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats as spst

# ───────────────────────── global experiment parameters ──────────────────────────
params: Dict[str, Any] = {
    # core trial settings
    'nTrials'       : 300,      # number of trials per (single‑block) file
    'nEvidence'     : 20,       # length of evidence sequence
    'xLim'          : 5,        # evidence truncation bounds ±xLim
    'Mu'            : 1,        # latent mean magnitude

    # Beta prior placeholder (overwritten inside the loop)
    'hazard_beta'   : (1.0, 1.0),

    # block settings
    'testSigmas'    : [1],      # per‑block evidence σ (len must equal block_list)
    'block_list'    : ['single'],

    # I/O
    'train_variants': 500,      # how many train CSVs per Beta prior
    'test_variants' : 50,       # how many   test CSVs per Beta prior

    # CSV column order (kept for legacy compatibility)
    'trial_fields'  : [
        'blockNum', 'blockDifficulty', 'sigma', 'trialInBlock',
        'trueHazard', 'evidence', 'states', 'trueReport', 'truePredict'
    ],
}

# ───────────────────────── helpers ──────────────────────────
def genEvidence(hz: float, sigma: float, p: Dict[str, Any]):  # -> (evidence, states)
    """Simulate one evidence sequence for a *single* trial.

    Parameters
    ----------
    hz : float
        Hazard rate for this trial (probability of flipping the latent mean
        after *each* evidence sample).
    sigma : float
        Observation noise σ (0 → deterministic evidence).
    p : Dict[str, Any]
        Convenience handle to global *params* dict.
    """
    mu_val = float(p['Mu']) * (1 if np.random.rand() > 0.5 else -1)
    ev, st = [], []
    for _ in range(p['nEvidence']):
        lw, hg = -p['xLim'], p['xLim']
        if sigma > 0:
            sample = float(spst.truncnorm((lw - mu_val)/sigma, (hg - mu_val)/sigma,
                                          mu_val, sigma).rvs())
        else:
            sample = float(mu_val)
        ev.append(sample)
        st.append(mu_val)
        if np.random.rand() < hz:      # latent sign flip
            mu_val = -mu_val
    return ev, st

def genBlockHazards(p: Dict[str, Any]):
    """Draw one hazard value per trial from the Beta prior (no snapping)."""
    nT = int(p['nTrials'])
    a, b = p['hazard_beta']
    return spst.beta.rvs(a, b, size=nT).tolist()

def makeBlockTrials(p: Dict[str, Any]):
    trials: List[Dict[str, Any]] = []
    for b, (blk, sig) in enumerate(zip(p['block_list'], p['testSigmas'])):
        for i, hz in enumerate(genBlockHazards(p), 1):
            ev, st = genEvidence(hz, sig, p)
            trials.append({
                'blockNum'      : b,
                'blockDifficulty': blk,
                'sigma'         : sig,
                'trialInBlock'  : i,
                'trueHazard'    : hz,
                'evidence'      : ev,
                'states'        : st,
                'trueReport'    : st[-1],
                'truePredict'   : -1 if hz < 0.5 else 1,   # for quick visual sanity‑checks
            })
    return trials

def export_variants(save_dir: str, p: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    for k in range(p['train_variants']):
        pd.DataFrame(makeBlockTrials(p)).to_csv(
            os.path.join(save_dir, f'trainConfig_{k}.csv'), index=False)
    for k in range(p['test_variants']):
        pd.DataFrame(makeBlockTrials(p)).to_csv(
            os.path.join(save_dir, f'testConfig_{k}.csv'), index=False)

    # save a lightweight *TaskConfig.csv* for bookkeeping / provenance
    pd.Series({k: str(v) for k, v in p.items()}).to_csv(
        os.path.join(save_dir, 'TaskConfig.csv'))

# ───────────────────────── batch generation loop ──────────────────────────
BETA_LIST = [0.1, 0.5, 1.0, 2.0, 10.0]   # symmetric Beta(x,x)

ROOT_VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variants')

if __name__ == '__main__':
    for x in BETA_LIST:
        params['hazard_beta'] = (x, x)
        sub = f"beta_{str(x).replace('.', 'p')}"       # e.g. beta_0p1
        params['saveDir'] = os.path.join(ROOT_VARIANTS_DIR, sub)

        print(f"[TCG] Beta({x},{x})  →  {params['saveDir']}  (train={params['train_variants']})")
        export_variants(params['saveDir'], params)

    print('[TCG] All done.')

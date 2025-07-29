#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig_Generator_Beta.py
===========================
Generate hazard‑grid configs for **several symmetric Beta(x, x) priors** in one go.
For each x in ``BETA_LIST`` the script:

1.  sets ``params['hazard_beta'] = (x, x)`` (with x = 1 giving the uniform prior);
2.  writes the CSVs into a dedicated sub‑folder
    ``variants/beta_<x>/`` (dots → “p” for filesystem safety, e.g. beta_0p1).

Adjust ``TRAIN_VARIANTS`` / ``TEST_VARIANTS`` if you need more or fewer files.
Run it once – everything is created.
"""

from __future__ import annotations

import os, numpy as np, pandas as pd, scipy.stats as spst
from typing import Dict, Any, Tuple, List

# ───────────────────────── global experiment parameters ──────────────────────────
params: Dict[str, Any] = {
    # core trial settings
    'nTrials': 300,
    'nEvidence': 20,
    'xLim': 5,
    'Mu': 1,

    # hazard grid
    'HazRes': 0.05,
    # filled below after HazRes known

    # beta prior placeholder (will be overwritten in loop)
    'hazard_beta': (1.0, 1.0),

    # block settings
    'testSigmas': [1],
    'block_list': ['single'],

    # will set saveDir dynamically per beta
    'train_variants': 500,
    'test_variants': 50,

    # CSV column order
    'trial_fields': [
        'blockNum', 'blockDifficulty', 'sigma', 'trialInBlock',
        'trueHazard', 'evidence', 'states', 'trueReport', 'truePredict'
    ],
}
# compute Hazards grid once
params['Hazards'] = np.round(np.arange(0, 1 + params['HazRes'] / 2, params['HazRes']), 2)

# legacy discrete path left off
HAZARDS_CUSTOM, PROBS_CUSTOM = [], []
GRID, RES = params['Hazards'], params['HazRes']

# ───────────────────────── helpers ──────────────────────────

def _snap_to_grid(values: np.ndarray) -> np.ndarray:
    idx = np.clip(np.rint(values / RES).astype(int), 0, len(GRID) - 1)
    return GRID[idx]

def genEvidence(hz: float, sigma: float, p: Dict[str, Any]):
    mu_val = float(p['Mu']) * (1 if np.random.rand() > 0.5 else -1)
    ev, st = [], []
    for _ in range(p['nEvidence']):
        lw, hg = -p['xLim'], p['xLim']
        if sigma > 0:
            sample = float(spst.truncnorm((lw - mu_val)/sigma, (hg - mu_val)/sigma, mu_val, sigma).rvs())
        else:
            sample = float(mu_val)
        ev.append(sample)
        st.append(mu_val)
        if np.random.rand() < hz:
            mu_val = -mu_val
    return ev, st

def genBlockHazards(p: Dict[str, Any]):
    nT = int(p['nTrials'])
    a, b = p['hazard_beta']
    raw = spst.beta.rvs(a, b, size=nT)
    return _snap_to_grid(raw).tolist()

def makeBlockTrials(p: Dict[str, Any]):
    trials: List[Dict[str, Any]] = []
    for b, (blk, sig) in enumerate(zip(p['block_list'], p['testSigmas'])):
        for i, hz in enumerate(genBlockHazards(p), 1):
            ev, st = genEvidence(hz, sig, p)
            trials.append({
                'blockNum': b,
                'blockDifficulty': blk,
                'sigma': sig,
                'trialInBlock': i,
                'trueHazard': hz,
                'evidence': ev,
                'states': st,
                'trueReport': st[-1],
                'truePredict': -1 if hz < 0.5 else 1,
            })
    return trials

def export_variants(save_dir: str, p: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    for k in range(p['train_variants']):
        pd.DataFrame(makeBlockTrials(p)).to_csv(os.path.join(save_dir, f'trainConfig_{k}.csv'), index=False)
    if p['test_variants']:
        for k in range(p['test_variants']):
            pd.DataFrame(makeBlockTrials(p)).to_csv(os.path.join(save_dir, f'testConfig_{k}.csv'), index=False)
    pd.Series({k: str(v) for k, v in p.items()}).to_csv(os.path.join(save_dir, 'TaskConfig.csv'))

# ───────────────────────── batch generation loop ──────────────────────────
BETA_LIST = [0.1, 0.5, 1.0, 2.0, 10.0]  # symmetric Beta(x,x)

ROOT_VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variants')

if __name__ == '__main__':
    for x in BETA_LIST:
        params['hazard_beta'] = (x, x)
        sub = f"beta_{str(x).replace('.', 'p')}"  # e.g. beta_0p1
        params['saveDir'] = os.path.join(ROOT_VARIANTS_DIR, sub)
        print(f"[TaskConfig_Gen] Beta({x},{x}) → {params['saveDir']}  (train={params['train_variants']})")
        export_variants(params['saveDir'], params)
    print('[TaskConfig_Gen] All done.')

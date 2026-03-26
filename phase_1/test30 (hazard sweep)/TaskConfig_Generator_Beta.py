#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig Generator — report + predict target with **grid‑snapped Beta hazards**
===============================================================================
Generates CSV config files where each trial’s hazard rate *h* is first sampled
from a Beta(α, β) prior and **then snapped exactly to the nearest value in a
fixed 0.05‑spaced grid**.  The snapping is now index‑based, so we never hit
binary floating‑point artefacts like 0.15000000000000002.

Usage
-----
▶ Edit `params[...]` below to suit your experiment (e.g. `train_variants`).  
▶ Run `python TaskConfig_Generator_Trial.py` – that’s it.
"""

# %% -------------------------------------------------------------------
# Imports
from __future__ import annotations

import os
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import scipy.stats as spst

# %% -------------------------------------------------------------------
# Experiment‑wide parameters --------------------------------------------------------------
params: Dict[str, Any] = {}

# — Core trial settings —
params.update({
    'nTrials'  : 300,    # trials per block
    'nEvidence': 20,     # evidential samples per trial
    'xLim'     : 5,      # truncation limit of evidence
    'Mu'       : 1,      # latent mean magnitude
})

# — Hazard grid definition (0 → 1 inclusive, step 0.05) —
params['HazRes']  = 0.05
params['Hazards'] = np.round(
    np.arange(0, 1 + params['HazRes'] / 2, params['HazRes']), 2
)  # round 2 dp so 0.3 not 0.30000000000000004

# — Continuous Beta prior parameters —
# type: Tuple[float, float] | None
params['hazard_beta'] = (1,1)  # (α, β) > 0

# — Discrete custom hazard‑probabilities (legacy; set to None to ignore) —
params['hazard_probs'] = None

# — Evidence noise (σ) values per difficulty block —
params['testSigmas']  = [1]
params['block_list']  = ['single']

# — Output / variants —
script_dir               = os.path.dirname(os.path.abspath(__file__))
params['saveDir']        = os.path.join(script_dir, 'variants')
os.makedirs(params['saveDir'], exist_ok=True)
params['train_variants'] = 10
params['test_variants']  = 0

# — Trial CSV columns (order preserved) —
params['trial_fields'] = [
    'blockNum', 'blockDifficulty', 'sigma', 'trialInBlock',
    'trueHazard', 'evidence', 'states', 'trueReport', 'truePredict'
]

# %% -------------------------------------------------------------------
# Prepare discrete custom hazards (legacy) ------------------------------------------

def _prep_hazard_distribution(p: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    hp = p.get('hazard_probs')
    if not isinstance(hp, dict):
        return [], []
    if not np.isclose(sum(hp.values()), 1.0):
        raise ValueError('hazard_probs must sum to 1.')
    hazards = np.array(list(hp.keys()), dtype=float)
    probs   = np.array(list(hp.values()), dtype=float)
    return hazards.tolist(), probs.tolist()

HAZARDS_CUSTOM, PROBS_CUSTOM = _prep_hazard_distribution(params)
GRID = params['Hazards']      # shorthand
RES  = params['HazRes']

# %% -------------------------------------------------------------------
# Evidence generation ---------------------------------------------------------------

def genEvidence(hz: float, sigma: float, p: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    mu_val = float(p['Mu']) * (1 if np.random.rand() > 0.5 else -1)
    evidence, states = [], []
    for _ in range(p['nEvidence']):
        lw, hg = -p['xLim'], p['xLim']
        if sigma > 0:
            sample = float(spst.truncnorm((lw - mu_val)/sigma, (hg - mu_val)/sigma, mu_val, sigma).rvs())
        else:
            sample = float(mu_val)
        evidence.append(sample)
        states.append(mu_val)
        if np.random.rand() < hz:
            mu_val = -mu_val
    return evidence, states

# %% -------------------------------------------------------------------
# Hazard list for a block -----------------------------------------------------------

def _snap_to_grid(values: np.ndarray) -> np.ndarray:
    """Return grid values nearest to *values* (index lookup avoids FP drift)."""
    idx = np.clip(np.rint(values / RES).astype(int), 0, len(GRID) - 1)
    return GRID[idx]

def genBlockHazards(p: Dict[str, Any]) -> List[float]:
    nT = int(p['nTrials'])

    # 1) Continuous Beta prior → snapped to grid
    if p.get('hazard_beta') is not None:
        a, b = p['hazard_beta']
        if a <= 0 or b <= 0:
            raise ValueError('hazard_beta parameters must be positive.')
        raw = spst.beta.rvs(a, b, size=nT)
        return _snap_to_grid(raw).tolist()

    # 2) Custom discrete distribution (legacy)
    if HAZARDS_CUSTOM:
        return np.random.choice(HAZARDS_CUSTOM, size=nT, p=PROBS_CUSTOM).tolist()

    # 3) Uniform grid fallback (equal counts)
    reps = nT // len(GRID)
    rest = nT - reps * len(GRID)
    return np.concatenate([np.repeat(GRID, reps), np.random.choice(GRID, rest)]).tolist()

# %% -------------------------------------------------------------------
# Assemble trials --------------------------------------------------------------

def makeBlockTrials(p: Dict[str, Any]) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    for b, (blockName, sigma) in enumerate(zip(p['block_list'], p['testSigmas'])):
        for i, hz in enumerate(genBlockHazards(p), 1):
            evidence, states = genEvidence(hz, sigma, p)
            trials.append({
                'blockNum'       : b,
                'blockDifficulty': blockName,
                'sigma'          : sigma,
                'trialInBlock'   : i,
                'trueHazard'     : hz,
                'evidence'       : evidence,
                'states'         : states,
                'trueReport'     : states[-1],
                'truePredict'    : -1 if hz < 0.5 else 1,
            })
    return trials

# %% -------------------------------------------------------------------
# CSV export ----------------------------------------------------------------

def export_variants(save_dir: str | None = None,
                    train_variants: int | None = None,
                    test_variants: int | None = None,
                    p: Dict[str, Any] = params) -> None:
    save_dir = save_dir or p['saveDir']
    os.makedirs(save_dir, exist_ok=True)
    n_train = p['train_variants'] if train_variants is None else train_variants
    n_test  = p['test_variants']  if test_variants  is None else test_variants

    # TRAIN configs
    for k in range(n_train):
        pd.DataFrame(makeBlockTrials(p)).to_csv(os.path.join(save_dir, f'trainConfig_{k}.csv'), index=False)

    # TEST configs
    for k in range(n_test):
        pd.DataFrame(makeBlockTrials(p)).to_csv(os.path.join(save_dir, f'testConfig_{k}.csv'), index=False)

    # provenance
    pd.Series({k: str(v) for k, v in p.items()}).to_csv(os.path.join(save_dir, 'TaskConfig.csv'))

# %% -------------------------------------------------------------------
# Main ---------------------------------------------------------------------

if __name__ == '__main__':
    print(f"[TaskConfig_Generator] Exporting {params['train_variants']} train + "
          f"{params['test_variants']} test variants → {params['saveDir']}")
    export_variants()
    print('[TaskConfig_Generator] Done.')

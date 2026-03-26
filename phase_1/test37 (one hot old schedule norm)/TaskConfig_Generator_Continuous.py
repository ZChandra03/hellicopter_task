#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig_Generator_Continuous.py (enhanced)
============================================
Generate *continuous‑hazard* configs for several symmetric Beta(x, x) priors
**and** append the binary decisions from the Bayesian normative observer.

Adds two new CSV columns per trial:
  - ``normReport``  ∈ {−1, +1}  (Bayesian report decision after all evidence)
  - ``normPredict`` ∈ {−1, +1}  (Bayesian stay/switch prediction; −1=stay, +1=switch)

You can toggle normative computation via ``params['add_normative']``.
The normative model is imported from ``NormativeModel.py`` and evaluated on
``params['hazard_grid']``.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as spst

# ───────────────────────── global experiment parameters ──────────────────────────
params: Dict[str, Any] = {
    # core trial settings
    'nTrials'       : 300,      # number of trials per (single‑block) file
    'nEvidence'     : 20,       # length of evidence sequence
    'xLim'          : 5,        # evidence truncation bounds ±xLim
    'Mu'            : 1,        # latent mean magnitude (±Mu are the two states)

    # Beta prior placeholder (overwritten inside the loop)
    'hazard_beta'   : (1.0, 1.0),

    # block settings
    'testSigmas'    : [1],      # per‑block evidence σ (len must equal block_list)
    'block_list'    : ['single'],

    # I/O
    'train_variants': 500,      # how many train CSVs per Beta prior
    'test_variants' : 50,       # how many   test CSVs per Beta prior

    # Normative observer controls
    'add_normative' : True,                     # compute normative binary decisions
    'hazard_grid'   : list(np.arange(0.0, 1.0, 0.05)),  # grid used by BayesianObserver

    # CSV column order (kept for legacy compatibility; extra fields appended)
    'trial_fields'  : [
        'blockNum', 'blockDifficulty', 'sigma', 'trialInBlock',
        'trueHazard', 'evidence', 'states', 'trueReport', 'truePredict',
        # new columns are *appended* in writeout (no need to list here strictly)
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

# ───────────────────────── normative observer wrapper ────────────────────

def normative_binary(evidence: List[float], sigma: float, p: Dict[str, Any]) -> Tuple[int, int]:
    """Run BayesianObserver on one trial and return (normReport, normPredict) in {−1,+1}.

    Raises a RuntimeError with a clear message if NormativeModel cannot be imported.
    """
    if not p.get('add_normative', True):
        return 0, 0  # placeholders (unused when not writing columns)

    try:
        # Import locally to keep module dependency optional
        from NormativeModel import BayesianObserver
    except Exception as e:
        raise RuntimeError(
            "Failed to import BayesianObserver from NormativeModel.py — "
            "ensure the file is in the same directory and importable."
        ) from e

    hs = np.asarray(p.get('hazard_grid', np.arange(0.0, 1.0, 0.05)), dtype=float)
    mu = float(p['Mu'])
    # BayesianObserver expects (ev, mu1, mu2, sigma, hs)
    _L_haz, _L_state, resp_Rep, resp_Pred = BayesianObserver(evidence, -mu, mu, sigma, hs)
    # Ensure int outputs in {−1, +1}
    r = int(np.sign(resp_Rep)) if resp_Rep != 0 else int(np.random.choice([-1, 1]))
    pbin = int(np.sign(resp_Pred)) if resp_Pred != 0 else int(np.random.choice([-1, 1]))
    return r, pbin

# ───────────────────────── trial making & export ─────────────────────────

def makeBlockTrials(p: Dict[str, Any]):
    trials: List[Dict[str, Any]] = []
    add_norm = bool(p.get('add_normative', True))

    for b, (blk, sig) in enumerate(zip(p['block_list'], p['testSigmas'])):
        for i, hz in enumerate(genBlockHazards(p), 1):
            ev, st = genEvidence(hz, sig, p)

            # ground truths (same conventions as before)
            true_report  = st[-1]
            true_predict = -1 if hz < 0.5 else 1

            row = {
                'blockNum'      : b,
                'blockDifficulty': blk,
                'sigma'         : sig,
                'trialInBlock'  : i,
                'trueHazard'    : hz,
                'evidence'      : ev,
                'states'        : st,
                'trueReport'    : true_report,
                'truePredict'   : true_predict,
            }

            if add_norm:
                try:
                    nrep, npred = normative_binary(ev, sig, p)
                    row['normReport']  = nrep
                    row['normPredict'] = npred
                except Exception as e:
                    # Provide explicit fallback if normative model is unavailable
                    row['normReport']  = np.nan
                    row['normPredict'] = np.nan

            trials.append(row)
    return trials


def export_variants(save_dir: str, p: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)

    cols = list(p['trial_fields'])
    # Append normative columns if requested and not already in list
    if p.get('add_normative', True):
        for extra in ['normReport', 'normPredict']:
            if extra not in cols:
                cols.append(extra)

    for k in range(p['train_variants']):
        df = pd.DataFrame(makeBlockTrials(p))
        df.to_csv(os.path.join(save_dir, f'trainConfig_{k}.csv'), index=False, columns=cols)

    for k in range(p['test_variants']):
        df = pd.DataFrame(makeBlockTrials(p))
        df.to_csv(os.path.join(save_dir, f'testConfig_{k}.csv'), index=False, columns=cols)

    # save a lightweight *TaskConfig.csv* for bookkeeping / provenance
    pd.Series({k: str(v) for k, v in p.items() if k != 'hazard_grid'}).to_csv(
        os.path.join(save_dir, 'TaskConfig.csv')
    )

# ───────────────────────── batch generation loop ──────────────────────────
BETA_LIST = [0.1, 0.5, 1.0, 2.0, 10.0]   # symmetric Beta(x,x)

ROOT_VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variants')

if __name__ == '__main__':
    for x in BETA_LIST:
        params['hazard_beta'] = (x, x)
        sub = f"beta_{str(x).replace('.', 'p')}"       # e.g. beta_0p1
        params['saveDir'] = os.path.join(ROOT_VARIANTS_DIR, sub)

        print(f"[TCG] Beta({x},{x})  →  {params['saveDir']}  (train={params['train_variants']}) | add_normative={params['add_normative']}")
        export_variants(params['saveDir'], params)

    print('[TCG] All done.')

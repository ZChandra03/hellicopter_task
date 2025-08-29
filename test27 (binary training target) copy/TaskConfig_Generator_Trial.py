#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig Generator — unified “report + predict” target with custom hazard‑rate ratios
====================================================================================
This module can **either** be *imported* (fast, no I/O side‑effects) **or** run as a
stand‑alone script to regenerate CSV configuration files for behavioural experiments
in which each *trial* consists of a sequence of noisy evidential samples emitted
from a latent two‑state process that may switch sign with probability *h* (the hazard
rate).

Public API
----------
• ``params`` – dictionary of experiment‑wide settings you may tweak from your own code.
• ``makeBlockTrials(params)`` – return a list of trial dictionaries for one variant.
• ``export_variants(save_dir: str | None = None, variants: int | None = None)`` –
  write ``trainConfig_*.csv`` and ``testConfig_*.csv`` files to *save_dir*.

The expensive CSV export **only** happens when the file is executed directly:

>>> python TaskConfig_Generator.py

Importing it from, e.g., ``train2.py`` will *not* trigger any heavy computation –
it merely makes the helper functions and parameters available immediately.
"""

# %% -------------------------------------------------------------------
# Imports
from __future__ import annotations

import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import scipy.stats as spst

# %% -------------------------------------------------------------------
# Experiment‑wide parameters --------------------------------------------------------------
params: Dict[str, Any] = {}

# — Core trial settings —
params.update({
    'nTrials'         : 300,    # trials per block
    'nEvidence'       : 20,     # evidential samples per trial
    'xLim'            : 5,      # truncation limit of evidence
    'Mu'              : 1,      # latent mean magnitude
    'responseTimeLimit_s': 5,   # (unused here, but written to CSV)
})

# — Uniform‑grid hazard generation (legacy fallback) —
params['HazRes']  = 0.05
params['Hazards'] = np.arange(0, 1.05, params['HazRes'])

# — Custom hazard‑ratio generation (***set to ``None`` to disable***) —
params['hazard_probs'] = {
    0.0: 0.1,
    0.1: 0.2,
    0.3: 0.2,
    0.7: 0.2,
    0.9: 0.2,
    1.0: 0.1,
}

# — Evidence noise (σ) values per difficulty block —
params['testSigmas']  = [1]
params['block_list']  = ['single']
params['nBlocks']     = len(params['testSigmas'])

# — Output / variants —
script_dir          = os.path.dirname(os.path.abspath(__file__))
params['saveDir']   = os.path.join(script_dir, 'variants')
os.makedirs(params['saveDir'], exist_ok=True)
params['variants']  = 40

# — Trial CSV column order —
params['trial_fields'] = [
    'blockNum', 'blockDifficulty', 'sigma', 'trialInBlock',
    'trueHazard', 'evidence', 'states', 'trueReport', 'truePredict'
]

# %% -------------------------------------------------------------------
# Helper: validate & prepare hazard‑probability dictionary -------------------------------

def _prep_hazard_distribution(p: Dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return (hazards, probs) arrays if using custom ratios else empty lists."""
    hp = p.get('hazard_probs')
    if hp is None or not isinstance(hp, dict):
        return [], []

    total_prob = float(sum(hp.values()))
    if not np.isclose(total_prob, 1.0, atol=1e-12):
        raise ValueError(f"hazard_probs must sum to 1 (got {total_prob:.6g}).")

    hazards = np.array(list(hp.keys()), dtype=float)
    probs   = np.array(list(hp.values()), dtype=float)
    return hazards.tolist(), probs.tolist()

HAZARDS_CUSTOM, PROBS_CUSTOM = _prep_hazard_distribution(params)

# %% -------------------------------------------------------------------
# Evidence generation ---------------------------------------------------------------

def genEvidence(hz: float, sigma: float, p: Dict[str, Any]) -> tuple[list[float], list[float]]:
    """Generate one trial; return (evidence, latent_state) lists."""
    mu_val = float(p['Mu'])
    mu     = mu_val if np.random.rand() > 0.5 else -mu_val  # random starting sign

    evidence, states = [], []
    for _ in range(p['nEvidence']):
        lw, hg  = -p['xLim'], p['xLim']
        if sigma > 0:
            xtrunc = spst.truncnorm((lw - mu) / sigma, (hg - mu) / sigma, mu, sigma)
            sample = float(xtrunc.rvs())
        else:
            sample = float(mu)  # zero noise ⇒ evidence equals latent mean

        evidence.append(sample)
        states  .append(mu)

        # latent state switch with probability h
        if np.random.rand() < hz:
            mu = -mu

    return evidence, states

# %% -------------------------------------------------------------------
# Build list of hazard rates for one block --------------------------------------------

def genBlockHazards(p: Dict[str, Any]) -> list[float]:
    """Return a list of hazard values for ``nTrials`` within one block."""
    nT = int(p['nTrials'])

    if HAZARDS_CUSTOM:  # custom distribution
        return np.random.choice(HAZARDS_CUSTOM, size=nT, p=PROBS_CUSTOM).tolist()

    # legacy uniform grid
    hazards_arr = np.array(p['Hazards'], dtype=float)
    n_each      = nT // len(hazards_arr)
    trialHaz    = [hz for hz in hazards_arr for _ in range(n_each)]
    while len(trialHaz) < nT:  # pad if not divisible
        trialHaz.append(float(np.random.choice(hazards_arr)))
    return trialHaz[:nT]

# %% -------------------------------------------------------------------
# Assemble trials for all blocks -------------------------------------------------------

def makeBlockTrials(p: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Generate trials for *one* variant (all difficulty blocks)."""
    trials: list[Dict[str, Any]] = []
    for b, blockName in enumerate(p['block_list']):
        sigma     = p['testSigmas'][b]
        haz_list  = genBlockHazards(p)
        for i, hz in enumerate(haz_list, start=1):
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
# CSV export helper (side‑effect free unless called) -------------------------------

def export_variants(
    save_dir : str | None = None,
    variants : int | None = None,
    p        : Dict[str, Any] = params,
) -> None:
    """Generate *variants* CSV pairs (train/test) and write them to *save_dir*."""
    if save_dir is None:
        save_dir = p['saveDir']
    if variants is None:
        variants = p['variants']

    os.makedirs(save_dir, exist_ok=True)

    for k in range(variants):
        tag      = f"var{k}"
        #train_df = pd.DataFrame(makeBlockTrials(p))
        test_df  = pd.DataFrame(makeBlockTrials(p))

        #train_df.to_csv(os.path.join(save_dir, f'trainConfig_{tag}.csv'), index=False)
        test_df .to_csv(os.path.join(save_dir, f'testConfig_{tag}.csv'),  index=False)

    # save the parameter dictionary for provenance
    pd.Series({k: str(v) for k, v in p.items()}).to_csv(os.path.join(save_dir, 'TaskConfig.csv'))

# %% -------------------------------------------------------------------
# Stand‑alone execution guard -------------------------------------------------------

if __name__ == "__main__":
    print("[TaskConfig_Generator] Exporting CSV variants…")
    export_variants()
    print("[TaskConfig_Generator] Done.")

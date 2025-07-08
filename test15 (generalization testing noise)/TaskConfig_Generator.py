#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig Generator — unified “report + predict” target with custom hazard-rate ratios
===============================================================================
This script generates CSV configuration files for behavioural experiments in
which each *trial* consists of a sequence of noisy evidential samples emitted
from a latent two-state process that may switch sign with probability *h* (the
hazard rate).  It supports two modes for drawing hazard rates per trial:

1. **Uniform grid (legacy)** — all values in ``params['Hazards']`` are repeated
evenly within a block, replicating the original behaviour.
2. **Custom ratios (new)** — supply a mapping ``params['hazard_probs']`` whose
*keys* are hazard values and *values* are their draw probabilities.  The
mapping **must sum to 1** (±1 e-12) or a ``ValueError`` is raised.

Each trial now outputs *two* target columns instead of separate blocks:
    • ``trueReport``  – the most-recent latent state (last element of ``states``)
    • ``truePredict`` – class label −1 or 1, derived from the hazard rate

All other behaviour (block order is now fixed rather than randomised,
per-sigma difficulty, multi-variant export, etc.) remains unchanged.

Created 29 Jan 2025, last modified 19 Jun 2025.
"""

# %% -------------------------------------------------------------------
# Imports
import numpy as np
import pandas as pd
import os
import scipy.stats as spst

from typing import Dict, List, Any

# %% -------------------------------------------------------------------
# Experiment-wide parameters -----------------------------------------------------------------
params: Dict[str, Any] = {}

# — Core trial settings —
params['nTrials']      = 100          # trials per block
params['nEvidence']    = 20           # evidential samples per trial
params['xLim']         = 5            # truncation limit of evidence
params['Mu']           = 1            # latent mean magnitude
params['responseTimeLimit_s'] = 5     # (unused here, but written to CSV)

# — Uniform-grid hazard generation (legacy fallback) —
params['HazRes']   = 0.05
params['Hazards']  = np.arange(0, 1.05, params['HazRes'])

# — Custom hazard-ratio generation (***set to ``None`` to disable***) —
params['hazard_probs'] = {
    0.0: 0.1,
    0.1: 0.2,
    0.3: 0.2,
    0.7: 0.2,
    0.9: 0.2,
    1.0: 0.1,
}

# — Block difficulty (σ for evidence noise) —
params['testSigmas']  = [1]
params['block_list']  = ['set']
params['nBlocks']     = len(params['testSigmas'])

# — Output / variants —
params['variants'] = 10
script_dir        = os.path.dirname(os.path.abspath(__file__))
params['saveDir'] = os.path.join(script_dir, 'variants')
os.makedirs(params['saveDir'], exist_ok=True)

# — Trial CSV column order —
params['trial_fields'] = [
    'blockNum', 'blockDifficulty', 'sigma', 'trialInBlock',
    'trueHazard', 'evidence', 'states', 'trueReport', 'truePredict'
]

# %% -------------------------------------------------------------------
# Helper: validate & prepare hazard-probability dictionary ------------------------------------

def _prep_hazard_distribution(
    params: Dict[str, Any]
) -> tuple[List[float], List[float]]:
    """Return (hazards, probs) arrays if using custom ratios else (None, None)."""
    hp = params.get('hazard_probs')
    if hp is None or not isinstance(hp, dict):
        return [], []

    total_prob = float(sum(hp.values()))
    if not np.isclose(total_prob, 1.0, atol=1e-12):
        raise ValueError(
            f"hazard_probs must sum to 1 (got {total_prob:.6g})."
        )

    hazards = np.array(list(hp.keys()), dtype=float)
    probs   = np.array(list(hp.values()), dtype=float)
    return hazards.tolist(), probs.tolist()

HAZARDS_CUSTOM, PROBS_CUSTOM = _prep_hazard_distribution(params)

# %% -------------------------------------------------------------------
# Evidence generation -------------------------------------------------------------------------

def genEvidence(
    hz: float,
    sigma: float,
    p: Dict[str, Any]
) -> tuple[List[float], List[float]]:
    """Return (evidence list, latent state list) for one trial."""
    mu_val = float(p['Mu'])
    mu = mu_val if np.random.rand() > 0.5 else -mu_val
    state, ev = [], []

    for _ in range(p['nEvidence']):
        lw, hg = -p['xLim'], p['xLim']
        if sigma > 0:
            xtrunc = spst.truncnorm((lw - mu) / sigma, (hg - mu) / sigma, mu, sigma)
            e = xtrunc.rvs()
        else:
            e = mu
        ev.append(float(e))
        state.append(mu)

        if np.random.rand() < hz:
            mu = -mu

    return ev, state

# %% -------------------------------------------------------------------
# Build list of hazard rates for one block ----------------------------------------------------

def genBlockHazards(
    p: Dict[str, Any]
) -> List[float]:
    """Return a list of hazard values for ``nTrials`` within one block."""
    nT = int(p['nTrials'])
    if HAZARDS_CUSTOM is not None:
        return np.random.choice(
            HAZARDS_CUSTOM, size=nT, p=PROBS_CUSTOM
        ).tolist()

    hazards_arr = np.array(p['Hazards'], dtype=float)
    n_each = nT // len(hazards_arr)
    trialHaz = [hz for hz in hazards_arr for _ in range(n_each)]
    # If nTrials not divisible, pad with random hazards
    while len(trialHaz) < nT:
        trialHaz.append(
            float(np.random.choice(hazards_arr))
        )
    return trialHaz[:nT]

# %% -------------------------------------------------------------------
# Assemble trials for all blocks ---------------------------------------------------------------

def makeBlockTrials(
    p: Dict[str, Any]
) -> List[Dict[str, Any]]:
    trials = []
    for b, blockName in enumerate(p['block_list']):
        sigma = p['testSigmas'][b]
        haz_list = genBlockHazards(p)
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
# Generate & export variants ------------------------------------------------------------------
for k in range(params['variants']):
    tag = f"var{k}"
    train_df = pd.DataFrame(makeBlockTrials(params))
    test_df  = pd.DataFrame(makeBlockTrials(params))
    train_df.to_csv(os.path.join(params['saveDir'], f'trainConfig_{tag}.csv'), index=False)
    test_df .to_csv(os.path.join(params['saveDir'], f'testConfig_{tag}.csv'),  index=False)

pd.Series({k: str(v) for k, v in params.items()}).to_csv(
    os.path.join(params['saveDir'], 'TaskConfig.csv')
)

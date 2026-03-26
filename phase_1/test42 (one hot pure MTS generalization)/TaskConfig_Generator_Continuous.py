#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig_Generator_Continuous.py
==================================
Generate *continuous-hazard* configs for several *uniform* hazard groups.

Groups:
1) 0.0–0.4
2) 0.6–1.0
3) edges (0.0–0.2) ∪ (0.8–1.0)
4) 0.3–0.7
5) flat 0.0–1.0

Within each group, hazards are i.i.d. uniform on the specified range(s).
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as spst

# ───────────────────────── global experiment parameters ──────────────────────────
params: Dict[str, Any] = {
    # core trial settings
    'nTrials'       : 300,      # number of trials per (single-block) file
    'nEvidence'     : 20,       # length of evidence sequence
    'xLim'          : 5,        # evidence truncation bounds ±xLim
    'Mu'            : 1,        # latent mean magnitude

    # block settings
    'testSigmas'    : [1],      # per-block evidence σ (len must equal block_list)
    'block_list'    : ['single'],

    # I/O
    'train_variants': 500,      # how many train CSVs per group
    'test_variants' : 50,       # how many   test CSVs per group

    # CSV column order (kept for legacy compatibility)
    'trial_fields'  : [
        'blockNum', 'blockDifficulty', 'sigma', 'trialInBlock',
        'trueHazard', 'evidence', 'states', 'trueReport', 'truePredict'
    ],
}

# ───────────────────────── hazard groups (uniform within ranges) ─────────────────
# Each group is (slug, [(lo, hi), ...]) with 0.0 <= lo < hi <= 1.0
HAZARD_GROUPS: List[Tuple[str, List[Tuple[float, float]]]] = [
    ("hz_0_0p4",            [(0.0, 0.4)]),
    ("hz_0p6_1",            [(0.6, 1.0)]),
    ("hz_edges_0_0p2_0p8_1",[(0.0, 0.2), (0.8, 1.0)]),
    ("hz_0p3_0p7",          [(0.3, 0.7)]),
    ("hz_flat_0_1",         [(0.0, 1.0)]),
]

# ───────────────────────── helpers ──────────────────────────
def genEvidence(hz: float, sigma: float, p: Dict[str, Any]):  # -> (evidence, states)
    """Simulate one evidence sequence for a *single* trial."""
    mu_val = float(p['Mu']) * (1 if np.random.rand() > 0.5 else -1)
    ev, st = [], []
    lw, hg = -p['xLim'], p['xLim']
    for _ in range(p['nEvidence']):
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

def sample_hazards_from_ranges(ranges: Sequence[Tuple[float, float]], n: int) -> List[float]:
    """Sample n hazards uniformly over a union of disjoint intervals in (0,1).
       Sampling is proportional to interval length to keep overall density uniform."""
    ranges = [(float(lo), float(hi)) for lo, hi in ranges]
    lens = np.array([hi - lo for lo, hi in ranges], dtype=float)
    if np.any(lens <= 0):
        raise ValueError("All (lo, hi) must satisfy 0 <= lo < hi <= 1.")
    weights = lens / lens.sum()
    # choose an interval index for each draw, weighted by its length
    idxs = np.random.choice(len(ranges), size=n, p=weights)
    draws = []
    for i in idxs:
        lo, hi = ranges[i]
        draws.append(float(np.random.uniform(lo, hi)))
    return draws

def makeBlockTrials_for_ranges(ranges: Sequence[Tuple[float, float]], p: Dict[str, Any]):
    trials: List[Dict[str, Any]] = []
    hz_list = sample_hazards_from_ranges(ranges, int(p['nTrials']))
    for b, (blk, sig) in enumerate(zip(p['block_list'], p['testSigmas'])):
        for i, hz in enumerate(hz_list, 1):
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
                'truePredict'   : -1 if hz < 0.5 else 1,   # quick visual sanity-check
            })
    return trials

def export_variants(save_dir: str, ranges: Sequence[Tuple[float, float]], p: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    for k in range(p['train_variants']):
        pd.DataFrame(makeBlockTrials_for_ranges(ranges, p)).to_csv(
            os.path.join(save_dir, f'trainConfig_{k}.csv'), index=False)
    for k in range(p['test_variants']):
        pd.DataFrame(makeBlockTrials_for_ranges(ranges, p)).to_csv(
            os.path.join(save_dir, f'testConfig_{k}.csv'), index=False)

    # save a lightweight *TaskConfig.csv* for bookkeeping / provenance
    # include the ranges so it’s self-describing
    meta = {k: str(v) for k, v in p.items()}
    meta.update({
        'hazard_mode': 'uniform_union',
        'hazard_ranges': str(ranges),
    })
    pd.Series(meta).to_csv(os.path.join(save_dir, 'TaskConfig.csv'))

# ───────────────────────── batch generation loop ──────────────────────────
ROOT_VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variants')

if __name__ == '__main__':
    for slug, ranges in HAZARD_GROUPS:
        save_dir = os.path.join(ROOT_VARIANTS_DIR, slug)
        print(f"[TCG] Uniform hazards {ranges}  →  {save_dir}  (train={params['train_variants']})")
        export_variants(save_dir, ranges, params)

    print('[TCG] All done.')

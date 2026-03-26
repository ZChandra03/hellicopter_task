#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaskConfig Generator — Bayesian‑filtered helicopter trials
==========================================================
This rewrite enforces *quality control* on every generated trial:
   1. The Bayesian ideal observer’s hazard estimate must lie within ±0.1
      of the ground‑truth hazard.
   2. The Bayesian observer’s final **report** (site‑1 vs site‑2) must
      match the ground truth.
   3. The hazard estimate must stay on the same side of 0.5 as the true
      hazard (no crossing from «stay»⇄«switch»).
Only trials that satisfy **all three** rules are kept; others are
discarded and resampled.  Down‑stream code therefore never sees
ambiguous or misleading training cases.

The public API is unchanged:
* ``params`` – experiment‑wide settings (see below).
* ``makeBlockTrials(params)`` – return a list[dict] of accepted trials.
* ``export_variants()`` – CSV helper (unchanged).

Run the module directly to (re)generate CSV files:

>>> python TaskConfig_Generator.py
"""

# %% -------------------------------------------------------------------
# Imports
from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats as spst

from NormativeModel import BayesianObserver  # Bayesian ideal observer

# ----------------------------------------------------------------------
HS_GRID = np.arange(0, 1.05, 0.05)  # same 0‥1 grid used by the observer

# %% -------------------------------------------------------------------
# Experiment‑wide parameters --------------------------------------------------------------
params: Dict[str, Any] = {}

# — Core trial settings —
params.update({
    "nTrials": 100,            # trials per difficulty block
    "nEvidence": 20,           # evidential samples per trial
    "xLim": 5,                 # evidence truncation limit
    "Mu": 1,                   # latent mean magnitude (→ ±Mu)
    "responseTimeLimit_s": 5,  # (unused here, but written to CSV)
})

# — Uniform hazard grid (fallback when *hazard_probs* is None) —
params["HazRes"] = 0.05
params["Hazards"] = np.arange(0, 1.05, params["HazRes"])

# — Custom hazard distribution (set to None to disable) —
params["hazard_probs"] = {
    0.0: 0.1,
    0.1: 0.2,
    0.3: 0.2,
    0.7: 0.2,
    0.9: 0.2,
    1.0: 0.1,
}

# — Evidence noise (σ) values per difficulty block —
params['testSigmas']  = [0, 0.1, 0.5]
params['block_list']  = ['preTest', 'easy', 'medium']
params["nBlocks"] = len(params["testSigmas"])

# — Output / variants —
script_dir = os.path.dirname(os.path.abspath(__file__))
params["saveDir"] = os.path.join(script_dir, "variants")
os.makedirs(params["saveDir"], exist_ok=True)
params["variants"] = 40

# — Trial CSV column order —
params["trial_fields"] = [
    "blockNum",
    "blockDifficulty",
    "sigma",
    "trialInBlock",
    "trueHazard",
    "evidence",
    "states",
    "trueReport",
    "truePredict",
]

# %% -------------------------------------------------------------------
# Helper: validate & prepare hazard‑probability dictionary -------------------------------

def _prep_hazard_distribution(p: Dict[str, Any]) -> tuple[List[float], List[float]]:
    """Return (hazards, probs) arrays if using custom ratios else empty lists."""
    hp = p.get("hazard_probs")
    if hp is None:
        return [], []

    probs_sum = float(sum(hp.values()))
    if not np.isclose(probs_sum, 1.0, atol=1e-12):
        raise ValueError("hazard_probs must sum to 1 (got %.6g)" % probs_sum)

    hazards = np.array(list(hp.keys()), dtype=float)
    probs = np.array(list(hp.values()), dtype=float)
    return hazards.tolist(), probs.tolist()

HAZARDS_CUSTOM, PROBS_CUSTOM = _prep_hazard_distribution(params)

# %% -------------------------------------------------------------------
# Evidence generation ---------------------------------------------------------------

def genEvidence(hz: float, sigma: float, p: Dict[str, Any]) -> tuple[list[float], list[float]]:
    """Generate one *raw* trial; return (evidence, latent_state) lists."""
    mu_val = float(p["Mu"])
    mu = mu_val if np.random.rand() > 0.5 else -mu_val  # random start sign

    evidence, states = [], []
    for _ in range(p["nEvidence"]):
        lw, hg = -p["xLim"], p["xLim"]
        if sigma > 0:
            xtrunc = spst.truncnorm((lw - mu) / sigma, (hg - mu) / sigma, mu, sigma)
            sample = float(xtrunc.rvs())
        else:  # sigma == 0 ⇒ perfect evidence
            sample = float(mu)

        evidence.append(sample)
        states.append(mu)

        # latent state switch with probability h
        if np.random.rand() < hz:
            mu = -mu

    return evidence, states

# %% -------------------------------------------------------------------
# Bayesian helper utilities ----------------------------------------------------------

def _normative_predict(evidence: List[float], sigma: float, mu_val: float = 1.0) -> tuple[int, float]:
    """Return (site_report ∈ {−1,1}, hazard_estimate ∈ [0,1]) for the sequence."""
    L_haz, _, rep, _ = BayesianObserver(evidence, -mu_val, +mu_val, sigma, HS_GRID.copy())
    haz_est = float(np.dot(HS_GRID, L_haz[:, -1]))  # posterior mean hazard
    return int(rep), haz_est


def _trial_passes(evidence: List[float], sigma: float, true_hz: float, true_rep: int, mu_val: float) -> bool:
    """Apply the three acceptance criteria described at top‑of‑file."""
    rep_norm, haz_norm = _normative_predict(evidence, sigma, mu_val)

    rule1 = abs(haz_norm - true_hz) <= 0.1
    rule2 = rep_norm == true_rep
    rule3 = (haz_norm < 0.5) == (true_hz < 0.5)

    return rule1 and rule2 and rule3

# %% -------------------------------------------------------------------
# Assemble trials for *all* difficulty blocks ----------------------------------------

def makeBlockTrials(p: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Generate *accepted* trials for **one variant** across all difficulty blocks."""
    trials: list[Dict[str, Any]] = []
    mu_val = float(p["Mu"])

    for b, blockName in enumerate(p["block_list"]):
        sigma = p["testSigmas"][b]
        accepted = 0

        while accepted < p["nTrials"]:
            # — sample a candidate hazard —
            if HAZARDS_CUSTOM:
                hz = float(np.random.choice(HAZARDS_CUSTOM, p=PROBS_CUSTOM))
            else:
                hz = float(np.random.choice(p["Hazards"]))

            # — simulate raw evidence —
            evidence, states = genEvidence(hz, sigma, p)
            true_rep = states[-1]

            # — acceptance test —
            if _trial_passes(evidence, sigma, hz, true_rep, mu_val):
                accepted += 1
                trials.append({
                    "blockNum": b,
                    "blockDifficulty": blockName,
                    "sigma": sigma,
                    "trialInBlock": accepted,
                    "trueHazard": hz,
                    "evidence": evidence,
                    "states": states,
                    "trueReport": true_rep,
                    "truePredict": -1 if hz < 0.5 else 1,
                })

    return trials

# %% -------------------------------------------------------------------
# CSV export helper (side‑effect free unless called) -------------------------------

def export_variants(save_dir: str | None = None, variants: int | None = None, p: Dict[str, Any] = params) -> None:
    """Generate *variants* CSV pairs (train/test) and write them to *save_dir*."""
    if save_dir is None:
        save_dir = p["saveDir"]
    if variants is None:
        variants = p["variants"]

    os.makedirs(save_dir, exist_ok=True)

    for k in range(variants):
        tag = f"var{k}"
        train_df = pd.DataFrame(makeBlockTrials(p))
        test_df = pd.DataFrame(makeBlockTrials(p))

        train_df.to_csv(os.path.join(save_dir, f"trainConfig_{tag}.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, f"testConfig_{tag}.csv"), index=False)

    # provenance: save the parameter dictionary
    pd.Series({k: str(v) for k, v in p.items()}).to_csv(os.path.join(save_dir, "TaskConfig.csv"))

# %% -------------------------------------------------------------------
# Script entry‑point ---------------------------------------------------------------

if __name__ == "__main__":
    print("[TaskConfig_Generator] Exporting CSV variants…")
    export_variants()
    print("[TaskConfig_Generator] Done.")

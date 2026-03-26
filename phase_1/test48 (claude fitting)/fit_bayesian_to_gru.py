#!/usr/bin/env python3
"""
fit_bayesian_to_gru.py
======================
Complete pipeline to fit a detuned Bayesian model to a trained GRU.

Steps:
1. Load trained GRU (models/trained_gru/final.pt)
2. Collect GRU predictions on test set (both choices and probabilities)
3. Fit Bayesian parameters (sigma_belief, bias) via maximum likelihood
4. Compare fitted Bayesian vs GRU vs optimal Bayesian
5. Generate comprehensive analysis plots

Expected directory structure:
    models/trained_gru/
        - final.pt
        - hp.json
    variants/sigma_1/  (or sigma_2, beta_1p0, etc.)
        - testConfig_*.csv
"""

import os
import glob
import json
import ast
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import beta as beta_dist, norm

from rnn_models import GRUModel

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VARIANTS_DIR = os.path.join(BASE_DIR, "variants")
OUT_DIR = os.path.join(BASE_DIR, "results", "bayesian_fitting")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TEST_CFGS = 20  # Number of test configs to use
MODEL_NAME = "trained_gru"  # Name of the trained model folder
VARIANT_NAME = "sigma_2"  # Which test set to use (sigma_1, sigma_2, beta_1p0, etc.)

# Bayesian observer parameters
MU1, MU2 = -1, 1
HS = np.arange(0, 1, 0.05)  # Hazard rate grid

# Binning for plots
BIN_WIDTH = 0.05
BIN_EDGES = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


# ============================================================================
# Phase 1: Data Collection - GRU Predictions
# ============================================================================
class HelicopterEvalDS(Dataset):
    """Dataset for evaluation - stores evidence sequences and ground truth."""
    def __init__(self, df: pd.DataFrame):
        xs, rep_targets, hazards, evidences = [], [], [], []
        for _, row in df.iterrows():
            evid = row["evidence"]
            if not isinstance(evid, list):
                evid = ast.literal_eval(str(evid))
            xs.append(torch.tensor(evid, dtype=torch.float32).unsqueeze(-1))  # (T,1)
            rep_targets.append(float(1.0 if row["trueReport"] > 0 else 0.0))
            hazards.append(float(row["trueHazard"]))
            evidences.append(evid)
        
        self.x = xs
        self.y_rep = torch.tensor(rep_targets, dtype=torch.float32).unsqueeze(1)
        self.haz = torch.tensor(hazards, dtype=torch.float32)
        self.evidences = evidences

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y_rep[i], self.haz[i], self.evidences[i]


def collate_fn(batch):
    """Collate function that handles variable length sequences."""
    xs, yr, hz, ev = zip(*batch)
    return torch.stack(xs, 0), torch.stack(yr, 0), torch.stack(hz, 0), list(ev)


def load_gru_model(model_dir: str) -> Tuple[GRUModel, Dict]:
    """Load trained GRU model and hyperparameters."""
    model_path = os.path.join(MODELS_DIR, model_dir)
    
    # Load hyperparameters
    with open(os.path.join(model_path, "hp.json"), "r") as f:
        hp = json.load(f)
    hp.setdefault("n_input", 1)
    hp.setdefault("n_rnn", 128)
    
    # Load model
    model = GRUModel(hp).to(DEVICE)
    state = torch.load(os.path.join(model_path, "final.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    
    print(f"✓ Loaded GRU model from {model_path}")
    print(f"  - n_input: {hp['n_input']}, n_rnn: {hp['n_rnn']}")
    
    return model, hp


def get_test_csvs(variant_name: str, n_configs: int = N_TEST_CFGS) -> List[str]:
    """Get test configuration CSV files."""
    variant_dir = os.path.join(VARIANTS_DIR, variant_name)
    paths = sorted(glob.glob(os.path.join(variant_dir, "testConfig_*.csv")))[:n_configs]
    
    if len(paths) < n_configs:
        print(f"[warn] only {len(paths)} test configs found in {variant_dir}")
    
    print(f"✓ Found {len(paths)} test configs in {variant_name}")
    return paths


@torch.no_grad()
def collect_gru_predictions(model: GRUModel, csvs: List[str]) -> pd.DataFrame:
    """
    Collect GRU predictions (both probabilities and choices) on test set.
    
    Returns DataFrame with columns:
        - evidence: list of observations
        - true_hazard: ground truth hazard
        - true_report: ground truth final state (+1 or -1)
        - gru_report_prob: P(state=+1) from sigmoid
        - gru_report_choice: binary choice (-1 or +1)
        - gru_hazard_prob: P(hazard>0.5) from sigmoid
        - gru_hazard_choice: binary choice (-1 or +1)
        - trial_idx: unique identifier
    """
    records = []
    trial_idx = 0
    
    print("\n" + "="*60)
    print("Phase 1: Collecting GRU Predictions")
    print("="*60)
    
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        dl = DataLoader(
            HelicopterEvalDS(df), 
            batch_size=256, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        for x, y_rep, hz, evidences in dl:
            x = x.to(DEVICE)
            loc_logits, haz_logits = model(x)  # (B, T, 1), (B, 1)
            
            # Report head: final step probabilities
            rep_probs = torch.sigmoid(loc_logits[:, -1, 0]).cpu().numpy()  # (B,)
            rep_choices = (rep_probs > 0.5).astype(int) * 2 - 1  # Convert to -1/+1
            
            # Hazard head: probability of hazard > 0.5
            haz_probs = torch.sigmoid(haz_logits[:, 0]).cpu().numpy()  # (B,)
            haz_choices = (haz_probs > 0.5).astype(int) * 2 - 1  # Convert to -1/+1
            
            # Ground truth
            true_reports = (y_rep.squeeze(1).numpy() > 0.5).astype(int) * 2 - 1  # -1/+1
            true_hazards = hz.numpy()
            
            # Store records
            for i in range(len(x)):
                records.append({
                    'trial_idx': trial_idx,
                    'evidence': evidences[i],
                    'true_hazard': float(true_hazards[i]),
                    'true_report': int(true_reports[i]),
                    'gru_report_prob': float(rep_probs[i]),
                    'gru_report_choice': int(rep_choices[i]),
                    'gru_hazard_prob': float(haz_probs[i]),
                    'gru_hazard_choice': int(haz_choices[i]),
                })
                trial_idx += 1
    
    predictions_df = pd.DataFrame(records)
    
    # Save predictions
    pred_csv = os.path.join(OUT_DIR, "gru_predictions.csv")
    predictions_df.to_csv(pred_csv, index=False)
    print(f"\n✓ Collected {len(predictions_df)} trial predictions")
    print(f"✓ Saved to: {pred_csv}")
    
    # Quick stats
    rep_acc = (predictions_df['gru_report_choice'] == predictions_df['true_report']).mean()
    haz_acc = ((predictions_df['gru_hazard_choice'] == 1) == 
               (predictions_df['true_hazard'] > 0.5)).mean()
    print(f"\n  GRU Performance:")
    print(f"    - Report accuracy: {rep_acc:.3f}")
    print(f"    - Hazard accuracy: {haz_acc:.3f}")
    
    return predictions_df


# ============================================================================
# Phase 2: Bayesian Observer with Probabilities
# ============================================================================
def BayesianObserver_with_probs(ev, mu1, mu2, sigma, hs, bias=0.0):
    """
    Extended Bayesian observer that returns choice probabilities.
    
    Returns:
    --------
    - L_haz: hazard beliefs over time (len(hs), nEvidence+1)
    - L_state: state beliefs over time (2, nEvidence+1)
    - resp_Rep: binary report choice (-1 or 1)
    - resp_Pred: binary hazard choice (-1 or 1)
    - P_report_state_pos: P(state=+1 | evidence) [NEW]
    - P_predict_switch: P(hazard>0.5 | evidence) [NEW]
    """
    nEvidence = len(ev)
    
    # Initialize the arrays
    L_n = [np.zeros((len(hs), nEvidence + 1)) for _ in range(2)]
    
    # Set initial alpha and beta for Beta distribution prior
    alpha = 1.0
    beta_param = 1.0
    
    # Adjust alpha and beta based on the bias
    if bias > 0:
        alpha += bias
    elif bias < 0:
        beta_param -= bias
    
    # Calculate the Beta distribution prior
    beta_prior = beta_dist.pdf(hs, alpha, beta_param)
    
    # Marginalization factor
    marg = 2 * len(hs)
    
    # Initialize the first column
    L_n[0][:, 0] = beta_prior / marg
    L_n[1][:, 0] = L_n[0][:, 0]
    
    # Check if the sum of probabilities is 1, and adjust if necessary
    P_check = np.sum(L_n[0][:, 0]) + np.sum(L_n[1][:, 0])
    P_diff = 1 - P_check
    P_marg = P_diff / (2 * len(hs))
    L_n[0][:, 0] += P_marg
    L_n[1][:, 0] += P_marg
    
    # Compute the normal pdf for all evidence once
    if sigma != 0:
        norm_P_S1 = norm.pdf(ev, mu1, sigma)
        norm_P_S2 = norm.pdf(ev, mu2, sigma)
    
    # Main loop - compute the likelihoods
    for n in range(nEvidence):
        for s in range(2):  # Two states
            for h in range(len(hs)):
                if s == 0:  # State 1 (mu = -1)
                    if sigma == 0:
                        P_S1 = 1 if int(ev[n]) == mu1 else 0
                    else:
                        P_S1 = norm_P_S1[n]
                    
                    L_n[0][h, n + 1] = P_S1 * ((1 - hs[h]) * L_n[0][h, n] + hs[h] * L_n[1][h, n])
                    
                elif s == 1:  # State 2 (mu = +1)
                    if sigma == 0:
                        P_S2 = 1 if int(ev[n]) == mu2 else 0
                    else:
                        P_S2 = norm_P_S2[n]
                    
                    L_n[1][h, n + 1] = P_S2 * ((1 - hs[h]) * L_n[1][h, n] + hs[h] * L_n[0][h, n])
        
        # Renormalization
        T = np.sum(L_n[0][:, n + 1]) + np.sum(L_n[1][:, n + 1])
        if T > 0:
            L_n[0][:, n + 1] /= T
            L_n[1][:, n + 1] /= T
    
    # Compute hazard likelihoods
    L_haz = np.zeros((len(hs), nEvidence + 1))
    for n in range(nEvidence + 1):
        for h in range(len(hs)):
            L_haz[h, n] = L_n[0][h, n] + L_n[1][h, n]
    
    # Compute state likelihoods
    L_state = np.zeros((2, nEvidence + 1))
    for n in range(nEvidence + 1):
        for s in range(2):
            L_state[s, n] = np.sum(L_n[s][:, n])
    
    # Report response (final state belief)
    P_s1 = L_state[0, -1]  # P(state = -1)
    P_s2 = L_state[1, -1]  # P(state = +1)
    
    P_report_state_pos = P_s2  # Probability of choosing state +1
    
    if P_s1 > P_s2:
        resp_Rep = -1
    elif P_s1 < P_s2:
        resp_Rep = 1
    else:
        resp_Rep = np.random.choice([-1, 1])
    
    # Prediction response (hazard belief)
    P_haz_switch = hs * L_haz[:, -1]
    P_haz_stay = (1 - hs) * L_haz[:, -1]
    P_stay = np.sum(P_haz_stay)
    P_switch = np.sum(P_haz_switch)
    
    P_predict_switch = P_switch / (P_stay + P_switch + 1e-10)  # Probability of switch
    
    if P_stay == 0.5 and P_switch == 0.5:
        resp_Pred = np.random.choice([-1, 1])
    elif P_stay > P_switch:
        resp_Pred = -1
    elif P_stay < P_switch:
        resp_Pred = 1
    else:
        resp_Pred = 1  # default to switch
    
    return L_haz, L_state, resp_Rep, resp_Pred, P_report_state_pos, P_predict_switch


# ============================================================================
# Phase 3: Likelihood Function for Fitting
# ============================================================================
def negative_log_likelihood(params, gru_data, true_sigma=1.0, verbose=False):
    """
    Compute negative log-likelihood of GRU choices under Bayesian model.
    
    Parameters:
    -----------
    params: [sigma_belief, bias]
    gru_data: DataFrame with GRU predictions
    true_sigma: actual noise level (for reference, not used in computation)
    
    Returns:
    --------
    nll: negative log-likelihood (to minimize)
    """
    sigma_belief, bias = params
    
    # Clip to reasonable bounds to prevent numerical issues
    sigma_belief = np.clip(sigma_belief, 0.01, 10.0)
    bias = np.clip(bias, -10.0, 10.0)
    
    nll_total = 0.0
    n_trials = 0
    
    for _, row in gru_data.iterrows():
        ev = row['evidence']
        if isinstance(ev, str):
            ev = ast.literal_eval(ev)
        
        # Run Bayesian observer with current parameters
        try:
            _, _, _, _, P_bay_state_pos, P_bay_switch = BayesianObserver_with_probs(
                ev, MU1, MU2, sigma_belief, HS, bias
            )
        except:
            # If Bayesian observer fails, return large penalty
            return 1e10
        
        # Likelihood of GRU's report choice
        if row['gru_report_choice'] == 1:
            p_report = P_bay_state_pos
        else:
            p_report = 1 - P_bay_state_pos
        
        # Likelihood of GRU's hazard choice
        if row['gru_hazard_choice'] == 1:
            p_hazard = P_bay_switch
        else:
            p_hazard = 1 - P_bay_switch
        
        # Combined log-likelihood (assuming independence)
        # Add epsilon to prevent log(0)
        nll_total -= (np.log(p_report + 1e-10) + np.log(p_hazard + 1e-10))
        n_trials += 1
    
    if verbose and n_trials % 100 == 0:
        print(f"  σ={sigma_belief:.3f}, bias={bias:.3f} → NLL={nll_total/n_trials:.4f}")
    
    return nll_total


# ============================================================================
# Phase 4: Optimization
# ============================================================================
def fit_bayesian_parameters(gru_data: pd.DataFrame, true_sigma: float = 1.0):
    """
    Fit Bayesian parameters to match GRU behavior.
    
    Returns:
    --------
    best_params: [sigma_belief, bias]
    optimization_result: scipy optimization result
    """
    print("\n" + "="*60)
    print("Phase 2: Fitting Bayesian Parameters")
    print("="*60)
    
    # Parameter bounds
    bounds = [
        (0.1, 5.0),   # sigma_belief: reasonable noise range
        (-5.0, 5.0),  # bias: prior bias range
    ]
    
    print(f"\nParameter bounds:")
    print(f"  σ_belief: [{bounds[0][0]}, {bounds[0][1]}]")
    print(f"  bias: [{bounds[1][0]}, {bounds[1][1]}]")
    
    # Strategy 1: Multiple local optimizations from random starts
    print(f"\nStrategy 1: Multiple random initializations...")
    best_result = None
    best_nll = np.inf
    
    n_starts = 5
    for i in range(n_starts):
        # Random initialization
        x0 = [
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1])
        ]
        
        print(f"\n  Start {i+1}/{n_starts}: σ={x0[0]:.3f}, bias={x0[1]:.3f}")
        
        result = minimize(
            negative_log_likelihood,
            x0=x0,
            args=(gru_data, true_sigma),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        print(f"    → Result: σ={result.x[0]:.3f}, bias={result.x[1]:.3f}, NLL={result.fun:.2f}")
        
        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result
    
    print(f"\n  Best local result: σ={best_result.x[0]:.3f}, bias={best_result.x[1]:.3f}")
    
    # Strategy 2: Global optimization (differential evolution)
    print(f"\nStrategy 2: Global optimization (differential evolution)...")
    result_global = differential_evolution(
        negative_log_likelihood,
        bounds=bounds,
        args=(gru_data, true_sigma),
        maxiter=50,
        seed=42,
        workers=1,
        disp=True
    )
    
    print(f"\n  Global result: σ={result_global.x[0]:.3f}, bias={result_global.x[1]:.3f}")
    
    # Choose the best result
    final_result = result_global if result_global.fun < best_nll else best_result
    
    print("\n" + "="*60)
    print("Optimization Complete")
    print("="*60)
    print(f"\nFitted parameters:")
    print(f"  σ_belief = {final_result.x[0]:.4f}")
    print(f"  bias = {final_result.x[1]:.4f}")
    print(f"  NLL = {final_result.fun:.2f}")
    print(f"  NLL per trial = {final_result.fun / len(gru_data):.4f}")
    
    # Save results
    results_dict = {
        'sigma_belief': float(final_result.x[0]),
        'bias': float(final_result.x[1]),
        'nll': float(final_result.fun),
        'nll_per_trial': float(final_result.fun / len(gru_data)),
        'true_sigma': float(true_sigma),
        'n_trials': int(len(gru_data)),
        'success': bool(final_result.success),
    }
    
    results_file = os.path.join(OUT_DIR, "fitted_parameters.json")
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Saved fitted parameters to: {results_file}")
    
    return final_result.x, final_result


# ============================================================================
# Phase 5: Model Comparison & Visualization
# ============================================================================
def evaluate_bayesian_on_trials(gru_data: pd.DataFrame, sigma_belief: float, bias: float):
    """Evaluate Bayesian model with given parameters on all trials."""
    records = []
    
    for _, row in gru_data.iterrows():
        ev = row['evidence']
        if isinstance(ev, str):
            ev = ast.literal_eval(ev)
        
        _, _, resp_Rep, resp_Pred, P_state_pos, P_switch = BayesianObserver_with_probs(
            ev, MU1, MU2, sigma_belief, HS, bias
        )
        
        records.append({
            'trial_idx': row['trial_idx'],
            'bay_report_choice': resp_Rep,
            'bay_hazard_choice': resp_Pred,
            'bay_report_prob': P_state_pos,
            'bay_hazard_prob': P_switch,
        })
    
    return pd.DataFrame(records)


def bin_accuracy(hazards, correct_mask):
    """Compute accuracy within hazard bins."""
    idx = np.digitize(hazards, BIN_EDGES) - 1
    idx = np.clip(idx, 0, len(BIN_EDGES) - 2)
    total = np.zeros(len(BIN_CENTERS), int)
    good = np.zeros(len(BIN_CENTERS), int)
    for i, ok in zip(idx, correct_mask):
        total[i] += 1
        if ok:
            good[i] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(total > 0, good / total, np.nan)
    return acc, total, good


def generate_comparison_plots(gru_data: pd.DataFrame, fitted_params: np.ndarray, 
                             true_sigma: float):
    """Generate comprehensive comparison plots."""
    print("\n" + "="*60)
    print("Phase 3: Generating Comparison Plots")
    print("="*60)
    
    sigma_fit, bias_fit = fitted_params
    
    # Evaluate fitted Bayesian
    print("\nEvaluating fitted Bayesian model...")
    fitted_bay = evaluate_bayesian_on_trials(gru_data, sigma_fit, bias_fit)
    
    # Evaluate optimal Bayesian
    print("Evaluating optimal Bayesian model...")
    optimal_bay = evaluate_bayesian_on_trials(gru_data, true_sigma, 0.0)
    
    # Merge data
    comparison_df = gru_data.copy()
    comparison_df['bay_fitted_report_choice'] = fitted_bay['bay_report_choice'].values
    comparison_df['bay_fitted_hazard_choice'] = fitted_bay['bay_hazard_choice'].values
    comparison_df['bay_optimal_report_choice'] = optimal_bay['bay_report_choice'].values
    comparison_df['bay_optimal_hazard_choice'] = optimal_bay['bay_hazard_choice'].values
    
    # Compute accuracies
    hazards = comparison_df['true_hazard'].values
    
    # GRU
    gru_rep_correct = (comparison_df['gru_report_choice'] == comparison_df['true_report']).values
    gru_haz_correct = ((comparison_df['gru_hazard_choice'] == 1) == (hazards > 0.5))
    
    # Fitted Bayesian
    fitted_rep_correct = (comparison_df['bay_fitted_report_choice'] == comparison_df['true_report']).values
    fitted_haz_correct = ((comparison_df['bay_fitted_hazard_choice'] == 1) == (hazards > 0.5))
    
    # Optimal Bayesian
    optimal_rep_correct = (comparison_df['bay_optimal_report_choice'] == comparison_df['true_report']).values
    optimal_haz_correct = ((comparison_df['bay_optimal_hazard_choice'] == 1) == (hazards > 0.5))
    
    # Bin accuracies
    gru_rep_acc, _, _ = bin_accuracy(hazards, gru_rep_correct)
    gru_haz_acc, _, _ = bin_accuracy(hazards, gru_haz_correct)
    
    fitted_rep_acc, _, _ = bin_accuracy(hazards, fitted_rep_correct)
    fitted_haz_acc, _, _ = bin_accuracy(hazards, fitted_haz_correct)
    
    optimal_rep_acc, _, _ = bin_accuracy(hazards, optimal_rep_correct)
    optimal_haz_acc, _, _ = bin_accuracy(hazards, optimal_haz_correct)
    
    # ========================================================================
    # Plot 1: Report Head Comparison
    # ========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(BIN_CENTERS, gru_rep_acc, 'o-', color='tab:blue', linewidth=2.5, 
            markersize=6, label='GRU (trained)', alpha=0.9)
    ax1.plot(BIN_CENTERS, fitted_rep_acc, 's--', color='tab:orange', linewidth=2.2,
            markersize=5, label=f'Fitted Bayesian (σ={sigma_fit:.2f}, bias={bias_fit:.2f})', alpha=0.8)
    ax1.plot(BIN_CENTERS, optimal_rep_acc, '^-.', color='tab:green', linewidth=2.0,
            markersize=5, label=f'Optimal Bayesian (σ={true_sigma:.2f}, bias=0)', alpha=0.7)
    
    ax1.set_xlabel("True hazard rate (bin centers, width=0.05)", fontsize=12)
    ax1.set_ylabel("Report-head accuracy", fontsize=12)
    ax1.set_title("Report Head: GRU vs Fitted vs Optimal Bayesian", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fontsize=11, loc='best')
    fig1.tight_layout()
    
    fig1_path = os.path.join(OUT_DIR, "comparison_report_head.png")
    fig1.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Saved: {fig1_path}")
    
    # ========================================================================
    # Plot 2: Hazard Head Comparison
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    ax2.plot(BIN_CENTERS, gru_haz_acc, 'o-', color='tab:blue', linewidth=2.5,
            markersize=6, label='GRU (trained)', alpha=0.9)
    ax2.plot(BIN_CENTERS, fitted_haz_acc, 's--', color='tab:orange', linewidth=2.2,
            markersize=5, label=f'Fitted Bayesian (σ={sigma_fit:.2f}, bias={bias_fit:.2f})', alpha=0.8)
    ax2.plot(BIN_CENTERS, optimal_haz_acc, '^-.', color='tab:green', linewidth=2.0,
            markersize=5, label=f'Optimal Bayesian (σ={true_sigma:.2f}, bias=0)', alpha=0.7)
    
    ax2.set_xlabel("True hazard rate (bin centers, width=0.05)", fontsize=12)
    ax2.set_ylabel("Hazard-head accuracy", fontsize=12)
    ax2.set_title("Hazard Head: GRU vs Fitted vs Optimal Bayesian", fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, fontsize=11, loc='best')
    fig2.tight_layout()
    
    fig2_path = os.path.join(OUT_DIR, "comparison_hazard_head.png")
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved: {fig2_path}")
    
    # ========================================================================
    # Plot 3: Probability Calibration
    # ========================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Report head calibration
    ax3a.hist(gru_data['gru_report_prob'], bins=20, alpha=0.6, color='tab:blue', 
             label='GRU', density=True)
    ax3a.hist(fitted_bay['bay_report_prob'], bins=20, alpha=0.6, color='tab:orange',
             label='Fitted Bayesian', density=True)
    ax3a.set_xlabel("P(report state = +1)", fontsize=11)
    ax3a.set_ylabel("Density", fontsize=11)
    ax3a.set_title("Report Head: Choice Probability Distribution", fontsize=12, fontweight='bold')
    ax3a.legend(frameon=True)
    ax3a.grid(True, alpha=0.3)
    
    # Hazard head calibration
    ax3b.hist(gru_data['gru_hazard_prob'], bins=20, alpha=0.6, color='tab:blue',
             label='GRU', density=True)
    ax3b.hist(fitted_bay['bay_hazard_prob'], bins=20, alpha=0.6, color='tab:orange',
             label='Fitted Bayesian', density=True)
    ax3b.set_xlabel("P(hazard > 0.5)", fontsize=11)
    ax3b.set_ylabel("Density", fontsize=11)
    ax3b.set_title("Hazard Head: Choice Probability Distribution", fontsize=12, fontweight='bold')
    ax3b.legend(frameon=True)
    ax3b.grid(True, alpha=0.3)
    
    fig3.tight_layout()
    fig3_path = os.path.join(OUT_DIR, "probability_distributions.png")
    fig3.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print(f"✓ Saved: {fig3_path}")
    
    # ========================================================================
    # Plot 4: Delta plots (GRU - Fitted Bayesian)
    # ========================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Report head delta
    delta_rep = gru_rep_acc - fitted_rep_acc
    ax4a.plot(BIN_CENTERS, delta_rep, 'o-', color='tab:purple', linewidth=2.5, markersize=6)
    ax4a.axhline(0, linestyle='--', color='black', alpha=0.5, linewidth=1.5)
    ax4a.set_xlabel("True hazard rate", fontsize=11)
    ax4a.set_ylabel("Δ Accuracy (GRU - Fitted Bayesian)", fontsize=11)
    ax4a.set_title("Report Head: Performance Difference", fontsize=12, fontweight='bold')
    ax4a.grid(True, alpha=0.3)
    
    # Hazard head delta
    delta_haz = gru_haz_acc - fitted_haz_acc
    ax4b.plot(BIN_CENTERS, delta_haz, 'o-', color='tab:purple', linewidth=2.5, markersize=6)
    ax4b.axhline(0, linestyle='--', color='black', alpha=0.5, linewidth=1.5)
    ax4b.set_xlabel("True hazard rate", fontsize=11)
    ax4b.set_ylabel("Δ Accuracy (GRU - Fitted Bayesian)", fontsize=11)
    ax4b.set_title("Hazard Head: Performance Difference", fontsize=12, fontweight='bold')
    ax4b.grid(True, alpha=0.3)
    
    fig4.tight_layout()
    fig4_path = os.path.join(OUT_DIR, "accuracy_deltas.png")
    fig4.savefig(fig4_path, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    print(f"✓ Saved: {fig4_path}")
    
    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    
    print("\nOverall Accuracies:")
    print(f"  Report Head:")
    print(f"    - GRU:             {gru_rep_correct.mean():.4f}")
    print(f"    - Fitted Bayesian: {fitted_rep_correct.mean():.4f}")
    print(f"    - Optimal Bayesian:{optimal_rep_correct.mean():.4f}")
    
    print(f"\n  Hazard Head:")
    print(f"    - GRU:             {gru_haz_correct.mean():.4f}")
    print(f"    - Fitted Bayesian: {fitted_haz_correct.mean():.4f}")
    print(f"    - Optimal Bayesian:{optimal_haz_correct.mean():.4f}")
    
    # Agreement statistics
    from sklearn.metrics import cohen_kappa_score
    
    rep_kappa = cohen_kappa_score(
        comparison_df['gru_report_choice'],
        comparison_df['bay_fitted_report_choice']
    )
    haz_kappa = cohen_kappa_score(
        comparison_df['gru_hazard_choice'],
        comparison_df['bay_fitted_hazard_choice']
    )
    
    print(f"\nCohen's Kappa (GRU vs Fitted Bayesian):")
    print(f"  - Report head: {rep_kappa:.4f}")
    print(f"  - Hazard head: {haz_kappa:.4f}")
    
    # Save summary
    summary = {
        'gru_report_acc': float(gru_rep_correct.mean()),
        'gru_hazard_acc': float(gru_haz_correct.mean()),
        'fitted_report_acc': float(fitted_rep_correct.mean()),
        'fitted_hazard_acc': float(fitted_haz_correct.mean()),
        'optimal_report_acc': float(optimal_rep_correct.mean()),
        'optimal_hazard_acc': float(optimal_haz_correct.mean()),
        'report_kappa': float(rep_kappa),
        'hazard_kappa': float(haz_kappa),
        'fitted_sigma': float(sigma_fit),
        'fitted_bias': float(bias_fit),
        'true_sigma': float(true_sigma),
    }
    
    summary_file = os.path.join(OUT_DIR, "comparison_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary to: {summary_file}")
    
    # Save detailed comparison CSV
    comparison_csv = os.path.join(OUT_DIR, "detailed_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"✓ Saved detailed comparison to: {comparison_csv}")


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" "*15 + "FITTING BAYESIAN MODEL TO GRU")
    print("="*70)
    
    # Configuration summary
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Test variant: {VARIANT_NAME}")
    print(f"  Number of test configs: {N_TEST_CFGS}")
    print(f"  Device: {DEVICE}")
    print(f"  Output directory: {OUT_DIR}")
    
    # Determine true sigma from variant name
    if "sigma_1" in VARIANT_NAME:
        true_sigma = 1.0
    elif "sigma_2" in VARIANT_NAME:
        true_sigma = 2.0
    elif "beta" in VARIANT_NAME:
        true_sigma = 1.0  # Default for beta variants
    else:
        true_sigma = 1.0
        print(f"[warn] Could not determine sigma from variant name, using {true_sigma}")
    
    print(f"  True σ: {true_sigma}")
    
    # Phase 1: Load model and collect predictions
    try:
        model, hp = load_gru_model(MODEL_NAME)
        csvs = get_test_csvs(VARIANT_NAME, N_TEST_CFGS)
        gru_predictions = collect_gru_predictions(model, csvs)
    except Exception as e:
        print(f"\n[ERROR] Failed to collect GRU predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 2: Fit Bayesian parameters
    try:
        fitted_params, opt_result = fit_bayesian_parameters(gru_predictions, true_sigma)
    except Exception as e:
        print(f"\n[ERROR] Failed to fit Bayesian parameters: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 3: Generate comparison plots
    try:
        generate_comparison_plots(gru_predictions, fitted_params, true_sigma)
    except Exception as e:
        print(f"\n[ERROR] Failed to generate comparison plots: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {OUT_DIR}")
    print("\nGenerated files:")
    print("  - gru_predictions.csv")
    print("  - fitted_parameters.json")
    print("  - comparison_summary.json")
    print("  - detailed_comparison.csv")
    print("  - comparison_report_head.png")
    print("  - comparison_hazard_head.png")
    print("  - probability_distributions.png")
    print("  - accuracy_deltas.png")


if __name__ == "__main__":
    main()

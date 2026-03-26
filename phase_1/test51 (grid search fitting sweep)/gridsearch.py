#!/usr/bin/env python3
# gridsearch_simple_multi_seeds.py
# Grid search to find best (sigma, bias) that makes Bayesian model match trained GRUs
# Now loops over 10 trained GRUs: hz_flat_0_1 seed_0 ... seed_9

import os
import glob
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, norm
import torch
from rnn_models import GRUModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "figures", "grid_search")
os.makedirs(OUT_DIR, exist_ok=True)

# ========================= CONFIGURATION =========================
# Where your multiple trained GRUs live:
# assumed structure: hz_flat_0_1/seed_0, ..., seed_9
MODEL_GROUP = "hz_flat_0_1"
SEEDS = list(range(10))      # seed_0 ... seed_9

N_TEST_CFGS = 50

# Grid search parameters
SIGMA_GRID = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
BIAS_GRID = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

# ========================= BAYESIAN OBSERVER =========================
def BayesianObserver_withBias(ev, mu1, mu2, sigma, hs, bias=0.0):
    """Bayesian observer with optional prior bias (from binned_accuracy.py)"""
    nEvidence = len(ev)
    L_n = [np.zeros((len(hs), nEvidence + 1)) for _ in range(2)]
    
    alpha = 1.0
    beta_param = 1.0
    
    if bias > 0:
        alpha += bias
    elif bias < 0:
        beta_param -= bias
    
    beta_prior = beta_dist.pdf(hs, alpha, beta_param)
    marg = 2 * len(hs)
    
    L_n[0][:, 0] = beta_prior / marg
    L_n[1][:, 0] = L_n[0][:, 0]
    
    P_check = np.sum(L_n[0][:, 0]) + np.sum(L_n[1][:, 0])
    P_diff = 1 - P_check
    P_marg = P_diff / (2 * len(hs))
    L_n[0][:, 0] += P_marg
    L_n[1][:, 0] += P_marg
    
    if sigma != 0:
        norm_P_S1 = norm.pdf(ev, mu1, sigma)
        norm_P_S2 = norm.pdf(ev, mu2, sigma)
    
    for n in range(nEvidence):
        for s in range(2):
            for h in range(len(hs)):
                if s == 0:
                    if sigma == 0:
                        P_S1 = 1 if int(ev[n]) == mu1 else 0
                    else:
                        P_S1 = norm_P_S1[n]
                    L_n[0][h, n + 1] = P_S1 * ((1 - hs[h]) * L_n[0][h, n] + hs[h] * L_n[1][h, n])
                elif s == 1:
                    if sigma == 0:
                        P_S2 = 1 if int(ev[n]) == mu2 else 0
                    else:
                        P_S2 = norm_P_S2[n]
                    L_n[1][h, n + 1] = P_S2 * ((1 - hs[h]) * L_n[1][h, n] + hs[h] * L_n[0][h, n])
        
        T = np.sum(L_n[0][:, n + 1]) + np.sum(L_n[1][:, n + 1])
        if T > 0:
            L_n[0][:, n + 1] /= T
            L_n[1][:, n + 1] /= T
    
    # Compute state likelihoods
    L_state = np.zeros((2, nEvidence + 1))
    for n in range(nEvidence + 1):
        for s in range(2):
            L_state[s, n] = np.sum(L_n[s][:, n])
    
    # Report response
    P_s1 = L_state[0, -1]
    P_s2 = L_state[1, -1]
    
    if P_s1 > P_s2:
        resp_Rep = -1
    elif P_s1 < P_s2:
        resp_Rep = 1
    else:
        resp_Rep = np.random.choice([-1, 1])
    
    # Hazard prediction
    L_haz = np.zeros((len(hs), nEvidence + 1))
    for n in range(nEvidence + 1):
        for h in range(len(hs)):
            L_haz[h, n] = L_n[0][h, n] + L_n[1][h, n]
    
    P_haz_switch = hs * L_haz[:, -1]
    P_haz_stay = (1 - hs) * L_haz[:, -1]
    P_stay = np.sum(P_haz_stay)
    P_switch = np.sum(P_haz_switch)
    
    if P_stay == 0.5 and P_switch == 0.5:
        resp_Pred = np.random.choice([-1, 1])
    elif P_stay > P_switch:
        resp_Pred = -1
    elif P_stay < P_switch:
        resp_Pred = 1
    else:
        resp_Pred = 1
    
    return resp_Rep, resp_Pred


# ========================= GRU MODEL =========================
def load_gru_model(hp_config, checkpoint_path):
    """Load trained GRU model for a specific seed"""
    hp = {
        'n_input': hp_config['n_input'],
        'n_rnn': hp_config['n_rnn'],
        'bidirectional': False
    }
    
    model = GRUModel(hp)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ Loaded GRU checkpoint from {checkpoint_path}")
    return model


def get_gru_predictions(model, evidence_seq):
    """Get GRU predictions for a single trial"""
    with torch.no_grad():
        # Convert evidence to tensor (B=1, T, n_input=1)
        x = torch.tensor(evidence_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        loc_logits, haz_logits = model(x)
        
        # Report prediction (final timestep location)
        loc_final = loc_logits[0, -1, 0].item()
        resp_Rep = 1 if loc_final > 0 else -1
        
        # Hazard prediction
        haz_final = haz_logits[0, 0].item()
        resp_Pred = 1 if haz_final > 0 else -1
        
    return resp_Rep, resp_Pred


# ========================= BINNED ACCURACY =========================
BIN_WIDTH = 0.05
BIN_EDGES = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def bin_accuracy(hazards, correct_mask):
    """Compute accuracy within hazard bins (from binned_accuracy.py)"""
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


# ========================= DATA LOADING =========================
def test_csvs(variant_dir):
    """Get first N_TEST_CFGS test CSV files (same as binned_accuracy.py)"""
    paths = sorted(glob.glob(os.path.join(variant_dir, "testConfig_*.csv")))[:N_TEST_CFGS]
    if len(paths) < N_TEST_CFGS:
        print(f"[warn] only {len(paths)} test configs found in {variant_dir}")
    return paths


# ========================= GRID SEARCH =========================
def eval_agreement(gru_model, csvs, belief_sigma, bias):
    """
    Evaluate agreement between GRU and Bayesian observer
    Returns both overall agreement and binned accuracy for both models
    """
    # Storage for agreement
    rep_matches = 0
    haz_matches = 0
    n_trials = 0
    
    # Storage for binned accuracy
    hazards = []
    gru_rep_correct = []
    gru_haz_correct = []
    bayes_rep_correct = []
    bayes_haz_correct = []
    
    # Parameters for Bayesian observer
    mu1, mu2 = -1, 1
    hs = np.arange(0, 1, 0.05)
    
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Parse evidence
            ev = row["evidence"]
            if not isinstance(ev, list):
                ev = ast.literal_eval(str(ev))
            
            # Get true answers
            true_report = row["trueReport"]
            true_hazard = row["trueHazard"]
            true_rep_label = 1 if true_report > 0 else -1
            true_haz_label = 1 if true_hazard > 0.5 else -1
            
            # Get GRU predictions
            gru_rep, gru_pred = get_gru_predictions(gru_model, ev)
            
            # Get Bayesian predictions
            bayes_rep, bayes_pred = BayesianObserver_withBias(
                ev, mu1, mu2, belief_sigma, hs, bias
            )
            
            # Check agreement between models
            rep_matches += (gru_rep == bayes_rep)
            haz_matches += (gru_pred == bayes_pred)
            n_trials += 1
            
            # Check accuracy against ground truth for binned plots
            hazards.append(true_hazard)
            gru_rep_correct.append(gru_rep == true_rep_label)
            gru_haz_correct.append(gru_pred == true_haz_label)
            bayes_rep_correct.append(bayes_rep == true_rep_label)
            bayes_haz_correct.append(bayes_pred == true_haz_label)
    
    # Overall agreement
    rep_agreement = rep_matches / n_trials if n_trials > 0 else 0
    haz_agreement = haz_matches / n_trials if n_trials > 0 else 0
    
    # Binned accuracy
    hazards = np.array(hazards)
    gru_rep_acc, _, _ = bin_accuracy(hazards, np.array(gru_rep_correct))
    gru_haz_acc, _, _ = bin_accuracy(hazards, np.array(gru_haz_correct))
    bayes_rep_acc, _, _ = bin_accuracy(hazards, np.array(bayes_rep_correct))
    bayes_haz_acc, _, _ = bin_accuracy(hazards, np.array(bayes_haz_correct))
    
    binned_data = {
        'gru_report': gru_rep_acc,
        'gru_hazard': gru_haz_acc,
        'bayes_report': bayes_rep_acc,
        'bayes_hazard': bayes_haz_acc,
    }
    
    return rep_agreement, haz_agreement, n_trials, binned_data


def run_grid_search(gru_model, csvs):
    """Run grid search over all (sigma, bias) combinations"""
    results = []
    binned_results = []  # Store binned accuracy for each grid point
    total = len(SIGMA_GRID) * len(BIAS_GRID)
    current = 0
    
    print(f"\nRunning grid search over {total} combinations...")
    print(f"Sigma values: {SIGMA_GRID}")
    print(f"Bias values: {BIAS_GRID}\n")
    
    for sigma in SIGMA_GRID:
        for bias in BIAS_GRID:
            current += 1
            rep_agree, haz_agree, n, binned_data = eval_agreement(gru_model, csvs, sigma, bias)
            combined = 0.5 * rep_agree + 0.5 * haz_agree
            
            results.append({
                'sigma': sigma,
                'bias': bias,
                'report_agreement': rep_agree,
                'hazard_agreement': haz_agree,
                'combined_score': combined,
                'n_trials': n
            })
            
            # Store binned accuracy for this grid point
            for i, center in enumerate(BIN_CENTERS):
                binned_results.append({
                    'sigma': sigma,
                    'bias': bias,
                    'bin_center': center,
                    'gru_report_acc': binned_data['gru_report'][i],
                    'gru_hazard_acc': binned_data['gru_hazard'][i],
                    'bayes_report_acc': binned_data['bayes_report'][i],
                    'bayes_hazard_acc': binned_data['bayes_hazard'][i],
                })
            
            print(f"[{current:2d}/{total}] σ={sigma:.2f}, bias={bias:+.1f} | "
                  f"rep={rep_agree:.3f}, haz={haz_agree:.3f}, comb={combined:.3f}")
    
    return pd.DataFrame(results), pd.DataFrame(binned_results)


# ========================= VISUALIZATION =========================
def plot_heatmaps(df, dataset_name):
    """Create heatmap visualizations"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        ('report_agreement', 'Report Agreement'),
        ('hazard_agreement', 'Hazard Agreement'),
        ('combined_score', 'Combined Score')
    ]
    
    for ax, (metric, title) in zip(axes, metrics):
        pivot = df.pivot(index='bias', columns='sigma', values=metric)
        
        im = ax.imshow(pivot, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, origin='lower')
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f'{s:.2f}' for s in pivot.columns], rotation=45)
        ax.set_yticklabels([f'{b:+.1f}' for b in pivot.index])
        
        ax.set_xlabel('Belief Sigma (σ)', fontsize=12)
        ax.set_ylabel('Bias', fontsize=12)
        ax.set_title(f'{title}\n(GRU vs Bayesian)', fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Agreement', rotation=270, labelpad=20)
        
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                color = "white" if value < 0.5 else "black"
                ax.text(j, i, f'{value:.2f}', ha="center", va="center", 
                        color=color, fontsize=8)
        
        # Mark best point on combined score plot
        if metric == 'combined_score':
            best_idx = df[metric].idxmax()
            best_row = df.loc[best_idx]
            sigma_idx = list(pivot.columns).index(best_row['sigma'])
            bias_idx = list(pivot.index).index(best_row['bias'])
            ax.plot(sigma_idx, bias_idx, 'r*', markersize=20, 
                    markeredgecolor='white', markeredgewidth=2)
    
    fig.suptitle(f'Grid Search: Fitting Bayesian Model to GRU ({dataset_name})', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(OUT_DIR, f"grid_search_heatmap_{dataset_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved heatmap: {save_path}")


def plot_1d_slices(df, dataset_name):
    """Create 1D slice plots"""
    best_idx = df['combined_score'].idxmax()
    best = df.loc[best_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Slice at best bias
    slice_sigma = df[df['bias'] == best['bias']].sort_values('sigma')
    ax1.plot(slice_sigma['sigma'], slice_sigma['report_agreement'], 
             'o-', label='Report', linewidth=2, markersize=8)
    ax1.plot(slice_sigma['sigma'], slice_sigma['hazard_agreement'], 
             's-', label='Hazard', linewidth=2, markersize=8)
    ax1.plot(slice_sigma['sigma'], slice_sigma['combined_score'], 
             '^-', label='Combined', linewidth=2, markersize=8)
    ax1.axvline(best['sigma'], color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Belief Sigma (σ)', fontsize=12)
    ax1.set_ylabel('Agreement', fontsize=12)
    ax1.set_title(f'Agreement vs Sigma (bias={best["bias"]:+.1f})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Slice at best sigma
    slice_bias = df[df['sigma'] == best['sigma']].sort_values('bias')
    ax2.plot(slice_bias['bias'], slice_bias['report_agreement'], 
             'o-', label='Report', linewidth=2, markersize=8)
    ax2.plot(slice_bias['bias'], slice_bias['hazard_agreement'], 
             's-', label='Hazard', linewidth=2, markersize=8)
    ax2.plot(slice_bias['bias'], slice_bias['combined_score'], 
             '^-', label='Combined', linewidth=2, markersize=8)
    ax2.axvline(best['bias'], color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Bias', fontsize=12)
    ax2.set_ylabel('Agreement', fontsize=12)
    ax2.set_title(f'Agreement vs Bias (σ={best["sigma"]:.2f})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"grid_search_slices_{dataset_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved slice plots: {save_path}")


def plot_binned_comparison(binned_df, best_sigma, best_bias, dataset_name):
    """
    Plot GRU vs best-fit Bayesian observer binned accuracy
    Similar to binned_accuracy.py output
    """
    # Get data for best fit
    best_data = binned_df[(binned_df['sigma'] == best_sigma) & 
                          (binned_df['bias'] == best_bias)]
    
    # Create two subplots: report and hazard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Report accuracy
    ax1.plot(best_data['bin_center'], best_data['gru_report_acc'], 
             'o-', color='tab:blue', linewidth=2.5, markersize=6, 
             label=f'GRU (trained)', alpha=0.8)
    ax1.plot(best_data['bin_center'], best_data['bayes_report_acc'], 
             's--', color='tab:orange', linewidth=2.5, markersize=6,
             label=f'Bayesian (σ={best_sigma:.2f}, bias={best_bias:+.1f})', alpha=0.8)
    
    ax1.set_xlabel('True Hazard Rate (bin centers, width=0.05)', fontsize=12)
    ax1.set_ylabel('Report Accuracy', fontsize=12)
    ax1.set_title(f'Report Head: GRU vs Best-Fit Bayesian\n({dataset_name})', 
                  fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    
    # Hazard accuracy
    ax2.plot(best_data['bin_center'], best_data['gru_hazard_acc'], 
             'o-', color='tab:blue', linewidth=2.5, markersize=6,
             label=f'GRU (trained)', alpha=0.8)
    ax2.plot(best_data['bin_center'], best_data['bayes_hazard_acc'], 
             's--', color='tab:orange', linewidth=2.5, markersize=6,
             label=f'Bayesian (σ={best_sigma:.2f}, bias={best_bias:+.1f})', alpha=0.8)
    
    ax2.set_xlabel('True Hazard Rate (bin centers, width=0.05)', fontsize=12)
    ax2.set_ylabel('Hazard Prediction Accuracy', fontsize=12)
    ax2.set_title(f'Hazard Head: GRU vs Best-Fit Bayesian\n({dataset_name})', 
                  fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"gru_vs_bayesian_binned_{dataset_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved binned comparison: {save_path}")


def plot_all_bayesian_fits(binned_df, results_df, dataset_name):
    """
    Plot GRU vs multiple Bayesian parameter settings
    Shows how different (sigma, bias) compare
    """
    # Get GRU data (same across all parameter settings)
    gru_data = binned_df[(binned_df['sigma'] == SIGMA_GRID[0]) & 
                         (binned_df['bias'] == BIAS_GRID[0])]
    
    # Select a few interesting parameter combinations to visualize
    best_combined = results_df.loc[results_df['combined_score'].idxmax()]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top candidates to visualize
    top_n = 5
    top_results = results_df.nlargest(top_n, 'combined_score')
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, top_n))
    
    # Report accuracy - top fits
    ax = axes[0, 0]
    ax.plot(gru_data['bin_center'], gru_data['gru_report_acc'], 
            'o-', color='black', linewidth=3, markersize=8, 
            label='GRU (trained)', alpha=0.9, zorder=100)
    
    for idx, (_, row) in enumerate(top_results.iterrows()):
        data = binned_df[(binned_df['sigma'] == row['sigma']) & 
                         (binned_df['bias'] == row['bias'])]
        ax.plot(data['bin_center'], data['bayes_report_acc'], 
                '--', color=colors[idx], linewidth=2, markersize=4,
                label=f"σ={row['sigma']:.2f}, b={row['bias']:+.1f} (score={row['combined_score']:.3f})",
                alpha=0.7)
    
    ax.set_xlabel('True Hazard Rate', fontsize=11)
    ax.set_ylabel('Report Accuracy', fontsize=11)
    ax.set_title('Report Head: Top 5 Bayesian Fits', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)
    
    # Hazard accuracy - top fits
    ax = axes[0, 1]
    ax.plot(gru_data['bin_center'], gru_data['gru_hazard_acc'], 
            'o-', color='black', linewidth=3, markersize=8,
            label='GRU (trained)', alpha=0.9, zorder=100)
    
    for idx, (_, row) in enumerate(top_results.iterrows()):
        data = binned_df[(binned_df['sigma'] == row['sigma']) & 
                         (binned_df['bias'] == row['bias'])]
        ax.plot(data['bin_center'], data['bayes_hazard_acc'], 
                '--', color=colors[idx], linewidth=2, markersize=4,
                label=f"σ={row['sigma']:.2f}, b={row['bias']:+.1f} (score={row['combined_score']:.3f})",
                alpha=0.7)
    
    ax.set_xlabel('True Hazard Rate', fontsize=11)
    ax.set_ylabel('Hazard Accuracy', fontsize=11)
    ax.set_title('Hazard Head: Top 5 Bayesian Fits', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)
    
    # Bottom: vary sigma at best bias
    ax = axes[1, 0]
    ax.plot(gru_data['bin_center'], gru_data['gru_report_acc'], 
            'o-', color='black', linewidth=3, markersize=8,
            label='GRU', alpha=0.9, zorder=100)
    
    best_bias = best_combined['bias']
    for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
        if sigma in SIGMA_GRID:
            data = binned_df[(binned_df['sigma'] == sigma) & 
                             (binned_df['bias'] == best_bias)]
            ax.plot(data['bin_center'], data['bayes_report_acc'], 
                    '--', linewidth=2, label=f'σ={sigma:.2f}', alpha=0.7)
    
    ax.set_xlabel('True Hazard Rate', fontsize=11)
    ax.set_ylabel('Report Accuracy', fontsize=11)
    ax.set_title(f'Varying Sigma (bias={best_bias:+.1f})', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)
    
    # Bottom right: vary bias at best sigma
    ax = axes[1, 1]
    ax.plot(gru_data['bin_center'], gru_data['gru_hazard_acc'], 
            'o-', color='black', linewidth=3, markersize=8,
            label='GRU', alpha=0.9, zorder=100)
    
    best_sigma = best_combined['sigma']
    for bias in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        if bias in BIAS_GRID:
            data = binned_df[(binned_df['sigma'] == best_sigma) & 
                             (binned_df['bias'] == bias)]
            ax.plot(data['bin_center'], data['bayes_hazard_acc'], 
                    '--', linewidth=2, label=f'bias={bias:+.1f}', alpha=0.7)
    
    ax.set_xlabel('True Hazard Rate', fontsize=11)
    ax.set_ylabel('Hazard Accuracy', fontsize=11)
    ax.set_title(f'Varying Bias (σ={best_sigma:.2f})', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"gru_vs_bayesian_variants_{dataset_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved variants comparison: {save_path}")


# ========================= MAIN =========================
def main():
    print("="*70)
    print("GRID SEARCH: Fitting Bayesian Observer to Trained GRUs (multiple seeds)")
    print("="*70)
    
    for seed in SEEDS:
        print("\n" + "="*70)
        print(f"PROCESSING SEED {seed}")
        print("="*70)
        
        model_dir = os.path.join(BASE_DIR, MODEL_GROUP, f"seed_{seed}")
        hp_path = os.path.join(model_dir, "hp.json")
        checkpoint_path = os.path.join(model_dir, "checkpoint_best.pt")
        
        if not os.path.exists(hp_path):
            print(f"[ERROR] hp.json not found for seed {seed}: {hp_path}")
            continue
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] checkpoint_best.pt not found for seed {seed}: {checkpoint_path}")
            continue
        
        # Load hyperparameters for this seed
        with open(hp_path, 'r') as f:
            hp_config = json.load(f)
        
        dataset_name = hp_config.get('sigma_key', MODEL_GROUP)
        dataset_seed_name = f"{dataset_name}_seed{seed}"
        
        print(f"Dataset: {dataset_name}")
        print(f"GRU architecture: n_input={hp_config['n_input']}, n_rnn={hp_config['n_rnn']}")
        
        # Load GRU model
        gru_model = load_gru_model(hp_config, checkpoint_path)
        
        # Load test data (same for all seeds, keyed by dataset_name)
        variant_dir = os.path.join(BASE_DIR, "variants", "sigma_1")
        if not os.path.isdir(variant_dir):
            print(f"\nError: Variant directory not found: {variant_dir}")
            continue
        
        csvs = test_csvs(variant_dir)
        print(f"✓ Found {len(csvs)} test configs in {variant_dir}")
        
        # Run grid search for this seed
        results_df, binned_df = run_grid_search(gru_model, csvs)
        
        # Save results (seed-specific filenames)
        csv_path = os.path.join(OUT_DIR, f"grid_search_results_{dataset_seed_name}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved overall results to {csv_path}")
        
        # Save binned accuracy for all grid points
        binned_csv_path = os.path.join(OUT_DIR, f"grid_search_binned_{dataset_seed_name}.csv")
        binned_df.to_csv(binned_csv_path, index=False)
        print(f"✓ Saved binned accuracy data to {binned_csv_path}")
        
        # Find and display best parameters
        best_combined = results_df.loc[results_df['combined_score'].idxmax()]
        best_report = results_df.loc[results_df['report_agreement'].idxmax()]
        best_hazard = results_df.loc[results_df['hazard_agreement'].idxmax()]
        
        print("\n" + "-"*70)
        print(f"RESULTS FOR SEED {seed}")
        print("-"*70)
        
        print(f"\n🏆 Best Combined Score:")
        print(f"   σ = {best_combined['sigma']:.2f}, bias = {best_combined['bias']:+.2f}")
        print(f"   Report:   {best_combined['report_agreement']:.3f}")
        print(f"   Hazard:   {best_combined['hazard_agreement']:.3f}")
        print(f"   Combined: {best_combined['combined_score']:.3f}")
        
        print(f"\n📊 Best Report Agreement:")
        print(f"   σ = {best_report['sigma']:.2f}, bias = {best_report['bias']:+.2f}")
        print(f"   Agreement: {best_report['report_agreement']:.3f}")
        
        print(f"\n📊 Best Hazard Agreement:")
        print(f"   σ = {best_hazard['sigma']:.2f}, bias = {best_hazard['bias']:+.2f}")
        print(f"   Agreement: {best_hazard['hazard_agreement']:.3f}")
        
        # Generate plots
        print("\nGenerating visualizations...")
        plot_heatmaps(results_df, dataset_seed_name)
        plot_1d_slices(results_df, dataset_seed_name)
        plot_binned_comparison(
            binned_df,
            best_combined['sigma'],
            best_combined['bias'],
            dataset_seed_name
        )
        plot_all_bayesian_fits(binned_df, results_df, dataset_seed_name)
        
        print(f"\n✓ Grid search complete for seed {seed}!")
        print(f"Results saved to: {OUT_DIR}")
        
        # Summary of files created for this seed
        print("\n" + "-"*70)
        print(f"OUTPUT FILES FOR SEED {seed}")
        print("-"*70)
        print(f"1. {csv_path}")
        print(f"2. {binned_csv_path}")
        print(f"3. {OUT_DIR}/grid_search_heatmap_{dataset_seed_name}.png")
        print(f"4. {OUT_DIR}/grid_search_slices_{dataset_seed_name}.png")
        print(f"5. {OUT_DIR}/gru_vs_bayesian_binned_{dataset_seed_name}.png")
        print(f"6. {OUT_DIR}/gru_vs_bayesian_variants_{dataset_seed_name}.png")
    
    print("\n" + "="*70)
    print("ALL SEEDS PROCESSED")
    print("="*70)


if __name__ == "__main__":
    main()

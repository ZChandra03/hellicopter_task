#!/usr/bin/env python3
# binned_accuracy.py
# Evaluates Bayesian normative model variants on sigma_1 and sigma_2 test configs.
# Includes: matched sigma, mismatched sigma, and biased priors.
# Saves binned accuracy figures showing performance across hazard rates.

import os
import glob
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_TEST_CFGS = 50
BIN_WIDTH = 0.01
BIN_EDGES = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# Dataset configurations (true environment parameters)
DATASET_CONFIGS = [
    ("sigma_1", 1.0),
    #("sigma_2", 2.0)
]

# Bayesian observer variants to test
# Format: (name, sigma_belief, bias, color, linestyle)
BAYESIAN_VARIANTS = [
    ("Matched σ", "matched", 0.0, "tab:blue", "-"),
    ("σ=0.5 (underestimate)", 0.5, 0.0, "tab:orange", "--"),
    ("σ=2.0 (overestimate)", 2.0, 0.0, "tab:green", "-."),
    ("Bias +2 (expect switch)", "matched", 2.0, "tab:red", (0, (3, 1, 1, 1))),
    ("Bias -2 (expect stay)", "matched", -2.0, "tab:purple", (0, (3, 1, 1, 1, 1, 1))),
    ("Bias +1 (expect switch)", "matched", 1.0, "tab:grey", (0, (3, 1, 1, 1))),
    ("Bias -1 (expect switch)", "matched", -1.0, "tab:brown", (0, (3, 1, 1, 1))),
]

def BayesianObserver_withBias(ev, mu1, mu2, sigma, hs, bias=0.0):
    """
    Bayesian observer with optional prior bias.
    
    Parameters:
    - ev: evidence sequence
    - mu1, mu2: means of the two states
    - sigma: noise level (standard deviation)
    - hs: array of possible hazard rates
    - bias: prior bias parameter (positive = expect switches, negative = expect stays)
    
    Returns:
    - L_haz: likelihood over hazard rates at each evidence step
    - L_state: likelihood over states at each evidence step
    - resp_Rep: report response (-1 or 1)
    - resp_Pred: prediction response (-1 or 1)
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
                if s == 0:  # State 1
                    if sigma == 0:
                        if int(ev[n]) == mu1:
                            P_S1 = 1
                        elif int(ev[n]) == mu2:
                            P_S1 = 0
                        else:
                            P_S1 = 0
                    else:
                        P_S1 = norm_P_S1[n]
                    
                    L_n[0][h, n + 1] = P_S1 * ((1 - hs[h]) * L_n[0][h, n] + hs[h] * L_n[1][h, n])
                    
                elif s == 1:  # State 2
                    if sigma == 0:
                        if int(ev[n]) == mu2:
                            P_S2 = 1
                        elif int(ev[n]) == mu1:
                            P_S2 = 0
                        else:
                            P_S2 = 0
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
    P_s1 = L_state[0, -1]
    P_s2 = L_state[1, -1]
    
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
    
    if P_stay == 0.5 and P_switch == 0.5:
        resp_Pred = np.random.choice([-1, 1])
    elif P_stay > P_switch:
        resp_Pred = -1
    elif P_stay < P_switch:
        resp_Pred = 1
    else:
        resp_Pred = 1  # default to switch
    
    return L_haz, L_state, resp_Rep, resp_Pred

def test_csvs(variant_dir):
    """Get first N_TEST_CFGS test CSV files from variant directory."""
    paths = sorted(glob.glob(os.path.join(variant_dir, "testConfig_*.csv")))[:N_TEST_CFGS]
    if len(paths) < N_TEST_CFGS:
        print(f"[warn] only {len(paths)} test configs found in {variant_dir}")
    return paths

def eval_bayesian_variant(csvs, true_sigma, belief_sigma, bias):
    """
    Evaluate Bayesian observer variant on all trials in CSVs.
    
    Parameters:
    - csvs: list of CSV file paths
    - true_sigma: actual noise level in the environment (not used in computation, just for tracking)
    - belief_sigma: sigma the observer believes (can be mismatched)
    - bias: prior bias parameter
    """
    hazards, rep_ok, haz_ok = [], [], []
    
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
            
            # Run Bayesian observer with specified parameters
            L_haz, L_state, resp_Rep, resp_Pred = BayesianObserver_withBias(
                ev, mu1, mu2, belief_sigma, hs, bias
            )
            
            # Check report accuracy
            true_report = row["trueReport"]
            rep_correct = (resp_Rep == (1 if true_report > 0 else -1))
            rep_ok.append(rep_correct)
            
            # Check hazard prediction accuracy
            true_hazard = row["trueHazard"]
            haz_correct = (resp_Pred == (1 if true_hazard > 0.5 else -1))
            haz_ok.append(haz_correct)
            
            hazards.append(true_hazard)
    
    return np.array(hazards), np.array(rep_ok, dtype=bool), np.array(haz_ok, dtype=bool)

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

def save_fig(fig, filename_base):
    """Save figure to output directory."""
    png = os.path.join(OUT_DIR, f"{filename_base}.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")

def main():
    # Loop over each dataset (sigma_1, sigma_2)
    for dataset_name, true_sigma in DATASET_CONFIGS:
        variant_dir = os.path.join(BASE_DIR, "variants", dataset_name)
        if not os.path.isdir(variant_dir):
            print(f"[warn] missing variant folder: {dataset_name} – skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name} (true σ={true_sigma})")
        print(f"{'='*60}")
        
        csvs = test_csvs(variant_dir)
        
        # Collect accuracy data for each Bayesian variant
        rep_acc_data = {}
        haz_acc_data = {}
        
        for variant_name, sigma_belief, bias, color, linestyle in BAYESIAN_VARIANTS:
            # Handle "matched" sigma (match to true environment)
            if sigma_belief == "matched":
                belief_sigma = true_sigma
            else:
                belief_sigma = sigma_belief
            
            print(f"  [{variant_name}] belief_σ={belief_sigma}, bias={bias}")
            hz, rep_ok, haz_ok = eval_bayesian_variant(csvs, true_sigma, belief_sigma, bias)
            
            acc_rep, _, _ = bin_accuracy(hz, rep_ok)
            acc_haz, _, _ = bin_accuracy(hz, haz_ok)
            
            rep_acc_data[variant_name] = (acc_rep, color, linestyle)
            haz_acc_data[variant_name] = (acc_haz, color, linestyle)
        
        # Plot report head accuracy
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        for variant_name, (acc, color, linestyle) in rep_acc_data.items():
            ax1.plot(BIN_CENTERS, acc, color=color, linewidth=2.2, 
                    label=variant_name, linestyle=linestyle, marker='o', markersize=4)
        ax1.set_xlabel("True hazard (bin centers, width = 0.05)", fontsize=12)
        ax1.set_ylabel("Report accuracy", fontsize=12)
        ax1.set_title(f"Bayesian Model Variants: Report Accuracy on {dataset_name} (true σ={true_sigma})", 
                     fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False, fontsize=10, loc='best')
        fig1.tight_layout()
        save_fig(fig1, f"bayesian_report_accuracy_{dataset_name}")
        
        # Plot hazard head accuracy
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        for variant_name, (acc, color, linestyle) in haz_acc_data.items():
            ax2.plot(BIN_CENTERS, acc, color=color, linewidth=2.2, 
                    label=variant_name, linestyle=linestyle, marker='o', markersize=4)
        ax2.set_xlabel("True hazard (bin centers, width = 0.05)", fontsize=12)
        ax2.set_ylabel("Hazard prediction accuracy", fontsize=12)
        ax2.set_title(f"Bayesian Model Variants: Hazard Prediction on {dataset_name} (true σ={true_sigma})", 
                     fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=10, loc='best')
        fig2.tight_layout()
        save_fig(fig2, f"bayesian_hazard_accuracy_{dataset_name}")
        
        # Save CSV of results for this dataset
        rows = []
        for variant_name in rep_acc_data.keys():
            acc_rep, _, _ = rep_acc_data[variant_name]
            acc_haz, _, _ = haz_acc_data[variant_name]
            for i, center in enumerate(BIN_CENTERS):
                rows.append({
                    "dataset": dataset_name,
                    "true_sigma": true_sigma,
                    "variant": variant_name,
                    "bin_center": float(center),
                    "report_accuracy": float(acc_rep[i]),
                    "hazard_accuracy": float(acc_haz[i])
                })
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, f"bayesian_binned_accuracy_{dataset_name}.csv"), index=False
        )
        print(f"  Saved results CSV -> figures/bayesian_binned_accuracy_{dataset_name}.csv")

if __name__ == "__main__":
    main()
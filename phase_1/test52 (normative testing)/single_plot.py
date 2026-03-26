#!/usr/bin/env python3
"""
plot_single_trial.py
====================
Pulls a single trial from sigma_1 test configs and visualizes:
- Evidence sequence over time
- True underlying states
- True hazard rate
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGMA_1_DIR = os.path.join(BASE_DIR, "variants", "sigma_1")
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def load_trial(config_idx=1, trial_idx=1):
    """Load a specific trial from sigma_1 test configs."""
    csv_path = os.path.join(SIGMA_1_DIR, f"testConfig_{config_idx}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Config file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if trial_idx >= len(df):
        raise ValueError(f"Trial index {trial_idx} out of range (max: {len(df)-1})")
    
    trial = df.iloc[trial_idx]
    
    # Parse evidence and states
    evidence = trial['evidence']
    if not isinstance(evidence, list):
        evidence = ast.literal_eval(str(evidence))
    
    states = trial['states']
    if not isinstance(states, list):
        states = ast.literal_eval(str(states))
    
    return {
        'evidence': np.array(evidence),
        'states': np.array(states),
        'trueHazard': trial['trueHazard'],
        'trueReport': trial['trueReport'],
        'truePredict': trial['truePredict'],
        'sigma': trial['sigma'],
        'config_idx': config_idx,
        'trial_idx': trial_idx
    }

def plot_trial(trial_data, save=True):
    """Create comprehensive visualization of a single trial."""
    evidence = trial_data['evidence']
    states = trial_data['states']
    hazard = trial_data['trueHazard']
    n_steps = len(evidence)
    time_steps = np.arange(n_steps)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # ============ Panel 1: Evidence Sequence ============
    ax1 = axes[0]
    ax1.plot(time_steps, evidence, 'o-', color='steelblue', 
             linewidth=2, markersize=8, alpha=0.7, label='Evidence')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=1, color='salmon', linestyle=':', alpha=0.5, linewidth=1.5, label='μ₂ = +1')
    ax1.axhline(y=-1, color='mediumseagreen', linestyle=':', alpha=0.5, linewidth=1.5, label='μ₁ = -1')
    ax1.set_ylabel('Evidence Value', fontsize=12, fontweight='bold')
    ax1.set_title(f'Single Trial Visualization (Config {trial_data["config_idx"]}, Trial {trial_data["trial_idx"]})\n'
                  f'σ = {trial_data["sigma"]}, True Hazard = {hazard:.3f}', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(-5.5, 5.5)
    
    # ============ Panel 2: True Hidden States ============
    ax2 = axes[1]
    ax2.plot(time_steps, states, 's-', color='darkviolet', 
             linewidth=2.5, markersize=10, alpha=0.8, label='True State')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_ylabel('True State (μ)', fontsize=12, fontweight='bold')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['State 1 (μ=-1)', '0', 'State 2 (μ=+1)'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Highlight state changes (switches)
    switches = []
    for i in range(1, n_steps):
        if states[i] != states[i-1]:
            switches.append(i)
            ax2.axvline(x=i-0.5, color='red', linestyle='--', 
                       alpha=0.6, linewidth=2)
    
    # ============ Panel 3: Switch Indicators ============
    ax3 = axes[2]
    switch_indicators = np.zeros(n_steps)
    for s in switches:
        switch_indicators[s] = 1
    
    ax3.bar(time_steps, switch_indicators, color='crimson', 
            alpha=0.7, width=0.8, label='State Switch')
    ax3.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Switch Event', fontsize=12, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No Switch', 'Switch'])
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.legend(loc='upper right', fontsize=10)
    
    # Add text annotation with trial info
    info_text = (f"Number of Switches: {len(switches)}\n"
                f"True Report: {trial_data['trueReport']}\n"
                f"True Predict: {'Switch' if trial_data['truePredict'] == 1 else 'Stay'}")
    ax3.text(0.02, 0.95, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        filename = f"single_trial_config{trial_data['config_idx']}_trial{trial_data['trial_idx']}.png"
        filepath = os.path.join(OUT_DIR, filename)
        fig.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    return fig

def main():
    """Main function to load and plot a trial."""
    # You can change these parameters to view different trials
    CONFIG_IDX = 0  # Which test config file (0 to 49)
    TRIAL_IDX = 0   # Which trial within that config (0 to 299)
    
    print(f"Loading trial from sigma_1...")
    print(f"Config: testConfig_{CONFIG_IDX}.csv, Trial: {TRIAL_IDX}")
    
    trial_data = load_trial(config_idx=CONFIG_IDX, trial_idx=TRIAL_IDX)
    
    print(f"\nTrial Summary:")
    print(f"  Sigma: {trial_data['sigma']}")
    print(f"  True Hazard: {trial_data['trueHazard']:.4f}")
    print(f"  True Report: {trial_data['trueReport']}")
    print(f"  True Predict: {trial_data['truePredict']}")
    print(f"  Evidence length: {len(trial_data['evidence'])}")
    print(f"  Number of switches: {np.sum(np.diff(trial_data['states']) != 0)}")
    
    plot_trial(trial_data, save=True)
    plt.show()

if __name__ == "__main__":
    main()
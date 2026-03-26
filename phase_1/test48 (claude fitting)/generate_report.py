#!/usr/bin/env python3
"""
generate_report.py
==================
Generate a human-readable text report from the fitting results.
Run this after fit_bayesian_to_gru.py completes.

Usage:
    python generate_report.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "bayesian_fitting")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "fitting_report.txt")

def format_header(text, level=1):
    """Format section headers."""
    if level == 1:
        return "\n" + "="*80 + f"\n{text.center(80)}\n" + "="*80 + "\n"
    elif level == 2:
        return "\n" + "-"*80 + f"\n{text}\n" + "-"*80 + "\n"
    else:
        return f"\n{text}\n" + "~"*len(text) + "\n"

def interpret_sigma(fitted_sigma, true_sigma):
    """Interpret the fitted sigma parameter."""
    ratio = fitted_sigma / true_sigma
    
    if 0.95 <= ratio <= 1.05:
        return f"✓ ACCURATE: Fitted σ matches true σ within 5% (ratio: {ratio:.3f})"
    elif ratio < 0.95:
        diff_pct = (1 - ratio) * 100
        return f"⚠ UNDERESTIMATE: Fitted σ is {diff_pct:.1f}% lower than true σ\n" \
               f"  → GRU is OVERCONFIDENT (treats evidence as more reliable than it is)"
    else:
        diff_pct = (ratio - 1) * 100
        return f"⚠ OVERESTIMATE: Fitted σ is {diff_pct:.1f}% higher than true σ\n" \
               f"  → GRU is UNDERCONFIDENT (treats evidence as less reliable than it is)"

def interpret_bias(bias):
    """Interpret the fitted bias parameter."""
    if abs(bias) < 0.2:
        return f"✓ UNBIASED: Bias ≈ 0 (actual: {bias:.3f})\n" \
               f"  → GRU has no systematic prior expectations about volatility"
    elif bias > 0:
        if bias > 2.0:
            strength = "STRONGLY"
        elif bias > 1.0:
            strength = "MODERATELY"
        else:
            strength = "SLIGHTLY"
        return f"⚠ SWITCH-BIASED: Positive bias = {bias:.3f}\n" \
               f"  → GRU {strength} expects the world to be MORE volatile\n" \
               f"  → Prior shifted toward expecting more state switches"
    else:
        if bias < -2.0:
            strength = "STRONGLY"
        elif bias < -1.0:
            strength = "MODERATELY"
        else:
            strength = "SLIGHTLY"
        return f"⚠ STAY-BIASED: Negative bias = {bias:.3f}\n" \
               f"  → GRU {strength} expects the world to be LESS volatile\n" \
               f"  → Prior shifted toward expecting fewer state switches"

def interpret_kappa(kappa, task):
    """Interpret Cohen's kappa value."""
    if kappa >= 0.8:
        quality = "EXCELLENT"
        desc = "GRU and Bayesian make nearly identical choices"
    elif kappa >= 0.6:
        quality = "GOOD"
        desc = "GRU and Bayesian generally agree, with some differences"
    elif kappa >= 0.4:
        quality = "MODERATE"
        desc = "GRU and Bayesian show partial agreement but notable differences"
    else:
        quality = "POOR"
        desc = "GRU and Bayesian use substantially different strategies"
    
    return f"{task}: κ = {kappa:.3f} ({quality})\n  → {desc}"

def interpret_accuracy_gap(gru_acc, fitted_acc, optimal_acc, task):
    """Interpret the accuracy differences."""
    gap_vs_fitted = gru_acc - fitted_acc
    gap_vs_optimal = gru_acc - optimal_acc
    
    report = f"{task} Accuracy:\n"
    report += f"  GRU:             {gru_acc:.4f}\n"
    report += f"  Fitted Bayesian: {fitted_acc:.4f} (Δ = {gap_vs_fitted:+.4f})\n"
    report += f"  Optimal Bayesian:{optimal_acc:.4f} (Δ = {gap_vs_optimal:+.4f})\n"
    
    if abs(gap_vs_fitted) < 0.02:
        report += "  ✓ Excellent fit: GRU accuracy matches fitted Bayesian\n"
    elif abs(gap_vs_fitted) < 0.05:
        report += "  ✓ Good fit: GRU and fitted Bayesian are close\n"
    else:
        report += "  ⚠ Moderate fit: Noticeable difference between GRU and fitted Bayesian\n"
        report += "     Consider: (1) different parameterization, (2) lapse rate, (3) non-Bayesian strategy\n"
    
    if gap_vs_optimal > 0.01:
        report += "  ✓ GRU slightly outperforms optimal Bayesian (possible overfitting)\n"
    elif gap_vs_optimal > -0.02:
        report += "  ✓ GRU performs at optimal Bayesian level\n"
    elif gap_vs_optimal > -0.05:
        report += "  ⚠ GRU slightly underperforms optimal Bayesian\n"
    else:
        report += "  ⚠ GRU significantly underperforms optimal Bayesian\n"
        report += "     → Either: (1) GRU learned sub-optimal strategy, or (2) model misspecification\n"
    
    return report

def generate_report():
    """Generate comprehensive text report."""
    
    # Check if results exist
    fitted_params_file = os.path.join(RESULTS_DIR, "fitted_parameters.json")
    summary_file = os.path.join(RESULTS_DIR, "comparison_summary.json")
    
    if not os.path.exists(fitted_params_file):
        print(f"Error: {fitted_params_file} not found.")
        print("Please run fit_bayesian_to_gru.py first.")
        return
    
    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found.")
        print("Please run fit_bayesian_to_gru.py first.")
        return
    
    # Load results
    with open(fitted_params_file, 'r') as f:
        fitted_params = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Start building report
    report = []
    
    # Title and metadata
    report.append(format_header("BAYESIAN MODEL FITTING REPORT", level=1))
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"Results directory: {RESULTS_DIR}\n")
    
    # Executive Summary
    report.append(format_header("EXECUTIVE SUMMARY", level=2))
    
    sigma_fit = fitted_params['sigma_belief']
    bias_fit = fitted_params['bias']
    true_sigma = fitted_params['true_sigma']
    
    report.append(f"Fitted Parameters:\n")
    report.append(f"  σ_belief = {sigma_fit:.4f} (true σ = {true_sigma:.4f})\n")
    report.append(f"  bias     = {bias_fit:.4f}\n")
    report.append(f"\nFitting Quality:\n")
    report.append(f"  Negative log-likelihood: {fitted_params['nll']:.2f}\n")
    report.append(f"  NLL per trial: {fitted_params['nll_per_trial']:.4f}\n")
    report.append(f"  Number of trials: {fitted_params['n_trials']}\n")
    report.append(f"  Optimization status: {'SUCCESS' if fitted_params['success'] else 'FAILED'}\n")
    
    # Parameter Interpretations
    report.append(format_header("PARAMETER INTERPRETATIONS", level=2))
    
    report.append(format_header("Noise Belief (σ)", level=3))
    report.append(interpret_sigma(sigma_fit, true_sigma) + "\n")
    
    report.append(format_header("Prior Bias", level=3))
    report.append(interpret_bias(bias_fit) + "\n")
    
    # Performance Comparison
    report.append(format_header("PERFORMANCE COMPARISON", level=2))
    
    report.append(interpret_accuracy_gap(
        summary['gru_report_acc'],
        summary['fitted_report_acc'],
        summary['optimal_report_acc'],
        "Report Head"
    ))
    
    report.append("\n")
    
    report.append(interpret_accuracy_gap(
        summary['gru_hazard_acc'],
        summary['fitted_hazard_acc'],
        summary['optimal_hazard_acc'],
        "Hazard Head"
    ))
    
    # Agreement Analysis
    report.append(format_header("AGREEMENT ANALYSIS", level=2))
    report.append("Cohen's Kappa measures how often GRU and fitted Bayesian make the same choice:\n\n")
    report.append(interpret_kappa(summary['report_kappa'], "Report head") + "\n\n")
    report.append(interpret_kappa(summary['hazard_kappa'], "Hazard head") + "\n")
    
    # Overall Conclusion
    report.append(format_header("OVERALL CONCLUSION", level=2))
    
    # Determine overall strategy
    sigma_accurate = 0.9 <= (sigma_fit / true_sigma) <= 1.1
    unbiased = abs(bias_fit) < 0.3
    good_fit = abs(summary['gru_report_acc'] - summary['fitted_report_acc']) < 0.03
    high_agreement = summary['report_kappa'] > 0.7
    
    if sigma_accurate and unbiased and good_fit and high_agreement:
        conclusion = "✓ OPTIMAL BAYESIAN STRATEGY\n\n" \
                    "The GRU has learned to implement nearly optimal Bayesian inference:\n" \
                    "  • Accurate noise model (σ_belief ≈ true σ)\n" \
                    "  • Unbiased prior expectations\n" \
                    "  • Excellent fit to Bayesian behavior\n" \
                    "  • High agreement in individual trial choices\n\n" \
                    "Interpretation: The network successfully discovered the optimal solution\n" \
                    "to the change-point detection task through learning."
    
    elif good_fit and high_agreement:
        conclusion = "⚠ SUB-OPTIMAL BAYESIAN STRATEGY\n\n" \
                    "The GRU implements a Bayesian strategy but with systematic deviations:\n"
        if not sigma_accurate:
            if sigma_fit < true_sigma:
                conclusion += "  • Overconfident: underestimates noise\n"
            else:
                conclusion += "  • Underconfident: overestimates noise\n"
        if not unbiased:
            if bias_fit > 0:
                conclusion += "  • Switch-biased: expects higher volatility\n"
            else:
                conclusion += "  • Stay-biased: expects lower volatility\n"
        conclusion += "\nInterpretation: The network learned a Bayesian-like strategy but with\n" \
                     "incorrect assumptions about the environment."
    
    else:
        conclusion = "⚠ NON-BAYESIAN OR POORLY FIT STRATEGY\n\n" \
                    "The fitted Bayesian model does not closely match GRU behavior:\n"
        if not good_fit:
            conclusion += "  • Large accuracy gap between GRU and fitted Bayesian\n"
        if not high_agreement:
            conclusion += "  • Low agreement in individual trial choices\n"
        conclusion += "\nPossible interpretations:\n" \
                     "  1. GRU uses a non-Bayesian strategy (e.g., heuristics, shortcuts)\n" \
                     "  2. Current parameterization is insufficient (need more parameters)\n" \
                     "  3. GRU has learned a more complex, context-dependent strategy\n\n" \
                     "Recommendations:\n" \
                     "  • Try extended models (separate σ for each head, lapse rate)\n" \
                     "  • Analyze GRU hidden states for representational similarity\n" \
                     "  • Check for systematic patterns in disagreement trials"
    
    report.append(conclusion + "\n")
    
    # Recommendations
    report.append(format_header("NEXT STEPS & RECOMMENDATIONS", level=2))
    
    recommendations = []
    
    if not sigma_accurate:
        recommendations.append(
            "1. NOISE MISESTIMATION ANALYSIS\n"
            "   Investigate why the network has incorrect noise beliefs:\n"
            "   • Plot GRU confidence vs. true evidence reliability\n"
            "   • Check if misestimation varies across hazard rates\n"
            "   • Consider architectural limitations (e.g., sigmoid saturation)"
        )
    
    if not unbiased:
        recommendations.append(
            "2. PRIOR BIAS INVESTIGATION\n"
            "   Understand the source of prior bias:\n"
            "   • Check if bias emerges from training data imbalance\n"
            "   • Analyze if bias is adaptive (helps in some conditions)\n"
            "   • Test if bias varies across different hazard regimes"
        )
    
    if not good_fit:
        recommendations.append(
            "3. EXTENDED PARAMETERIZATION\n"
            "   Try more flexible Bayesian models:\n"
            "   • Separate σ_belief for report vs. hazard predictions\n"
            "   • Add lapse rate parameter (random choices)\n"
            "   • Allow time-varying parameters within trial"
        )
    
    if not high_agreement:
        recommendations.append(
            "4. DISAGREEMENT ANALYSIS\n"
            "   Study trials where GRU and Bayesian disagree:\n"
            "   • Are disagreements random or systematic?\n"
            "   • Do disagreements cluster by hazard rate?\n"
            "   • Check evidence patterns in disagreement trials"
        )
    
    recommendations.append(
        f"{len(recommendations)+1}. GENERALIZATION TESTING\n"
        "   Test fitted parameters across different conditions:\n"
        "   • Fit on σ=1, test on σ=2 (or vice versa)\n"
        "   • Fit on one hazard prior, test on another\n"
        "   • Check if fitted parameters are training-set dependent"
    )
    
    recommendations.append(
        f"{len(recommendations)+1}. REPRESENTATIONAL ANALYSIS\n"
        "   Compare GRU internal representations to Bayesian beliefs:\n"
        "   • Extract GRU hidden states over time\n"
        "   • Compute Bayesian L_state and L_haz beliefs\n"
        "   • Calculate representational similarity (RSA)\n"
        "   • Identify which Bayesian quantities are encoded"
    )
    
    report.append("\n".join(recommendations) + "\n")
    
    # Output files reference
    report.append(format_header("OUTPUT FILES", level=2))
    report.append("Data files:\n")
    report.append("  • gru_predictions.csv       - All GRU predictions on test set\n")
    report.append("  • fitted_parameters.json    - Fitted parameters and metadata\n")
    report.append("  • comparison_summary.json   - Summary statistics\n")
    report.append("  • detailed_comparison.csv   - Trial-by-trial comparison\n")
    report.append("\nVisualizations:\n")
    report.append("  • comparison_report_head.png       - Report accuracy vs. hazard\n")
    report.append("  • comparison_hazard_head.png       - Hazard accuracy vs. hazard\n")
    report.append("  • probability_distributions.png    - Choice probability histograms\n")
    report.append("  • accuracy_deltas.png              - GRU - Bayesian differences\n")
    
    # Footer
    report.append("\n" + "="*80 + "\n")
    report.append("End of report\n")
    report.append("="*80 + "\n")
    
    # Write report
    full_report = "".join(report)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(full_report)
    
    # Also print to console
    print(full_report)
    print(f"\n✓ Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()

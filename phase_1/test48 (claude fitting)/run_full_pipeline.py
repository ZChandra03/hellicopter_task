#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
Master script that runs the complete Bayesian fitting pipeline:
1. Check setup
2. Fit Bayesian model to GRU
3. Generate comprehensive report

Usage:
    python run_full_pipeline.py
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_name}")
        return False

def main():
    print("\n" + "="*70)
    print(" "*15 + "BAYESIAN GRU FITTING PIPELINE")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Step 1: Check setup
    print("\n" + "Step 1/3: Verifying setup...")
    if not run_script("check_setup.py", "Setup verification"):
        print("\n✗ Setup check failed. Please fix the issues above before continuing.")
        sys.exit(1)
    
    # Ask user to confirm before proceeding
    print("\n" + "-"*70)
    response = input("Setup check passed. Proceed with fitting? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted by user.")
        sys.exit(0)
    
    # Step 2: Run fitting
    print("\n" + "Step 2/3: Fitting Bayesian model to GRU...")
    if not run_script("fit_bayesian_to_gru.py", "Bayesian parameter fitting"):
        print("\n✗ Fitting failed. Check error messages above.")
        sys.exit(1)
    
    # Step 3: Generate report
    print("\n" + "Step 3/3: Generating analysis report...")
    if not run_script("generate_report.py", "Report generation"):
        print("\n✗ Report generation failed. Check error messages above.")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults saved to: results/bayesian_fitting/")
    print("\nKey files:")
    print("  • fitting_report.txt               - Human-readable summary")
    print("  • fitted_parameters.json           - Fitted σ and bias")
    print("  • comparison_report_head.png       - Report accuracy plot")
    print("  • comparison_hazard_head.png       - Hazard accuracy plot")
    print("  • probability_distributions.png    - Probability histograms")
    print("\nNext steps:")
    print("  1. Read fitting_report.txt for interpretations")
    print("  2. Examine plots to understand GRU behavior")
    print("  3. Consider running with different test variants")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

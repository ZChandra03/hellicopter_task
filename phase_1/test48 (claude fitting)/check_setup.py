#!/usr/bin/env python3
"""
check_setup.py
==============
Quick diagnostic script to verify all files are in place before running the fitting pipeline.
Run this first to catch any configuration issues early.
"""

import os
import glob
import json

def check_directory(path, description):
    """Check if directory exists."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_file(path, description):
    """Check if file exists."""
    exists = os.path.isfile(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("\n" + "="*70)
    print(" "*20 + "SETUP DIAGNOSTICS")
    print("="*70 + "\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_ok = True
    
    # Check for required Python files
    print("Required Python files:")
    print("-" * 70)
    required_files = [
        ("rnn_models.py", "GRU model definitions"),
        ("fit_bayesian_to_gru.py", "Main fitting script"),
    ]
    
    for filename, desc in required_files:
        path = os.path.join(base_dir, filename)
        if not check_file(path, desc):
            all_ok = False
    
    # Check models directory
    print("\n" + "Models directory:")
    print("-" * 70)
    models_dir = os.path.join(base_dir, "models")
    if check_directory(models_dir, "Models base directory"):
        trained_gru = os.path.join(models_dir, "trained_gru")
        if check_directory(trained_gru, "trained_gru model folder"):
            final_pt = os.path.join(trained_gru, "final.pt")
            hp_json = os.path.join(trained_gru, "hp.json")
            
            if check_file(final_pt, "Model checkpoint (final.pt)"):
                # Check file size
                size_mb = os.path.getsize(final_pt) / (1024 * 1024)
                print(f"    → Size: {size_mb:.2f} MB")
            else:
                all_ok = False
            
            if check_file(hp_json, "Hyperparameters (hp.json)"):
                # Try to load and display
                try:
                    with open(hp_json, 'r') as f:
                        hp = json.load(f)
                    print(f"    → n_input: {hp.get('n_input', '?')}")
                    print(f"    → n_rnn: {hp.get('n_rnn', '?')}")
                except Exception as e:
                    print(f"    ⚠ Warning: Could not load hp.json: {e}")
            else:
                all_ok = False
        else:
            all_ok = False
            print("    ⚠ Tip: Rename your model folder to 'trained_gru' or update MODEL_NAME in the script")
    else:
        all_ok = False
    
    # Check variants directory
    print("\n" + "Test configurations:")
    print("-" * 70)
    variants_dir = os.path.join(base_dir, "variants")
    if check_directory(variants_dir, "Variants base directory"):
        # Check for common variants
        test_variants = ["sigma_1", "sigma_2", "beta_1p0", "beta_2p0", "beta_10p0"]
        found_variants = []
        
        for variant in test_variants:
            variant_path = os.path.join(variants_dir, variant)
            if os.path.isdir(variant_path):
                test_configs = glob.glob(os.path.join(variant_path, "testConfig_*.csv"))
                if test_configs:
                    print(f"✓ {variant}: {len(test_configs)} test configs")
                    found_variants.append(variant)
                else:
                    print(f"⚠ {variant}: directory exists but no testConfig_*.csv files")
            else:
                print(f"  {variant}: not found (optional)")
        
        if not found_variants:
            print("✗ No test configurations found!")
            all_ok = False
        else:
            print(f"\n  Recommended: Set VARIANT_NAME = \"{found_variants[0]}\" in fit_bayesian_to_gru.py")
    else:
        all_ok = False
    
    # Check Python dependencies
    print("\n" + "Python dependencies:")
    print("-" * 70)
    dependencies = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "scikit-learn"),
    ]
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - Run: pip install {module_name}")
            all_ok = False
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✓ ALL CHECKS PASSED - Ready to run fit_bayesian_to_gru.py!")
        print("\nNext steps:")
        print("  1. Verify VARIANT_NAME in fit_bayesian_to_gru.py")
        print("  2. Run: python fit_bayesian_to_gru.py")
        print("  3. Check results in: results/bayesian_fitting/")
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print("\nCommon fixes:")
        print("  - Missing model: Ensure final.pt and hp.json are in models/trained_gru/")
        print("  - Missing test configs: Check variants/sigma_X/ directories")
        print("  - Missing dependencies: pip install numpy pandas torch scipy matplotlib scikit-learn")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

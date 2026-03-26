# Quick Start Guide: Fitting Bayesian Model to GRU

## What This Does
Finds the Bayesian observer parameters (σ_belief, bias) that best match your trained GRU's behavior, revealing what "internal model" the network learned.

## Prerequisites
```bash
pip install numpy pandas torch scipy matplotlib scikit-learn
```

## File Setup
Ensure you have:
```
your_project/
├── models/trained_gru/
│   ├── final.pt       # Your trained GRU checkpoint
│   └── hp.json        # Model hyperparameters
├── variants/sigma_2/   # Or sigma_1, beta_1p0, etc.
│   └── testConfig_*.csv
├── rnn_models.py
└── [scripts from outputs/]
```

## Three Ways to Run

### Option 1: Full Pipeline (Recommended)
```bash
python run_full_pipeline.py
```
Runs everything automatically: checks setup → fits model → generates report

### Option 2: Step-by-Step
```bash
# 1. Check setup
python check_setup.py

# 2. Fit parameters
python fit_bayesian_to_gru.py

# 3. Generate report
python generate_report.py
```

### Option 3: Just Fitting
```bash
python fit_bayesian_to_gru.py
```

## Configuration

Edit `fit_bayesian_to_gru.py` near the top:

```python
MODEL_NAME = "trained_gru"     # Your model folder name
VARIANT_NAME = "sigma_2"       # Test set to use
N_TEST_CFGS = 20              # Number of test configs
```

## Expected Runtime
- **Setup check:** < 1 second
- **Fitting:** 10-20 minutes (depends on # of test configs)
- **Report generation:** < 1 second

## What You'll Get

### Files (in `results/bayesian_fitting/`)
1. **fitting_report.txt** - Human-readable interpretation
2. **fitted_parameters.json** - σ_belief and bias values
3. **comparison_report_head.png** - Report accuracy plot
4. **comparison_hazard_head.png** - Hazard accuracy plot
5. **probability_distributions.png** - Choice probability histograms
6. **accuracy_deltas.png** - Where GRU differs from Bayesian
7. **gru_predictions.csv** - All GRU predictions
8. **detailed_comparison.csv** - Trial-by-trial comparison

### Key Metrics
- **σ_belief:** How much noise the GRU thinks there is
  - `< true_σ`: Overconfident (treats evidence as too reliable)
  - `≈ true_σ`: Accurate noise model
  - `> true_σ`: Underconfident (treats evidence as too noisy)

- **bias:** Prior expectation about volatility
  - `< 0`: Expects fewer switches (stay-biased)
  - `≈ 0`: No prior bias
  - `> 0`: Expects more switches (switch-biased)

- **Cohen's κ:** Agreement between GRU and fitted Bayesian
  - `> 0.8`: Excellent (GRU ≈ Bayesian)
  - `0.6-0.8`: Good agreement
  - `< 0.6`: Different strategies

## Interpreting Results

### Case 1: Optimal Performance
```
σ_belief = 2.05 (true σ = 2.0)
bias = 0.12
κ_report = 0.85
```
**→ GRU learned optimal Bayesian inference!**

### Case 2: Overconfident Network
```
σ_belief = 1.23 (true σ = 2.0)
bias = -0.89
κ_report = 0.71
```
**→ GRU underestimates noise and expects more stability**

### Case 3: Poor Fit
```
σ_belief = 2.45 (true σ = 2.0)
bias = 0.23
κ_report = 0.54
```
**→ GRU may use non-Bayesian strategy; consider extended models**

## Troubleshooting

### "No module named 'rnn_models'"
- Ensure `rnn_models.py` is in the same directory
- Or add to Python path: `export PYTHONPATH=$PYTHONPATH:.`

### "Model file not found"
- Check that `models/trained_gru/final.pt` exists
- Verify `hp.json` is also present
- Update `MODEL_NAME` in script if folder has different name

### "No test configs found"
- Check that `variants/sigma_2/testConfig_*.csv` exists
- Update `VARIANT_NAME` to match your available variants
- Run `python check_setup.py` to see available variants

### Fitting takes too long
- Reduce `N_TEST_CFGS` from 20 to 10
- Comment out `differential_evolution` (keep only L-BFGS-B)
- Use fewer starting points in random initialization

### Poor fit quality
This is actually informative! It means:
1. GRU learned a non-Bayesian strategy, OR
2. Need more flexible parameterization, OR
3. GRU has context-dependent behavior

**Try:** Add lapse rate, separate σ for each head, or analyze disagreement trials

## Next Steps After Fitting

1. **Read the report:**
   ```bash
   cat results/bayesian_fitting/fitting_report.txt
   ```

2. **Examine plots:**
   Open PNG files in `results/bayesian_fitting/`

3. **Test generalization:**
   - Fit on σ=1, test on σ=2
   - Fit on Beta(1,1), test on Beta(10,10)

4. **Compare architectures:**
   Fit to GRU, vanilla RNN, and LSTM to see if architecture affects strategy

5. **Deeper analysis:**
   - Analyze GRU hidden states
   - Study disagreement trials
   - Try extended Bayesian models

## Common Questions

**Q: Why likelihood fitting instead of matching accuracies?**
A: Captures full probability distributions, more sensitive to strategy differences.

**Q: What if fitted params vary across seeds?**
A: Good question! Fit each seed separately to check consistency. High variance suggests training instability.

**Q: Can I fit different σ for report vs. hazard?**
A: Yes! Modify `negative_log_likelihood` to use two sigma parameters.

**Q: How do I know if the fit is good?**
A: Check: (1) Fitted acc ≈ GRU acc, (2) κ > 0.7, (3) Similar probability distributions

## Getting Help

1. Run `python check_setup.py` for diagnostics
2. Check `README_fitting.md` for detailed documentation
3. Review `bayesian_gru_fitting_plan.md` for full methodology
4. Examine example outputs in the report

## Citation

If you use this fitting approach, consider citing the change-point detection literature:

```bibtex
@article{nassar2012rational,
  title={Rational regulation of learning dynamics by pupil-linked arousal systems},
  author={Nassar, Matthew R and others},
  journal={Nature Neuroscience},
  year={2012}
}
```

---

**Ready?** Run `python run_full_pipeline.py` and let's see what your GRU learned!

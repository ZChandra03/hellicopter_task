# Fitting Detuned Bayesian Model to Trained GRU

## Overview
This script implements a complete pipeline to fit a detuned Bayesian observer model to a trained GRU network. It finds the Bayesian parameters (σ_belief and bias) that best explain the GRU's behavior using maximum likelihood estimation.

## Quick Start

### Prerequisites
```bash
pip install numpy pandas torch scipy matplotlib scikit-learn
```

### Directory Structure
```
project/
├── models/
│   └── trained_gru/          # Your trained model
│       ├── final.pt           # Model checkpoint
│       └── hp.json            # Hyperparameters
├── variants/
│   ├── sigma_1/               # Test configs for σ=1
│   │   └── testConfig_*.csv
│   └── sigma_2/               # Test configs for σ=2
│       └── testConfig_*.csv
├── rnn_models.py              # GRU model definition
└── fit_bayesian_to_gru.py     # Main fitting script
```

### Basic Usage
```bash
python fit_bayesian_to_gru.py
```

## Configuration

Edit these variables at the top of `fit_bayesian_to_gru.py`:

```python
MODEL_NAME = "trained_gru"     # Name of model folder in models/
VARIANT_NAME = "sigma_2"       # Which test set to use
N_TEST_CFGS = 20              # Number of test configs (default: 20)
```

## What It Does

### Phase 1: Collect GRU Predictions
- Loads your trained GRU from `models/trained_gru/final.pt`
- Runs inference on test configurations
- Extracts both **choice probabilities** and **binary decisions**
- Saves predictions to `results/bayesian_fitting/gru_predictions.csv`

**Output columns:**
- `evidence`: sequence of observations
- `true_hazard`: ground truth hazard rate
- `true_report`: ground truth final state
- `gru_report_prob`: P(state=+1) from GRU
- `gru_report_choice`: -1 or +1
- `gru_hazard_prob`: P(hazard>0.5) from GRU
- `gru_hazard_choice`: -1 or +1

### Phase 2: Fit Bayesian Parameters
Finds the parameters that maximize:
```
P(GRU choices | Bayesian model with σ_belief, bias)
```

Uses two optimization strategies:
1. **Multiple random starts** with L-BFGS-B (fast, local)
2. **Differential evolution** (slower, global)

**Parameter bounds:**
- `σ_belief`: [0.1, 5.0] - how much noise the observer believes
- `bias`: [-5.0, 5.0] - prior expectation about volatility

**Saves to:** `results/bayesian_fitting/fitted_parameters.json`

### Phase 3: Generate Comparison Plots

Creates 4 comprehensive plots:

#### 1. Report Head Comparison (`comparison_report_head.png`)
Shows report accuracy vs hazard rate for:
- GRU (trained network)
- Fitted Bayesian (with optimal σ and bias)
- Optimal Bayesian (true σ, no bias)

#### 2. Hazard Head Comparison (`comparison_hazard_head.png`)
Shows hazard prediction accuracy vs hazard rate

#### 3. Probability Distributions (`probability_distributions.png`)
Histograms comparing choice probability distributions between GRU and fitted Bayesian

#### 4. Accuracy Deltas (`accuracy_deltas.png`)
Shows where GRU deviates from fitted Bayesian (GRU - Fitted)

## Interpreting Results

### Fitted Parameters

**σ_belief (noise belief):**
- `σ_belief ≈ true_σ` → GRU has accurate noise model
- `σ_belief < true_σ` → GRU underestimates noise (overconfident)
- `σ_belief > true_σ` → GRU overestimates noise (underconfident)

**bias (prior bias):**
- `bias ≈ 0` → No systematic bias
- `bias > 0` → Expects world to be MORE volatile (switch-biased)
- `bias < 0` → Expects world to be LESS volatile (stay-biased)

### Example Interpretations

**Case 1: Optimal Performance**
```json
{
  "sigma_belief": 2.05,
  "bias": 0.12,
  "fitted_report_acc": 0.847,
  "optimal_report_acc": 0.849
}
```
→ GRU has nearly perfect Bayesian inference!

**Case 2: Overconfident**
```json
{
  "sigma_belief": 1.23,
  "bias": -0.89,
  "fitted_report_acc": 0.782,
  "optimal_report_acc": 0.849
}
```
→ GRU underestimates noise (σ_belief < 2.0) and is biased toward stability

**Case 3: Pessimistic**
```json
{
  "sigma_belief": 3.45,
  "bias": 1.67,
  "fitted_report_acc": 0.801,
  "optimal_report_acc": 0.849
}
```
→ GRU overestimates noise and expects more switches than warranted

### Cohen's Kappa
Measures agreement between GRU and fitted Bayesian choices:
- `κ > 0.8`: Excellent agreement (GRU ≈ Bayesian)
- `0.6 < κ < 0.8`: Good agreement
- `κ < 0.6`: Moderate/poor agreement (different strategies)

## Output Files

All results saved to `results/bayesian_fitting/`:

### Data Files
- `gru_predictions.csv` - All GRU predictions on test set
- `fitted_parameters.json` - Fitted σ and bias with metadata
- `comparison_summary.json` - Summary statistics
- `detailed_comparison.csv` - Trial-by-trial comparison

### Plots
- `comparison_report_head.png` - Report accuracy comparison
- `comparison_hazard_head.png` - Hazard accuracy comparison
- `probability_distributions.png` - Probability histograms
- `accuracy_deltas.png` - Performance differences

## Advanced Usage

### Testing Multiple Models

To compare multiple trained models:

```python
models = ["trained_gru_seed0", "trained_gru_seed1", "trained_gru_seed2"]

for model_name in models:
    MODEL_NAME = model_name
    main()
```

### Custom Parameter Bounds

If you want to test specific hypotheses:

```python
# Hypothesis: GRU severely underestimates noise
bounds = [
    (0.1, 1.5),    # σ_belief: only test small values
    (-5.0, 5.0),   # bias: full range
]
```

### Using Different Test Sets

```python
# Test on σ=1 data
VARIANT_NAME = "sigma_1"
true_sigma = 1.0

# Test on Beta(10,10) data
VARIANT_NAME = "beta_10p0"
true_sigma = 1.0
```

## Troubleshooting

### Error: "Model file not found"
- Check that `models/trained_gru/final.pt` exists
- Verify `hp.json` is in the same directory

### Error: "No test configs found"
- Check that `variants/sigma_2/testConfig_*.csv` files exist
- Adjust `N_TEST_CFGS` if you have fewer files

### Optimization takes too long
- Reduce `N_TEST_CFGS` from 20 to 10
- Use only local optimization (comment out differential_evolution)
- Reduce `maxiter` in optimization calls

### Poor fit (very different accuracies)
This might indicate:
1. GRU learned a non-Bayesian strategy
2. Need different parameterization (e.g., separate σ for report vs hazard)
3. GRU has lapse rate (random choices on some trials)

Try:
```python
# Add lapse rate parameter
def negative_log_likelihood_with_lapse(params, gru_data):
    sigma_belief, bias, lapse = params
    # Mix Bayesian predictions with random choices
    p_report = (1-lapse) * P_bay_state_pos + lapse * 0.5
    ...
```

## Computational Cost

**Typical runtime:**
- 10 test configs (~3000 trials): 5-10 minutes
- 20 test configs (~6000 trials): 10-20 minutes
- 50 test configs (~15000 trials): 30-60 minutes

**Memory usage:**
- GRU inference: ~500MB GPU memory
- Optimization: ~1GB RAM

**Parallelization:**
- Set `workers=-1` in `differential_evolution` to use all CPU cores
- Careful: may cause issues with some NumPy installations

## Citation

If you use this fitting approach, consider citing:

```bibtex
@article{nassar2010rational,
  title={Rational regulation of learning dynamics by pupil-linked arousal systems},
  author={Nassar, Matthew R and Rumsey, Kendra M and Wilson, Robert C and Parikh, Kalpana and Heasly, Benjamin and Gold, Joshua I},
  journal={Nature neuroscience},
  year={2012}
}
```

(Seminal work on Bayesian modeling of change-point detection tasks)

## Next Steps

After fitting:

1. **Check goodness-of-fit**: Is fitted Bayesian accuracy close to GRU?
2. **Interpret parameters**: What strategy did the GRU learn?
3. **Test generalization**: Fit on one σ, test on another
4. **Representational analysis**: Compare GRU hidden states to Bayesian beliefs
5. **Ablation studies**: How do fitted parameters change with architecture/training?

## Questions?

Common questions:

**Q: Why fit to choices instead of matching accuracies?**
A: Fitting to individual trial choices captures the full probability distribution, not just summary statistics. This is more sensitive to subtle differences in strategy.

**Q: Can I fit different σ for report vs hazard predictions?**
A: Yes! Modify `negative_log_likelihood` to use two sigma parameters:
```python
def negative_log_likelihood(params, gru_data):
    sigma_report, sigma_hazard, bias = params
    # Use sigma_report for report head predictions
    # Use sigma_hazard for hazard head predictions
```

**Q: What if my GRU has multiple seeds/checkpoints?**
A: Fit each seed separately and compare fitted parameters across seeds to assess consistency.

**Q: How do I know if the fit is good?**
A: Check:
1. Fitted Bayesian accuracy ≈ GRU accuracy (within 2-3%)
2. Cohen's kappa > 0.7
3. Probability distributions look similar
4. NLL per trial is reasonable (typically 0.5-1.5)

## Contact

For bugs or questions about this fitting pipeline, please check:
- The detailed plan document: `bayesian_gru_fitting_plan.md`
- Your GRU training code and hyperparameters
- Bayesian observer implementation in `NormativeModel.py`

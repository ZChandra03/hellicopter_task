# Plan: Fitting Detuned Bayesian Models to Trained GRU

## Project Overview
Fit a Bayesian observer model with tunable parameters (σ_belief and bias) to match the behavior of a trained GRU network on the helicopter decision-making task. This will help understand whether the GRU has learned to implement an optimal (or sub-optimal) Bayesian inference strategy.

---

## Current Infrastructure Analysis

### What You Have:
1. **GRU Model** (`rnn_models.py`)
   - Two-head architecture: location prediction + hazard prediction
   - Trained on continuous hazard tasks with different σ values

2. **Bayesian Observer** (`NormativeModel.py`, `binned_accuracy.py`)
   - Implements optimal Bayesian inference with tunable parameters:
     - `sigma_belief`: observer's belief about noise level
     - `bias`: prior bias toward expecting switches/stays
   - Currently configured to test multiple variants independently

3. **Task Configurations** (`TaskConfig_Generator_Continuous.py`)
   - Generates training/test datasets with σ ∈ {1, 2}
   - Flat Beta(1,1) prior on hazard rates
   - 20-step evidence sequences

4. **Evaluation Framework** (`binned_accuracy.py`)
   - Bins performance by true hazard rate
   - Separate metrics for report and hazard prediction accuracy

---

## Fitting Strategy

### Objective
Find the Bayesian parameters (σ_belief, bias) that best match the GRU's behavior on test data.

### Approach: Maximum Likelihood Estimation (MLE)

**Key Insight:** Rather than just comparing final accuracies, we'll fit the Bayesian model by maximizing the likelihood of the GRU's actual choices under the Bayesian model's decision probabilities.

---

## Implementation Plan

### Phase 1: Data Collection & GRU Evaluation

**File: `collect_gru_predictions.py`**

```python
# Pseudocode structure:
def collect_gru_predictions(model_path, test_configs):
    """
    Load trained GRU and collect predictions on test set.
    
    Returns:
    --------
    predictions_df with columns:
        - evidence: list of observations
        - true_hazard: ground truth hazard rate
        - true_report: ground truth final state
        - gru_report_prob: sigmoid(loc_logits[-1])
        - gru_report_choice: -1 or 1
        - gru_hazard_prob: sigmoid(haz_logits)
        - gru_hazard_choice: -1 or 1
        - trial_idx: unique trial identifier
    """
```

**Tasks:**
- [ ] Load trained GRU checkpoint
- [ ] Run inference on all test configurations
- [ ] Extract both choice probabilities and binary decisions
- [ ] Save predictions to CSV for fitting

---

### Phase 2: Bayesian Model Modification

**File: `bayesian_observer_with_probs.py`**

Currently `BayesianObserver` returns only binary choices. We need to modify it to return **choice probabilities** for likelihood calculation.

```python
def BayesianObserver_with_probs(ev, mu1, mu2, sigma_belief, hs, bias=0.0):
    """
    Extended Bayesian observer that returns choice probabilities.
    
    Returns:
    --------
    - L_haz: hazard beliefs over time
    - L_state: state beliefs over time
    - resp_Rep: binary report choice (-1 or 1)
    - resp_Pred: binary hazard choice (-1 or 1)
    - P_report_state1: P(state=1 | evidence) ← NEW
    - P_predict_switch: P(switch | evidence) ← NEW
    """
```

**Key additions:**
- Return `P_report_state1 = L_state[0, -1]` (probability of choosing state 1)
- Return `P_predict_switch` from hazard marginalization
- These probabilities enable likelihood calculations

---

### Phase 3: Likelihood Function

**File: `fitting_objective.py`**

```python
def negative_log_likelihood(params, gru_data, test_configs):
    """
    Compute -log P(GRU choices | Bayesian model with params).
    
    Parameters:
    -----------
    params: [sigma_belief, bias]
    gru_data: DataFrame with GRU predictions
    test_configs: list of test CSV paths
    
    Returns:
    --------
    nll: negative log-likelihood (to minimize)
    """
    sigma_belief, bias = params
    
    nll_total = 0.0
    
    for trial in gru_data.iterrows():
        # Run Bayesian observer with current params
        _, _, _, _, P_bay_s1, P_bay_switch = BayesianObserver_with_probs(
            trial['evidence'], mu1=-1, mu2=1, 
            sigma_belief=sigma_belief, hs=hs, bias=bias
        )
        
        # Likelihood of GRU's report choice
        if trial['gru_report_choice'] == -1:
            p_report = P_bay_s1  # chose state 1
        else:
            p_report = 1 - P_bay_s1  # chose state 2
            
        # Likelihood of GRU's hazard choice
        if trial['gru_hazard_choice'] == 1:
            p_hazard = P_bay_switch
        else:
            p_hazard = 1 - P_bay_switch
            
        # Combined log-likelihood (assuming independence)
        nll_total -= (np.log(p_report + 1e-10) + np.log(p_hazard + 1e-10))
    
    return nll_total
```

**Note:** Add small epsilon (1e-10) to prevent log(0).

---

### Phase 4: Optimization

**File: `fit_bayesian_to_gru.py`**

```python
from scipy.optimize import minimize, differential_evolution

def fit_bayesian_to_gru(gru_predictions_csv, test_configs_dir):
    """
    Find best-fitting Bayesian parameters for GRU behavior.
    """
    
    # Load GRU predictions
    gru_data = pd.read_csv(gru_predictions_csv)
    
    # Define parameter bounds
    bounds = [
        (0.1, 5.0),   # sigma_belief: reasonable noise range
        (-5.0, 5.0),  # bias: prior bias range
    ]
    
    # Initial guess
    x0 = [1.0, 0.0]  # matched sigma, no bias
    
    # Option 1: Local optimization (fast)
    result_local = minimize(
        negative_log_likelihood,
        x0=x0,
        args=(gru_data, test_configs_dir),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    # Option 2: Global optimization (slower but more robust)
    result_global = differential_evolution(
        negative_log_likelihood,
        bounds=bounds,
        args=(gru_data, test_configs_dir),
        maxiter=100,
        seed=42
    )
    
    return result_global  # or compare both
```

**Optimization strategies:**
- Start with local optimization (L-BFGS-B) for speed
- Use global optimization (differential_evolution) for robustness
- Try multiple random initializations
- Consider grid search over coarse parameter space first

---

### Phase 5: Model Comparison & Validation

**File: `compare_models.py`**

After fitting, compare:

1. **Accuracy Comparison**
   - Binned accuracy plots (like `binned_accuracy.py` currently does)
   - GRU vs fitted Bayesian vs optimal Bayesian

2. **Choice Distribution Analysis**
   - Histogram of choice probabilities
   - Confidence calibration curves

3. **Parameter Interpretation**
   - What does fitted σ_belief tell us?
   - Does bias reveal systematic overestimation of volatility?

4. **Goodness-of-Fit Metrics**
   - Log-likelihood ratio vs optimal Bayesian
   - AIC/BIC for model comparison
   - Cohen's κ (agreement between GRU and fitted Bayesian)

```python
def model_comparison_report(gru_data, fitted_params, test_configs):
    """
    Generate comprehensive comparison report.
    """
    sigma_fit, bias_fit = fitted_params
    
    # 1. Get predictions from fitted Bayesian
    bay_preds = evaluate_bayesian(test_configs, sigma_fit, bias_fit)
    
    # 2. Get predictions from optimal Bayesian
    optimal_preds = evaluate_bayesian(test_configs, sigma_true, bias=0.0)
    
    # 3. Compute metrics
    metrics = {
        'accuracy_gru': compute_accuracy(gru_data),
        'accuracy_fitted': compute_accuracy(bay_preds),
        'accuracy_optimal': compute_accuracy(optimal_preds),
        'log_likelihood_fitted': -negative_log_likelihood([sigma_fit, bias_fit], ...),
        'log_likelihood_optimal': -negative_log_likelihood([sigma_true, 0.0], ...),
        'cohens_kappa': cohen_kappa(gru_data.choices, bay_preds.choices),
        'fitted_sigma': sigma_fit,
        'fitted_bias': bias_fit,
    }
    
    return metrics
```

---

## Advanced Extensions

### Extension 1: Trial-by-Trial Fitting
Instead of fitting to all trials jointly, fit separately to different hazard bins to see if GRU's "internal model" changes with task statistics.

### Extension 2: Cross-Validation
- Fit parameters on 80% of test data
- Validate on held-out 20%
- Repeat with k-fold CV

### Extension 3: Multi-Dimensional Search
Add more Bayesian parameters:
- Prior parameters (α, β for Beta distribution)
- Lapse rate (probability of random choice)
- Different σ_belief for report vs hazard predictions

### Extension 4: Information-Theoretic Metrics
- Mutual information between GRU and Bayesian predictions
- KL divergence between choice probability distributions

---

## File Structure

```
project/
│
├── variants/
│   ├── sigma_1/
│   │   ├── trainConfig_*.csv
│   │   └── testConfig_*.csv
│   └── sigma_2/
│       ├── trainConfig_*.csv
│       └── testConfig_*.csv
│
├── models/
│   └── gru_trained.pth  (your trained GRU checkpoint)
│
├── scripts/
│   ├── collect_gru_predictions.py       [NEW - Phase 1]
│   ├── bayesian_observer_with_probs.py  [NEW - Phase 2]
│   ├── fitting_objective.py             [NEW - Phase 3]
│   ├── fit_bayesian_to_gru.py          [NEW - Phase 4]
│   └── compare_models.py                [NEW - Phase 5]
│
└── results/
    ├── gru_predictions.csv
    ├── fitted_parameters.json
    ├── comparison_report.txt
    └── figures/
        ├── binned_accuracy_comparison.png
        ├── choice_probability_histograms.png
        └── calibration_curves.png
```

---

## Implementation Checklist

### Phase 1: Setup
- [ ] Create project directory structure
- [ ] Locate trained GRU checkpoint file
- [ ] Verify test configs are available

### Phase 2: Data Collection
- [ ] Implement `collect_gru_predictions.py`
- [ ] Run GRU inference on test set
- [ ] Verify predictions CSV format
- [ ] Sanity check: GRU accuracy should match training metrics

### Phase 3: Bayesian Modifications
- [ ] Extend `BayesianObserver` to return probabilities
- [ ] Unit test: verify probabilities sum to 1
- [ ] Verify binary choices match original implementation

### Phase 4: Fitting Infrastructure
- [ ] Implement likelihood function
- [ ] Test on synthetic data (known parameters)
- [ ] Implement optimization wrapper
- [ ] Add logging/progress tracking

### Phase 5: Run Fitting
- [ ] Fit to σ=1 dataset
- [ ] Fit to σ=2 dataset
- [ ] Save fitted parameters
- [ ] Generate comparison plots

### Phase 6: Analysis
- [ ] Create model comparison report
- [ ] Interpret fitted parameters
- [ ] Statistical significance tests
- [ ] Write up findings

---

## Expected Outcomes

### Hypothesis 1: Matched Model
**If GRU is optimal:** σ_belief ≈ σ_true, bias ≈ 0
- GRU has learned true Bayesian inference
- High likelihood under fitted model

### Hypothesis 2: Conservative Model
**If GRU underestimates noise:** σ_belief < σ_true
- GRU treats evidence as more reliable than it is
- May show overconfidence in state estimates

### Hypothesis 3: Pessimistic Model
**If GRU overestimates noise:** σ_belief > σ_true
- GRU is more uncertain than it should be
- May show underconfidence, more switches

### Hypothesis 4: Biased Model
**If bias ≠ 0:**
- Positive bias: GRU expects world to be more volatile
- Negative bias: GRU expects world to be more stable

---

## Practical Considerations

### Computational Cost
- Each Bayesian evaluation takes ~1-10ms per trial
- Fitting 10,000 trials × 100 iterations ≈ 10-100 minutes
- Use multiprocessing to parallelize trial evaluations

### Numerical Stability
- Add small epsilon to probabilities before log
- Monitor for NaN/Inf in likelihood calculations
- Use log-space arithmetic where possible

### Hyperparameter Sensitivity
- Parameter bounds matter! Start conservative
- Sigma: [0.1, 5.0] covers wide range
- Bias: [-5, 5] covers strong priors
- Consider prior regularization if needed

### Validation
- Always compare fitted model accuracy to GRU accuracy
- If fitted accuracy << GRU accuracy, model is misspecified
- Try different parameterizations (e.g., log-scale for σ)

---

## Alternative Approaches

### Approach A: Match Summary Statistics
Instead of MLE, minimize distance between summary statistics:
- Mean accuracy per hazard bin
- Response time distributions (if available)
- Confidence distributions

### Approach B: Representational Similarity Analysis (RSA)
Compare GRU hidden states to Bayesian belief states:
- Extract GRU hidden vectors h_t
- Compute Bayesian L_state, L_haz
- Correlate representations across trials

### Approach C: Behavioral Cloning
Train a "Bayesian surrogate" neural network:
- Input: Bayesian parameters
- Output: predictions matching GRU
- Learn non-linear mapping from parameters to behavior

---

## Questions to Answer

1. **Does the GRU implement Bayesian inference?**
   - How close are fitted parameters to optimal?

2. **Are there systematic deviations?**
   - Noise misestimation? Prior bias?

3. **Does it depend on training σ?**
   - Do σ=1 and σ=2 trained models have different "internal models"?

4. **Can we predict when the GRU will fail?**
   - Compare errors on trials where Bayesian is confident vs uncertain

5. **How does architecture matter?**
   - Fit to vanilla RNN and LSTM as well
   - Do different architectures learn different strategies?

---

## Timeline Estimate

- **Phase 1 (Data Collection):** 2-4 hours
- **Phase 2 (Bayesian Mods):** 2-3 hours
- **Phase 3 (Likelihood):** 3-4 hours
- **Phase 4 (Optimization):** 4-6 hours (including debugging)
- **Phase 5 (Analysis):** 4-8 hours

**Total:** ~2-3 days of focused work

---

## Next Steps

1. **Immediate:** Locate and verify trained GRU checkpoint
2. **Today:** Implement Phase 1 (GRU prediction collection)
3. **Tomorrow:** Implement Phases 2-3 (Bayesian mods + likelihood)
4. **Day 3:** Run fitting and generate first results

Would you like me to start implementing any of these phases?

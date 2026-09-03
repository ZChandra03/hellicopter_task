# Bayes-optimal hazard comparison

This analysis compares the trained GRU's prefix-level hazard predictions with
two Bayesian observers.

## Observers

The **evidence-matched oracle** receives the same noisy evidence prefix as the
GRU. For each candidate hazard, it evaluates the two-state hidden Markov model
using the generator's truncated-normal emissions. It then integrates the
likelihood over the low-hazard interval `[0, 0.5)` and high-hazard interval
`[0.5, 1]` under the uniform hazard prior.

The **latent-state oracle** sees the true states. If a prefix contains `k`
switches in `n = L - 1` transition opportunities, its posterior is

```text
P(high | k, n) = 1 - I_0.5(k + 1, n - k + 1)
```

where `I` is the regularized incomplete beta function. This is a
privileged-information ceiling, not a fair input-matched comparison.

Exact posterior ties receive 0.5 accuracy credit. This avoids an arbitrary
high/low tie-breaking rule when `L = 1`, where no hazard information exists.
NLL and Brier score always use the posterior probability directly.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\phase_4\test82_oracle\oracle_hazard_analysis.py
```

The default model predictions come from:

```text
phase_4/test82_input_length/prefix_length_outputs/prefix_trial_predictions.csv.gz
```

Use `--predictions` to analyze another prefix experiment and `--help` for the
remaining options.

## Main outputs

- `oracle_model_comparison.png`: accuracy, NLL, Brier score, and model NLL
  regret relative to the evidence-matched oracle.
- `model_vs_evidence_oracle_probability.png`: per-trial probability agreement.
- `oracle_calibration.png`: reliability curves at representative lengths.
- `oracle_accuracy_by_hazard_bin.png`: performance across true hazard ranges.
- `state_oracle_posterior_lookup.png`: exact switch-count posterior lookup.
- `oracle_comparison_summary.csv`: compact numerical comparison.
- `model_oracle_trial_predictions.csv.gz`: paired model and oracle predictions.

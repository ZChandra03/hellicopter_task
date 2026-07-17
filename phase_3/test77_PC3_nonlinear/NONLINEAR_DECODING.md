# Test 77: Linear Versus Nonlinear PC Decoding

This analysis compares two feature sets:

- linear: `PC1`, `PC2`, `PC3`;
- quadratic: the linear terms plus `PC1^2`, `PC2^2`, `PC3^2`, and every
  pairwise interaction `PCi * PCj`.

Both models use the same trial-grouped cross-validation folds, so every
timestep from a trial stays entirely in either training or testing data.

## Run

```powershell
python .\phase_3\test77_PC3_nonlinear\correlate_pcs_with_task.py `
  --skip-by-timestep --no-posterior-bins
```

Continuous variables use Ridge regression and report cross-validated R2.
Two-valued variables use logistic regression and report ROC-AUC. The default
scopes are `evidence_timesteps` and `final_timestep`.

To test only uncertainty:

```powershell
python .\phase_3\test77_PC3_nonlinear\correlate_pcs_with_task.py `
  --decode-targets norm_intermediate_hazard_sd,norm_intermediate_hazard_entropy,norm_intermediate_state_entropy `
  --skip-by-timestep --no-posterior-bins
```

## Outputs

- `decoder_fold_metrics.csv`: held-out metrics for every fold and feature set.
- `decoder_summary.csv`: mean, SD, and SEM across folds.
- `linear_vs_quadratic_decoders.csv`: paired quadratic-minus-linear gains.
- `decoder_feature_sets.json`: exact terms included in each model.
- `quadratic_decoder_gain_<scope>.png`: nonlinear improvement by target.

A positive quadratic gain indicates information accessible through curvature,
magnitude, or PC interactions beyond an ordinary linear readout.

# Test 77: PC–Task Nonlinear Decoding

See [NONLINEAR_DECODING.md](NONLINEAR_DECODING.md) for the trial-grouped
linear-versus-quadratic decoder analysis added in this test. The material below
documents the correlation and normative-matching stage retained from test 76.

This analysis asks which task variables and Bayesian-observer beliefs are
represented by the leading recurrent-state PCs.

## Run

Generate the per-timestep PCA table, then run the matching/correlation analysis:

```powershell
python .\phase_3\test77_PC3_nonlinear\pca_checkpoint_ep010.py
python .\phase_3\test77_PC3_nonlinear\correlate_pcs_with_task.py
```

The correlation script reads the model's `hp.json`, so its default temporal
matching respects the four null timesteps inserted between evidence samples.
At an evidence timestep and its following null steps, the hidden state is
matched to the Bayesian posterior after that evidence sample.

Useful options:

```powershell
# Analyze only PC1–PC3 and omit the larger per-timestep correlation table.
python .\phase_3\test77_PC3_nonlinear\correlate_pcs_with_task.py `
  --top-n-pcs 3 --skip-by-timestep

# Analyze a final-timestep PCA export.
python .\phase_3\test77_PC3_nonlinear\correlate_pcs_with_task.py `
  --input .\phase_3\test77_PC3_nonlinear\pca_outputs\pca_ep010_final_timestep_hidden_states.csv
```

## Matched Variables

The output contains recorded task/model variables, time-varying task variables,
and quantities derived from all four `BayesianObserver` returns:

- current evidence, latent state, state switch, run length, cumulative evidence,
  and empirical hazard;
- the time-matched `L_haz` posterior at every hazard-grid value;
- the time-matched `L_state` probabilities;
- returned final report/predict responses and whether they match the targets;
- posterior hazard mean/MAP/SD/entropy, `P(switch)`, `P(stay)`, and log odds;
- state signed belief, confidence, entropy, log odds, and interim responses.

## Outputs

- `pc_task_normative_matched.csv.gz`: every PCA row plus its matched task and
  normative features.
- `pc_parameter_correlations.csv`: Pearson and Spearman correlations for all
  timesteps, evidence-only timesteps, and one final timestep per trial.
- `pc_parameter_correlations_by_timestep.csv`: correlations computed separately
  at each recurrent timestep.
- `pc_parameter_correlation_heatmap.png`: strongest evidence-timestep effects.
- `strongest_final_pc_correlations.png`: strongest trial-level final effects.
- `run_config.json`: exact input paths and normative-analysis settings.

The all-timestep table is descriptive: rows from the same trial are not
independent, and null steps repeat the same normative posterior. Use the
`final_timestep` scope for trial-level associations and the
`evidence_timesteps` scope for belief trajectories without null-step
duplication. P-values are accompanied by Benjamini–Hochberg FDR-adjusted
q-values within each analysis scope.

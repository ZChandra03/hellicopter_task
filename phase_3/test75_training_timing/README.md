# Test 75: Training Timing And Last-Evidence Heuristic

This folder tests whether the n5 network behaves like a last-evidence report
heuristic before the shared recurrent state learns the predict task.

The comparison is intentionally restricted to n5 models:

- `models/models_n5/bce_both/sigma_1`: report trained on `trueReport`, predict
  trained on `truePredict`.
- `models/models_n5_hueristic/bce_both/sigma_1`: report trained directly on
  `sign(last evidence)`, predict trained on `truePredict`.

The diagnostic subset is the important one: trials where `sign(last evidence)`
disagrees with `trueReport`. A model that follows only the last evidence should
score near zero report accuracy on these conflict trials. A model that has moved
beyond the heuristic should rescue some of these trials by predicting
`trueReport` instead.

## Run

```powershell
python .\phase_3\test75_training_timing\behavioral_timing_analysis.py
python .\phase_3\test75_training_timing\plot_behavioral_timing.py
```

The behavioral sweep defaults to CUDA and a large inference batch. Use
`--device cpu` only for debugging on a machine without a GPU.

For a quick smoke run:

```powershell
python .\phase_3\test75_training_timing\behavioral_timing_analysis.py --seeds 0 --max-variant-csvs 1 --checkpoints init,1,final --output-dir .\phase_3\test75_training_timing\smoke_outputs
python .\phase_3\test75_training_timing\plot_behavioral_timing.py --input-dir .\phase_3\test75_training_timing\smoke_outputs --figure-dir .\phase_3\test75_training_timing\smoke_outputs\figures
```

## Outputs

- `checkpoint_metrics.csv`: one row per model role, seed, checkpoint.
- `subset_metrics.csv`: long-form metrics for diagnostic and control subsets.
- `aggregate_checkpoint_metrics.csv`: mean/std/SEM across seeds.
- `aggregate_subset_metrics.csv`: subset means/std/SEM across seeds.
- `training_history.csv`: train/validation losses from the model folders.
- `timing_correlations.csv`: simple correlations across checkpoint means.
  Includes all-checkpoint, post-init, and epoch-only versions because the random
  init checkpoint can obscure the training-time trend.
- `figures/accuracy_and_heuristic_agreement_over_training.png`
- `figures/diagnostic_conflict_rescue_over_training.png`
- `figures/predict_report_coupling.png`
- `figures/predict_accuracy_vs_conflict_rescue.png`
- `figures/training_loss_histories.png`
- `figures/true_report_training_metric_heatmap.png`

## Reading The Figures

The strongest evidence for the hypothesis would be:

1. Early true-report checkpoints have high report agreement with
   `sign(last evidence)`.
2. Conflict-trial rescue accuracy is initially low.
3. Predict accuracy rises over training.
4. Conflict-trial rescue rises after, or with, predict accuracy.
5. The heuristic-trained comparison keeps high last-evidence agreement and low
   conflict rescue.

The `predict_report_coupling.png` figure asks a stricter version of the same
question: on conflict trials, does report rescue improve more when the predict
head is correct than when it is wrong?

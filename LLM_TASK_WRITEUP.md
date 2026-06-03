# Helicopter Task Writeup for LLMs

This repository studies recurrent neural networks on a sequential "helicopter" task. Each trial contains a sequence of 20 noisy evidence samples generated from a hidden binary state, represented as `-1` or `+1`. The hidden state can switch sign after each sample with a trial-specific hazard rate `trueHazard`, drawn continuously from `[0, 1]`. Evidence is sampled around the current hidden state mean with noise level `sigma`, with current variants using `sigma_1`, `sigma_2`, and `sigma_3`.

The model sees only the evidence sequence. It is trained to produce two possible outputs:

- `trueReport`: the final hidden state at the end of the trial.
- `truePredict`: whether the trial hazard was high or low, coded as `+1` when `trueHazard >= 0.5` and `-1` otherwise.

The current models are simple recurrent networks, mainly GRUs, with a shared recurrent encoder and two linear readout heads. The report head reads out state evidence over time and is usually evaluated at the final timestep. The hazard head reads from the final recurrent state. Training variants include report-only, hazard-only, and joint report-plus-hazard objectives, usually with binary cross-entropy.

## What A Good Policy Would Need To Learn

The normative solution is a Bayesian observer that jointly tracks belief over the current latent state and possible hazard rates. For report, it should infer whether the final state is more likely `-1` or `+1` after integrating all evidence. For prediction, it should infer whether the latent process was more likely stable or volatile, which depends on the whole sequence, not just the last sample.

Because the task is sequential, the model can improve by learning several related quantities:

- A denoised estimate of the current latent state.
- An estimate of how often the latent state appears to switch.
- A confidence or uncertainty signal that depends on `sigma`.
- A memory of recent evidence, especially near the final timestep.
- A longer-timescale volatility representation useful for hazard prediction.

## Likely Strategies And Shortcuts

A key research question is whether the RNN learns the Bayesian-like latent-state/hazard computation or simpler heuristics that perform well on the generated data.

For the report output, the simplest shortcut is a last-evidence heuristic: report the sign of the final evidence sample. This is surprisingly strong because the final observation is sampled from the final latent state. The checked-in heuristic evaluation shows report accuracies of about `0.844`, `0.679`, and `0.609` for `sigma_1`, `sigma_2`, and `sigma_3`. The Bayesian observer is better but close, with report accuracies of about `0.867`, `0.700`, and `0.620`. This means a model can look competent on report while relying heavily on the final sample rather than maintaining a full posterior.

For the hazard output, a plausible shortcut is to count apparent sign changes or alternations in the evidence sequence. High-hazard trials tend to flip latent state more often, so volatility in the observations is informative. However, high `sigma` can create apparent flips even when the latent state did not switch, so a stronger strategy must discount noisy evidence and track uncertainty.

Another possible strategy is a recency-weighted state tracker: the model may use earlier evidence to smooth the final report, but still weight late evidence heavily. This sits between the last-evidence heuristic and a full Bayesian observer. It may be especially attractive because the report target depends only on the final hidden state, while the hazard target depends on the full transition statistics.

Joint training can encourage shared representations: units useful for denoising state may also help estimate switch frequency, while hazard estimation may encourage the recurrent state to preserve longer history than report-only training requires. Conversely, report-only training may collapse toward a local final-evidence strategy, and hazard-only training may prioritize transition/volatility features without preserving the final state as cleanly.

## How To Interpret Analyses In This Repo

The comparison baselines are important. If a model matches true reports but also matches the last-evidence heuristic, it may not be learning the intended latent-state inference. Trials where the final evidence is weak, near zero, or on the wrong side of the true final state are especially diagnostic. Good performance on those subsets is stronger evidence for integration over the sequence.

For hazard, useful diagnostics should ask whether model outputs track true hazard, apparent evidence alternations, Bayesian hazard belief, or simple summary statistics such as the number of sign changes. PCA and decoder analyses in `phase_3` are intended to inspect whether recurrent activity separates report-related state information from hazard or volatility information over training.

In short: the task is not just binary classification from a sequence. It is designed to reveal whether recurrent models learn latent-state inference, hazard inference, or cheaper correlations in the data-generating process.

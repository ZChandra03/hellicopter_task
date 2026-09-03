# Test 84: How the helicopter-task manifold changes during learning

This folder tests the specific hypothesis that the seed-0 GRU changes from a
mostly two-dimensional, last-evidence solution into a three-dimensional,
Bayesian-like solution when it learns the hazard task. It also tests two
alternatives:

1. PC3 gains variance because an existing curved 2-D manifold rotates or bends,
   without gaining intrinsic dimension.
2. The states called "high/low-hazard basins" are actually positions on a slow
   manifold or transient channels rather than discrete attractor basins.

Only these model directories are used:

- `models/models_n5/bce_both/sigma_1/seed_0`: true-report-trained model.
- `models/models_n5_hueristic/bce_both/sigma_1/seed_0`: last-evidence-trained
  control.

There is deliberately no seed sweep. All analyses use the same validation
trials at every checkpoint, so changes across epochs are paired changes in the
same network on the same inputs.

## Why PCA variance is not enough

PC3 variance can increase when a 2-D surface curves, when the leading plane
rotates, when nuisance variance grows, or when a genuine third latent variable
appears. The geometry analysis therefore combines:

- the full covariance eigenspectrum and participation-ratio dimension;
- TWO-NN local intrinsic dimension;
- separate `all_evidence` and `final_evidence` scopes, so temporal progression
  cannot masquerade as another trial-state dimension;
- trial-grouped, held-out reconstruction by two versus three PCs;
- trial-grouped decoding of Bayesian state belief, hazard mean, hazard
  uncertainty, state uncertainty, and last evidence from two versus three PCs;
- principal angles, linear CKA, and Procrustes shape disparity across
  checkpoints;
- variance escaping the epoch-1 top-2 plane;
- alignment of each PC with the report and hazard readout weights;
- the task's exact sign symmetry, evaluated by pairing every sequence `e` with
  `-e`.

The strongest evidence for a real 2-D to 3-D transition is a coordinated rise
in PC3 variance, participation ratio/TWO-NN dimension, and held-out 3-PC versus
2-PC reconstruction. If the new direction specifically carries hazard posterior
or uncertainty, its 3-PC decoding gain should rise with predict accuracy and
Bayesian report rescue. A PC3 variance rise without these controls is only a
variance redistribution result.

## Experiment 1: checkpoint-resolved geometry

Run the complete seed-0 analysis:

```powershell
.\.venv\Scripts\python.exe .\phase_5\test84_manifold\manifold_geometry.py
```

A quicker targeted run around the behavioral transition is:

```powershell
.\.venv\Scripts\python.exe .\phase_5\test84_manifold\manifold_geometry.py `
  --checkpoints 1,5,6,7,8,10,final --max-csvs 2 --max-cv-trials 300
```

Important outputs in `geometry_outputs/`:

- `dimension_metrics.csv`: participation ratio, TWO-NN, variance dimensions,
  PC3 variance, and mirror-symmetry errors at both scopes.
- `reconstruction_metrics.csv`: held-out 2-PC and 3-PC reconstruction.
- `decoder_metrics.csv`: held-out Bayesian/task-variable decoding.
- `alignment_metrics.csv`: subspace angles, CKA, Procrustes distance, and
  variance outside the early top-2 plane.
- `behavior_metrics.csv`: report/predict accuracy plus last-evidence and
  Bayesian policy matches on all, ambiguous-final-evidence, and conflict trials.
- `readout_alignment.csv`: report/hazard readout alignment with PCs.
- `manifold_atlas_*.png`: local 3-D views colored by Bayesian hazard belief.

Interpretation checks:

- A genuine added latent dimension should improve held-out reconstruction with
  PC3 and raise both local and global dimension estimates.
- A curved 2-D sheet may raise PC3 variance and 3-PC linear reconstruction while
  TWO-NN remains near two. That result is still mechanistically interesting, but
  it is a bending/unfolding claim rather than a dimensionality claim.
- If PC3's added decoding is selective for Bayesian hazard mean/uncertainty and
  not last evidence, it supports the proposed Bayesian computation.
- The heuristic control reveals changes caused by generic hazard-head learning;
  the true-report model must exceed that control to link PC3 to Bayesian report
  rescue.

## Experiment 2: matched-suffix history relaxation

The prefix-length heatmap shows that very short sequences lead to a stay
prediction, but that observation does not by itself prove a stay attractor. This
experiment separates an initial bias, persistent history state, and incoming
evidence.

For every replicate it generates low- and high-hazard 20-item histories whose
final latent sign is matched. It then appends the *identical* low-hazard suffix
to both histories and, in a second pair, the identical high-hazard suffix to
both histories. Suffix latent states, evidence, and observation noise are shared
within each pair. Sign-mirrored copies balance report state. Thus any paired
difference during the suffix is caused by prior history rather than different
current inputs.

```powershell
.\.venv\Scripts\python.exe .\phase_5\test84_manifold\history_switch_experiment.py
```

Important outputs in `history_switch_outputs/`:

- `paired_suffix_contrasts.csv`: hazard-output, report-output, and hidden-state
  differences at each shared suffix item.
- `relaxation_metrics.csv`: exponential time constants, empirical half-lives,
  and area under each history-contrast curve.
- `condition_summary.csv`: short-prefix bias and pre/post-switch diagnostics.
- `controlled_trajectories.csv.gz`: PC1-PC3 and outputs for a reproducible
  subset of the controlled sequences.
- `sequence_bank.npz`: the exact stimuli and latent paths.

A high-hazard history followed by low-hazard evidence should begin above the
low-to-low control and relax toward it if the network maintains and updates a
volatility belief. No decay indicates persistent memory or an attractor over the
tested horizon. Immediate collapse indicates little usable hazard memory. A
large history contrast in hidden state with little hazard-output contrast means
the distinction exists but is not read out by the hazard head.

The 40-item sequence is intentionally outside the 20-item training horizon. It
is a causal dynamical probe, not an in-distribution accuracy test.

## Experiment 3: slow points and basin census

This analysis does not assume that high and low hazard are basins. It asks
whether empirical states converge to a few autonomous endpoints and separately
finds fixed/slow points by minimizing

`q(h) = 0.5 * ||F(h) - h||^2`.

The `null` map is one no-input update. The evidence-cycle maps are
phase-matched: starting immediately after an evidence item, they apply four null
updates and then the next canonical evidence update. This avoids treating a
repeated evidence-present input as though it were the trained temporal regime.

```powershell
.\.venv\Scripts\python.exe .\phase_5\test84_manifold\slow_point_tracking.py `
  --maps null,neg_cycle,zero_cycle,pos_cycle
```

Important outputs in `slow_point_outputs/`:

- `slow_points.csv`: speed, stability, Jacobian spectrum, readout labels, and
  distance to real trajectories for every deduplicated candidate.
- `slow_point_states.npz`: full 128-D candidate states.
- `null_basin_assignments.csv`: endpoint and cluster for each empirical initial
  state after a long null rollout.
- `null_basin_summary.csv`: occupancy by source hazard/report class.
- `state_space_*png`: on-trajectory drift plus slow points in local PCA space.

The long null rollout is also out of distribution because the trained task has
only four null steps between observations. Interpret discrete endpoint clusters
as properties of the autonomous GRU, then verify that their slow points lie near
real trajectories. If most states remain unconverged, form a continuum, or sit
near a chain/sheet of slow points, "slow manifold" is better language than
"basins." Stable fixed points with distinct hazard readouts and reproducible
occupancy support a basin interpretation. Saddles and near-unit eigenvalues can
instead organize transient channels and long relaxation times.

## Smoke tests

These commands exercise every pipeline without a long run:

```powershell
.\.venv\Scripts\python.exe .\phase_5\test84_manifold\manifold_geometry.py `
  --models bayesian --checkpoints 1,7,10 --max-csvs 1 --max-trials 120 `
  --max-cv-trials 80 --cv-folds 2 `
  --output-dir .\phase_5\test84_manifold\smoke_geometry

.\.venv\Scripts\python.exe .\phase_5\test84_manifold\history_switch_experiment.py `
  --models bayesian --checkpoints 1,10 --n-sequences 32 `
  --max-trajectory-replicates 8 `
  --output-dir .\phase_5\test84_manifold\smoke_history

.\.venv\Scripts\python.exe .\phase_5\test84_manifold\slow_point_tracking.py `
  --models bayesian --checkpoints 1,10 --maps null,zero_cycle `
  --max-csvs 1 --max-trials 80 --n-inits 24 --opt-steps 120 `
  --patience 40 --max-points-per-map 4 --basin-inits 24 --basin-steps 30 `
  --output-dir .\phase_5\test84_manifold\smoke_slow_points
```

## Methodological sources

- Sussillo & Barak introduced optimization of fixed and slow points plus local
  linearization for reverse engineering trained RNNs:
  <https://pubmed.ncbi.nlm.nih.gov/23272922/>.
- Mante et al. showed how approximate line attractors and selection directions
  can implement low-dimensional recurrent computation:
  <https://www.nature.com/articles/nature12742>.
- Maheswaranathan et al. used fixed-point topology and linearized dynamics to
  identify a line-attractor mechanism in GRUs/LSTMs/RNNs:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7416638/>.
- Their population study also cautions that representational geometry can vary
  while the underlying computational scaffold remains similar:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7416639/>.
- TWO-NN estimates local intrinsic dimension from first/second-neighbor distance
  ratios and is designed to reduce sensitivity to curvature and density changes:
  <https://www.nature.com/articles/s41598-017-11873-y>.
- Participation-ratio dimensionality is developed in the theory of neural
  population dimensionality:
  <https://ganguli-gang.stanford.edu/pdf/17.theory.measurement.pdf>.
- Shape metrics formalize comparisons of neural representations modulo
  transformations such as rotations:
  <https://papers.nips.cc/paper/2021/file/252a3dbaeb32e7690242ad3b556e626b-Paper.pdf>.
- Dynamical Similarity Analysis explains why trajectory dynamics can differ from
  static representational geometry; it is a possible follow-up if the present
  fixed-point and matched-history results disagree:
  <https://papers.nips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html>.

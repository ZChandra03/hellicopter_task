# Epoch-10 hazard-to-report mechanism

This experiment turns the epoch-10 hidden state into a task-aligned dynamical
atlas. It is deliberately restricted to `sigma_1`, seed 0, and checkpoint
`ep010` unless command-line arguments override those values.

The analysis asks four concrete questions:

1. What does the occupied low-dimensional structure look like in coordinates
   tied directly to the two trained outputs?
2. Where does the stored hazard state change the report computation?
3. Which fixed, slow, saddle, transition, and separatrix structures organize
   trajectories?
4. Are the apparent high/low-hazard regions classical basins, or only basins of
   an input-conditioned map?

## Task-aligned coordinates

The three displayed coordinates are orthonormal directions in the 128-unit
hidden state:

- **report readout**: the normalized report-head weight;
- **hazard-exclusive**: the component of the hazard-head weight orthogonal to
  the report readout;
- **residual variance**: the leading variance direction after removing the
  first two directions.

Coordinates are reported in empirical standard-deviation units. This is a
targeted dimensionality-reduction view, not a claim that all activity is
exactly three-dimensional. It follows the task-axis logic used by
[Mante et al. (2013)](https://www.nature.com/articles/nature12742), while the
fixed/slow-point search follows the dynamical-systems approach of
[Sussillo and Barak (2013)](https://pubmed.ncbi.nlm.nih.gov/23272922/).

## Central causal probe

For each sampled real hidden state immediately after evidence item 19, the
script applies the phase-matched sequence

```text
four null steps -> controlled final evidence
```

over final evidence values from -2 to +2. It differentiates the final report
logit with respect to the prefix hidden state and projects that gradient onto
the hazard-exclusive direction. Because this direction is orthogonal to the
report readout, it has zero instantaneous effect on report. Any report effect
after the five-step update is therefore recurrent hazard-to-report coupling.

The script also measures ordinary evidence gain, report transition thresholds,
and on-manifold psychometric curves split by the naturally occurring hazard
state. The Jacobian intervention is mechanistically sharper; the natural-state
split is its on-manifold corroboration.

## Fixed points and basins

Fixed/slow points are optimized for the stroboscopic map

```text
post-evidence state -> four null steps -> repeated evidence value
```

at nine repeated-evidence values. Candidate initializations cover the real
manifold, hazard/report extremes, and states generated from sign-mirrored
evidence. Exact 128-dimensional Jacobian eigenspectra are computed for the
three fastest candidates at each evidence value.

The zero-evidence map is then rolled forward for 100 cycles from all 1,500 real
penultimate prefix states. Terminal states are assigned to the two stable
zero-evidence endpoints, and a linear three-dimensional separatrix is fit to
the resulting assignments. These are basins of the repeated ambiguous-input
map; they should not be confused with autonomous null-input basins or with the
continually changing maps encountered on natural trials.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe `
  .\phase_5\test85_epoch10_mechanism\epoch10_mechanism.py
```

Quick integration test:

```powershell
.\.venv\Scripts\python.exe `
  .\phase_5\test85_epoch10_mechanism\epoch10_mechanism.py `
  --smoke `
  --output-dir .\phase_5\test85_epoch10_mechanism\smoke_outputs
```

The heuristic comparison can be run without changing code:

```powershell
.\.venv\Scripts\python.exe `
  .\phase_5\test85_epoch10_mechanism\epoch10_mechanism.py `
  --model heuristic `
  --output-dir .\phase_5\test85_epoch10_mechanism\heuristic_outputs
```

## Main outputs

| Output | Meaning |
|---|---|
| `task_aligned_manifold_atlas.png` | trajectories, zero-evidence flow, manifold sheet/fold, and local dimension |
| `hazard_to_report_gating.png` | matched psychometrics, causal hazard gain, transition line, and evidence gain |
| `fixed_point_branches.png` | input-conditioned fixed/slow loci and tested stability |
| `zero_cycle_basins.png` | finite-horizon basin labels, candidate separatrix, and relaxation paths |
| `trajectory_states.csv.gz` | all 30,000 real post-evidence states in task coordinates |
| `trajectory_examples.csv` | small trajectory subset for interactive visualization |
| `report_transition_line.csv` | one controlled report threshold per sampled prefix state |
| `hazard_to_report_surface.csv` | binned local hazard-to-report gain surface |
| `controlled_vector_field.csv` | task-coordinate flow under repeated evidence -1, 0, and +1 |
| `conditioned_fixed_points.csv` | fixed/slow candidates, outputs, distances, and eigenspectrum labels |
| `zero_cycle_basin_assignments.csv` | 100-cycle endpoints and separatrix scores for all real prefixes |
| `points_of_interest.csv` | unified point/line/region landmarks for a trajectory viewer |
| `summary.json` | compact numerical results |

`run_config.json` records all resolved paths and arguments. Large raw probe
arrays and the task axes are retained in compressed NumPy files so follow-up
plots do not need to rerun the network.

## Interpretation guardrails

- This is one trained network, not an estimate over training seeds.
- A slow point one or two empirical neighbor spacings away is a plausible
  organizing scaffold, not proof that natural trajectories visit it exactly.
- An infinitesimal hazard-axis perturbation can have a component transverse to
  the occupied manifold. Its sign and location should be read together with
  the natural-state psychometrics and transition-line shift.
- Failure to find a point is not proof it does not exist; the search is broad
  but numerical.
- The nearest-neighbor branch labels in the CSV are visualization aids, not a
  formal bifurcation continuation result. The figure emphasizes the
  minimum-speed locus and individually verified points.

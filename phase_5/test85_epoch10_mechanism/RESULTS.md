# Epoch-10 results: hazard gates how prior state reaches report

All numbers below use the five `sigma_1` validation files (1,500 trials), the
true-report-trained seed 0 network, and checkpoint `ep010`.

## Main result

The epoch-10 network does not appear to use hazard by simply adding a hazard
bias to the report logit. Instead, hazard changes how strongly the state carried
from the first 19 evidence items survives the last update.

The clearest statistic is the final-evidence value at which the report logit
crosses zero. Multiplying that threshold by the sign of the prefix report state
makes positive values mean “the old state has been erased and must be
re-established by congruent final evidence.” Its median moves monotonically
across prefix network-hazard quartiles:

| prefix hazard | Q1 low | Q2 | Q3 | Q4 high |
|---|---:|---:|---:|---:|
| signed transition evidence | -0.610 | -0.236 | 0.118 | 0.490 |

The signed threshold has Spearman correlation `0.802` with the prefix hazard
logit. After controlling linearly for absolute prefix-report strength, a
one-standard-deviation hazard increase shifts it by `0.346` evidence units;
the hazard-by-prior-strength interaction is another `0.157`.

This gives a direct geometric interpretation: **low hazard preserves the old
state across ambiguous or mildly opposing final evidence; high hazard contracts
that memory and lets the new evidence determine report.**

## Direct hazard-to-report coupling

The report and hazard readout weights are nearly orthogonal (cosine `-0.072`).
The local intervention axis is explicitly the part of the hazard readout with
its report component removed, so moving along it cannot change the report
instantaneously.

After four null updates and one controlled final-evidence update:

- increasing hazard weakens the signed prefix report at `83.3%` of empirical
  report-transition points;
- under exactly zero final evidence, it weakens the prior on `79.8%` of sampled
  prefix states;
- the median absolute probability gain at the transition is `0.0295` change in
  report probability per unit of prefix hazard logit;
- the strongest binned mean has magnitude `0.0594`, near final evidence
  `-0.333` for the strongest positive-prefix bin;
- the median ordinary final-evidence gain at the decision transition is `1.93`
  report-logit units per evidence unit.

The sign reversal across the report axis is important. Raising hazard pushes a
positive prior downward and a negative prior upward: it contracts the old
state toward a new-evidence-sensitive transition channel. This is the expected
qualitative operation for Bayesian hazard use, and is more specific than merely
decoding hazard from activity.

## Manifold structure

The orthonormal report, hazard-exclusive, and residual-variance axes explain
`83.2%` of empirical hidden-state variance. Local participation-ratio dimension
has median `3.90` and 95th percentile `4.57`, consistent with the previous
result that the converged system has three prominent global directions but
local dimension closer to four.

The residual coordinate forms a curved sheet over the report-by-hazard plane,
not a flat rectangular surface. Local dimensional expansion is distributed,
with pockets around trajectory bends rather than one single third-axis branch.
The saved `manifold_sheet.csv` and `local_structure.csv` identify those pockets
for overlay in a trajectory viewer.

## Fixed and slow scaffold

Across nine repeated-evidence maps, the search retained 18 fixed/slow points;
their median distance from the empirical state cloud is `1.82` empirical
nearest-neighbor spacings. The zero-evidence cycle contains three distinct
organizing structures among its three fastest candidates:

| type | P(report +) | P(high hazard) | spectral radius | interpretation |
|---|---:|---:|---:|---|
| unstable fixed point | 0.609 | 0.150 | 1.139 | saddle separating channels |
| stable slow point | 0.504 | 0.804 | 0.961 | high-hazard, report-neutral endpoint |
| stable slow point | 0.824 | 0.050 | 0.983 | low-hazard, positive-report endpoint |

The asymmetry is real in the numerical search: despite including trajectories
from sign-mirrored evidence, no negative-report stable zero-evidence point was
found. The high-hazard endpoint is almost report-neutral, whereas the low-hazard
endpoint carries a strong positive report bias. This is evidence against a
simple symmetric pair of high/low hazard wells. It is still a failure-to-find
result, so a denser continuation search would be needed before claiming the
negative point is mathematically absent.

The minimum-speed locus across repeated evidence moves from stable
negative-report/low-hazard points for negative inputs, through the saddle and
two stable modes at zero, into a positive-report/low-hazard slow channel for
positive inputs. That organization is better described as an input-conditioned
branched scaffold than as a single autonomous attractor landscape.

## Finite-horizon zero-evidence basins

Starting from every real penultimate prefix state and repeating an ambiguous
zero-evidence cycle 100 times gives:

| endpoint | fraction | terminal P(report +) | terminal P(high hazard) |
|---|---:|---:|---:|
| high-hazard/report-neutral | 0.795 | 0.503 | 0.796 |
| low-hazard/positive-report | 0.205 | 0.824 | 0.049 |

The terminal basin margin correlates `0.846` (Spearman) with the initial
hazard-exclusive coordinate. Assignment to the high-hazard endpoint rises from
`55.2%` for true hazards at most 0.2 to `98.7%` for true hazards above 0.8.
Thus the stored hazard state strongly determines the long ambiguous-input
relaxation route.

A linear separatrix in the three task coordinates classifies the two endpoints
with `98.1%` training accuracy. Its decision plane is
`-7.36 report + 8.62 hazard - 3.44 residual + 9.98 = 0`. The boundary is
therefore not a threshold on hazard alone: report position and the residual
manifold coordinate both materially tilt it. Prefix states closest to this
plane are saved as separatrix landmarks.

This does **not** imply that 79.5% of natural trials occupy a permanent
high-hazard autonomous basin. It is a basin census for the repeated ambiguous
input map. Natural evidence changes the map on every cycle, and the earlier
history-switch experiment already showed relaxation times comparable to or
longer than a trial.

## Landmarks for the trajectory viewer

`points_of_interest.csv` unifies four useful overlays:

- conditioned fixed/slow points;
- centroids along the empirical report-transition line;
- the maximum hazard-leverage ridge for each prefix-report bin;
- the top 2% of local dimension-expansion points;
- prefix states nearest the fitted zero-cycle separatrix.

Together with `trajectory_examples.csv`, `controlled_vector_field.csv`, and
`zero_cycle_relaxation.csv.gz`, these are sufficient to build an interactive
viewer with projection switching, trajectory selection, vector-field toggles,
and landmark overlays without rerunning the model.

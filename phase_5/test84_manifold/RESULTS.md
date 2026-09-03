# Seed-0 results

These results were generated on the five `sigma_1` validation files (1,500
trials) using only seed 0. Cross-validation estimates trial generalization
within this seed; they are not uncertainty estimates over independently trained
networks.

## Main conclusion

The true-report network undergoes a real low-dimensional expansion at the same
time that it learns the hazard task. This is not explained by PC rotation alone:
the third component improves held-out reconstruction on new trials, the result
appears both across the full belief trajectory and at final trial endpoints, and
participation-ratio dimension rises from about 2 to about 3. The expansion
begins at epoch 6 and becomes large at epoch 7.

The expansion is not unique to true-report training: the last-evidence control
also becomes more locally complex as it learns hazard. However, the expansion
is much larger in the true-report model and is accompanied by a behavioral
shift away from last evidence toward the Bayesian report policy. This supports
a stronger claim than "PC3 is hazard": hazard learning creates the extra
dynamical scaffold in both networks, while true-report training recruits that
scaffold more strongly for joint hazard/state inference.

## 1. Geometry and behavior

Selected `all_evidence` metrics:

| model | epoch | predict accuracy | participation ratio | PC3 variance | held-out gain: 3 PCs vs 2 |
|---|---:|---:|---:|---:|---:|
| true report | 1 | 0.496 | 1.993 | 0.003 | 0.003 |
| true report | 5 | 0.502 | 1.368 | 0.015 | 0.015 |
| true report | 6 | 0.628 | 1.964 | 0.056 | 0.056 |
| true report | 7 | 0.786 | 2.813 | 0.170 | 0.169 |
| true report | 10 | 0.814 | 3.100 | 0.206 | 0.206 |
| heuristic | 1 | 0.496 | 1.988 | 0.004 | 0.004 |
| heuristic | 5 | 0.542 | 1.430 | 0.015 | 0.016 |
| heuristic | 6 | 0.577 | 1.465 | 0.028 | 0.029 |
| heuristic | 7 | 0.725 | 1.620 | 0.062 | 0.063 |
| heuristic | 10 | 0.790 | 1.871 | 0.081 | 0.082 |

The endpoint-only analysis gives the same qualitative result. In the true-report
model, final-state PC3 variance rises from `0.0004` at epoch 1 to `0.167` at
epoch 10; held-out 3-PC versus 2-PC reconstruction gain rises from `0.0005` to
`0.164`. The heuristic endpoint gains are smaller (`0.0005` to `0.057`). This
rules out temporal-position pooling as the sole source of PC3.

TWO-NN returns values near four after hazard learning rather than exactly three.
Thus the safest description is an expansion from an almost linear/planar early
solution to a manifold with at least three important variance directions and
local dimension around four. A literal globally 3-D manifold is too strong.

### What the third direction represents

PC3's function changes during the transition:

- At epoch 6, adding PC3 raises held-out decoding of Bayesian hazard mean by
  `0.206 R2` over the trajectory and `0.334 R2` at trial endpoints. PC3 is also
  strongly aligned with the hazard readout at this checkpoint (squared
  alignment `0.400`).
- By epochs 7–10, hazard belief is already well represented in the leading two
  PCs, so the marginal hazard benefit of PC3 declines.
- At epoch 10, adding PC3 improves Bayesian state-belief decoding by `0.028 R2`
  over trajectories and `0.048 R2` at endpoints. Its marginal last-evidence
  decoding also rises, while direct hazard-head alignment is near zero.

Therefore "PC3 is the hazard axis" is accurate during the epoch-6 transition
but not at convergence. The manifold rotates/reorganizes after the new direction
appears; hazard information moves into the leading plane while the third
direction contributes to state/report structure.

### Behavioral coupling

For the true-report model, predict accuracy rises from `0.502` at epoch 5 to
`0.628` at epoch 6 and `0.786` at epoch 7. Over the same interval PC3 variance
rises from `0.015` to `0.056` and `0.170`.

On the 154 ambiguous-final-evidence trials (`abs(final evidence) <= 0.2`):

- true-report accuracy rises from `0.461` at epoch 1 to `0.675` at epoch 10;
- report agreement with the Bayesian policy rises from `0.487` to `0.870`;
- report agreement with last evidence falls to `0.571`;
- the heuristic model remains last-evidence-like at epoch 10 (`0.929`
  agreement with last evidence) and reaches only `0.526` report accuracy.

This is the strongest link between the geometric transition and the intended
Bayesian-like computation.

## 2. Matched-suffix history dynamics

The 512-replicate controlled experiment produces a strong history state after
hazard learning. At epoch 10 in the true-report model:

- after low-hazard history, the low-to-low control ends at `P(high)=0.099`;
- after high-hazard history, the *identical* low-hazard suffix ends at
  `P(high)=0.626`;
- low-to-high crosses `P(high)=0.5` after 13 suffix observations;
- high-to-low never crosses below `0.5` within the 20-observation suffix;
- the paired high-history versus low-history hazard contrast falls only from
  `0.751` to `0.527` under the low suffix (fitted time constant about 52
  evidence items), but falls to `0.226` under the high suffix (time constant
  about 18 items).

This directional hysteresis rejects a simple rapidly leaky hazard estimate.
Low-to-high transitions occur within a normal trial length, whereas a learned
high-hazard state is much harder to erase with low-hazard evidence. The same
asymmetry exists in the heuristic model, so it is principally a hazard-task
mechanism, not by itself the source of Bayesian report behavior.

Very short sequences do not start in a literal high/low attractor: after one
evidence item the final model averages `P(high)=0.425`, consistent with the
observed initial stay response. High-volatility evidence then drives the network
into a persistent high-hazard region.

## 3. Slow points and basins

The early epoch-1 null dynamics converge to a single neutral endpoint from all
sampled states. After learning begins, that simple basin disappears:

| model | epoch | fraction converged after 200 null updates | discrete endpoint clusters |
|---|---:|---:|---:|
| true report | 1 | 1.000 | 1 |
| true report | 5 | 0.160 | 4 |
| true report | 6 | 0.000 | 0 |
| true report | 7 | 0.000 | 0 |
| true report | 10 | 0.016 | 2 |
| heuristic | 1 | 1.000 | 1 |
| heuristic | 5 | 0.000 | 0 |
| heuristic | 6 | 0.000 | 0 |
| heuristic | 7 | 0.000 | 0 |
| heuristic | 10 | 0.000 | 0 |

At epoch 7 the true-report model's ambiguous zero-evidence cycle contains a
stable high-hazard slow point (`P(high)=0.808`), a stable low-hazard point
(`P(high)=0.152`), and a family of low/intermediate-hazard saddles near
`P(high)=0.395`. At epoch 10 the retained candidates include stable low-hazard
points near `P(high)=0.050` and an unstable point family near `P(high)=0.150`.
These points are slower than every sampled empirical state and lie about one to
two empirical nearest-neighbor spacings from the real manifold.

Under the null map, the trained model has near-unit directions and both stable
and unstable candidates, but almost no empirical states settle into discrete
endpoints within 200 steps. The best description is therefore:

> a slow, weakly attracting/repelling scaffold with saddle-organized channels
> and hysteresis, not two clean high/low autonomous basins.

The high-hazard persistence seen in the matched-suffix experiment can still be
"basin-like" over the task horizon without being a classical autonomous fixed
point basin. Incoming evidence changes the relevant stroboscopic map, and the
20-item horizon is much shorter than the observed relaxation time.

## Files

- Full geometry outputs: `geometry_outputs/`
- Full history-switch outputs: `history_switch_outputs/`
- Cross-checkpoint slow-point outputs: `slow_point_outputs/`

The exact arguments and resolved device/model paths are recorded in each
folder's `run_config.json`.

# The shape of the two-dimensional success region

Phase 12's hard gate: does opening `T` buy structure that the one-dimensional space lacked?

Data: `outputs/logs/landscape_2d_fine.json` (40 448 episodes, 93.7 % valid).
Analysis: `outputs/logs/landscape_2d_fine_analysis.json`, `outputs/logs/parameter_targets_fine.json`.
Code: `src/probe_drawer/analysis/{landscape_2d,parameter_targets}.py`.
Parameter space and physical bounds: `docs/PARAMETER_SPACE_2D.md`.

---

## 1. The verdict, up front

**Structure exists, and it is weaker than the strong form of the hypothesis.** The gate
passes, and the honest statement of what it passes on is:

| Claim | Verdict | Evidence |
|---|---|---|
| The region's location depends on the hidden state | **Yes, unambiguously** | centroid `F` vs `µ_d`: ρ = **+0.973**; regions span 91 % of the box |
| The region is genuinely non-convex | **Yes, in a minority** | midpoint failure median **6.4 %**, 16 of 31 states above 5 %, worst **66.7 %** |
| Non-convexity is physical, not grid noise | **Yes** | survives a 5× resolution increase; correlates with `µ_s` at **+0.556** |
| The region is disconnected | **Sometimes** | 12 of 31 states under 8-connectivity |
| Averaging two good answers is unsafe | **Rarely** | the centroid itself fails for only **2 of 32** states (6.2 %) |
| `T` is determined by the hidden state | **No** | target `T` spans 1.0–2.1 s with sd 0.12–0.21; ρ with `µ_d` ≤ 0.47 |

So the second axis adds moderate structure, concentrated in high-friction drawers, while
leaving `T` broadly permissive. That is not the clean "the landscape is necessary" result the
phase was looking for, and it is what the data says.

## 2. What the region is

Figure E is the explanation. The task is `|d(T) − d_goal| ≤ ε_d` **and** `|v(T)| ≤ ε_v`, and
the two conditions carve the box differently:

* the `d = d_goal` contour is a **hyperbola-like curve** — low force needs long `T`, high
  force needs short `T`;
* the `|v| = ε_v` contour is a **near-vertical wall** truncating it on the fast side.

The success region is their intersection: a **thin curved band** hugging the `d_goal` contour,
cut off where the drawer would still be moving. Median area is 5.0 % of the box, elongation
10.6, orientation 99.8° in the normalised box — i.e. long and near-vertical, tilting as
friction rises (orientation vs `µ_d`: +0.717).

## 3. Why resolution had to come first

The coarse sweep's `F` step was 0.25 N — *larger than the entire 0.20 N success band* the 1-D
Oracle measured. At that resolution the median region spanned 3 columns by 9 rows and no claim
about its shape would have meant anything. Two numbers moved substantially when the grid was
refined to 0.05 N:

| | coarse (0.25 N) | fine (0.05 N) |
|---|---|---|
| median region size | 3 × 9 cells | **17 × 14 cells** |
| states resolvable (≥ 4 × 4) | 16 of 46 | **31 of 32** |
| centroid failure rate | 19.6 % | **6.2 %** |
| midpoint failure (median) | 9.6 % | **6.4 %** |

The centroid's apparent 19.6 % failure was largely an artefact: a 3-cell-wide region's mean
rounds badly. The midpoint failure rate, by contrast, **survived** — and it is measured over
twice as many states, so the fine number is the one to quote.

This is why the phase measured before claiming. Asserting non-convexity from the coarse grid
would have been asserting the grid.

## 4. Non-convexity, where it lives

The midpoint failure rate is the operational test: for pairs of succeeding points whose mean
lands exactly on a swept grid point, does the mean succeed? Fine grid, 31 resolvable states:

* median **6.4 %**, mean 11.5 %, max **66.7 %**
* 16 of 31 states above 5 %
* correlates with `µ_s` at **+0.556** and `µ_d` at +0.442; with mass at −0.283

**It is not uniform — it is concentrated in sticky drawers, and the worst case has a physical
signature.** The state at 57.7 % midpoint failure with 6 disconnected components under
8-connectivity is `m = 7.0, µ_s = 1.99, µ_d = 0.75, b = 9.9` — a **stick-slip ratio of 2.65**.
A drawer that grips much harder than it slides has an unstable breakaway, and its success
region fragments. That is a coherent mechanism rather than a numerical accident, and it is why
the correlation with `µ_s` is the more informative statistic than the median.

## 5. Disconnection, carefully

Reported at two connectivities on purpose. A thin band that steps one column per row is
4-disconnected and 8-connected, and is physically one band; counting only the first would
manufacture topology out of resolution.

| | 4-connectivity | 8-connectivity |
|---|---|---|
| states with more than one component | 19 of 31 | **12 of 31** |
| orthogonally convex (every row *and* column one unbroken run) | — | 10 of 31 |

So 12 states are genuinely fragmented at this resolution and 19 look worse than they are. The
detectors are validated against hand-built masks first
(`tests/unit/test_landscape_2d.py`): a staircase must be one region under 8-connectivity and
several under 4, and two islands must stay two under both.

## 6. `T` is the permissive axis

The finding that most weakens the 2-D case, recorded for that reason.

Every rule for choosing a single target parameter lands `T` in a narrow band:

| rule | target `F` range | target `T` range | `T` sd | `F` vs `µ_d` | `T` vs `µ_d` |
|---|---|---|---|---|---|
| centroid | 0.43–3.70 N | 1.34–1.86 s | 0.120 | +0.973 | +0.176 |
| min_cost | 0.30–4.00 N | 1.00–1.70 s | 0.175 | +0.914 | +0.230 |
| max_margin | 0.50–4.10 N | 1.10–2.10 s | 0.209 | +0.968 | +0.467 |

The force coordinate sweeps almost the whole range and is almost perfectly predicted by
dynamic friction. The duration coordinate barely moves and is barely predicted by anything.

**A regressor that outputs a constant `T` and a friction-driven `F` will hit the target.** The
second axis is a real physical trade-off (§2) and a nearly-degenerate *prediction* problem.

## 7. What a single point can and cannot do

| rule | succeeds for | note |
|---|---|---|
| centroid | 30 of 32 (93.8 %) | the natural least-squares target |
| min_cost | 32 of 32 (100 %) | tautological — selected *from* the success set |
| max_margin | 32 of 32 (100 %) | tautological, and has 2 grid steps of margin against min_cost's 1 |

The 100 % figures are labelled tautological because they are: both rules pick an actual
succeeding point. So the claim "a landscape is needed because no single answer exists" is
**false here** — a single answer always exists, and `max_margin` is the robust one.

What remains true is narrower and still worth testing: the good targets are a *less smooth*
function of the hidden state than the centroid, and a regressor's own error moves it off
whatever point it aims at, into a band whose median width is a few grid steps. Whether a
model that represents the whole region handles that better than one that names a point is an
**empirical** question about learning, not a structural one about representability — and it is
what Dataset v1 and the closed-loop comparison are for.

The fair baseline is therefore trained on `max_margin`, the highest-ceiling and most robust
target. Training it on the centroid would hand it a 6.2 % ceiling for free, which is the
mirror image of letting ACE see `ξ`.

## 8. Coverage and the formal box

* **32 of 32** hidden states have a succeeding `(F, T)` on the fine grid — up from 46 of 48 on
  the coarse one, because the fine grid resolves bands the coarse one stepped over.
* Proposed box after trimming to where some state succeeds: `F ∈ [0.30, 4.20] N`,
  `T ∈ [1.10, 2.50] s`. Only `T = 1.00 s` was dropped; every swept force earns its place.

## 9. What this licenses

Dataset v1, with the honest expectation set: the phase should measure whether a landscape model
beats a *fairly targeted* single-point regressor, and should be prepared for the answer to be
"only modestly, and mostly on the sticky drawers where the region fragments". Section 6 is the
reason to expect that, and it was found before the training rather than after.

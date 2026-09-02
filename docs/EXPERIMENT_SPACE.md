# Experiment space: what is usable, and the parameters selected from it

**Everything in this document is measured.** The figures that used to appear in the project's
own instructions -- 2 N, 10 N, 5 mm, 100 mm, 2 s, 5 N -- were examples for getting the
pipeline working. They are relabelled throughout as a *provisional validation operating
point* and none of them survived unchanged (`docs/DECISIONS.md` D021).

Data: `outputs/logs/sweep_execution_coarse.json` and `sweep_fine_fall{010,015,020,030,035}.json`.
Reports: `outputs/logs/oracle_landscape.json`, `probe_calibration.json`.
Collected 2026-09-02 on Isaac Sim 5.1.0.0 / Isaac Lab 2.3.0, RTX 5080.

---

## 1. The valid operating region

An episode can miss the goal and still be perfectly good evidence. What the *validity mask*
rejects is episodes whose physics or control quality make them unusable either way. Every
threshold is anchored to a measurement, not chosen for roundness.

| Threshold | Value | Anchored to |
|---|---|---|
| `mechanical_margin_fraction` | 0.80 of the 0.4 m travel, i.e. 0.32 m | At 0.326 m (0.82 of travel) the TCP lateral drift was 14-15 mm, against 0.36-0.66 mm below 0.2 m. Behaviour near the stop is qualitatively different. |
| `max_peak_velocity` | 0.25 m/s | Clean Phase 8 runs peaked at 0.054-0.132 m/s; the drifting one at 0.418 m/s. |
| `max_lateral_drift` | 5 mm | Clean runs stayed under 0.7 mm: a 7x margin, and 10x inside the 50 mm safety limit. |
| `max_orientation_drift_deg` | 5 deg | Clean runs stayed under 0.41 deg. |
| `min_displacement` | 1 mm | About 20x the residual zero-command creep (D010). |

Authoritative values: `probe_drawer.evaluation.operating_region.OperatingRegionCfg`,
mirrored in `configs/evaluation.yaml`. They are deliberately far tighter than
`SafetyLimits`: safety stops the simulation from diverging, validity decides whether an
episode may become an Oracle label.

### Where the region is

Coarse sweep, 11 hidden states x 9 forces x 5 durations = 495 points, 48.1 % valid. Fraction
of hidden states usable, by `(T, F_peak)`:

| T \ F | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.36 | 0.45 | 0.73 | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 0.82 |
| 1.0 | 0.36 | 0.55 | **0.91** | **0.91** | **1.00** | 0.82 | 0.36 | 0.18 | 0.00 |
| 1.5 | 0.36 | 0.64 | **0.91** | 0.73 | 0.82 | 0.36 | 0.18 | 0.00 | 0.00 |
| 2.0 | 0.36 | 0.45 | 0.82 | 0.73 | 0.36 | 0.27 | 0.18 | 0.00 | 0.00 |
| 3.0 | 0.36 | 0.45 | 0.45 | 0.36 | 0.36 | 0.18 | 0.00 | 0.00 | 0.00 |

Read off this table: **the usable force range narrows as `T` grows**, because a longer pull
at the same force runs the drawer into its end stop. `T = 1.0` and `T = 1.5` s have the
widest usable force range, and above 8 N nothing is usable for any `T >= 1 s`. Those bounds
set the fine sweep: forces 1.0-8.0 N, durations 1.0 and 1.5 s.

Figure: `outputs/plots/experiment_space_validity.png`.

### Why points are rejected (fine sweep, 4536 points, 84.7 % valid)

| Reason | Count |
|---|---|
| no measurable motion | 432 |
| excessive velocity | 242 |
| excessive lateral drift | 141 |
| excessive orientation drift | 113 |
| mechanical limit | 1 |
| safety abort | 1 |

The two dominant reasons are the two ends of the force axis: too little to move a stiff
drawer, too much for a free one. That is the shape a well-chosen range should have.

---

## 2. The Phase 8 drift question, answered

Phase 8 left an open question: was the 14-15 mm held-axis drift on the `easy` preset the
*operating point* or the *controller*?

**It is the operating point.** Figure
`outputs/plots/execution_drift_vs_operating_point.png` plots peak lateral drift against
force, displacement and peak velocity for all 4536 fine-sweep points. Drift stays below
1 mm across the whole force range as long as the displacement and speed stay moderate, and
rises by more than an order of magnitude only as the drawer approaches its end stop at high
speed. At small displacement, high force alone does not produce drift.

Consequence: the hybrid controller does not need fixing. The validity mask already excludes
the regime where it degrades, and the selected operating point keeps the worst succeeding
episode at 16 % of travel.

---

## 3. The execution force profile's ramp-down

This was not a tuning knob but a design parameter, and it had to be measured.

With the original `fall_fraction = 0.1`, a drawer that travels 50 mm in 1.5 s is still
moving at roughly 0.16 m/s when the force reaches zero: the ramp-down lasts 0.15 s, and a
low-resistance drawer cannot decelerate in that time. "Reach the goal **and** come to rest"
was therefore unreachable for most hidden states.

Largest `d(T)` achievable with `|v(T)| <= 0.05` m/s, measured over the 108-point grid:

| `fall_fraction` | T = 1.0 s | T = 1.5 s |
|---|---|---|
| 0.10 | 29.7 mm | 49.4 mm |
| 0.15 | 32.5 mm | 54.9 mm |
| 0.20 | 36.5 mm | 65.2 mm |
| 0.30 | 39.7 mm | 71.3 mm |
| 0.35 | 43.8 mm | 79.4 mm |

The relationship is monotone and substantial: a longer ramp-down raises the reachable
at-rest distance by about 60 % over this range. Each row is a full 4536-point sweep; all five
datasets are in `outputs/logs/`.

Selected: **`fall_fraction = 0.20`**. It produced the highest-discrimination accepted task
of the five (1.568 against 1.557 at 0.15, 1.564 at 0.10, 1.549 at 0.30, 1.531 at 0.35) with
coverage 0.98, and it sits inside the 0.15-0.30 band. `docs/DECISIONS.md` D023.

---

## 4. The selected experiment parameters

Authoritative values: `probe_drawer.experiment_plan`, mirrored in
`configs/experiment_plan.yaml`, drift-tested by `tests/unit/test_config_snapshots.py`.

### Main task

| Parameter | Value | How it was chosen |
|---|---|---|
| `T_goal` | **1.5 s** | Widest usable force range together with T = 1.0; selected by the Oracle score. |
| `d_goal` | **50 mm** | The largest goal reachable *at rest* by nearly every hidden state at this profile. |
| `epsilon_d` | **15 mm** | The tightest tolerance whose success band the 0.25 N force grid can resolve. |
| `epsilon_v` | **0.08 m/s** | Tighter values drop coverage sharply: a low-resistance drawer physically cannot stop faster. |
| `F_peak` range | **1.0 - 5.0 N** | The union of all per-hidden-state success bands. |
| execution `fall_fraction` | **0.20** | Section 3. |

### Standardised probe

| Parameter | Value | How it was chosen |
|---|---|---|
| `initial_force` | **1.0 N** | Below the weakest breakaway in the grid, so the ramp starts under every drawer. |
| `max_force` | **6.0 N** | Above the strongest breakaway, so the ramp brackets the whole grid. |
| `target_displacement` | **3 mm** | 6.9 % of the goal: the least intrusive of seven candidates whose predictive power was statistically tied. |
| `max_velocity` | **0.08 m/s** | Never the binding stop condition on this grid; present as a guard. |
| `ramp_duration` | **1.0 s** | Reaches `max_force` inside the budget; slower and faster ramps scored the same. |
| `max_probe_duration` | **1.5 s** | Backstop. All 108 hidden states stop on displacement, median 0.467 s. |

Measured: coverage 1.00 (every drawer breaks away), all 108 terminate on
`displacement_reached`, best feature correlates with the required force at |rho| = 0.969.

### Hidden-state ranges

| Dimension | Training | Out of distribution |
|---|---|---|
| `drawer_mass` m | 4.0 - 12.0 kg | 2.0 - 18.0 kg |
| `joint_static_friction` mu_s | 0.5 - 3.0 N | 0.25 - 4.5 N |
| `mu_d / mu_s` ratio | 0.3 - 1.0 | 0.15 - 1.0 |
| `joint_damping` b | 2.0 - 10.0 N s/m | 1.0 - 16.0 N s/m |

The training bounds are the swept grid's. The friction ceiling is set by the *operating
point*, not the simulator: only 40-80 % of a commanded force reaches the drawer
(`docs/FORCE_CHANNEL_AUDIT.md`), so a static friction much above 3 N cannot be broken away
inside the task's force range.

The OOD ranges extend every axis by one step while staying inside the physically valid
region -- the coarse sweep reached 12 N and 3 s without the simulation misbehaving, so these
extrapolate the *task*, not the simulator. `mu_d <= mu_s` is preserved by construction
because the ratio is what is sampled (`docs/DECISIONS.md` D016).

---

## 5. On the role of T

Fixed at 1.5 s for the first paper. The question the paper poses is then exactly:

> Given one standardised probe, predict the peak force that brings the drawer to
> `d_goal = 50 mm` within `T = 1.5 s` and leaves it at rest there.

The sweep gives no reason to make `T` a task condition: both usable durations produce
accepted tasks with almost the same discrimination, so varying `T` would add a dimension
without adding a phenomenon. Adding it later needs no API change -- `run(peak_force,
duration)` already takes it.

---

## 6. Phase 10 — the parameters re-selected against the sequential Oracle

Everything in §4 was selected against the reset Oracle. It is kept for the comparison in
`docs/SEQUENTIAL_PROTOCOL.md` §5 and is **not** the paper's parameter set. The values below
are, and they come from `scripts/refine_task_space.py` scoring
`outputs/logs/sequential_oracle_fall{020,030,035}.json` (report
`outputs/logs/task_refinement.json`, 300 candidates over 3 datasets).

### The selection rule, in priority order

The user's priority order, encoded in `_select`:

1. the task can truly stop (`ε_v` satisfiable at all)
2. position precision (`ε_d` as small as possible)
3. coverage (fraction of hidden states with a succeeding force)
4. the force differences between hidden states are real
5. the success band is wide enough to aim at
6. **only then** discrimination

Discrimination is the last tie-breaker, not the objective. A task every drawer solves at the
same force would be well-behaved and worthless; a task no regression can hit is worse.

### Selected

| Parameter | Value | Phase 9 value |
|---|---|---|
| `T_goal` | 1.5 s | 1.5 s (unchanged) |
| `d_goal` | **40 mm** | 50 mm |
| `ε_d` | **7.5 mm** | 15 mm |
| `ε_v` | **0.03 m/s** | 0.08 m/s |
| `fall_fraction` | **0.35** | 0.20 |
| inference gap | **8 steps** (133 ms) | n/a (reset) |
| `F_peak` envelope | 0.15 – 4.50 N | 1.0 – 5.0 N |

Position tolerance halved and terminal-velocity tolerance cut by 2.7×, at a *higher* coverage
than Phase 9 achieved with the looser task.

### Measured at this operating point (108-point grid)

| Quantity | Value |
|---|---|
| coverage | **0.972** (105/108) |
| required force, closest to goal | 0.20 – 4.30 N, median 1.50 N — a **21.5×** range |
| required force, band centre | 0.25 – 4.30 N, median 1.50 N — a 17.2× range |
| median success band | 0.20 N (0.14 of the required force) |
| bands contiguous | 100 of 105 (0.952) |
| grid step | 0.05 N, resolves the band |
| max travel used | 0.119 of the drawer's range |
| discrimination | 2.70 |
| valid rows | 97.2 % of 5616 |

Hidden states with no succeeding force: `[4.0, 2.0, 0.6, 2.0]`, `[4.0, 3.0, 0.9, 2.0]`,
`[4.0, 3.0, 0.9, 6.0]` — light, high-friction drawers, which overshoot as soon as they break
away at all.

### Why not `ε_d = 5 mm`

Coverage is *higher* at 5 mm (0.981), so coverage is not the constraint. The band is: it
collapses to 0.10 N — one grid step, about 7 % of the required force — which fails the
project's own `min_relative_width = 0.10` floor. 7.5 mm is also about 7× the protocol's
intrinsic `d_total(T)` noise of roughly 1 mm, against 4.5× at 5 mm (D033, D028).

### Why `fall_fraction = 0.35` and not 0.20

Re-compared over 0.20 / 0.30 / 0.35 as instructed. At the selected goal (figure D), coverage
against `ε_d` at 7.5 mm is 0.71 at fall = 0.20 and 0.98 at both 0.30 and 0.35; against `ε_v`
at 0.03 m/s it is 0.53 against 0.95–0.96. The mechanism is the terminal-velocity condition: a
short ramp-down leaves a low-resistance drawer no time to decelerate before `T`.

Between 0.30 and 0.35 the curves nearly coincide; jointly scored at the selected point,
coverage is 0.972 at 0.35 against 0.954 at 0.30, so 0.35 was taken. 0.20 was **not** taken
despite its marginally higher discrimination in the Phase 9 sweep, per the priority order
above.

### The coverage ceiling, and how it was raised

The first sequential pass reached coverage 0.93 and stalled. The cause was the force grid, not
the physics: 10 of the 13 failing hidden states already **overshot** at the grid's lowest
force of 1.0 N. Supplementing 0.4–0.9 N and 0.15–0.35 N and merging on exact force equality
raised coverage to 0.972. This is what `analysis.sweep.force_grid` exists for — a grid built
by repeated addition drifts and the passes would not merge.

### What still bounds the friction range

Unchanged from Phase 9: only about 40–80 % of a commanded force reaches the drawer, so a
static friction much above 3 N cannot be broken away inside the task's force range. The
training range's upper friction bound is set by the operating point, not by the simulator.

# Oracle success landscape

**What this answers.** Before any model is trained, two things must be true of the physics,
and neither can be assumed:

1. **the task must be achievable** -- for most hidden states there is a peak force that
   lands the drawer on the goal and leaves it at rest, and the band of such forces is wide
   enough that a predictor has some tolerance;
2. **adaptation must be necessary** -- those bands must sit at *different* forces for
   different hidden states, or a constant would solve the task and a probe would be
   pointless.

Both hold. This document records the measurement.

`S_oracle(F_peak, T | xi, d_goal)` here is not a model. It is the label the simulation
gives: run the execution, apply the success definition, record the answer.

Produced by `python scripts/build_oracle_landscape.py`. Report:
`outputs/logs/oracle_landscape.json`. Data: five 4536-row sweeps. 2026-09-02.

---

## Success definition

```
success  =  |d(T) - d_goal| <= epsilon_d
        AND |v(T)|          <= epsilon_v
        AND the operating point is valid
```

The terminal-velocity term is not decoration. A drawer that arrives at the goal at 0.2 m/s
has not been placed there; a moment later it is somewhere else, or against its end stop.
Requiring `|v(T)|` to be small is what makes "reached the goal" mean "came to rest at the
goal" (`docs/DECISIONS.md` D020). Validity subsumes the safety check: a safety-aborted
episode is never a valid operating point.

Implemented in `probe_drawer.evaluation.task_evaluator`, deliberately outside the execution
controller, which never learns what the goal was (D004).

---

## How a task definition is judged

840 candidates were scored -- five ramp-down fractions x two durations x seven goals x four
position tolerances x three velocity tolerances. **22 were accepted.**

| Condition | Threshold | Why |
|---|---|---|
| coverage | >= 0.80 | The task must be achievable for most of the training distribution. |
| discrimination = `(max - min) / median` of the required force | >= 0.50 | The point of the study. If the force barely varies, a constant would do. |
| median band width, relative | >= 0.10 | A predictor outputs a number; a 2 % tolerance is a knife edge. |
| median band width, relative | <= 0.60 | A band covering the force axis means adaptation buys nothing. |
| max travel fraction of any success | <= 0.70 | The goal must not sit next to the mechanical end stop. |
| contiguity | >= 0.95 | A band with holes in it is not a band. |
| `epsilon_d / d_goal` | <= 0.30 | A 20 mm goal with a 15 mm tolerance is not a positioning task. |
| band width vs force grid | >= 1.5 grid steps | Not a property of the task: when this fails the *sweep* is too coarse. Reported separately so the remedy is a finer grid, not a different task. |

Authoritative values: `probe_drawer.analysis.oracle.AcceptanceThresholds`. Among accepted
candidates the recommendation maximises **discrimination**, because that is the property the
research question depends on.

A note on the first attempt: with a 0.5 N force grid only *one* grid force succeeded per
hidden state, and an earlier acceptance rule that demanded at least two rejected almost
everything. The band was real; the grid could not see it. Measured `dd/dF` is 20.3 mm/N at
`T = 1.0` and 44.1 mm/N at `T = 1.5`, so a 10 mm tolerance corresponds to a 0.5 N band at
`T = 1.0` and 0.23 N at `T = 1.5`. Refining the grid to 0.25 N resolved it. The rule was
changed to measure the band in force units and to flag grid resolution separately -- a
correction of the criterion, not of the bar.

---

## Result

**Recommended: `fall_fraction = 0.20`, `T = 1.5 s`, `d_goal = 50 mm`,
`epsilon_d = 15 mm`, `epsilon_v = 0.08 m/s`.**

| Measure | Value |
|---|---|
| Hidden states with a succeeding force | **106 / 108** (coverage 0.981) |
| Required peak force, best per state | **1.00 - 4.50 N**, median 2.25 N -- a **4.5x** range |
| Band centre range | 1.00 - 4.62 N -- a 4.62x range (this is `discrimination` = 1.568) |
| Median band width | **0.50 N**, i.e. 0.164 relative |
| Overall band envelope | 1.00 - 5.00 N |
| Contiguous bands | **105 / 106** |
| Largest travel fraction of any success | **0.162** -- far from the end stop |
| Force grid | 0.25 N, which resolves the 0.50 N band |

Best accepted candidate per ramp-down fraction, all at `T = 1.5 s` and
`epsilon_v = 0.08 m/s`:

| `fall_fraction` | `d_goal` | coverage | discrimination |
|---|---|---|---|
| 0.10 | 50 mm | 0.95 | 1.564 |
| 0.15 | 50 mm | 0.96 | 1.557 |
| **0.20** | **50 mm** | **0.98** | **1.568** |
| 0.30 | 60 mm | 0.98 | 1.549 |
| 0.35 | 50 mm | 1.00 | 1.531 |

The five are close, which is itself informative: the ramp-down changes *what distance* is
reachable at rest much more than it changes how discriminating the task is.

### Representative success bands

| m (kg) | mu_s (N) | mu_d (N) | b (N s/m) | F_low | F_high | F_best | d(T) at best | v(T) at best |
|---|---|---|---|---|---|---|---|---|
| 4.0 | 0.50 | 0.15 | 2.0 | 1.00 | 1.00 | 1.00 | 47.6 mm | 0.059 m/s |
| 8.0 | 0.50 | 0.33 | 2.0 | 1.25 | 1.50 | 1.50 | 57.6 mm | 0.067 m/s |
| 4.0 | 1.25 | 0.38 | 10.0 | 1.50 | 1.75 | 1.50 | 44.2 mm | 0.047 m/s |
| 12.0 | 0.50 | 0.33 | 10.0 | 1.50 | 2.00 | 1.75 | 48.2 mm | 0.053 m/s |
| 8.0 | 2.00 | 0.60 | 2.0 | 2.00 | 2.00 | 2.00 | 50.6 mm | 0.066 m/s |
| 8.0 | 1.25 | 0.81 | 10.0 | 2.00 | 2.50 | 2.25 | 53.7 mm | 0.055 m/s |
| 4.0 | 2.00 | 1.30 | 10.0 | 2.25 | 2.75 | 2.50 | 49.6 mm | 0.047 m/s |
| 8.0 | 3.00 | 0.90 | 6.0 | 2.75 | 2.75 | 2.75 | 50.1 mm | 0.066 m/s |
| 4.0 | 2.00 | 2.00 | 10.0 | 3.00 | 3.25 | 3.25 | 53.5 mm | 0.044 m/s |
| 8.0 | 3.00 | 1.95 | 2.0 | 3.25 | 3.50 | 3.25 | 54.9 mm | 0.059 m/s |

Figures: `outputs/plots/oracle_success_landscape.png` (the labels themselves) and
`outputs/plots/oracle_force_intervals.png` (the bands, ordered by required force). The
second is the figure that makes the case: the bands march steadily from 1.0 N to 4.5 N and
never merge into one.

### The two hidden states with no succeeding force

`(m = 4.0, mu_s = 3.0, mu_d = 0.9, b = 2.0)` and `(m = 12.0, mu_s = 3.0, mu_d = 0.9, b = 2.0)`.

Both have high static friction with a large drop to dynamic friction and almost no damping:
they need a large force to break away and then have nothing to slow them, so they overshoot
50 mm on the same force that started them. That is a real physical corner of the range, not
a bug, and it is a reason to keep `mu_d / mu_s` bounded below at 0.3 rather than allowing
arbitrarily slippery-once-moving drawers.

---

## What this licenses, and what it does not

Licensed: proceeding to dataset generation and then to a predictor. The physics question --
does one force serve every drawer? -- is answered no, with a 4.5x spread, and the bands are
wide enough (0.50 N, 16 % relative) to be a regression target.

Not licensed by this document:

* **Damping identifiability.** The calibrated probe barely responds to `b`
  (`docs/HIDDEN_STATE_AUDIT.md`). The required force also depends only weakly on `b`, so the
  task remains predictable, but a claim that the probe identifies all four dimensions would
  be false.
* **A tighter `epsilon_d`.** 15 mm is 30 % of the goal, which is the loosest this project's
  own acceptance rule permits. It is set by the force-grid resolution, not by physics:
  `dd/dF` = 44 mm/N at `T = 1.5`, so a 0.125 N grid would support roughly 7.5 mm. Refining
  the grid is the next improvement if a tighter task is wanted.
* **Sequential probe-then-execute.** Every episode here resets between the probe and the
  execution. The probe travels 3.4 mm, 6.9 % of the goal, so a sequential protocol is
  plausible -- but it has not been measured, and `d_goal` would have to be interpreted
  relative to the post-probe position.

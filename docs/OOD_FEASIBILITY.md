# Is the out-of-distribution range a solvable test domain?

A pilot, run before any model is asked to generalise. If a large share of `OOD_XI_RANGES` had
no succeeding force, a low OOD score would be measuring the *task* and no model work would
move it.

Sweep: `scripts/sweep_ood_feasibility.py`. Report: `scripts/analyze_ood_feasibility.py`.
Data: `outputs/logs/ood_feasibility.json`, summary `ood_feasibility_summary.json`.
Setting: frozen, [PROBE_V1.md](PROBE_V1.md) — probe 3.5 N / 0.3 s, `d_goal` 100 mm, `T` 1.5 s.

**Nothing here evaluates a model, and nothing here changes the OOD range.**

---

## 1. What "out of distribution" was made to mean

The OOD box *contains* the training box, so roughly 13 % of its volume is in-distribution — and
those are the easy states. Counting them as OOD would flatter any result measured on the mixture.
`sample_ood_hidden_states` draws a Sobol sequence over the OOD box and **keeps only states with
at least one axis outside the training range**, by rejection rather than by construction, so
*which* axis is novel stays varied rather than fixed.

| axis | training | OOD | can be novel |
|---|---|---|---|
| `mass` | 4–12 kg | 2–18 | both ends |
| `static_friction` | 0.5–3.0 N | 0.25–4.5 | both ends |
| `dynamic_friction_ratio` | 0.3–1.0 | 0.15–1.0 | **low only** — the ceiling is shared |
| `damping` | 2–10 | 1–16 | both ends |

64 states drawn, all 64 verified novel, spread over all seven possible directions, 35 of them
novel on more than one axis.

## 2. Is it solvable? Yes.

Swept `F_peak` from 0.25 to 10.05 N at 0.10 N — **deliberately past the task's own 6.5 N
ceiling**, because "no force reaches the goal" and "no force *the task allows* reaches the goal"
are different findings with different remedies.

| | |
|---|---|
| solvable within the task's range, 0.5–6.5 N | **61 / 64 = 95.3 %** |
| solvable at some force, any magnitude | **64 / 64 = 100 %** |
| solvable only above the ceiling | 3 |
| **unsolvable at any swept force** | **0** |

Not one of the 64 states is a drawer this rig cannot open. Every failure is a *range* failure.

**Required force: 1.05–6.85 N, median 3.75** (in-distribution: 0.64–5.51, median 2.91). Shifted
up by about 0.8 N, as a wider friction range should. **Band width 0.10–1.00 N, median 0.40** —
wider than the in-distribution 0.30 N, so the OOD targets are if anything easier to hit once the
right force is known.

## 3. Is the `F_peak` range truncating it? Marginally, yes.

Three states are solvable only above 6.5 N, and they need **6.55–6.85 N** — between 0.05 and
0.35 N past the ceiling. Three further states are solvable *at* the ceiling. None sits at the
floor.

So the ceiling is the binding constraint for 4.7 % of the OOD domain, and it binds by a very
small margin. Ten of 64 states need more than the in-distribution maximum of 5.51 N, which is
the expected consequence of a friction range that reaches 4.5 N.

## 4. Where the failures are

Failure rate per novel axis, against how often each axis appears:

| novel axis | states | failed | rate |
|---|---|---|---|
| `static_friction_high` | 27 | 3 | **11.1 %** |
| `mass_high` | 28 | 2 | 7.1 % |
| `damping_high` | 30 | 1 | 3.3 % |
| `dynamic_friction_ratio_low` | 13 | 0 | 0 % |
| `static_friction_low` | 4 | 0 | 0 % |
| `mass_low` | 9 | 0 | 0 % |
| `damping_low` | 4 | 0 | 0 % |

All three failures carry `static_friction_high`, with `µ_s` of 3.62, 4.12 and 4.48 against a
training maximum of 3.0. Two also carry `mass_high`. Nothing fails on the low side of any axis,
and nothing fails on the ratio axis.

| `m` | `µ_s` | ratio | `b` | novel | probe moved | closest | needs |
|---|---|---|---|---|---|---|---|
| 10.37 | 4.48 | 0.93 | 4.21 | µ_s high | **no** | 1.9 mm at 7.05 N | 6.85 N |
| 16.22 | 3.62 | 0.97 | 6.78 | m, µ_s high | **no** | 0.3 mm at 7.05 N | 6.85 N |
| 12.87 | 4.12 | 0.76 | 13.38 | m, µ_s, b high | **no** | 1.6 mm at 6.75 N | 6.55 N |

## 5. The finding that matters more than the headline

**The frozen probe does not break away 17 of the 64 OOD states (26.6 %).** Every one has
`µ_s ≥ 3.32`, above the training maximum. Their probe displacement is **0.22–0.83 mm** against a
6.7 mm in-distribution median — the probe barely moves them, so it returns almost no information
about them.

And yet **14 of those 17 are solvable**, needing 3.95–6.85 N. So for a quarter of this domain the
task is achievable while the probe is nearly silent.

The probe calibration enforced a "responsive" gate — every hidden state must break away — and it
held for all 24 in-distribution states it was selected on ([D044](DECISIONS.md#d044)). Out of
distribution that gate fails for a quarter of the range, because breakaway is a force threshold
and 3.5 N commanded delivers roughly 2.1–2.8 N to the drawer, below a `µ_s` of 3.3 and above.

This matters for how an OOD result should be read: a low OOD score would conflate *a hard task*
with *an uninformative probe*, and on this evidence a quarter of the domain is the second. Any
OOD evaluation should report the breakaway fraction beside its score.

Median OOD probe displacement is 3.4 mm against 6.7 mm in-distribution, and the range widens to
0.22–19.9 mm — both tails are further out than anything the probe was calibrated against.

## 6. Safety

**2 safety aborts in 6 336 state-force evaluations** (64 states × 99 forces, run as 198
parallel executions), across 2 states. The median state is invalid for 41.4 %
of the *swept* forces, but the grid runs to 10 N on purpose; the dominant reasons are
`excessive_velocity` (1 629), lateral drift (968) and orientation drift (805), all concentrated
at forces far above what any state needs. Setting V1's own range is much narrower.

## 7. Recommendation

**Keep `OOD_XI_RANGES` as it is.** It is a reasonable test domain on the evidence: entirely
solvable in physics, 95.3 % solvable inside the frozen action range, with required forces and
band widths that are shifted but not degenerate.

Two things to record rather than fix, both of them decisions for a person:

1. The 6.5 N ceiling truncates 4.7 % of the domain, by 0.05–0.35 N. Whether to report OOD on the
   61 in-range states or to widen `peak_force_range` is a framing choice; widening it would
   change Setting V1, which is frozen.
2. The probe is under-powered for `µ_s > 3.3`, which is 27 % of this domain. This is a **probe**
   limitation, not a range problem, and the probe is frozen. It should be reported alongside any
   OOD number rather than worked around.

## 8. Reproducing

```bash
python scripts/sweep_ood_feasibility.py --headless --num-xi 64 --num_envs 32
python scripts/analyze_ood_feasibility.py
```

# Is the task condition worth conditioning on?

An Oracle audit, run before any multi-goal model is trained. Setting V1 feeds the model
`(d_goal, T_goal)` but holds both constant across Dataset v1, so they carry no information yet
([D044](DECISIONS.md#d044)). The question is whether a multi-goal experiment would measure
anything.

Script: `scripts/audit_task_conditioning.py`. Analysis: `probe_drawer.analysis.task_conditioning`.
Data: `outputs/logs/task_conditioning_audit.json`. Nothing trained, no dataset written, the
probe / controllers / `T_goal` / hidden-state ranges all frozen.

---

## 1. One sweep, three readings

**Neither controller reads the goal.** `ExecutionPullController.run` takes a force and a
duration ([D004](DECISIONS.md#d004)); the fixed-budget probe takes an amplitude and a budget
(D044); and validity bounds travel and drift, not the target. So one force sweep produces the
episodes for *every* goal, and the goals differ only in how those episodes are scored.

The three numbers below are therefore three views of **identical physics** — cheaper than three
sweeps, and free of any between-run confound.

32 representative in-distribution hidden states, one frozen probe each, `F_peak` swept
0.25–9.05 N at 0.10 N (89 values, deliberately past the 6.5 N action ceiling), `T_goal` = 1.5 s,
`ε_d` = 7.5 mm.

## 2. Results

| goal | solvable | band width (N) | required force (N) | disconnected | at action ceiling |
|---|---|---|---|---|---|
| 80 mm | **32/32** | 0.10–0.50, med **0.35** | 0.55–5.15, med **2.20** | 3 | 0 |
| 100 mm | 31/32 | 0.20–0.50, med **0.40** | 0.75–5.75, med **2.65** | 3 | 0 |
| 120 mm | **32/32** | 0.10–0.50, med **0.30** | 1.05–6.35, med **3.20** | 2 | 0 |

1. **Solvable fraction** — essentially complete at all three goals. The single miss at 100 mm is
   a grid artefact on an unusually sharp state (`m` 4.40, `µ_s` 2.88, `b` 2.40) whose bands at 80
   and 120 mm are one 0.10 N cell wide; at 100 mm its band fell between grid points.
2. **Band width** — 0.30–0.40 N median, near-identical across goals. The goal does not make the
   task harder to hit, it moves where to hit.
3. **Required force** — rises clearly and monotonically: median 2.20 → 2.65 → 3.20 N, and the
   whole distribution shifts with it (min 0.55 → 0.75 → 1.05, max 5.15 → 5.75 → 6.35). Nothing
   reaches the 6.5 N ceiling, so the frozen action range covers 80–120 mm.

## 3. How far the optimum moves

| step | median \|ΔF*\| | signed mean | max | **shift / band width** |
|---|---|---|---|---|
| 80 → 100 mm | **0.500 N** | +0.494 N | 0.750 N | **1.29×** |
| 100 → 120 mm | **0.500 N** | +0.481 N | 0.600 N | **1.29×** |

Per state, `dF*/dd_goal` = 15.0–31.2 N/m, median **23.8**, mean 24.4 ± 3.9 — so **0.475 N per
20 mm**, against a band 0.35–0.40 N wide. The mapping is close to affine in the goal and the
per-state variation in slope is modest.

## 4. Does the 100 mm optimum transfer? No — 0 of 31, twice

| deployed at | reached | success |
|---|---|---|
| 80 mm | 0 / 31 | **0.0 %** |
| 100 mm | 31 / 31 | 100.0 % |
| 120 mm | 0 / 31 | **0.0 %** |

Not marginal. Using the 100 mm optimum lands at **+19.8 mm median error at 80 mm** (range +16.3
to +24.5) and **−20.2 mm at 120 mm** (−23.7 to −15.5), against a ±7.5 mm tolerance — the error
exceeds `ε_d` for **31 of 31** states in both directions.

The number is close to the goal change itself, and that is the clearest way to say it: a model
that ignores the goal **lands at the distance it was trained on**, whatever it was asked for.

## 5. The mechanism, and why the conclusion is robust

For a locally affine response `d = d₀ + (F − c)/k`, the success band is `2·ε_d·k` wide and the
optimum moves `Δd_goal·k` between goals. Both scale with `k`, so the ratio is

```
shift / band  =  Δd_goal / (2·ε_d)  =  20 mm / 15 mm  =  1.33
```

**independent of the drawer's dynamics.** The measured 1.29× is that arithmetic. So the transfer
failure is not a property of these drawers — it follows from the goal spacing exceeding the
tolerance window, and no drawer stiffness could rescue it. Equally, goals spaced *inside* the
window would transfer for free; the audit's unit tests pin both directions.

## 6. Verdict (superseded by §7 — read both)

> This section's conclusion was reversed by the affine-baseline pilot in §7. It is kept
> because the measurement behind it stands; only the recommendation changed.


**Yes — the task condition genuinely changes the action mapping, and by enough to be worth a
multi-goal experiment.**

* The shift (0.5 N per 20 mm) is **1.29× the band it has to leave**, so a single-goal force is
  not merely suboptimal at a new goal, it fails.
* Transfer is **0 %** in both directions across 31 states — as unambiguous as this measurement
  gets.
* All three goals are essentially fully solvable inside the frozen action range, with band
  widths that do not degrade, so a multi-goal study would be measuring adaptation rather than
  feasibility.
* The mapping is close to affine with a modest spread in slope (24 ± 4 N/m), so the structure a
  multi-goal model would have to learn is simple — which makes the experiment a fair test of
  conditioning rather than a capacity contest.

One caveat worth carrying: because the relationship is near-affine and `d_goal` enters almost
linearly, a multi-goal model could do well by learning little more than that slope. A
multi-goal experiment should therefore report the comparison against an explicit *affine-in-goal*
baseline, or it risks crediting conditioning for arithmetic.

## 7. The follow-up: one global slope is already enough

§6's caveat has now been tested, and it decides the question the other way round.

A **fresh, independent** 64-state plain-Sobol draw over the in-distribution ranges (seed
20260904, `--sampler sobol`) reproduced §2–§4 exactly: 64/64 solvable at all three goals, median
band 0.40 N throughout, `|ΔF*|` 0.500 N median per 20 mm at **1.29×** the band, and transfer of
the 100 mm optimum **0/64** in both directions. So the earlier 32-state result was not a draw
artefact.

Then the affine baseline. The 64 states were split **32 calibration / 32 held out** by a
content-addressed permutation fixed before any physics was read; the slope was fitted on the
calibration half only, and applied as `F(g) = F_100 + k_global·(g − 0.10)`.

Two allowances **in the baseline's favour**, because the question is whether a global slope
*suffices* rather than whether it is deployable: it is handed each held-out drawer's **correct**
`F_100` from that drawer's own Oracle band centre, and the slope is fitted on the same
distribution it is tested on. These numbers are an upper bound.

**`k_global` = 24.53 N/m** — 0.491 N per 20 mm — from 32/32 calibration drawers whose own slopes
span 18.8–30.0 (sd 3.0).

| goal | affine reach | Oracle | gap | median \|d−goal\| | signed median |
|---|---|---|---|---|---|
| 80 mm | **32/32 = 100.0 %** | 100.0 % | **+0.0 pp** | 2.19 mm | +0.20 mm |
| 100 mm | 32/32 = 100.0 % | 100.0 % | +0.0 pp | 0.77 mm | −0.36 mm |
| 120 mm | **31/32 = 96.9 %** | 100.0 % | **+3.1 pp** | 2.21 mm | −0.34 mm |

Held-out per-drawer slopes run **15.0–28.8 N/m** (median 23.8, sd 3.1), so `|k_i − k_global|` is
**2.97 N/m median (12.1 % relative), 9.53 N/m worst (38.9 %)**.

### Why the variation does not matter

Hidden dynamics **do** change the slope — a 1.9× spread across drawers — and the reason it is
absorbed is arithmetic again:

| | |
|---|---|
| correction applied over 20 mm | 0.491 N |
| force error from a median slope error | **0.059 N** — 30 % of the half-band |
| force error from the worst slope error | 0.191 N — **95 %** of the half-band |
| band half-width | 0.200 N |

So the median drawer's slope error is a third of the way to the band edge and the single worst
drawer sits right on it. That drawer is the only failure: `m` 4.96, `µ_s` 2.87, `b` 4.13, whose
true slope of **15.0 N/m is the population minimum**, 39 % below `k_global`. The affine rule
asks it for 4.05 N where the Oracle wants 3.75 N, and it overshoots to 131.5 mm (+11.5 mm).

### Verdict, reversing §6

**Hidden dynamics measurably change the goal→force mapping, and a single global slope is
nonetheless sufficient.** Both halves of that are true and neither cancels the other: the slope
is genuinely drawer-dependent (15–29 N/m) but the induced force error is small against the
success band, so one constant recovers 100 / 100 / 96.9 % of the Oracle.

**A multi-goal task-conditioned experiment is therefore not worth running as posed.** It would
be competing against a one-parameter rule that already loses only 3.1 pp at the hardest goal,
and any model beating that would be doing so on a single outlier drawer. §6's recommendation
stands but its conclusion inverts: the affine-in-goal baseline is not merely a control to report
*beside* a multi-goal result — measured, it removes the reason to produce one.

What would change the answer: a goal range wide enough that the mapping stops being affine
(the goal-distance study found the terminal-velocity term failing from ~100 mm and the drawer's
own stop from ~350 mm, so 200–400 mm is where curvature would live), or a tighter `ε_d` that
stops absorbing a 12 % slope error. Both change Setting V1, which is frozen.

## 8. Reproducing

```bash
# the 32-state audit (sections 2-6)
python scripts/audit_task_conditioning.py --headless

# the 64-state fresh draw and the affine baseline (section 7)
python scripts/audit_task_conditioning.py --headless --num-xi 64 --num_envs 32 \
    --sampler sobol --seed 20260904 --output outputs/logs/affine_goal_sweep.json
python scripts/audit_affine_goal_baseline.py
```

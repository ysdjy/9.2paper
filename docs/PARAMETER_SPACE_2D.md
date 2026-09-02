# The two-dimensional execution parameter space

What `p = [F_peak, T]` means, where it is physically usable, and what bounds it.

Sweeps: `scripts/sweep_parameter_space_2d.py`. Analysis: `scripts/analyze_landscape_2d.py`.
Data: `outputs/logs/landscape_2d_{coarse,fine}.json`. Shape findings: `docs/LANDSCAPE_2D.md`.
Decisions: `docs/DECISIONS.md` D041.

---

## 1. What changed, and what deliberately did not

`T` moves out of the task definition and becomes the second predicted parameter. That is the
only change. In particular:

| Unchanged | Why it matters |
|---|---|
| the normalised profile `φ(τ)`, `τ = t/T` | a longer `T` stretches the identical curve; it does not reshape it |
| `rise_fraction = 0.1`, `fall_fraction = 0.35` | scanning these too would make the space three-dimensional |
| `d_goal = 40 mm`, `ε_d = 7.5 mm`, `ε_v = 0.03 m/s` | one variable at a time; the task stays the Phase 11 task |
| `ξ = [m, µ_s, µ_d, b]` | no fifth hidden dimension |
| the probe, the 8-step gap, the sequential protocol | the baseline this phase measures against |
| `ExecutionPullController.run(peak_force, duration)` | the API already took both; no controller change was needed |

That `φ(τ)` is genuinely `T`-independent is asserted rather than assumed:
`tests/integration/test_branching.py::test_the_normalised_profile_is_the_same_curve_at_every_duration`
checks each execution's commanded force against the analytic `φ` at its own sample times, to
`1e-4`.

The check found something worth recording. Comparing a 0.8 s and a 2.0 s execution at matching
normalised times showed a 0.475 N disagreement between what should be one curve. The cause is
a recording convention: `history.time[k]` is the time *after* step `k` while
`commanded_force[k]` was computed from the time before it. That pairing is right for a
model — that force is what produced that position — but it means the force at index `k` was
issued at `time[k] − step_dt`, and comparing against `time[k]` shifts the curve by
`step_dt/T`: 2.6 % of the profile at `T = 0.8 s` against 0.8 % at `T = 2.0 s`. On the steep
rise that accounts for the entire discrepancy, to three decimals. The physics was never
wrong.

## 2. Why `T` is a real axis and not a formality

The task is `|d(T) − d_goal| ≤ ε_d` **and** `|v(T)| ≤ ε_v`, so opening `T` opens a genuine
trade-off rather than a rescaling. Figure E shows it directly:

* the `d = d_goal` contour is a **hyperbola-like curve** — low force needs long `T`, high
  force needs short `T`;
* the `|v| = ε_v` contour is a **near-vertical wall** that truncates it on the fast side.

The success region is their intersection: a thin curved band hugging the `d_goal` contour,
cut off where the drawer would still be moving at `T`.

## 3. The valid operating region, in two dimensions

Coarse sweep: 48 representative hidden states, `F ∈ [0.15, 5.90] N` at 0.25 N, `T ∈ [0.40,
2.50] s` at 0.15 s, 17 280 episodes, **88.8 % valid**.

Invalid reasons, and where each lives:

| reason | episodes | where |
|---|---|---|
| excessive lateral drift | 1 795 | long `T` and high `F` — the drawer runs far and the held axes drift |
| excessive velocity | 1 665 | high `F` — the drawer is still moving at `T` |
| excessive orientation drift | 1 506 | with the lateral drift, same cause |
| mechanical limit | 1 079 | high `F` and long `T` — the drawer reaches its end stop |
| safety abort | 148 | the extreme high-force corner |

Validity along each axis, marginalised over hidden states:

| `F` (N) | 0.15–1.15 | 1.40–2.40 | 2.65–3.65 | 3.90–4.65 | 4.90–5.90 |
|---|---|---|---|---|---|
| valid | 100 % | 97–100 % | 89–96 % | 79–86 % | 61–74 % |

| `T` (s) | 0.40–1.00 | 1.15–1.45 | 1.60–1.90 | 2.05–2.50 |
|---|---|---|---|---|
| valid | ~100 % | 95–100 % | 81–92 % | 63–77 % |

**Two hard physical bounds fall out, and they are the reason `T` matters.**

**Short `T` is forbidden by the terminal-velocity condition.** Success is 0.0 % for
`T ≤ 0.70 s` and 0.1 % at 0.85 s, despite validity being 100 % there. Reaching 40 mm that
quickly leaves the drawer moving faster than `ε_v = 0.03 m/s` allows — there is no force that
both travels far enough and stops. This is not a limit of the rig; it is the task.

**Long `T` is bounded by held-axis drift.** Validity falls monotonically from 100 % at
`T ≤ 1.0 s` to 62.6 % at 2.5 s as the drawer travels further and the five pose-held degrees of
freedom accumulate error. That is the known Phase 8 limitation, now quantified on the `T`
axis.

## 4. The formal candidate box

Kept where *some* hidden state succeeds — trimming on validity alone would keep large regions
that are physically fine and useless (0.15 N is perfectly valid and never reaches the goal),
and trimming on a high success rate would discard the forces only the stiffest drawers need,
which are exactly the discriminative ones.

```
F_peak ∈ [0.30, 4.20] N
T      ∈ [1.00, 2.50] s
```

Dropped by the rule: `F ≤ 0.15` and `F ≥ 4.15` (0 % success above 4.40 N in the coarse
sweep), `T ≤ 1.00` (0 % success). The box is then extended slightly downward in `F` to 0.30 N
and its `T` floor kept at 1.00 s so the fine sweep straddles the boundary rather than starting
inside it.

Fine sweep: `F` at **0.05 N** (79 values, matching the resolution the 1-D Oracle resolved the
0.20 N success band at) and `T` at **0.10 s** (16 values), 32 hidden states, 40 448 episodes.

The coarse grid's 0.25 N step is *larger than the entire 1-D success band*, which is why no
claim about the region's shape is made from it — see `docs/LANDSCAPE_2D.md`.

## 5. Where the region sits, and what puts it there

The one thing the coarse grid settles beyond doubt.

| descriptor | `m` | `µ_s` | `µ_d` | `b` |
|---|---|---|---|---|
| centroid `F` | +0.02 | **+0.83** | **+0.98** | +0.05 |
| centroid `T` | +0.04 | −0.08 | +0.17 | −0.23 |
| success area | +0.26 | −0.04 | −0.01 | +0.24 |
| orientation | +0.07 | +0.47 | **+0.57** | +0.32 |

(Spearman, 46 solvable hidden states.)

**Dynamic friction determines where on the force axis a drawer succeeds, at ρ = +0.98.** The
success intervals march monotonically from `F ∈ [0.15, 0.65] N` at `µ_d = 0.21` to
`F ∈ [3.40, 4.40] N` at `µ_d = 2.77`, with only mild overlap between neighbours — figure B's
right panel is that partition drawn.

**Nothing determines where on the `T` axis it succeeds.** Every correlation with the centroid's
duration is under 0.25, and across all 46 states the duration centroid spans only 1.21–1.97 s.

So the hidden state moves the region along `F` and leaves `T` broadly permissive within its
window. That is a finding, and it is one that *weakens* the case for a two-dimensional
landscape rather than strengthening it. It is recorded here for that reason.

## 6. Reproducing

```bash
python scripts/sweep_parameter_space_2d.py --headless --stage coarse --num-xi 48 --num_envs 24
python scripts/analyze_landscape_2d.py --dataset outputs/logs/landscape_2d_coarse.json

python scripts/sweep_parameter_space_2d.py --headless --stage fine --num-xi 32 --num_envs 32 \
    --force-low 0.30 --force-high 4.20 --force-step 0.05 \
    --duration-low 1.00 --duration-high 2.50 --duration-step 0.10
python scripts/analyze_landscape_2d.py --dataset outputs/logs/landscape_2d_fine.json
python scripts/plot_phase12.py --dataset outputs/logs/landscape_2d_fine.json
```

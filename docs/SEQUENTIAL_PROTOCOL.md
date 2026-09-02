# The sequential probe-then-execute protocol

This is the protocol the paper runs. It replaces the Phase 9 arrangement in which the drawer
was reset between the probe and the execution.

Implementation: `src/probe_drawer/protocols/sequential_pull_protocol.py`.
Validation report: `outputs/logs/sequential_protocol_validation.json`.
Decisions: `docs/DECISIONS.md` D026–D030.

---

## 1. Why the reset had to go

Phase 9's Oracle probed the drawer, reset it, and then ran the execution from a closed drawer
at rest. That made the sweep clean and the analysis easy, and it measures the wrong thing.

A real robot does not get a reset. It pulls the handle to feel the drawer, and it is then
holding a drawer that has already moved a few millimetres and is still creeping. Whatever
force it commands next acts on *that* state. Two consequences follow, and both were measured
rather than assumed (§5):

* the probe has already broken static friction, so the drawer starts the execution in the
  sliding regime, and needs **less** force to reach the goal;
* the probe has already covered part of the distance, so part of `d_goal` is already spent.

A reset erases both. Training on reset data would teach a model to predict a force that is
systematically too large for the situation it will actually face.

## 2. The five stages

| Stage | What happens | Recorded as |
|---|---|---|
| `INITIAL` | One reset into the recorded grasp, drawer closed. `x_initial` is read here. | — |
| `PROBE` | The standardised probe: force ramps from 1 N at 5 N/s until the drawer has moved 3 mm (or 1.5 s, 6 N, or 0.08 m/s). | `ProbeResult`, the model's input |
| `PROBE_END` | Nothing. The probe's last commanded force simply stops. | `post_probe_state` |
| transition | A fixed 8 control steps (133 ms) of **zero** pull force. The five non-pull axes are still held. | `InferenceTransition`, in neither history |
| `EXECUTION` | `F(t) = F_peak · φ(t/T)` for the full `T = 1.5 s`. No stopping condition. | `ExecutionResult` |
| `EVALUATE` | `d_total(T)` and `v(T)` against the task. | `ExecutionVerdict` |

There is **exactly one** `reset()` in the whole protocol, at `INITIAL`. The protocol class
contains no physics; it sequences the two existing controllers and reads the existing sensor.

## 3. The three prohibitions, and how each is enforced

> 整个过程中：禁止 reset drawer。禁止恢复 drawer position。禁止恢复 drawer velocity。禁止用
> privileged xi 修正系统状态。

**Nothing is reset after the probe.** `SequentialPullProtocol.run()` calls `system.reset()`
once, before the probe, and never again. Verified by
`tests/integration/test_sequential_protocol.py::test_the_drawer_is_not_reset_after_the_probe`,
which asserts the execution starts at or beyond where the probe left the drawer.

**No velocity is written.** The post-probe velocity decays because the drawer's own friction
and damping decelerate it — the transition commands zero force, it does not brake. The
distinction is load-bearing and it is why the protocol *refuses to run* with an execution
controller whose `settle_steps != 0`: a settle applies a velocity-proportional braking force
of up to 15 N to the pull axis, which would erase exactly the state the probe left behind.
That refusal is a `ValueError` in `__init__`, not a warning.

Measured at the operating point: the drawer retains **under 0.2 %** of its probe-end velocity
after the 8-step gap — against 54 % at 2 steps and 0.1 % at 4. The decay is a physical time
constant, not a threshold, and nothing was written to get there.

**No privileged state is used.** `xi` is applied by the randomiser before the episode and
read back from `root_physx_view` for the record. Nothing in the protocol, the controllers or
the evaluator reads it.

**The goal never reaches the controller.** `ExecutionPullController.run` still takes exactly
`(peak_force, duration)`. Checked by signature inspection in
`test_the_execution_controller_never_sees_the_goal`, and by
`tests/unit/test_execution_has_no_goal_feedback.py`.

## 4. The task reference frame

`d_goal` is measured from `x_initial`, the drawer's position **before** the probe:

```
d_total(T) = x_drawer(T) − x_initial
           = d_probe + d_coast + d_execution(T)
```

So a probe that travels 3.5 mm has already delivered 3.5 mm of a 40 mm goal, and the
execution must supply the remaining 36.5 mm. This is the only reference frame in which the
robot's total behaviour is what is judged; measuring from the post-probe position would let a
model bank an arbitrary amount of free displacement by probing harder.

`SweepRecord.final_displacement` always holds the quantity the task is judged on, for both
protocols, which is what lets one Oracle analysis read both datasets.

`test_ignoring_the_probe_would_change_the_label` demonstrates the frame is not bookkeeping:
the same episode passes with the probe counted and fails without it.

## 5. What the probe actually did to the task

`scripts/compare_reset_vs_sequential.py`, report `outputs/logs/reset_vs_sequential.json`.

Compared on the Phase 9 task (`d_goal = 50 mm`, `ε_d = 15 mm`, `ε_v = 0.08 m/s`, `T = 1.5 s`),
which both force grids can express, over all 108 hidden states:

| Quantity | Reset | Sequential |
|---|---|---|
| coverage | 1.000 | 1.000 |
| median success band | 0.50 N | 0.60 N |
| median required force | — | **0.80×** the reset value |

* The required force falls by a median of **0.45 N**, a factor of **0.80**. Per hidden state
  the ratio ranges from **0.32 to 1.02** — for the drawer most affected, the reset Oracle
  demanded three times the force the sequential protocol needs.
* The largest single shift is **1.40 N**.
* The *ordering* of hidden states survives: the rank correlation between the two protocols'
  required forces is **+0.95**.

Read together: the reset was a good proxy for *which* drawer is stiff and a biased one for
*how much* force it takes, by about 20 % on average and up to 3× in the worst case. That is
far too large to absorb into a tolerance, which is why the sequential Oracle is the
authoritative ground truth (D026) and the reset one is kept only as this comparison.

On the Phase 10 task the reset dataset covers only 0.287, but that number is **not** a
physical result: the Phase 9 grid starts at 1.00 N with 0.25 N spacing and simply cannot
express a 7.5 mm tolerance. That inexpressibility is why the landscape was re-swept.

## 6. Why the gap is 8 steps

A deployed system needs wall-clock time to run its adaptation model, so the gap is reserved
explicitly and identically in every episode rather than left implicit.

Its length was chosen on **repeatability of the finished task**, not on being short. Over six
identical episodes at the operating point, the spread of `d_total(T)` was:

| gap (steps) | 0 | 2 | 4 | **8** | 12 |
|---|---|---|---|---|---|
| `d_total(T)` spread (mm) | 3.58 | 2.61 | 3.29 | **0.90** | 1.40 |
| velocity retained | 1.000 | 0.539 | 0.001 | **0.000** | 0.000 |

The knee at 8 has a mechanism. Just above breakaway `dd/dF` reaches roughly 40 mm/N, so the
few hundred µm/s of residual velocity the probe leaves is amplified into millimetres of
finished displacement. Letting the drawer coast to a near-stop under its own friction removes
the amplification. Beyond 8 steps nothing further is gained.

A second run over 4, 8 and 12 steps gave 1.66, 1.14 and 1.40 mm. The absolute spreads move by
a few tenths of a millimetre between runs — six episodes is a small sample — but 8 steps is the
minimum in both, so the choice does not rest on one measurement.

About **1 mm** is therefore the protocol's intrinsic episode noise, and it is why `ε_d = 7.5 mm`
rather than 5 mm is a defensible tolerance: 7.5 mm is roughly 7× the noise floor, 5 mm only
about 4.5×.

The gap belongs to **neither** history. It is not part of the probe the model sees, and it is
not part of the commanded `T`. `test_the_gap_does_not_appear_in_either_history` asserts both.

## 7. Candidate comparison inside one episode

Environments are the hidden-state axis and the sweep loops over `F_peak`, so every hidden
state receives an identical command at each point. Each `(xi, F_peak)` is a **complete
episode, probe included** — a candidate force is only meaningful together with the probe that
preceded it, and re-running the probe is what makes each row a genuine episode rather than a
shared starting state reused.

When several candidates *are* run in one episode (the fairness test, and any future
comparison), the execution applies a per-environment amplitude to one shared unit-amplitude
profile, so `φ(t/T)` is bit-identical across candidates and only the scale differs.

The post-probe spread across environments is **245 µm** at the chosen gap, and 176–421 µm
across the five gaps measured. It is not zero, and it cannot be: repeating the same probe in a
*single* environment gives 264–464 µm of spread, which is the same magnitude. So the
variability is intrinsic to the probe's stopping rule — a probe that stops on a displacement
threshold can cross it a step early or late, and the probe duration itself varies by one to
three control steps — and not an artefact of running environments in parallel.

What matters is that it cannot flip a label on its own: 245 µm is **3 %** of `ε_d = 7.5 mm`.
`test_candidates_start_from_comparable_states` asserts it stays under 20 % of `ε_d`, which is
loose enough not to fail on sampling noise and tight enough to catch a real regression.

## 8. Known limitation: damping is not identified

The calibrated probe does not distinguish damping. Sweeping `b` from 2 to 11 N·s/m leaves the
probe duration and the breakaway force essentially unchanged, because the probe moves the
drawer 3 mm at a few mm/s and a viscous term is proportional to velocity.

This is recorded, not fixed. Adding a second, faster probe segment would identify `b`, and it
would also change the protocol, the task and the dataset all at once. This round keeps the
Phase 9 probe as the baseline so the sequential protocol is the only thing that changed
(D032). Whether `b` needs to be identifiable at all is an open question: it is a hidden
dimension of the *dynamics* regardless, and the measured effect of `b` on the required force
is what decides whether a model must resolve it.

## 9. Running it

```bash
# validate the protocol's properties and re-derive the gap length
python scripts/validate_sequential_protocol.py --headless

# build the authoritative Oracle (about 2.5 h for 5616 rows)
python scripts/build_sequential_oracle.py --headless

# compare against Phase 9 and recompute the probe-feature correlation
python scripts/compare_reset_vs_sequential.py
```

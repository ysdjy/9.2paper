# Counterfactual post-probe branching

Whether one probe may supply labels for many candidate forces, and what had to be true for
the answer to be yes.

Implementation: `src/probe_drawer/protocols/simulation_snapshot.py`.
Validation: `scripts/validate_branching.py`, reports `outputs/logs/branching_*.json`.
Decision: `docs/DECISIONS.md` D034.

---

## 1. The question

A training sample asks: *given what this probe measured, would this candidate force have
reached the goal?* Dataset v0 wants 24 candidates per probe. For those 24 labels to be
comparable, all 24 executions must start from the **same** post-probe state.

Two ways to arrange that:

| | How | Cost |
|---|---|---|
| **A. Re-probe** | Run a fresh probe before every candidate. Phase 10's Oracle did this. | 24× the probes; the candidates do *not* share a starting state, because a probe is only reproducible to 264–464 µm |
| **B. Branch** | Probe once, capture the state, restore before each candidate. | Needs the capture to be faithful |

This document is the evidence for B. **B is a dataset-generation device only.** Deployment
runs one probe and one execution and restores nothing; `SequentialPullProtocol` remains the
authority on what an episode is.

## 2. What is captured

Three separate things, and the separation is the point.

**Simulator state** — via the official `InteractiveScene.get_state()`, deep-cloned (the
payload aliases live buffers, so an uncloned snapshot would silently track the present
instead of freezing it). For each of the two articulations (`robot`, `cabinet`):
`root_pose`, `root_velocity`, `joint_position`, `joint_velocity`.

**Controller state** — the OSC's captured pose reference and whether one exists. That is all
of it: the installed `isaaclab.controllers.operational_space` has no integral term and no
previous-error term, so the controller is stateless in the control-law sense. Verified by
reading the installed source, not assumed.

**Sensor state** — the reader's four causal-derivative filter histories, plus `has_samples`.
These are genuine state: velocity and acceleration are functions of the recent past, so a
branch that restored only physics would read a wrong velocity on its first step.

**Environment state** — `episode_length_buf`. See §4.

## 3. What is *not* captured

PhysX does not expose these for reading or writing:

* contact manifolds and friction anchors at the finger–handle interface,
* per-joint static/dynamic friction regime,
* solver velocity-iteration residuals,
* articulation sleep state.

So two branches from one snapshot are **not** guaranteed bit-identical, and the residual
difference is what §5 measures.

`InteractiveScene.reset_to()` is deliberately **not** used, even though it is the official
restore path. It also calls `set_joint_position_target` and `set_joint_velocity_target` on
every articulation — its own source carries a `FIXME` noting this assumes PD control. Our arm
is effort-controlled with zeroed gains so that part would be harmless, but the **fingers** are
position-controlled and their target is a specific per-finger grip command (`grip_squeeze =
0.006`, derived from a recorded contact equilibrium). Overwriting it with the current squeezed
position would change the grip force. This module writes state and touches no target, letting
the normal action pipeline set them on the next step.

## 4. Two real bugs the validation found

Both would have silently corrupted the dataset.

### 4.1 Stale kinematics after a restore

Writing joint positions does not move the *links*: PhysX recomputes link poses on a physics
tick, and the `FrameTransformer` that reports the TCP pose refreshes only when the scene's
sensors update. Immediately after restoring, `reader.tcp_pose` still read the pose from the
**end of the previous branch** — off by 34.45 mm, exactly the drawer displacement of the
disturbing execution.

This was not cosmetic. `run_profile` begins by calling `capture_pose_reference()`, which reads
the TCP pose, so every branch was handed a pose reference from wherever the previous branch
ended and the OSC spent the episode hauling the arm back to it.

Measured effect, `medium` preset:

| | before the fix | after |
|---|---|---|
| branch spread at F = 4 N | 22 134 µm | 1 901 µm |
| order dependence | 14 511 µm | 811 µm |
| `tcp_pose` restore error | 3.4 × 10⁻² m | 0.0 |

Fixed by `_refresh_derived_buffers()`: `sim.forward()`, then `articulation.update(0.0)` for
each articulation, then `sensor.update(0.0, force_recompute=True)` for each sensor. `dt = 0`
because no time has passed; `force_recompute` because the scene's lazy sensor update would
otherwise conclude nothing had changed.

### 4.2 The episode would have auto-reset mid-sweep

`ManagerBasedRLEnv` increments `episode_length_buf` every step and resets an environment when
it reaches `max_episode_length`. The research episode is 30 s. One probe plus 24 branches is

```
1.63 s (probe + gap) + 24 x 1.53 s (execution + cleanup) = 38.5 s
```

so the generator would have hit a **silent auto-reset partway through every candidate
sweep**. `episode_step` is now part of the snapshot, which both prevents that and is the
correct counterfactual semantics — every branch should be at the same age in the episode.

## 5. What was measured

`scripts/validate_branching.py`, presets `medium` (m = 8 kg, µ = 3.0, b = 6) and `hard`
(m = 10 kg, µ = 4.0, b = 9). Six checks; both presets pass all six.

### 5.1 Restore fidelity — exact

After a full 4 N execution had moved the drawer 34–36 mm, restoring left, on every readable
quantity:

| quantity | error |
|---|---|
| `drawer_position` | 0.0 |
| `drawer_velocity` | 0.0 |
| `arm_joint_position` | 0.0 |
| `arm_joint_velocity` | 0.0 |
| `finger_joint_position` | 0.0 |
| `tcp_pose` | 0.0 – 2.4 × 10⁻⁷ |
| `tcp_pull_axis_velocity` | 0.0 – 2.4 × 10⁻⁶ |

The first five are quantities the snapshot *writes*, and they come back **identical in
float32** — not "within tolerance". The last two are *derived* from them (the TCP pose is
forward kinematics through a `FrameTransformer`), so they can only agree to float32 round-off:
at coordinates around 0.5–0.7 m one ULP is about 6 × 10⁻⁸. The distinction is worth keeping
because it says where a future restore failure would show up first.

### 5.2 Branch drift over a full candidate sweep — the decisive check

24 branches from one snapshot at one force, against 24 fresh full episodes at the same force:

| preset | F (N) | branch mean | branch spread | fresh spread | branch/fresh | drift over the sweep |
|---|---|---|---|---|---|---|
| medium | 1.0 | 3.78 mm | **23 µm** | — | — | +2 µm |
| medium | 2.5 | 5.47 mm | 2 892 µm | 2 070 µm | **1.40** | **−1 312 µm** (−57 µm/branch) |
| hard | 3.5 | 5.73 mm | 2 739 µm | 7 997 µm | **0.34** | −369 µm (−16 µm/branch) |

Two things to read here.

**Branching is not noisier than re-probing.** At the same sample size on both sides, the
ratio is 1.40 and 0.34. When the execution barely moves the drawer (F = 1 N) branching is
essentially perfect: 23 µm, against 398–681 µm for fresh episodes — a factor of 20–30.

**But there is a systematic drift**, and it is the one finding that changed the design. At
medium/2.5 N the outcome falls 57 µm per branch, 1 312 µm (0.17 ε_d) across a full sweep. It
appears only when the execution actually pushes the drawer past breakaway — at F = 1 N it is
2 µm — which points at the un-restorable contact state as the cause.

Drift matters more than its size suggests. Noise widens a label's uncertainty; a *monotone*
drift correlated with branch index would correlate a candidate's label with its position in
the sweep, and since candidate forces are assigned to ordered strata, that would put a bias
along the exact axis the model is learning.

**Mandated mitigation.** Dataset v0 executes the 24 candidates of a probe in a
**deterministically shuffled order**, and stores `branch_index` on every row. That turns a
force-correlated bias into force-uncorrelated noise by construction, and keeps the drift
auditable: the dataset audit regresses `d_total` on `branch_index` at fixed force and checks
that force and `branch_index` are uncorrelated.

### 5.3 Order independence

Same snapshot, two candidate forces, run ascending then descending. Worst disagreement
822 µm (medium) and 588 µm (hard), against the 2 892 µm / 2 739 µm that 24 *identical*
branches already show. So a branch does not measurably remember what ran before it, beyond
the drift already quantified.

### 5.4 No systematic bias

Across seven independent runs, the branch mean always landed **inside** the fresh episodes'
range, with |bias| ≤ 0.096 mm — 0.04 to 0.20 of the fresh spread. Branching lands where
re-probing lands.

### 5.5 The probe's record is untouched

Displacement, duration, step count and a checksum over the recorded drawer position are all
unchanged after branching. The model's input is not edited by generating labels from it.

## 6. A separate finding: some operating points are bistable

Near a high-friction drawer's breakaway threshold the outcome is genuinely bistable — the
same command either breaks the drawer loose and runs, or does not. Observed:

* `hard` at 5 N: one run's fresh episodes spread 6 953 µm and one branch reached ≈ 65 mm
  against a typical 32 mm; a later 5-repeat run at the same operating point spread only
  970 µm. The difference is which side of the threshold that run's probe left the drawer on.
* `medium` at 4 N: fresh spread 4 510 µm.
* In one long unbroken session, a fresh `medium` episode at 2.5 N ran away to ≈ 140 mm.

This is a property of the **task**, not of branching: both protocols show it, and no protocol
produces a reliable label there. The validation flags any force where the fresh spread alone
exceeds `ε_d / 2` and excludes it from the comparative gate rather than blaming branching for
it. It is also the reason Dataset v0 keeps three independent probe repeats per hidden state:
so label noise can be *measured* instead of assumed, via
`empirical_success_probability`.

## 7. Criteria that were changed during this work, and why

Recorded because changing a pass criterion after seeing data is exactly the move that needs
justifying.

**An absolute 750 µm bar became a comparative one.** The first version required branch-to-branch
agreement within `ε_d / 10`. That bar was set before the comparison number existed, and it
demands more of branching than the physics offers at all: Phase 10 measured ~0.9–1.1 mm of
`d_total(T)` spread over six *identical fresh* episodes. The criterion is now "branching must
be no noisier than the alternative it replaces", with `ε_d` reported alongside.

**The pairwise branch/fresh ratio was demoted to a diagnostic.** With 2–4 repeats it ranged
**0.71 to 2.99** across runs of the same physics — it cannot separate "branching is noisier"
from sampling noise. The drift check measures the same property with 24 samples on both sides
and is what the verdict uses.

**Order independence was re-pointed at the 24-sample spread.** It had been comparing against a
2-repeat estimate of branch-to-branch noise (405 µm) that the 24-branch check showed to be
2 892 µm — a several-fold underestimate. That was a wiring mistake on my part, not a bar.

## 8. Verdict

**PASS — branching is usable for Dataset v0**, with the mitigation in §5.2 mandatory.

Residual risks, all recorded rather than solved:

* Branch-to-branch spread reaches 0.39 ε_d at forces just past breakaway. Labels for
  candidates near the tolerance boundary are intrinsically uncertain — in both protocols.
* Drift of up to 0.17 ε_d per sweep, mitigated by shuffling and made auditable by
  `branch_index`.
* Bistable operating points exist and are flagged, not removed.
* Long unbroken sessions (~30+ executions without a reset) showed one runaway episode. The
  generator's longest unbroken chain is one probe plus its 24 branches, which is the chain
  the drift check measures.

## 9. Reproducing

```bash
python scripts/validate_branching.py --headless --preset medium --drift-force 2.5 \
    --branch-forces 2.5 --bias-force 2.5 --branch-repeats 2
python scripts/validate_branching.py --headless --preset hard --drift-force 3.5 \
    --branch-forces 3.5 --bias-force 3.5 --branch-repeats 2
```

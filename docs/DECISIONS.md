# Design decisions

Settled questions. Do not relitigate these without adding a superseding entry.

---

### D001 — Manager-based cabinet environment, not the direct RL environment

**Decision.** Build on `isaaclab_tasks.manager_based.manipulation.cabinet`
(`Isaac-Open-Drawer-Franka-*`) rather than `Isaac-Franka-Cabinet-Direct-v0`.

**Reason.** The manager-based workflow lets us swap a single action term (joint position ->
hybrid OSC) and add one sensor by *inheriting* the official config. The direct environment
hard-codes its control and observation logic in one class, so the same change would mean
copying it.

**Alternatives.** Direct workflow (rejected: forces duplication); writing our own scene
(rejected: loses the validated official asset setup).

**Date.** 2026-09-02

---

### D002 — IK-Abs is a reference baseline, not the research controller

**Decision.** `Isaac-Open-Drawer-Franka-IK-Abs-v0` is used to validate the environment and
to *record* the grasped arm configuration. The research experiments run a force-driven
hybrid OSC instead.

**Reason.** The research question is about force, not motion: a differential-IK position
controller would make the drawer's dynamics almost invisible, because it would servo to a
position regardless of how hard that turned out to be.

**Alternatives.** IK-Rel (same objection); impedance control on all six axes (rejected: the
pull axis must be force-controlled, not stiffness-controlled).

**Date.** 2026-09-02

---

### D003 — Probe and Execution share one hybrid OSC, which is Isaac Lab's

**Decision.** `HybridPullOSC` wraps Isaac Lab's `OperationalSpaceController` (through the
official `OperationalSpaceControllerAction` term) with `target_types=["pose_abs",
"wrench_abs"]`, `motion_control_axes_task=(0,1,1,1,1,1)` and
`contact_wrench_control_axes_task=(1,0,0,0,0,0)`. Both public controllers own an instance;
neither contains robot control code.

**Reason.** The official controller already implements exactly the required split, so there
is nothing to reimplement. One shared instance also means the two controllers differ *only*
in force profile and stop conditions, which is what makes probe and execution histories
comparable.

**Alternatives.** A custom OSC (rejected: no missing capability was found); two independent
controllers (rejected: duplicated control code with no benefit).

**Date.** 2026-09-02

---

### D004 — `d_goal` is not an input to the execution controller

**Decision.** `ExecutionPullController.run(peak_force, duration)`. The goal displacement is
not an argument, and no stop condition in the execution path refers to drawer displacement.
The only permitted early stop is an absolute safety violation, implemented in the base
class.

**Reason.** Stopping when the goal is reached would turn an open-loop force-execution study
into a closed-loop position-control study — a different research question. Success is an
*evaluation* property: `abs(d(T) - d_goal) <= epsilon`, computed by the caller.

**Enforcement.** `tests/unit/test_execution_has_no_goal_feedback.py` parses the execution
controller's AST and fails if any of `d_goal`, `goal`, `target_displacement`, `epsilon` or
`success` appears as an identifier, if `run`'s signature changes, or if `_stop_conditions`
starts reading state off `self`.

**Alternatives.** Passing `d_goal` "for logging only" (rejected: it would be one edit away
from entering the loop).

**Date.** 2026-09-02

---

### D005 — Git is the backup mechanism

**Decision.** No `*_old.py` / `*_new.py` / `*_backup.py` files. Commits, branches and tags
only. `backups/` exists for large binary experiment snapshots that genuinely cannot live in
git, and every entry must be documented in `backups/README.md`.

**Date.** 2026-09-02

---

### D006 — `measured_force` is the wrist joint reaction wrench, not the commanded force and not the handle contact force

**Decision.** `DrawerStateReader.measured_pull_force` is the pull-axis component of
`Articulation.data.body_incoming_joint_wrench_b` at `panda_hand`, rotated out of the
`panda_link7` frame into the world frame. `commanded_force` and `measured_force` are always
recorded as separate signals.

**Reason.** This is the quantity a real Franka's wrist force/torque sensor measures, so the
same signal exists on hardware. It was validated against Newton's second law: under a 4 N
command on the `nominal` preset the wrist force averaged 1.40 N over the second half of the
run, against 1.53 N predicted by `m*a + f*sign(v) + c*v` from the recorded trajectory.

**Rejected alternative — a `ContactSensor` on `drawer_handle_top`.** Measured and discarded:
its `net_forces_w` reported 0.22-0.37 N whether the commanded pull was 4 N or 12 N, because
the pull is transmitted through *tangential* finger friction which the net-contact-force
report does not include. It is still logged as `handle_contact_force_w`, as a witness of
grip load, but nothing decides anything from it.

**Caveat.** The wrist wrench includes the force needed to accelerate the hand and fingers
(~0.9 kg, so ~0.2 N at typical accelerations) and is noisy in contact-rich phases
(observed swings of a few newtons at 5 N command level). It is an *estimate* of the force
delivered to the drawer, and is documented as such.

**Date.** 2026-09-02

---

### D007 — The probe stops the moment the command reaches `max_force`

**Decision.** `hold_after_max_force = 0.0` by default: the probe terminates with
`max_force_reached` on the step after the command first equals `max_force`, so
`final_commanded_force == max_force` exactly and `duration == ramp_duration + step_dt`.
The hold is configurable for future studies.

**Reason.** Simple, deterministic and reproducible, which was the stated priority for the
first version. Holding at maximum force would collect more information about an immovable
drawer, but it makes the probe's duration depend on a second configuration knob.

**Note.** The stop condition compares the *issued* command (one step back from the current
elapsed time), not `elapsed` directly. Comparing `elapsed` stopped the probe one step early
at 9.867 N instead of 10 N; this was found by an integration test.

**Date.** 2026-09-02

---

### D008 — The drawer drive stiffness is removed

**Decision.** `ProbeDrawerEnvCfg` sets the `drawers` actuator stiffness to 0 (the official
cabinet uses 10 N/m) and keeps damping at the official 1.0 N s/m as the nominal value.

**Reason.** A drive stiffness acts as a spring pulling the drawer shut, i.e. a fourth,
position-dependent hidden parameter on top of `xi = [mass, friction, damping]`. Removing it
makes the drawer exactly the mass-friction-damper system the hidden state describes.

**Alternatives.** Keeping 10 N/m (rejected: at 0.3 m of travel it contributes 3 N, the same
order as the pull itself, and it is not part of `xi`); making stiffness a fourth element of
`xi` (deferred: not required by the current research question).

**Date.** 2026-09-02

---

### D009 — Drawer velocity is a finite difference, not PhysX's `joint_vel`

**Decision.** `DrawerStateReader.drawer_velocity` is a two-step moving average of the
drawer position's finite difference across control steps. PhysX's own reading remains
available and is logged as `drawer_velocity_raw`.

**Reason.** Measured: with the drawer demonstrably opening at +0.0073 m/s (14.4 mm over
2.0 s), `Articulation.data.joint_vel` sampled once per 60 Hz control step averaged
**-0.0076 m/s** — right magnitude, wrong sign — with a step-to-step correlation of 0.18
against the finite difference. Gripper contact chatter at about half the control rate
aliases the sample. After the grip was rebalanced (D010) the same reading became
*identically zero* for 120 consecutive steps while the drawer moved 2.5 mm. Either way it
cannot be used for a stop condition.

The two-step average is chosen because it exactly cancels a two-step alternation, at the
cost of one step of lag.

**Alternatives.** Raising the physics rate (rejected for now: it changes every other
measurement and the discretisation the controllers run at); a longer filter (rejected:
adds lag to a stop condition with no accuracy benefit).

**Date.** 2026-09-02

---

### D010 — The grip is balanced per finger, with an explicit grip force

**Decision.** `ProbeDrawerEnvCfg` replaces the official gripper close command (both fingers
to 0 m) with a per-finger command derived from the *recorded contact equilibrium* of each
finger: `command = equilibrium - grip_squeeze`, with `grip_squeeze = 0.006 m` giving 12 N
per finger at the official 2000 N/m finger stiffness. The startup randomisation of the
robot and handle friction coefficients is also pinned to the midpoints of the official
ranges.

**Reason.** The hand does not sit exactly on the handle centre (measured offset 3.1 mm in
z), so a shared position target of 0 m made the two fingers deflect by 7.7 mm and 15.2 mm
and squeeze with roughly 15 N and 30 N. That imbalance leaked a **steady 0.68 N bias force
onto the pull axis** — comparable to what a 2 N probe actually delivers to the drawer — and
opened the drawer by 14.4 mm in 2 s with *zero* commanded force. It also drove the contact
chatter behind D009. Balancing the deflections reduced the bias to 0.25 N, the net vertical
grip load from 12.1 N to 0.99 N, and the zero-command drift to 2.5 mm over 2 s. Pinning the
friction coefficients removes the last source of episode-to-episode variation, so two
episodes with the same `xi` are genuinely comparable.

**Alternatives.** Iterating the grasp waypoint until the hand centres itself (tried:
converged at about 0.28 of the commanded correction per iteration, so it needed many
Isaac Sim runs to reach a result the per-finger command gives exactly); lowering the finger
stiffness (rejected: it scales the bias down but scales the grip strength down with it, and
the handle then slips under larger pulls).

**Residual.** A 0.25 N bias and a ~1.3 mm/s creep remain, mostly absorbed by the drawer's
own rail contact friction. Over a 2 s execution that is a few millimetres against
displacements of 60-330 mm. It is identical in every episode because the initialisation is
deterministic, and it is visible in every logged history.

**Date.** 2026-09-02

---

### D011 — Configuration lives in dataclasses; `configs/*.yaml` is a tested snapshot

**Decision.** The dataclasses under `envs/`, `controllers/` and `sensors/` are the single
source of truth. `configs/controller.yaml`, `probe.yaml`, `execution.yaml` and
`dynamics.yaml` are generated snapshots of their defaults, each declaring its `_source`.
They are never loaded at runtime. `tests/unit/test_config_snapshots.py` rebuilds the
expected contents from the dataclasses and fails, printing the correction, if they drift.

**Reason.** The project layout calls for `configs/`, and a reviewer should be able to read
the current settings without opening code. Two live sources of truth would drift; a
snapshot with a drift test cannot.

**Exception.** `configs/grasp_pose.yaml` *is* loaded at runtime — it is recorded
measurement data, not configuration, produced by
`scripts/run_official_drawer.py --export-grasp-pose`.

**Date.** 2026-09-02

---

### D012 — Structural additions to the prescribed layout

**Decision.** Three modules exist that the original layout did not list:

| Module | Why |
|---|---|
| `controllers/base_pull_controller.py` | the step loop, safety limits and history recording are shared by both public controllers; the layout offered no home for them and duplicating them was prohibited |
| `pull_system.py` | every script needs the same environment + reader + OSC + controllers wiring; it is the "Pull System" box of the architecture diagram |
| `sensors/pull_axis.py`, `envs/hybrid_pull_cfg.py` | small pure modules split out of larger Isaac-Lab-dependent ones, so the pull-axis logic and the OSC gains can be unit-tested without launching Isaac Sim |

**Date.** 2026-09-02

---

### D013 — Controllers are vectorised over environments

**Decision.** One `run()` call drives every environment; per-environment result fields are
`numpy` arrays indexed by environment, and `PullHistory` carries an `active` mask so an
environment that stopped early can be separated from its zero-padded tail.

**Reason.** The dynamics comparison needs the same force applied under different `xi` with
*nothing else* differing — same solver, same step, same grasp. Running the presets in
parallel environments of one simulation gives that by construction, and the data-generation
phase needs the throughput anyway. Designing for one environment now would mean rewriting
the API later.

**Cost.** Results are length-1 arrays rather than scalars in the single-environment case;
`result.summary(env_index)` exists for readable output.

**Date.** 2026-09-02

---

### D014 — Every simulation step goes through `HybridPullOSC.step`

**Decision.** `HybridPullOSC.step(action)` calls a caller-supplied `stepper` (the *wrapped*
gym environment) and then `DrawerStateReader.update()`. Nothing in the project calls
`env.step` directly.

**Reason.** Two things must happen exactly once per step: the finite-difference velocity
update (D009), and the step must pass through the gym wrapper stack. Stepping the unwrapped
environment silently bypassed `RecordVideo`, which is why the first probe and execution
recordings were a single frame long. Centralising the call makes both impossible to forget.

**Date.** 2026-09-02

---

### D015 — The main paper's hidden state is exactly four dimensional

**Decision.** `xi = [drawer_mass, joint_static_friction, joint_dynamic_friction, joint_damping]`.
Joint stiffness, per-DOF viscous friction, armature, centre of mass, inertia tensor, contact
materials, restitution, joint limits and every robot-side parameter are held at fixed,
documented values.

**Reason.** All fifteen candidates examined are writable and read back correctly
(`docs/HIDDEN_STATE_AUDIT.md`), so this is a modelling choice rather than a limitation. The
four selected are the ones that are simultaneously invisible to the robot, physically
independent of one another along a single prismatic degree of freedom, and individually
observable in a pull: mass in the acceleration transient, static friction in the breakaway,
dynamic friction in the drag once sliding, damping in the velocity-dependent part of it.

**Alternatives considered.** Adding viscous friction or armature was rejected as
*unidentifiable*: along one translational DOF they have the same signature as damping and
mass respectively, so no probe could ever separate them. Adding handle contact friction was
rejected as *confounding*: it changes the grasp, not the drawer, and would make a failed
pull ambiguous between a stiff drawer and a slipping grip. Joint stiffness and centre of
mass were deferred to out-of-distribution testing.

**Date.** 2026-09-02

---

### D016 — PhysX requires `mu_s >= mu_d`, and discards violating writes silently

**Decision.** `DynamicsParameters` refuses `joint_dynamic_friction > joint_static_friction`,
and `DynamicsRandomizer.sample` draws `mu_d` as a *fraction* of `mu_s` so every sample is
valid by construction.

**Reason.** Measured: writing `static = 1, dynamic = 5` makes PhysX log
`Static friction effort must be greater than or equal to dynamic friction effort` and keep
the previous values, while Isaac Lab's `Articulation.data` buffers report the requested ones.
`data` shows `(1.0, 5.0)`; the simulator holds `(5.0, 1.0)`. A silently discarded write is
indistinguishable from a successful one unless the readback comes from the simulator.

**Consequences.** Every readback in `dynamics_randomization.py` now comes from
`root_physx_view`, and the Isaac Lab mirror is reported separately as `mirror_agrees` so a
disagreement is visible. The previous `consistent` check compared against the mirror and
could therefore have passed on a write that never landed.

The constraint is also physically correct -- real Coulomb friction has `mu_s >= mu_d` -- so
the requirement in the original task description for a "low static, high dynamic" case
cannot be met and should not be.

**Date.** 2026-09-02

---

### D017 — Every logged channel is classified deployable, diagnostic, or sim-only

**Decision.** `probe_drawer.observations.OBSERVATION_SPECS` names, for all 25 channels, the
unit, the source, any filtering, and one of three deployability classes.
`validate_model_input()` raises if a non-deployable channel is proposed as a model input.

**Reason.** "The simulator can read it" is not the same as "the robot will have it". Without
an explicit classification, a future agent wiring up the adaptation model could reach for a
privileged channel and produce something that cannot be deployed -- and the mistake would be
invisible, because the training numbers would look fine.

**Date.** 2026-09-02

---

### D018 — `commanded_force` is a mandatory model input

**Decision.** The first adaptation model's observation vector always contains
`commanded_force`. It is `DEFAULT_ACE_INPUT[0]`, and a unit test asserts it.

**Reason.** A robot always knows the force it asked its controller for; withholding it would
be an artificial handicap. It is also load-bearing information: two drawers that both move
5 mm, one under 4 N and one under 8 N, have very different dynamics, and without the command
the two histories are nearly identical.

**Date.** 2026-09-02

---

### D019 — Rich logging, selective model input

**Decision.** Log every channel every episode; feed the model a documented subset. The
subset is `DEFAULT_ACE_INPUT`, and the ablation ladder ACE-1 through ACE-5 is expressible
from the logged channels without recollecting data.

**Reason.** Recollecting a sweep costs hours; storing a few more arrays costs megabytes. The
risk this creates -- someone using a channel they should not -- is handled by D017 rather
than by logging less.

**Date.** 2026-09-02

---

### D020 — Success requires the drawer to be at rest, not merely at the goal

**Decision.** `success = |d(T) - d_goal| <= eps_d AND |v(T)| <= eps_v AND valid`.

**Reason.** Position alone is not task completion. Measured: at `T = 1.5 s` with the
original profile, every episode that landed within 10 mm of 100 mm was still moving at
0.16-0.22 m/s -- passing through the goal, not placed at it. Requiring a small terminal
speed is what makes the label mean what it says.

**Consequence.** This is what forced the ramp-down redesign (D023): the criterion is
unsatisfiable at large distances with a 10 % ramp-down, and the honest response was to change
the profile rather than to drop the criterion.

**Date.** 2026-09-02

---

### D021 — The earlier 2 N / 10 N / 5 mm / 100 mm / 2 s / 5 N figures were provisional

**Decision.** Those numbers are labelled a *provisional validation operating point*
throughout the code and documentation. The paper's parameters come from the Phase 9 sweeps
and live in `probe_drawer.experiment_plan`.

**Reason.** They were chosen during development to make the pipeline exercise its own code
paths, not by any experiment-design criterion. Presenting them as final would misrepresent
how they were obtained. None survived unchanged: `T` moved from 2.0 to 1.5 s, `d_goal` from
100 to 50 mm, the probe target from 5 to 3 mm, and the probe's force range from 2-10 N to
1-6 N.

**Date.** 2026-09-02

---

### D022 — The execution result is snapshotted at `T`, before the zero-force cleanup

**Decision.** `ExecutionResult` is captured at `t = T`. Zero pull force is then commanded
for `zero_force_cleanup_steps` (default 2) plus `post_execution_settle_steps` (default 0)
further steps, which appear nowhere in the history, the duration, `d(T)`, `v(T)`, or the
success evaluation.

**Reason.** The profile satisfies `phi(1) = 0`, but a command is held for a whole control
step, so the last command of the episode is the one issued at `T - dt`: about 2 % of the peak
with a 10 % smoothstep fall. That command is *correct* for the interval it covers, and
`d(T)` is right. What would be wrong is leaving it standing after `T`, because the
environment holds the last action it was given until something replaces it.

**Verification.** Integration tests assert that the pull force is zero after the run, that
without cleanup a residual command *does* remain (so the cleanup is what zeroes it), that
the history contains exactly `T / dt` steps, and that `d(T)` predates the cleanup -- the
drawer has visibly coasted further by the time the run returns.

**Date.** 2026-09-02

---

### D023 — The execution profile's ramp-down is 20 % of the duration

**Decision.** `ExecutionControllerCfg.fall_fraction = 0.20` for the paper's experiments, up
from 0.10.

**Reason.** Measured over the 108-point hidden-state grid, the largest `d(T)` reachable with
`|v(T)| <= 0.05 m/s` at `T = 1.5 s`: 49.4 mm at `fall = 0.10`, 54.9 at 0.15, **65.2 at
0.20**, 71.3 at 0.30, 79.4 at 0.35. A 10 % ramp-down lasts 0.15 s, and a low-resistance
drawer cannot decelerate in that time, so D020's terminal-velocity requirement made larger
goals unreachable. 0.20 produced the highest-discrimination accepted task of the five
(1.568) with coverage 0.98.

**Note.** The normalised profile `phi(t/T)` is still identical across every experiment, which
is what the design requires; what changed is which fixed shape is used, once, on the basis
of a sweep.

**Date.** 2026-09-02

---

### D024 — Task parameters are selected by a scored rule, not by judgement

**Decision.** `probe_drawer.analysis.oracle` scores every candidate task definition against
eight explicit conditions and recommends the accepted candidate with the greatest spread of
required force. The probe is chosen the same way: the least intrusive of the candidates whose
predictive power is within 0.02 of the best.

**Reason.** A parameter chosen by eye is a parameter that cannot be defended or reproduced.
Encoding the preference makes the choice re-derivable from the datasets, and makes a
rejection reviewable: when nothing is accepted the report says which condition eliminated
what, so the response is to change the experiment rather than to lower the bar.

**Also.** Discrimination is the tie-breaker rather than coverage or precision because it is
the property the research question depends on. A task every drawer solves at the same force
would be a well-behaved task and a worthless one.

**Date.** 2026-09-02

---

### D025 — Drawer velocity, acceleration and every other derivative are causal

**Decision.** All derivatives come from `probe_drawer.sensors.CausalDerivative`: a moving
average of one-step finite differences, using only current and past samples. Both the
smoothed and the unsmoothed value are exposed, and the method, window and lag are recorded
with every episode.

**Reason.** A filter that looks ahead cannot run on a robot, so a model trained on
non-causally smoothed observations would not transfer. Keeping the raw channel alongside
makes the filter's effect auditable after the fact rather than a matter of trust.

**Date.** 2026-09-02

---

### D026 — The sequential Oracle is the ground truth; the reset Oracle is kept as a comparison

**Decision.** The paper's labels come from `outputs/logs/sequential_oracle_fall035.json`, in
which the probe, an inference gap and the execution run without a reset in between. Phase 9's
reset datasets are kept, are not superseded in the repository, and are used for exactly one
thing: measuring what the reset was hiding.

**Reason.** A real robot does not get a reset between feeling a drawer and pulling it, and the
difference is not small. Measured on the Phase 9 task over all 108 hidden states, the required
force falls by a median of **0.45 N — a factor of 0.80** — with per-state ratios from **0.32 to
1.02** and a largest single shift of **1.40 N**. That is far too large to absorb into a
tolerance.

**Also.** The ratio distribution is **bimodal**, not centred: one cluster near 0.55–0.65 and
another near 0.95–1.00 (figure E). So the reset was not a uniform rescaling that a calibration
constant could undo — how much the probe helps depends on the hidden state, which is precisely
the quantity a model has to infer. Meanwhile the *ranking* survives almost intact (rank
correlation **+0.95**), so the reset Oracle remains a fair answer to "which drawer is stiff"
and a biased answer to "by how much".

**Date.** 2026-09-02

---

### D027 — `d_goal` is measured from before the probe

**Decision.** `d_total(T) = x_drawer(T) − x_initial`, where `x_initial` is read once at the
start of the episode, before the probe. The probe's own displacement therefore counts towards
the goal. `SweepRecord.final_displacement` holds this quantity for both protocols.

**Reason.** It is the only frame in which the robot's total behaviour is what is judged.
Measuring from the post-probe position would let a policy bank arbitrary free displacement by
probing harder, turning an information-gathering action into a covert part of the task.

**Also.** `tests/integration/test_sequential_protocol.py::test_ignoring_the_probe_would_change_the_label`
demonstrates the frame is load-bearing rather than bookkeeping: the same episode passes with
the probe counted and fails without it.

**Date.** 2026-09-02

---

### D028 — The inference gap is 8 control steps, chosen on repeatability

**Decision.** A fixed 8 steps (133 ms at 60 Hz) of zero pull force separates the probe from
the execution, in every episode, identically.

**Reason.** A deployed system needs wall-clock time to run its adaptation model, so the time
is reserved explicitly rather than left implicit. The *length* was measured, not assumed: over
six identical episodes at the operating point (F = 4.25 N), the spread of `d_total(T)` was
3.58 mm at 0 steps, 2.61 mm at 2, 3.29 mm at 4, **0.90 mm at 8** and 1.40 mm at 12. A second
run over 4, 8 and 12 steps gave 1.66, 1.14 and 1.40 mm: the absolute values move by a few
tenths of a millimetre between runs, and 8 steps is the minimum in both.

**Also.** The mechanism is that `dd/dF` reaches about 40 mm/N just above breakaway, so the few
hundred µm/s of residual velocity the probe leaves is amplified into millimetres of finished
displacement. Coasting to a near-stop under the drawer's own friction removes the
amplification. About **1 mm** (0.90–1.14 mm over two runs) is the protocol's intrinsic
episode noise, and it is the reason `ε_d` is 7.5 mm rather than 5 mm.

**Date.** 2026-09-02

---

### D029 — The execution controller must not settle in the sequential protocol

**Decision.** `SequentialPullProtocol.__init__` raises `ValueError` if
`system.execution.cfg.settle_steps != 0`. The protocol refuses to run rather than warning.

**Reason.** The settle applies a velocity-proportional braking force of up to 15 N to the pull
axis. In Phase 9 that was correct — it quieted the rig between independent sweep points. In
the sequential protocol it would erase exactly the post-probe velocity the protocol exists to
preserve, and it would do so silently, producing data that looked like sequential data and was
not.

**Also.** This is why the post-probe velocity is allowed to decay by physics and is never
written. Measured: the drawer retains **under 0.2 %** of its probe-end velocity after the
gap (0.19 % in one run, below the reporting precision in another; 54 % at 2 steps and 0.1 % at
4, so the decay is a physical time constant and not a threshold). The
number is small; how it got small is the point.

**Date.** 2026-09-02

---

### D030 — The inference gap belongs to neither history

**Decision.** The gap is recorded as its own `InferenceTransition`. It is not appended to the
probe history a model reads, and it is not counted inside the commanded `T`.

**Reason.** The probe history must contain exactly what a deployed robot would have measured
*before* it had to decide, or the model is trained on evidence it will not have. And `T` is a
task parameter: if the gap were inside it, changing the model's inference budget would silently
change the task.

**Date.** 2026-09-02

---

### D031 — The dataset splits by group, and a per-row split is not offered

**Decision.** `SPLIT_LEVELS = ("xi_id", "probe_id")`; `SplitCfg(level="candidate_id")` raises.
Group assignment is by hashing the group key, not by shuffling with a seed.

**Reason.** One probe is expensive and is naturally paired with many candidate forces, so those
rows share a hidden state, a probe recording and a post-probe state. A random row split puts
near-duplicates of a training row into the test set and reports memorisation as
generalisation. Making the leaking option raise is cheaper than catching it in review.

**Also.** Hashing rather than seeding makes the split *stable*: adding hidden states later does
not move existing ones between subsets, so a model trained on an earlier version of the
dataset can still be evaluated honestly on the later version's test set.

**Date.** 2026-09-02

---

### D032 — Damping stays unidentified this round, and is recorded as a limitation

**Decision.** No second probe segment is added. The Phase 9 probe is kept unchanged as the
baseline so that the protocol is the only thing that changed this round.

**Reason.** The calibrated probe does not distinguish damping: sweeping `b` from 2 to
11 N·s/m leaves the probe duration and the breakaway force essentially unchanged, because the
probe moves the drawer 3 mm at a few mm/s and a viscous term scales with velocity. A second,
faster segment would identify `b` and would also change the probe, the task and the dataset
simultaneously, leaving no way to attribute any change in the results.

**Also.** Figure F is the consolation as well as the limitation. Required force is driven
almost entirely by **dynamic friction**, with static friction secondary; mass and damping
barely move the median at all. A hidden dimension that does not change the answer costs little
to leave unidentified — but that is a measured claim about this task, not a general one, and it
should be re-checked if the task changes.

**Date.** 2026-09-02

---

### D033 — `ε_d` is 7.5 mm even though 5 mm has higher coverage

**Decision.** The task is `d_goal = 40 mm`, `ε_d = 7.5 mm`, `ε_v = 0.03 m/s`, `T = 1.5 s`, with
`fall_fraction = 0.35`.

**Reason.** At `ε_d = 5 mm` the coverage is actually *higher* (0.981 against 0.972), so
coverage is not what rules it out. The success band is: at 5 mm it collapses to **0.10 N** —
one force grid step, and about **7 %** of the required force. That is a knife edge no
regression could be expected to hit, and it fails the project's own learnability floor
(`min_relative_width = 0.10`). At 7.5 mm the median band is 0.20 N, 14 % of the required force.

**Also.** `ε_d = 7.5 mm` is about **7×** the protocol's intrinsic `d_total(T)` noise of
roughly 1 mm (D028); 5 mm would be about 4.5×, which is too close to the floor for a label to
be reliable.

**And on the priority order.** `fall_fraction = 0.20` was *not* chosen despite marginally
higher discrimination in the Phase 9 sweep. At the Phase 10 task its coverage is 0.71 against
0.98 for 0.35 (figure D), because a short ramp-down leaves a low-resistance drawer no time to
decelerate before `T` and the terminal-velocity condition then fails. Discrimination is the
last tie-breaker, not the objective.

**Date.** 2026-09-02

---

### D034 — The adapted skill parameter is `p = [F_peak, T]`, two-dimensional

**Decision.** The parameter the adaptation methods predict is `p = [F_peak, T]`. `T` moves out
of the task definition and becomes a predicted parameter; the task is then `(d_goal, eps_d,
eps_v)` alone. The Oracle is re-swept over a `(F_peak, T)` grid and `MAIN_TASK` re-selected
against it by the existing scored rule (D024).

**Reason.** Measured, not preferred. On the one-dimensional landscape
(`outputs/logs/sequential_oracle_fall035.json`, `scripts/audit_adaptation_premise.py`) each
hidden state's succeeding forces form a contiguous interval: median band 0.20 N, median 3
succeeding forces on a 0.05 N grid, only 5 of 105 bands contain any interior failure, and
**the midpoint of the succeeding set succeeds for 104 of 105 hidden states**. In one dimension
averaging is therefore safe by construction, so a model that predicts the whole success
landscape cannot beat a single-output regressor on the grounds of multi-modality -- there is
none to exploit. That is the paper's central claim, and a one-dimensional parameter space
cannot test it.

In two dimensions the success set is a region that can be curved or disconnected, and the mean
of two succeeding parameter pairs need not succeed. Whether the regions *are* multi-modal is
an empirical question the re-sweep will answer; the point of the decision is that the question
becomes askable.

**Alternatives considered.** `p = [F_peak]` was rejected for the reason above, despite being
free -- the Oracle on disk is already its ground truth. `p = [F_max, v_cmd]`, which the
original task description specified, was rejected on cost: the pull axis is force-controlled
throughout (`HybridPullOSC`) and there is no velocity command anywhere in `src/`, so it would
need a new control mode, a re-run of all of Phase 6's controller validation, a new Oracle and
a new task selection -- a new benchmark rather than a new parameter. `[F_peak, T]` needs **no
new control code at all**: `ExecutionPullController.run` already takes `(peak_force, duration)`
and `SweepRecord` already carries `duration` as a first-class axis.

**Consequences.** `MainTask.duration` becomes a range rather than a constant.
`analysis/adaptation_premise.py` generalises from bands to regions, and gains a connected-
component count per hidden state -- the direct measurement of multi-modality. The oracle
regression target becomes the point of the success region furthest from its boundary, and
hidden states whose region is disconnected have more than one such point; those must be
counted and reported, because they are exactly the states where asking a regressor for one
answer is ill-posed. The Phase 10 sequential Oracle remains valid evidence about the physics
and about `F_peak` at `T = 1.5 s`, but it is no longer the task's ground truth.

**Date.** 2026-09-02

---

### D040 — One probe supplies many candidate labels by snapshot-and-restore, with the branch order shuffled

**Decision.** Dataset v0 runs a probe once, captures the state with
`protocols.simulation_snapshot`, and restores it before each of the 24 candidate executions.
The candidates are executed in a **deterministically shuffled order** and every row records
its `branch_index`. This is a dataset-generation device only: deployment runs one probe and
one execution and restores nothing.

**Reason.** The alternative — re-running the probe before every candidate, as Phase 9 and 10
did — means the 24 candidates do not share a starting state, because a probe is reproducible
only to 264–464 µm of post-probe displacement. Then the label attached to a candidate force is
partly a property of that candidate's own probe, and the counterfactual is only approximate.

Measured, with 24 samples on both sides at the same force: branch-to-branch spread is 23 µm
where the execution barely moves the drawer (against 398–681 µm for fresh episodes) and
2 739–2 892 µm just past breakaway (against 2 070–7 997 µm fresh). So branching is between
comparable and 30× more reproducible than the alternative, and its bias is negligible — across
seven runs the branch mean always landed inside the fresh episodes' range, with |bias| ≤
0.096 mm.

**Also — why the order is shuffled.** There is a *systematic* drift: at `medium`/2.5 N the
outcome falls 57 µm per branch, 1 312 µm (0.17 `ε_d`) across a full sweep. It appears only when
the execution pushes the drawer past breakaway, which points at the un-restorable PhysX
contact state. Drift matters more than its size suggests: candidate forces are assigned to
ordered strata, so a drift correlated with branch index would put a bias along the exact axis
the model is learning. Shuffling turns that into force-uncorrelated noise by construction, and
`branch_index` keeps it auditable.

**Two bugs this found**, both of which would have silently corrupted the dataset: a restore
left the TCP pose stale by 34 mm (the execution reads its pose reference from it), and 24
branches of 1.5 s exceed the 30 s episode, so the environment would have auto-reset partway
through every sweep. Both are fixed and regression-tested.

**Not solved, recorded.** Some operating points near a high-friction drawer's breakaway
threshold are genuinely bistable — the same command either breaks the drawer loose or does
not, and both protocols show it. Those are flagged, not removed, and are the reason each
hidden state keeps three independent probe repeats: so label noise is measured rather than
assumed.

Full evidence, including the criteria that were changed during the work and why:
`docs/COUNTERFACTUAL_BRANCHING.md`.

**Date.** 2026-09-02

---

### D035 — Candidate forces are sampled without reference to any label

**Decision.** Each hidden state's candidate forces are a stratified sample over the whole
task force range (0.15–4.5 N), jittered deterministically from that hidden state's own
identifier. The sampler is given a `xi_id` and a config, and nothing else: it cannot read a
success label, an Oracle band, or that drawer's best force.

**Reason.** Concentrating candidates near each drawer's success band would spend the budget
far better — most of the range is a foregone conclusion for any given drawer. It would also
make the training distribution a function of the labels, and a model trained on it would be
answering "given that someone already told you roughly where the answer is, refine it". That
is a different and much easier experiment than the one the paper claims to run.

**Also.** The jitter is per hidden state rather than global, so the drawers do not all share
one force grid; between-stratum forces get sampled across the dataset even though no single
drawer sees them. Zero jitter would put every sample at its stratum centre and lose that.

**Date.** 2026-09-02

---

### D036 — Every repeat of a hidden state is asked the same candidate forces

**Decision.** The three independent probe repeats of a drawer share one candidate force set.
Labels stay binary on disk; the empirical success probability is computed in analysis and
never written back.

**Reason.** It makes `(xi, F)` a *repeated measurement*. The protocol's intrinsic episode
noise is about 1 mm against `ε_d = 7.5 mm` (D028), so a single episode's label is right most
of the time and not always. Three repeats of the same question turn that from an unmeasured
worry into a number — and one that matters, because the bistable operating points found in
D040 are exactly where it will be large.

**Also.** Storing the average instead would destroy the information: a row records what
happened in one episode, and `0.667` is not something that happened. The audit and the
calibration analysis compute the probability when they need it.

**Date.** 2026-09-02

---

### D037 — Probe histories are stored ragged, at the raw control rate

**Decision.** Each probe's recording is written at its true length (16–46 steps in the pilot)
in its own compressed `.npz`, referenced by its candidates. Padding happens in the
DataLoader, per batch, to that batch's own longest sequence.

**Reason.** A probe stops when the drawer has moved 3 mm, so its length *is* information
about the drawer — resampling to a fixed grid would discard part of the signal, and padding
on disk would bake one model's convenience irreversibly into the data. The encoder consumes
the batch through `pack_padded_sequence`, so padding is never visited; the test suite asserts
the batched output equals the one-sequence-at-a-time output, because a mask applied slightly
wrongly degrades a model without failing anything.

**Also.** Storage is normalised for the same reason it is ragged: one probe answers 32
candidates, so writing the recording into each row would multiply the largest part of the
dataset by 32 for no information, and would make it possible for a probe to disagree with
itself.

**Date.** 2026-09-02

---

### D038 — "No positive was observed" is not "no force exists"

**Decision.** `oracle_feasible` is `None` for every hidden state in Dataset v0. The dataset
records `observed_positive_count` implicitly (the rows are all there); it does not claim
infeasibility.

**Reason.** This was nearly got wrong. Two of 32 pilot hidden states had no succeeding
candidate, and the first reading of their data — displacement jumping from 6.6 mm straight to
140 mm — looked like proof that no force lands the drawer at 40 ± 7.5 mm. Looking properly at
the force-sorted rows showed otherwise: one of them reached 40.1 mm at 2.61 N and failed only
on terminal velocity (0.032 against `ε_v = 0.03`), with its neighbour at 2.42 N giving
31.7 mm and a compliant 0.023 m/s. A force in the 0.19 N gap between them would have
succeeded. The grid missed it; the physics did not forbid it.

Only a dense sweep can establish infeasibility, and Dataset v0 has no dense sweep behind its
Sobol draws. Labelling a drawer infeasible on 32 negatives would silently remove the hardest
and most interesting cases from every metric.

**Date.** 2026-09-02

---

### D039 — The student is trained on the task, not on the teacher's latent

**Decision.** The student's loss is binary cross-entropy on success, optionally plus *logit*
distillation from the teacher. `latent_weight` on `||z_ace - z_priv||^2` exists, defaults to
0, and any run that raises it records that in its config.

**Reason.** Latent matching would set the student a target it provably cannot see. Phase 10
measured that the calibrated probe barely responds to damping — `b` from 2 to 11 N·s/m leaves
the probe duration and the breakaway force essentially unchanged — so a teacher free to encode
`b` in `z_priv` would demand the student reconstruct an unobservable quantity. The same phase
also showed `b` barely affects the required force, so encoding it would not even help.

The objective is *task-relevant context*, not system identification. Logit distillation asks
the student to reproduce the teacher's success landscape, which is the thing that matters,
without prescribing the coordinates it uses to get there.

**Also.** Class imbalance (about 6 % positives) is handled with `pos_weight` rather than
resampling, so the evaluation set stays the real distribution. Resampling the training set
and then evaluating on a resampled set would report a success rate no drawer has.

**Date.** 2026-09-02

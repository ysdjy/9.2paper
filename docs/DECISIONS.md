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

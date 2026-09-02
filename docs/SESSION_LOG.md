# Session log

One entry per work session. Newest first.

---

## 2026-09-02 — `agent/phase9-oracle-audit` — Phase 9A through 9M

**Agent / task.** Claude Opus 5 — fix the hidden state at four dimensions, expand and
classify the observations, audit the force channels, correct the execution's post-`T`
behaviour, then sweep the experiment space and select every experimental parameter from the
data rather than by hand. No learning components; this phase establishes the physics.

**Modified.**

* `envs/dynamics_randomization.py` rewritten: `xi` is now
  `[m, mu_s, mu_d, b]`; readback comes from `root_physx_view`, not Isaac Lab's mirror.
* `sensors/`: new `causal_derivative.py`; `drawer_state.py` gained acceleration channels,
  TCP pull-axis velocity/acceleration, orientation, and the two privileged drawer force
  channels.
* `controllers/`: `PullHistory` 16 -> 25 channels with a generic recorder; `ExecutionResult`
  gained `peak_velocity`; the execution controller now snapshots at `T` and releases the
  pull force afterwards.
* New: `observations.py`, `experiment_plan.py`, `evaluation/` (2 modules), `analysis/`
  (5 modules).
* New scripts: `audit_hidden_states.py`, `audit_force_channels.py`,
  `sweep_execution_space.py`, `build_oracle_landscape.py`, `calibrate_probe.py`,
  `plot_experiment_space.py`, `plot_probe_identifiability.py`.
* New docs: `HIDDEN_STATE_AUDIT.md`, `FORCE_CHANNEL_AUDIT.md`, `EXPERIMENT_SPACE.md`,
  `ORACLE_LANDSCAPE.md`. Existing docs updated; 11 new decisions (D015-D025).
* New configs: `evaluation.yaml`, `experiment_plan.yaml`, both drift-tested.
* Isaac Lab's own source tree: **unmodified**.

**API introduced.**

```python
DynamicsParameters(drawer_mass, joint_static_friction, joint_dynamic_friction, joint_damping)
assess_validity(result, OperatingRegionCfg()) -> ValidityReport
evaluate_execution(result, SuccessCriteria(...)) -> EvaluationReport
observations.validate_model_input(channels)
experiment_plan.MAIN_TASK / RECOMMENDED_PROBE_TASK / TRAINING_XI_RANGES / OOD_XI_RANGES
```

`ProbePullController.run` and `ExecutionPullController.run` are unchanged, and `d_goal` still
does not appear anywhere near the execution controller.

**Verification.** 248 tests pass (196 unit, 52 integration), up from 124. 23 175 simulated
episodes across six sweeps. Full table in `docs/VALIDATION.md`.

**Headline results.**

* At the selected task the required peak force spans **1.00-4.50 N across the hidden-state
  grid — a 4.5x range — with success bands 0.50 N wide**, and 106 of 108 hidden states are
  achievable. One force cannot serve every drawer.
* A single standardised probe's best feature correlates with the required force at
  **|rho| = 0.969**, using deployable channels only.
* Selected: `T = 1.5 s`, `d_goal = 50 mm`, `eps_d = 15 mm`, `eps_v = 0.08 m/s`, execution
  ramp-down 20 %, probe 1.0 -> 6.0 N over 1.0 s stopping at 3 mm.

**Simulator findings that changed the design.**

1. **PhysX requires `mu_s >= mu_d` and silently discards violating writes**, while Isaac
   Lab's `data` buffers report the request. The previous readback check could have passed on
   a write that never landed (D016).
2. **`get_dof_projected_joint_forces` is the drawer's internal resistance**,
   `-(mu_d*sign(v) + b*v)`, verified to 0.0099 N. The joint reaction wrench is structurally
   zero along a prismatic joint's own axis, so it cannot give the drawer-axis force.
3. **The Phase 8 17-23 N wrist force is the mechanical end stop**: a deliberate episode
   reached 59.2 N, 9.87x the command, at 84.4 % of travel.
4. **A 10 % force ramp-down cannot bring the drawer to rest.** With the terminal-velocity
   requirement, the reachable-at-rest distance at `T = 1.5 s` was 49 mm at `fall = 0.10`
   against 65 mm at 0.20 (D023).
5. **The Phase 8 held-axis drift was the operating point, not the controller.** Drift stays
   under 1 mm across the whole force range at moderate displacement and speed.

**Commit.** See `git log` on `agent/phase9-oracle-audit`; SHA recorded below after push.

**Remaining issues.** See "Outstanding" in `docs/VALIDATION.md`. The one that matters most
for the next phase: **the calibrated probe does not identify damping** — sweeping `b` from 2
to 11 N s/m leaves the probe response unchanged, because the probe stops before the drawer
reaches speeds where viscous drag matters. The required force also depends weakly on `b`, so
the task stays predictable, but any claim that one probe identifies all four dimensions
would be false.

**Next.** Dataset generation at the selected operating point: sample `xi` from
`TRAINING_XI_RANGES`, run the calibrated probe, then the execution at a grid of `F_peak`, and
store paired `(probe history, F_peak, d(T), v(T), success, xi)`. The APIs need no change.

---

## 2026-09-02 — `agent/phase0-8-bootstrap` — Phases 0 through 8

**Agent / task.** Claude Opus 5 — bootstrap the project, validate the official Isaac Lab
drawer environment, build the shared hybrid OSC and the two public pull controllers, add
dynamics randomisation, and validate all of it physically.

**Modified.**

* Project created at `/home/zbh/Downloads/IsaacLab/9.2paper` (moved mid-session from
  `~/Documents/isaaclab/9.2paper` after the user corrected the intended parent directory;
  nothing needed editing because every path is derived from
  `probe_drawer.utils.project_root()`). Added `9.2paper/` to the Isaac Lab checkout's
  `.git/info/exclude` so the nested repository never appears in Isaac Lab's `git status`.
* New package `src/probe_drawer/` with `envs/`, `controllers/`, `state_machines/`,
  `sensors/`, `logging/`, `utils/` and `pull_system.py`.
* New scripts: `inspect_isaaclab.py`, `run_official_drawer.py`, `test_probe_pull.py`,
  `test_execution_pull.py`, `test_dynamics_randomization.py`, `visualize_response.py`.
* New tests: `tests/unit/` (80) and `tests/integration/` (44).
* New docs: `README.md`, `CLAUDE.md`, `docs/{ARCHITECTURE,API,OFFICIAL_BASELINE,VALIDATION,DECISIONS,SESSION_LOG,CHANGELOG}.md`,
  `backups/README.md`.
* New configuration: `configs/{controller,probe,execution,dynamics}.yaml` (generated
  snapshots, drift-tested) and `configs/grasp_pose.yaml` (recorded measurement).
* Isaac Lab's own source tree: **unmodified**.

**API introduced.**

```python
ProbePullController.run(initial_force, max_force, target_displacement, max_velocity) -> ProbeResult
ExecutionPullController.run(peak_force, duration) -> ExecutionResult
DynamicsRandomizer.sample(num_envs) / .apply(env, params) / .get_current_params()
PullSystem.build(PullSystemCfg) -> PullSystem
```

**Verification.** All of Phases 0-8 pass; 80 unit tests and 44 integration tests pass.
Headline measurements: official drawer opens to 308.7 mm with a travel direction 0.000°
from the configured pull axis; all four probe stop conditions demonstrated; execution
profile invariant in `F_peak` and duration accurate to one control step; `F_peak = 5 N`,
`T = 2 s` gives `d(T) =` 326.1 / 141.7 / 59.6 mm on easy / medium / hard. Full table with
commands and observed values in `docs/VALIDATION.md`.

**Physical problems found and fixed** (each is a decision entry):

1. The official gripper close command squeezed the handle with ~15 N and ~30 N because the
   hand is 3.1 mm off centre. That leaked a **0.68 N bias onto the pull axis** and opened
   the drawer 14.4 mm in 2 s with *zero* commanded force. Fixed with a per-finger command
   derived from the recorded contact equilibrium (D010).
2. PhysX's drawer `joint_vel` sampled at 60 Hz reported **-0.0076 m/s while the drawer was
   opening at +0.0073 m/s** (contact-chatter aliasing), and after the grip fix went
   identically zero. Replaced by a finite-difference estimate (D009).
3. A `ContactSensor` on the handle reads only normal load, so it measured 0.22-0.37 N
   whether the pull was 4 N or 12 N. Replaced by the wrist joint reaction wrench, validated
   against `m*a + f + c*v` (D006).
4. The pull axis is force-controlled and therefore has no damping of its own, so grasp
   momentum persisted into the probe. Added an initialisation-only velocity brake plus a
   discarded warm-up episode.
5. `HybridPullOSC` stepped the *unwrapped* environment, silently bypassing `RecordVideo`
   (videos were one frame long). All stepping now goes through `HybridPullOSC.step` on the
   wrapped environment (D014).
6. The probe's max-force stop fired one control step early (9.867 N instead of 10.000 N).
7. Writing the drawer's static and dynamic joint friction in two calls made PhysX reject the
   update; they are now written together.

**Commit.** `b85d069edb6ea6033181460b2dcc655da9dae44d` on `agent/phase0-8-bootstrap`; `main` points at the same commit; tagged
`v0.1.0-phase8`.

**Push status.** Pushed. `main`, `agent/phase0-8-bootstrap` and the tag `v0.1.0-phase8`
all resolve to `de31ea9011a2db6f99993cfc19218d5ee7953fae` on
`https://github.com/ysdjy/9.2paper.git`, confirmed with `git ls-remote`. (The first attempt
failed because no GitHub credential was configured on this machine; the user configured one
and the push then succeeded.)

**Remaining issues.** See "Known limitations" in `docs/VALIDATION.md`. The ones that would
most affect the next phase:

* the held axes drift up to 15 mm / 7.5° when the `easy` drawer approaches its end stop;
* a residual 0.25 N pull-axis bias and ~1.3 mm/s creep at zero command;
* only ~40 % of the commanded force reaches the drawer during acceleration, so the useful
  `F_peak` range for a 2 s execution is roughly 2-8 N.

**Next.** Training-data generation: sample `xi`, run probe then execution, and store paired
`(probe history, F_peak, T, d(T))` episodes. The controller APIs should not need to change.

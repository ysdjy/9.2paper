# Validation

Every entry below was produced on this machine on **2026-09-02** (Isaac Sim 5.1.0.0,
Isaac Lab 2.3.0, RTX 5080). Observed values are copied from the logged episodes under
`outputs/logs/` and from the commands' own output — none is estimated.

Environment used for every controller test: `Probe-Drawer-Franka-v0`
(`probe_drawer.envs.ProbeDrawerEnvCfg`), `step_dt = 0.016667 s`.

---

## Phase 1 — installation inspection

| | |
|---|---|
| Command | `python scripts/inspect_isaaclab.py` |
| Expected | Isaac Sim and Isaac Lab importable; all four `Isaac-Open-Drawer-Franka-*` IDs plus `Isaac-Franka-Cabinet-Direct-v0` registered |
| Observed | Isaac Sim 5.1.0.0, Isaac Lab 0.49.0 (repo 2.3.0), Python 3.11.15, torch 2.7.1+cu128, CUDA 12.8, RTX 5080; all five IDs `true` |
| Result | **PASS** |
| Artefact | `outputs/logs/isaaclab_inspection.json` |

## Phase 2 — official drawer environment

### 2a. Official state machine script runs

| | |
|---|---|
| Command | `./isaaclab.sh -p scripts/environments/state_machine/open_cabinet_sm.py --num_envs 8 --headless` |
| Expected | runs without error (the script loops forever and prints nothing) |
| Observed | ran 240 s, no errors; terminated by us. The carb assertion in the log is SIGTERM landing during shutdown, at t = 239.9 s |
| Result | **PASS** |

### 2b. Full approach -> grasp -> open, with measurements

| | |
|---|---|
| Command | `python scripts/run_official_drawer.py --num_envs 4 --headless` |
| Expected | the documented phase sequence; the drawer opens; the travel direction is axis-aligned with the configured pull axis |
| Observed phases | `REST -> APPROACH_INFRONT_HANDLE -> APPROACH_HANDLE -> CLOSE_GRIPPER -> SETTLE -> OPEN_DRAWER`, grasp complete at t = 3.983 s |
| Observed displacement | 308.55, 308.42, 309.16, 307.92 mm (4 environments) |
| Observed travel direction | `(-1.0, 0.0, 0.0)` in world/base frame, all environments |
| Angle from configured pull axis `-x` | **0.000°** (tolerance 5°) |
| Result | **PASS** |
| Artefact | `outputs/logs/official_drawer_validation.json`, `outputs/videos/official_open_drawer-step-0.mp4` (9.02 s) |

### 2c. Grasp configuration export

| | |
|---|---|
| Command | `python scripts/run_official_drawer.py --num_envs 1 --headless --deterministic-init --export-grasp-pose` |
| Expected | a reproducible grasped arm configuration written to `configs/grasp_pose.yaml` |
| Observed | captured at t = 3.983 s; TCP minus handle `(10.6, -0.1, 3.2) mm`; finger equilibria 7.66 mm / 15.21 mm (asymmetry 7.55 mm) |
| Result | **PASS** |

---

## Phase 6 — Probe controller

### Probe Test 1 — force ramp

| | |
|---|---|
| Command | `python scripts/test_probe_pull.py --headless --preset medium --experiment-id probe_displacement_stop` |
| Expected | commanded force rises monotonically from 2 N towards 10 N at 8 N/s |
| Observed | 2.000 N -> 6.133 N over 0.533 s; monotone non-decreasing; fitted rate matched `(10-2)/1.0 = 8 N/s` to within 2 % (integration test `test_force_rate_matches_the_configured_ramp`) |
| Result | **PASS** |
| Plot | `outputs/plots/probe_displacement_stop_force.png` |

### Probe Test 2 — displacement stop

| | |
|---|---|
| Command | as above |
| Expected | `termination_reason = displacement_reached`, final displacement ≥ 5 mm and within one control step of it |
| Observed | `displacement_reached`, **5.292 mm** at 0.533 s, final velocity 30.96 mm/s, final command 6.133 N |
| Result | **PASS** |

### Probe Test 3 — velocity stop

| | |
|---|---|
| Command | `python scripts/test_probe_pull.py --headless --preset easy --max-velocity 0.004 --target-displacement 0.2 --experiment-id probe_velocity_stop` |
| Expected | `termination_reason = velocity_limit`, final velocity ≥ 4 mm/s |
| Observed | `velocity_limit`, final velocity **5.05 mm/s** at 0.100 s, displacement 0.28 mm, `reached_target = False` |
| Result | **PASS** |

### Probe Test 4a — max-force stop

| | |
|---|---|
| Command | `python scripts/test_probe_pull.py --headless --preset hard --target-displacement 0.35 --max-velocity 5.0 --experiment-id probe_force_stop` |
| Expected | `termination_reason = max_force_reached` with the command exactly at `max_force` |
| Observed | `max_force_reached`, final command **10.000 N** at 1.017 s (= `ramp_duration + step_dt`), displacement 27.0 mm |
| Result | **PASS** |

### Probe Test 4b — timeout stop

| | |
|---|---|
| Command | `python scripts/test_probe_pull.py --headless --preset hard --ramp-duration 5.0 --max-probe-duration 0.5 --target-displacement 0.35 --max-velocity 5.0 --experiment-id probe_timeout_stop` |
| Expected | `termination_reason = timeout` at 0.5 s; the probe never runs forever |
| Observed | `timeout` at **0.500 s**, displacement 0.03 mm, command 2.773 N |
| Result | **PASS** |

### Probe Test 5 — five held degrees of freedom

| | |
|---|---|
| Expected | y, z and orientation hold while the pull axis moves freely |
| Observed (probe episodes) | peak lateral drift 0.115-0.656 mm; peak orientation drift 0.031-0.311° |
| Observed (`probe_displacement_stop`) | 0.514 mm, 0.232° |
| Integration bound asserted | < 2 mm and < 2° |
| Result | **PASS** |
| Plot | `outputs/plots/probe_displacement_stop_lateral_error.png` |

### Probe history integrity

| | |
|---|---|
| Expected | every signal recorded every step, finite, time advancing by exactly one control step; `measured_force` distinct from `commanded_force` |
| Observed | all 16 signals present and finite; `diff(time) == step_dt` to 1e-9; `max|commanded - measured|` > 0.1 N in every episode |
| Result | **PASS** (integration tests `TestHistory`) |

---

## Phase 6 — Execution controller

### Execution Test 1 — force profile invariance

| | |
|---|---|
| Command | `python scripts/test_execution_pull.py --headless --preset hard --peak-force {3,5,7} --duration 2.0` |
| Expected | `F(t)/F_peak` identical for every `F_peak` |
| Observed | normalised curves coincided to `atol = 1e-5` across 3, 5 and 7 N (integration test `test_normalised_shape_is_the_same_for_every_peak_force`); plateau reached `1.0000` in all runs; first command exactly 0, last command 1.97 % of peak (zero-order hold at `T - step_dt`) |
| Result | **PASS** |
| Plots | `outputs/plots/execution_profile_invariance.png`, `execution_hard_F3_force.png`, `execution_hard_F7_force.png` |

### Execution Test 2 — duration

| | |
|---|---|
| Command | `python scripts/test_execution_pull.py --headless --preset medium --peak-force 5.0 --duration {1.0,2.0}` |
| Expected | the commanded duration is executed, and it does not depend on displacement |
| Observed | `duration = 2.000 s` / 120 steps and `1.000 s` / 60 steps, both `duration_completed`; `easy` (317.9 mm) and `hard` (59.8 mm) both executed 2.000 s exactly |
| Result | **PASS** |

### Execution Test 3 — no goal feedback

| | |
|---|---|
| Command | `python -m pytest tests/unit/test_execution_has_no_goal_feedback.py -q` |
| Expected | no `d_goal` / `goal` / `target_displacement` / `epsilon` / `success` identifier anywhere in the execution controller's code; `run` takes only `peak_force` and `duration`; `_stop_conditions` returns nothing and reads no state |
| Observed | 9 assertions pass |
| Result | **PASS** |

### Execution Test 4 — different dynamics, same command

See Phase 8 below.

### Safety termination

| | |
|---|---|
| Expected | an absolute violation aborts; the same run completes under the project limits; a profile above the force cap is refused |
| Observed | with `max_drawer_velocity = 0.02 m/s` the `easy` execution aborted with `safety_abort` before 2 s; with the project limits the identical run completed (`duration_completed`); `peak_force = 61 N` raised `ValueError: ... above the absolute safety limit`; a 30 N pull produced no non-finite state |
| Result | **PASS** (integration tests `TestSafety`) |

### Execution held-axis stability

| | |
|---|---|
| Observed | `medium` at 5 N: 0.659 mm / 0.322°. `hard` at 5 N: 0.362 mm / 0.160°. `hard` at 7 N: 0.650 mm / 0.405° |
| Observed | `easy` at 5 N: **14.4 mm / 7.50°** — see Known limitations |
| Result | **PASS** with the caveat recorded below |

---

## Phase 8 — hidden dynamics

### Presets

Calibrated by sweeping candidates through `ExecutionPullController` at the reference
operating point `peak_force = 5 N`, `duration = 2 s`, and keeping the triple that separates
cleanly without reaching the drawer's 0.4 m travel limit.

| Preset | `drawer_mass` (kg) | `joint_friction` | `joint_damping` (N s/m) | `joint_stiffness` |
|---|---|---|---|---|
| `nominal` | 5.175 (official) | 0.0 | 1.0 (official) | 0.0 |
| `easy` | 5.0 | 1.5 | 4.0 | 0.0 |
| `medium` | 8.0 | 3.0 | 6.0 | 0.0 |
| `hard` | 10.0 | 4.0 | 9.0 | 0.0 |

### Parameters reach the intended simulation quantities

| | |
|---|---|
| Command | `python scripts/test_dynamics_randomization.py --headless` |
| Expected | every requested value reads back out of PhysX; the target is the *top drawer*; other cabinet joints untouched |
| Observed readback | mass `[5.0, 8.0, 10.0]`, static friction `[1.5, 3.0, 4.0]`, dynamic friction `[1.5, 3.0, 4.0]`, damping `[4.0, 6.0, 9.0]`, stiffness `[0.0, 0.0, 0.0]`; `consistent = True` |
| Observed target | `drawer_joint = drawer_top_joint`, `drawer_body = drawer_top`; handle mass 0.1486 kg, total moving mass `[5.149, 8.149, 10.149]` kg |
| Observed | `drawer_bottom_joint`, `door_left_joint`, `door_right_joint` friction still 0.0 |
| Result | **PASS** |

### Same `(F_peak, T)`, different response

`peak_force = 5 N`, `duration = 2 s`, all three presets in parallel environments of one
simulation, identical commanded force (asserted).

| Preset | `d(T)` (mm) | peak \|velocity\| (m/s) | final velocity (m/s) | peak measured force (N) |
|---|---|---|---|---|
| `easy` | **326.08** | 0.4183 | 0.1682 | 17.58 |
| `medium` | **141.73** | 0.1321 | 0.1171 | 5.41 |
| `hard` | **59.62** | 0.0542 | 0.0397 | 5.54 |

Consecutive `d(T)` ratios **2.301** and **2.377** (required ≥ 1.5); monotone
`easy > medium > hard`. **PASS**

Plots: `outputs/plots/dynamics_execution_presets_displacement.png`,
`dynamics_execution_presets_force.png`, `dynamics_execution_presets_velocity.png`.

### The standardised probe distinguishes them

`initial_force = 2 N`, `max_force = 10 N`, `target_displacement = 5 mm`,
`max_velocity = 0.05 m/s`, same parallel environments.

| Preset | probe duration (s) | final command (N) | `d` at stop (mm) | reason |
|---|---|---|---|---|
| `easy` | **0.350** | 4.667 | 5.59 | `displacement_reached` |
| `medium` | **0.533** | 6.133 | 5.13 | `displacement_reached` |
| `hard` | **0.650** | 7.067 | 5.08 | `displacement_reached` |

Both probe duration and the force needed increase strictly with difficulty. **PASS**

Plots: `outputs/plots/dynamics_probe_presets_force.png`,
`dynamics_probe_presets_displacement.png`, `dynamics_probe_presets_velocity.png`.

### Random sampling

`DynamicsRandomizer(seed=0).sample(3)` -> mass `[7.970, 5.056, 7.921]`, friction
`[3.957, 1.883, 4.534]`, damping `[3.619, 7.439, 6.189]`; applying sampled parameters read
back consistent. Seeded sampling is reproducible (unit test). **PASS**

---

## Phase 9 — hidden state, observations, audits and the Oracle sweep

All commands were run on 2026-09-02. Reports referenced below are in `outputs/logs/`.

### 9A — baseline re-validation

| | |
|---|---|
| Command | `python -m pytest tests -q` on the Phase 8 tree |
| Observed | 124 passed; working tree clean; branch created from `4b0a815` |
| Result | **PASS** |

### 9B — four-dimensional hidden state

| Check | Command / method | Observed | Result |
|---|---|---|---|
| static and dynamic friction are independent channels | write `(5, 1)`, read `get_dof_friction_properties` | data `(5.0, 1.0)`, PhysX `[5.0, 1.0, 0.0]` | **PASS** |
| `mu_s < mu_d` is rejected | write `(1, 5)` | PhysX logs `Static friction effort must be greater than or equal to dynamic friction effort` and **keeps** `[5.0, 1.0]` while `data` reports `(1.0, 5.0)` | **PASS** (finding, D016) |
| readback comes from the simulator | `scripts/test_dynamics_randomization.py --headless` | mass `[5.0, 8.0, 10.0]`, static `[1.5, 3.0, 4.0]`, dynamic `[1.5, 3.0, 4.0]`, damping `[4.0, 6.0, 9.0]`; `consistent=True`, `mirror_agrees=True` | **PASS** |
| other joints untouched | same | `drawer_bottom_joint`, `door_left_joint`, `door_right_joint` static and dynamic friction still 0.0 | **PASS** |
| the split produces different behaviour | `scripts/plot_probe_identifiability.py` | `mu_s` 0.5 -> 3.0 N moved breakaway from 0.150 s / 1.67 N to 0.400 s / 2.92 N; `mu_d` 0.15 -> 1.25 N moved probe duration from 0.350 to 0.467 s | **PASS** |

### 9C — hidden-state capability audit

| | |
|---|---|
| Command | `python scripts/audit_hidden_states.py --headless` |
| Expected | every candidate probed by write, readback and restore |
| Observed | 15 candidates; **15 writable, 15 read back correctly**; roles: 4 in `xi`, 3 held fixed, 2 OOD candidates, 6 unsuitable |
| Result | **PASS** — table in `docs/HIDDEN_STATE_AUDIT.md` |

### 9D — observation expansion

| Check | Observed | Result |
|---|---|---|
| channel count | `PullHistory` went from 16 to **25** channels | **PASS** |
| registry agreement | `HISTORY_CHANNELS`, `PullHistory` fields, `OBSERVATION_SPECS` keys and the recorder's sampler all describe the same set (unit test) | **PASS** |
| derivatives are causal | `CausalDerivative` uses only current and past samples; a moving average over 2 steps recovers exactly 2.0 from a 3/1 alternating ramp | **PASS** |
| filter metadata recorded | method, window and lag recorded per channel and per episode | **PASS** |
| no privileged channel in the model input | `validate_model_input(DEFAULT_ACE_INPUT)` passes; adding either drawer force channel raises | **PASS** |

### 9E — force-channel audit

| | |
|---|---|
| Command | `python scripts/audit_force_channels.py --headless` |
| Resistance identity | `get_dof_projected_joint_forces` equals `-(mu_d*sign(v) + b*v)`: worst mean residual over six cases **0.0099 N** against 2-3 N forces (tolerance 0.05) | **PASS** |
| Delivered force vs wrist | `m*a - resistance` agrees with the wrist wrench to within **0.049-0.134 N**, against a hand-and-finger inertial bound of 0.342 N | **PASS** |
| Command share | `F_delivered / F_cmd` = 0.40 (free) to 0.80 (heavy) — the arm's own inertia absorbs the rest | recorded |
| Joint reaction wrench | `get_link_incoming_joint_force` reads exactly 0.000 along a prismatic joint's own axis — structural, not a bug | recorded |
| The Phase 8 anomaly | a deliberate end-stop episode reached **59.2 N wrist force, 9.87x the 6 N command, at 84.4 % of travel** | **explained** |

Full table in `docs/FORCE_CHANNEL_AUDIT.md`.

### 9F — execution snapshot and zero-force cleanup

| Check | Observed | Result |
|---|---|---|
| pull force is zero after `T` | commanded pull force `0.0` after the run | **PASS** |
| the cleanup is what zeroes it | with `zero_force_cleanup_steps=0` a residual of under 3 % of peak remains | **PASS** (control) |
| cleanup absent from the history | `history.num_steps == T / dt`, `history.time[-1] == T` | **PASS** |
| snapshot predates the cleanup | `final_displacement` equals the last history sample, and the drawer has coasted *further* by the time `run` returns | **PASS** |
| step count configurable and recorded | 3 + 4 steps -> `post_execution_steps_excluded_from_result = 7` | **PASS** |
| next episode inherits nothing | first commanded force of the next run is 0.0 | **PASS** |
| `peak_velocity` reported | equals `max|drawer_velocity|` and grows with force | **PASS** |

### 9G — TaskEvaluator

| Case | Expected | Result |
|---|---|---|
| on goal, at rest | PASS | **PASS** |
| on goal, still moving | FAIL on velocity | **PASS** |
| off goal, at rest | FAIL on position | **PASS** |
| on goal and at rest, but safety-aborted | FAIL | **PASS** |
| on goal and at rest, but outside the valid region | FAIL | **PASS** |

Exercised on synthetic episodes, which is the only way to produce these combinations on
demand. 40 unit tests in `tests/unit/test_task_evaluator.py`.

### 9H — execution sweep

| Stage | Command | Rows | Valid | Time |
|---|---|---|---|---|
| coarse | `sweep_execution_space.py --headless --stage coarse` | 495 (11 xi x 9 F x 5 T) | 238 (48.1 %) | 85 s |
| fine, `fall=0.10` | `--stage fine --forces 1.0..6.0 --durations 1.0 1.5` | 4536 (108 xi x 21 F x 2 T) | 3798 (83.7 %) | 359 s |
| fine, `fall=0.15` | as above `--fall-fraction 0.15` | 4536 | 3816 (84.1 %) | 326 s |
| fine, `fall=0.20` | as above `--fall-fraction 0.20` | 4536 | 3842 (84.7 %) | 365 s |
| fine, `fall=0.30` | as above `--fall-fraction 0.30` | 4536 | 3890 (85.8 %) | 360 s |
| fine, `fall=0.35` | as above `--fall-fraction 0.35` | 4536 | 3912 (86.2 %) | 352 s |

**23 175 episodes in total.** The identical command is applied to every hidden state at each
point (asserted by an integration test), so a difference between rows can only come from `xi`.

### 9I — valid operating region

| | |
|---|---|
| Thresholds | each anchored to a Phase 8 measurement; table in `docs/EXPERIMENT_SPACE.md` |
| Coarse region | forces 3-5 N keep 82-100 % of hidden states usable at `T = 1.0-1.5 s`; nothing is usable above 8 N for `T >= 1 s` |
| Dominant rejections (fine, `fall=0.20`) | no measurable motion 432, excessive velocity 242, lateral drift 141, orientation drift 113, mechanical limit 1, safety abort 1 |
| Phase 8 drift question | **answered: it is the operating point.** Drift stays under 1 mm across the whole force range at moderate displacement and speed, and rises by an order of magnitude only near the end stop | **PASS** |

### 9J/9K — Oracle landscape and task selection

| | |
|---|---|
| Command | `python scripts/build_oracle_landscape.py` |
| Candidates | 840 scored (5 ramp-downs x 2 durations x 7 goals x 4 position tolerances x 3 velocity tolerances), **22 accepted** |
| Recommended | `fall_fraction = 0.20`, `T = 1.5 s`, `d_goal = 50 mm`, `eps_d = 15 mm`, `eps_v = 0.08 m/s` |
| Coverage | **106 / 108** hidden states have a succeeding force |
| Required force | **1.00 - 4.50 N**, median 2.25 N — a **4.5x range** |
| Median success band | **0.50 N** (0.164 relative), **105 / 106 contiguous** |
| Largest travel fraction of a success | **0.162** |
| Result | **PASS** — one force cannot serve every drawer |

### 9L — probe re-calibration

| | |
|---|---|
| Command | `python scripts/calibrate_probe.py --headless` |
| Recommended | `initial_force = 1.0 N`, `max_force = 6.0 N`, `target_displacement = 3 mm`, `max_velocity = 0.08 m/s`, `ramp_duration = 1.0 s`, budget `1.5 s` |
| Coverage | **1.00** — every one of the 108 hidden states breaks away |
| Terminations | `displacement_reached` for all 108 |
| Intrusion | probe travels 3.44 mm, **6.9 %** of the 50 mm goal |
| Median probe duration | **0.467 s** |
| Best feature correlation with the required force | `displacement_per_newton`, **\|rho\| = 0.969** (ceiling across candidates 0.978) |
| `duration` / `final_commanded_force` | \|rho\| = 0.968 |
| `breakaway_time` / `breakaway_force` | \|rho\| = 0.945 |
| Result | **PASS** — a single standardised probe carries the information |

Identifiability, one dimension at a time (`outputs/plots/probe_identifiability.png`):

| Varied | Probe response |
|---|---|
| `mu_s` 0.5 -> 3.0 N | breakaway 0.150 -> 0.400 s at 1.67 -> 2.92 N; duration 0.350 -> 0.600 s — **strong** |
| `mu_d` 0.15 -> 1.25 N | breakaway 0.167 -> 0.267 s; duration 0.350 -> 0.467 s — **clear** |
| `m` 4 -> 12 kg | peak acceleration 0.133 -> 0.078 m/s²; breakaway essentially unchanged — **as predicted** |
| `b` 2 -> 11 N s/m | duration 0.400 -> 0.400 s, breakaway 1.92 -> 1.92 N — **not identified** (see Known limitations) |

## Test suites

| Suite | Command | Count | Time | Result |
|---|---|---|---|---|
| Unit (no Isaac Sim) | `python -m pytest tests/unit -q` | 196 | 1.0 s | **196 passed** |
| Integration (launches Isaac Sim once) | `python -m pytest tests/integration -q` | 52 | 74 s | **52 passed** |

Phase 9 added 116 unit tests and 8 integration tests. New unit files cover the observation
registry, the evaluator, the sweep and Oracle acceptance logic, causal differentiation, the
probe features and the selected experiment plan.

---

## Artefacts

### Plots (`outputs/plots/`)

Phase 9 figures, each carrying one step of the argument:

| Figure | File | What it shows |
|---|---|---|
| Hidden-state identifiability | `probe_identifiability.png` | probe force, displacement, velocity and acceleration with one hidden dimension varied at a time |
| `(F_peak, T)` surfaces | `experiment_space_surfaces.png` | `d(T)` and `v(T)` for four hidden states, with the goal and `eps_v` marked |
| Valid operating region | `experiment_space_validity.png` | usable fraction over `(F_peak, T)`, plus why points are rejected |
| **Oracle success landscape** | `oracle_success_landscape.png` | the labels themselves: invalid / valid-but-misses / success |
| **Success force bands** | `oracle_force_intervals.png` | per-hidden-state bands, ordered by required force -- the figure that shows one force cannot serve all |
| Drift vs operating point | `execution_drift_vs_operating_point.png` | drift against force, displacement and speed; answers the Phase 8 question |
| Force channels | `force_channel_comparison.png` | all four channels on a clean execution and on an end-stop impact |

Phase 6-8 figures:

| Figure | File |
|---|---|
| Probe force vs time | `probe_displacement_stop_force.png` |
| Probe displacement vs time | `probe_displacement_stop_displacement.png` |
| Probe velocity vs time | `probe_displacement_stop_velocity.png` |
| easy/medium/hard displacement, same execution | `dynamics_execution_presets_displacement.png` |
| easy/medium/hard force, same execution | `dynamics_execution_presets_force.png` |
| profile invariance in `F_peak` | `execution_profile_invariance.png` |

### Videos (`outputs/videos/`)

| File | Duration | Content |
|---|---|---|
| `official_open_drawer-step-0.mp4` | 9.02 s | official IK-Abs approach, grasp and motion-driven opening |
| `probe_easy-step-60.mp4` | 0.87 s | settle plus probe on `easy` |
| `probe_hard-step-60.mp4` | 1.18 s | settle plus probe on `hard` |
| `execution_easy-step-60.mp4` | 2.50 s | settle plus 5 N / 2 s execution on `easy` |
| `execution_hard-step-60.mp4` | 2.50 s | settle plus 5 N / 2 s execution on `hard` |

### Episode logs and datasets (`outputs/logs/`)

14 logged episodes, each a directory with `metadata.json` (versions, git commit, controller
parameters, hidden dynamics, per-environment result) and `trajectory.npz` (now 25 signals).

Phase 9 datasets and reports:

| File | Contents |
|---|---|
| `hidden_state_audit.json` | 15 candidates, their APIs, roles and measured round trips |
| `force_channel_audit.json` | the six force-audit cases and the end-stop episode |
| `sweep_execution_coarse.json` | 495 rows, 11 hidden states x 9 forces x 5 durations |
| `sweep_fine_fall{010,015,020,030,035}.json` | 4536 rows each: 108 hidden states x 21 forces x 2 durations |
| `oracle_landscape.json` | 840 scored task candidates, the 22 accepted, and the recommendation |
| `probe_calibration.json` | 7 probe candidates over the 108-point grid, with per-feature correlations |

Sweep rows carry the hidden state, the command, the response and the validity verdict but
not trajectories: any row is reproducible from its own `(xi, F_peak, T)`.

---

## Failed or unresolved checks

None outstanding. Three checks failed during development and were fixed rather than
weakened; recorded here because the fixes are load-bearing:

| Check | Failure | Cause | Fix |
|---|---|---|---|
| probe max-force stop | stopped at 9.867 N, not 10.000 N | the stop condition compared `elapsed` instead of the time the command was *issued*, i.e. one step late | D007; assertion kept at `rel=1e-3` |
| probe force monotonicity | last recorded command was 0 | environments that stop early are zero-padded to the longest one; the test was reading the padding | added `PullHistory.active` and `active_steps()`, which downstream analysis needs anyway |
| execution safety abort | a 30 N pull did not trip any limit | high-force behaviour is chaotic, so no particular limit is guaranteed to trip | test now trips a deliberately tight limit deterministically, plus a control test and a finiteness test at 30 N |

---

## Known limitations

### Resolved in Phase 9

* **Held-axis drift** was an open question after Phase 8. It is the operating point, not the
  controller: drift stays under 1 mm across the whole force range at moderate displacement
  and speed (`docs/EXPERIMENT_SPACE.md` section 2). No controller change is needed.
* **The 17-23 N wrist force** is the mechanical end stop. A deliberate episode reached
  59.2 N at 84.4 % of travel. The validity mask already excludes that regime.
* **`Articulation.data` friction buffers mirror the request, not the simulator.** The old
  `consistent` check could have passed on a write PhysX discarded. Readback now comes from
  `root_physx_view` (D016).

### Outstanding

1. **The calibrated probe does not identify damping.** Sweeping `b` from 2 to 11 N s/m left
   the probe duration and breakaway force unchanged to three significant figures, because the
   probe stops at 3 mm where the drawer is moving at only ~0.013 m/s and `b*v` is ~0.14 N
   against a 2 N command. The required peak force also depends only weakly on `b`, so the task
   stays predictable — but the probe identifies three of the four dimensions, not four.
   A second, higher-speed probe phase would be needed to see `b`.

2. **`epsilon_d = 15 mm` is 30 % of the goal**, the loosest this project's own acceptance
   rule permits. It is set by the 0.25 N force-grid resolution rather than by physics: at
   `dd/dF = 44 mm/N` a 0.125 N grid would support roughly 7.5 mm. Refining the grid is the
   next improvement if a tighter task is wanted.

3. **Two of 108 hidden states have no succeeding force** — both `mu_s = 3.0`, `mu_d = 0.9`,
   `b = 2.0`, i.e. hard to start and then nothing to slow them, so they overshoot on the force
   that started them. A real corner of the range, and a reason to keep `mu_d/mu_s` bounded
   below at 0.3.

4. **Probe and execution are separate episodes.** Every result here resets between them. The
   probe travels 6.9 % of the goal so a sequential protocol looks plausible, but it has not
   been measured and `d_goal` would need interpreting relative to the post-probe position.

5. **Held-axis drift on the `easy` preset at large travel.** At `F_peak = 5 N`, `T = 2 s`
   the `easy` drawer travels 318-326 mm at up to 0.42 m/s and approaches its 0.4 m end
   stop; the TCP lateral drift reaches 14-15 mm and the orientation drift 7.5°, against
   0.36-0.66 mm and 0.16-0.41° for `medium` and `hard`. Still well inside the 50 mm / 30°
   safety limits, but the hybrid hold degrades in that regime. Keep validation operating
   points away from the end stop.

6. **Residual pull-axis bias of about 0.25 N.** The gripper is not perfectly centred on the
   handle, so the balanced grip still leaks ~0.25 N onto the pull axis and the drawer
   creeps ~1.3 mm/s with zero commanded force (2.5 mm over 2 s). It is deterministic and
   identical in every episode, and it is visible in every logged history. See D010.

7. **`measured_force` is noisy.** The wrist reaction wrench swings by a few newtons at 5 N
   command level in contact-rich phases. The mean matches `m*a + f + c*v` to about 10 %.
   Downstream consumers should expect to filter. See D006.

8. **Only about 40 % of the commanded force reaches the drawer during acceleration.** The
   arm's own reflected inertia (~9 kg along the pull axis) absorbs the rest. This is real
   physics rather than a defect, but it means `F_peak` and the force delivered to the drawer
   are not interchangeable, and the operating range for a 2 s execution is roughly 2-8 N.

9. **The 0.4 m drawer travel limit saturates above about 8 N.** At `F_peak = 12 N` all three
   presets reach 313-354 mm and the preset separation collapses. The reference operating
   point is therefore 5 N rather than the 12 N originally suggested.

10. **`nominal` is a reference, not a fourth difficulty level.** With zero joint friction the
   drawer reaches the probe's 5 mm target in about 0.2 s at barely above `initial_force`,
   which makes it a poor discriminator.

11. **PhysX's own drawer `joint_vel` is unusable at this control rate** and is only logged for
   transparency. See D009.

---

## Phase 10 — the sequential protocol

All numbers below are observed values from the runs named, not targets. Reports:
`outputs/logs/sequential_protocol_validation.json`, `sequential_oracle_fall035.json`,
`task_refinement.json`, `reset_vs_sequential.json`.

### Seq Test 1 — nothing is reset after the probe

`scripts/validate_sequential_protocol.py --headless --transition-steps 0 2 4 8 12 --repeats 6`,
preset `medium` (m = 8 kg, µ_s = µ_d = 3.0, b = 6 N·s/m), F_peak = 2.5 N, T = 1.5 s, 8 envs.

| gap (steps) | probe `d` (mm) | coast in gap (mm) | `v` at execution start (m/s) | `d_total` (mm) |
|---|---|---|---|---|
| 0 | 3.454 | +0.000 | +0.01150 | 7.12 |
| 2 | 3.728 | +0.190 | +0.00571 | 6.59 |
| 4 | 3.668 | +0.099 | +0.00001 | 6.06 |
| **8** | 3.331 | +0.413 | +0.00000 | 5.99 |
| 12 | 3.352 | +0.531 | +0.00000 | 5.54 |

At every gap: `position kept = True`, `probe left it moving = True`,
`velocity decayed not erased = True`, `total = parts = True`. **Pass.**

### Seq Test 2 — the velocity decays, it is not written

Velocity retained across the gap, as a fraction of the probe-end value: **1.000** at 0 steps,
**0.539** at 2, **0.001** at 4, **0.000** at 8, **0.000** at 12. A smooth decay with a
physical time constant, which is what a coast under friction and damping looks like. An
erasure would show as a step to zero at whatever step the write happened. **Pass.**

### Seq Test 3 — the gap length, by measurement

Repeatability at the operating point, F = 4.25 N, six identical episodes per gap:

| gap | mean `d_total` (mm) | `d_total` spread (mm) | post-probe `v` spread (mm/s) | `v(T)` spread |
|---|---|---|---|---|
| 0 | 51.235 | 3.584 | 4.793 | 0.00249 |
| 2 | 48.669 | 2.614 | 9.879 | 0.00197 |
| 4 | 48.091 | 3.285 | 7.362 | 0.00298 |
| **8** | 47.304 | **0.898** | 0.128 | 0.00107 |
| 12 | 47.796 | 1.402 | 0.010 | 0.00245 |

An earlier run over gaps 4, 8 and 12 only gave spreads of 1.656, **1.143** and 1.402 mm. The
absolute values move by a few tenths of a millimetre between runs — six episodes is a small
sample — but 8 steps is the minimum in both. Recorded floor: **0.9–1.1 mm**, which is the
lower bound on any position tolerance. **Pass.**

### Seq Test 4 — candidate fairness

Post-probe spread across environments sharing one hidden state and one probe:

| gap | displacement spread (µm) | velocity spread (mm/s) | probe duration spread (s) |
|---|---|---|---|
| 0 | 420.9 | 10.690 | 0.0500 |
| 2 | 279.9 | 6.614 | 0.0333 |
| 4 | 323.6 | 7.676 | 0.0500 |
| **8** | **244.8** | 0.109 | 0.0167 |
| 12 | 175.5 | 0.060 | 0.0333 |

`fair = True` at every gap. Within a *single* environment, repeating the same probe gives
264–464 µm of spread — the same magnitude — so the variability is intrinsic to the probe's
displacement stopping rule (which can be crossed a step early or late), not an artefact of
parallel environments. 244.8 µm is 3 % of `ε_d = 7.5 mm`. **Pass.**

### Seq Test 5 — the task counts the probe

`tests/integration/test_sequential_protocol.py`, 17 tests, all passing. The load-bearing one
is `test_ignoring_the_probe_would_change_the_label`: with the probe's displacement counted the
episode passes, without it the same episode fails. Also asserted: `total = pre_execution +
execution` to 1e-12; the probe history ends at the probe's own duration and the execution
history at exactly `T`, so the gap is in neither; a settling execution raises `ValueError`;
`ExecutionPullController.run`'s signature is still exactly `(peak_force, duration)`.

### Seq Test 6 — the Oracle

`scripts/build_sequential_oracle.py --headless`, plus low-force supplements merged on exact
force equality.

| | fall = 0.20 | fall = 0.30 | fall = 0.35 |
|---|---|---|---|
| rows | 4428 | 5616 | 5616 |
| valid | 94.6 % | 96.7 % | **97.2 %** |
| forces | 1.00–5.00 N (41) | 0.15–5.00 N (52) | 0.15–5.00 N (52) |
| coverage at the selected task | 0.71 | 0.954 | **0.972** |

Selected: `T = 1.5 s`, `d_goal = 40 mm`, `ε_d = 7.5 mm`, `ε_v = 0.03 m/s`, `fall = 0.35`.
Measured at that point: coverage **0.972** (105/108), required force **0.20–4.30 N** (median
1.50 N, a 21.5× range), median band 0.20 N (0.14 relative), 100 of 105 bands contiguous, grid
step 0.05 N resolving the band, max travel 0.119 of the drawer's range, discrimination 2.70.

Three hidden states have no succeeding force: `[4.0, 2.0, 0.6, 2.0]`, `[4.0, 3.0, 0.9, 2.0]`,
`[4.0, 3.0, 0.9, 6.0]`.

### Seq Test 7 — what the reset was hiding

`scripts/compare_reset_vs_sequential.py`, on the Phase 9 task (both force grids express it),
108 hidden states:

| | reset | sequential |
|---|---|---|
| coverage | 1.000 | 1.000 |
| median success band | 0.50 N | 0.60 N |

Required force under the sequential protocol, relative to the reset: median ratio **0.800**
(median shift **−0.45 N**), per-state ratios **0.32 to 1.02**, largest single shift **1.40 N**.
Rank correlation of the required force between protocols: **+0.9522**. The ratio distribution
is bimodal (clusters near 0.55–0.65 and near 0.95–1.00), so the difference is not a rescaling.

### Seq Test 8 — probe features against the sequential answer

105 hidden states with a succeeding force. Strongest: `displacement_per_newton`, Spearman
**−0.9097**, Pearson −0.8413. Then `final_commanded_force` and `duration` at +0.9017/+0.8985,
`breakaway_time` and `breakaway_force` at +0.8799/+0.8615, `mean_speed_after_breakaway` at
−0.7709, `final_velocity` −0.6751, `final_displacement` −0.4928, `peak_acceleration` −0.4061.

Down from 0.969 against the reset Oracle, still strong. **Not sufficient**: at a given
`displacement_per_newton` the residual spread in required force reaches roughly ±0.3 N in the
mid-range, wider than the 0.20 N success band (figure G).

### Seq Test 9 — test suite

`python -m pytest tests/unit -q` → **233 passed**.
`python -m pytest tests/integration -q` → **69 passed** (launches Isaac Sim, 106 s).

### A bug this phase found in its own analysis

`scripts/refine_task_space.py` globbed `sequential_oracle_fall*.json` and excluded supplements
with `stem.endswith("_low")`, which does not match `..._vlow`. The two `_vlow` supplements were
therefore loaded as full datasets, and because the tolerance curves were keyed by
`f"fall={fall_fraction:g}"`, the curves for fall = 0.30 and 0.35 ended up computed from a
540-row supplement spanning only 0.15–0.35 N — reporting coverage 0.06 where the selection
said 0.972.

The **selection was not affected** (candidates are scored per dataset, and the merged file won
on coverage), but figure D was wrong and contradicted the selection, which is how it was
caught. Fixed with an exact-match regex; re-running left the selection identical.

---

## Phase 11 — dataset generation, training, deployment

All numbers are observed values from the runs named. Reports:
`outputs/logs/branching_*.json`, `outputs/dataset_v0/audit.json`,
`outputs/training/run_v0/{comparison.json,closed_loop_seed0.json}`.

### P11 Test 1 — snapshot restore fidelity

`scripts/validate_branching.py --headless`, presets `medium` and `hard`. After a full 4 N
execution had moved the drawer 34–36 mm, restoring left, on every readable quantity:

| quantity | error |
|---|---|
| `drawer_position`, `drawer_velocity` | **0.0** |
| `arm_joint_position`, `arm_joint_velocity` | **0.0** |
| `finger_joint_position` | **0.0** |
| `tcp_pose` (derived, forward kinematics) | 0.0 – 2.4 × 10⁻⁷ |
| `tcp_pull_axis_velocity` (derived) | 0.0 – 2.4 × 10⁻⁶ |

Identical in float32 for everything the snapshot writes; one float32 ULP for what is computed
from it. **Pass.**

### P11 Test 2 — branch drift over a full candidate sweep

24 branches from one snapshot against 24 fresh full episodes at the same force:

| preset | F (N) | branch spread | fresh spread | ratio | drift over the sweep |
|---|---|---|---|---|---|
| medium | 1.0 | **23 µm** | 398–681 µm | 0.03 | +2 µm |
| medium | 2.5 | 2 892 µm | 2 070 µm | 1.40 | **−1 312 µm** (−57 µm/branch) |
| hard | 3.5 | 2 739 µm | 7 997 µm | 0.34 | −369 µm (−16 µm/branch) |

Branching is between comparable and 30× more reproducible than re-probing. The systematic
drift is real and is why the branch order is shuffled. **Pass** on both presets, all six
checks.

### P11 Test 3 — two bugs the validation caught

Before the fix, a restore left the TCP pose stale by **34.45 mm** (the execution reads its
pose reference from it), giving 22 134 µm of branch spread at 4 N and 14 511 µm of order
dependence. After: 1 901 µm and 811 µm. And 24 branches of 1.5 s exceed the 30 s episode
(38.5 s), so the environment would have auto-reset partway through every sweep. Both fixed and
regression-tested.

### P11 Test 4 — candidate budget, measured before committing

32-hidden-state pilots:

| | 24 candidates | 32 candidates |
|---|---|---|
| probes with ≥1 positive | 85.4 % | **93.8 %** |
| probes with ≥2 | 54.2 % | **72.9 %** |
| hidden states with no positive | 2/32 | **0/32** |

The two failures at 24 were grid-resolution misses, not infeasibility: one reached 40.1 mm at
2.61 N and failed only on `|v(T)| = 0.032` against 0.03, with its neighbour at 2.42 N giving
31.7 mm at a compliant 0.023 m/s. Raised to 32 (D038).

### P11 Test 5 — Dataset v0 audit

`scripts/audit_dataset.py --dataset outputs/dataset_v0`. **Nine gates, all passed.**

49 152 rows / 1 536 probes / 512 hidden states; 6.54 % positive; 0.88 % invalid; histories
6–56 steps over 41 distinct lengths; `µ_d ≤ µ_s` for all 512. Splits 359/65/88 hidden states
with positive rates 6.51 / 6.52 / 6.68 %. Force against branch position: mean correlation
**−0.0005 = 0.11 σ** over 1 536 probes.

Per hidden state, **5 of 512 (0.98 %)** have no positive in any of three repeats; 447 (87.3 %)
have one in all three.

### P11 Test 6 — padding does not change the encoder

`tests/unit/test_training_dataloader.py`. A padded batch's encoder output equals the
one-sequence-at-a-time output to `atol = 1e-6`, in any batch order, and adding a longer
sequence to the batch does not perturb the shorter ones. **Pass.**

### P11 Test 7 — the student cannot see `xi`

Corrupting `batch.xi` with `randn * 100` leaves the student's output **bit-identical** and
changes the teacher's. Structural, not conventional. **Pass.**

### P11 Test 8 — the teacher gate

Privileged `E_priv + PSP`, test split: AUROC **0.9934 / 0.9940** across seeds, AUPRC
0.914 / 0.926, selecting a succeeding force for **92.0 % / 90.8 %** of feasible probes, force
MAE 0.064 / 0.067 N. The success landscape is learnable from `xi`. **Pass** — training the
student was licensed.

### P11 Test 9 — closed-loop deployment on unseen drawers

64 test hidden states, none seen in any split. All methods share one probe via the snapshot.

| method | physical success | median `|d−goal|` | forces chosen |
|---|---|---|---|
| teacher (privileged) | **89.1 %** | 1.85 mm | 0.50–3.35 N, sd 0.71 |
| ACE + PSP | **87.5 %** | 2.25 mm | 0.45–3.30 N, sd 0.70 |
| D GRU (history → force) | 81.2 % | 1.72 mm | 0.45–3.40 N, sd 0.74 |
| A linear (one feature) | 18.8 % | 14.28 mm | 0.95–2.60 N, sd 0.45 |
| fixed force (1.31 N) | 14.1 % | 24.09 mm | — |

No invalid episodes. ACE + PSP chose 41 distinct forces across 64 drawers.

### P11 Test 10 — test suite

`python -m pytest tests/unit -q` → **383 passed** (~5 s).
`python -m pytest tests/integration -q` → **84 passed** (297 s, launches Isaac Sim).

### An error I made and corrected

Reading the two zero-positive pilot hidden states, I first concluded they were physically
infeasible — their displacement jumped from 6.6 mm to 140 mm between adjacent candidates.
Looking at the force-sorted rows instead of the summary showed one of them reaching 40.1 mm
inside the position tolerance and failing only on terminal velocity by 0.002 m/s. The claim
was wrong and the correction is recorded as D038: "no positive was observed" is not "no force
exists", and only a dense sweep can establish the latter.

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

## Test suites

| Suite | Command | Count | Time | Result |
|---|---|---|---|---|
| Unit (no Isaac Sim) | `python -m pytest tests/unit -q` | 80 | 0.9 s | **80 passed** |
| Integration (launches Isaac Sim once) | `python -m pytest tests/integration -q` | 44 | 57 s | **44 passed** |

---

## Artefacts

### Plots (`outputs/plots/`, 71 files)

Required figures:

| Figure | File |
|---|---|
| 1. Probe force vs time | `probe_displacement_stop_force.png` |
| 2. Probe displacement vs time | `probe_displacement_stop_displacement.png` |
| 3. Probe velocity vs time | `probe_displacement_stop_velocity.png` |
| 4. easy/medium/hard displacement, same execution | `dynamics_execution_presets_displacement.png` |
| 5. easy/medium/hard force, same execution | `dynamics_execution_presets_force.png` |
| extra. profile invariance in `F_peak` | `execution_profile_invariance.png` |

### Videos (`outputs/videos/`)

| File | Duration | Content |
|---|---|---|
| `official_open_drawer-step-0.mp4` | 9.02 s | official IK-Abs approach, grasp and motion-driven opening |
| `probe_easy-step-60.mp4` | 0.87 s | settle plus probe on `easy` |
| `probe_hard-step-60.mp4` | 1.18 s | settle plus probe on `hard` |
| `execution_easy-step-60.mp4` | 2.50 s | settle plus 5 N / 2 s execution on `easy` |
| `execution_hard-step-60.mp4` | 2.50 s | settle plus 5 N / 2 s execution on `hard` |

### Episode logs (`outputs/logs/`)

14 episodes, each a directory with `metadata.json` (versions, git commit, controller
parameters, hidden dynamics, per-environment result) and `trajectory.npz` (16 signals).

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

1. **Held-axis drift on the `easy` preset at large travel.** At `F_peak = 5 N`, `T = 2 s`
   the `easy` drawer travels 318-326 mm at up to 0.42 m/s and approaches its 0.4 m end
   stop; the TCP lateral drift reaches 14-15 mm and the orientation drift 7.5°, against
   0.36-0.66 mm and 0.16-0.41° for `medium` and `hard`. Still well inside the 50 mm / 30°
   safety limits, but the hybrid hold degrades in that regime. Keep validation operating
   points away from the end stop.

2. **Residual pull-axis bias of about 0.25 N.** The gripper is not perfectly centred on the
   handle, so the balanced grip still leaks ~0.25 N onto the pull axis and the drawer
   creeps ~1.3 mm/s with zero commanded force (2.5 mm over 2 s). It is deterministic and
   identical in every episode, and it is visible in every logged history. See D010.

3. **`measured_force` is noisy.** The wrist reaction wrench swings by a few newtons at 5 N
   command level in contact-rich phases. The mean matches `m*a + f + c*v` to about 10 %.
   Downstream consumers should expect to filter. See D006.

4. **Only about 40 % of the commanded force reaches the drawer during acceleration.** The
   arm's own reflected inertia (~9 kg along the pull axis) absorbs the rest. This is real
   physics rather than a defect, but it means `F_peak` and the force delivered to the drawer
   are not interchangeable, and the operating range for a 2 s execution is roughly 2-8 N.

5. **The 0.4 m drawer travel limit saturates above about 8 N.** At `F_peak = 12 N` all three
   presets reach 313-354 mm and the preset separation collapses. The reference operating
   point is therefore 5 N rather than the 12 N originally suggested.

6. **`nominal` is a reference, not a fourth difficulty level.** With zero joint friction the
   drawer reaches the probe's 5 mm target in about 0.2 s at barely above `initial_force`,
   which makes it a poor discriminator.

7. **PhysX's own drawer `joint_vel` is unusable at this control rate** and is only logged for
   transparency. See D009.

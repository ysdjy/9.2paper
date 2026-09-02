# API

Two public controllers. Both are vectorised over environments: one call drives every
environment of the simulation, and every per-environment field of the result is a
`numpy` array indexed by environment.

Units throughout: **force N, position m, velocity m/s, time s, mass kg, angle rad**
(degrees only where a name says so).

```python
from probe_drawer.controllers import ExecutionPullController, ProbePullController
```

Both are normally obtained pre-wired from `PullSystem`:

```python
from probe_drawer.pull_system import PullSystem, PullSystemCfg

system = PullSystem.build(PullSystemCfg(num_envs=1))
system.probe       # ProbePullController
system.execution   # ExecutionPullController
```

---

## ProbePullController

The standardised physical probe: a known, reproducible, monotonically non-decreasing pull
force, applied until one stop condition fires.

```python
ProbePullController(env, osc, reader, cfg=None, safety=None)

run(
    initial_force: float,        # pull force at the start of the ramp (N)
    max_force: float,            # pull force at the end of the ramp (N)
    target_displacement: float,  # drawer opening at which the probe stops (m)
    max_velocity: float,         # drawer opening speed at which the probe stops (m/s)
) -> ProbeResult
```

### Stop conditions, in priority order

| # | Condition | `termination_reason` |
|---|---|---|
| 0 | any absolute safety limit violated | `safety_abort` |
| 1 | `displacement >= target_displacement` | `displacement_reached` |
| 2 | `abs(velocity) >= max_velocity` | `velocity_limit` |
| 3 | the command has reached `max_force` (held for `hold_after_max_force`) | `max_force_reached` |
| 4 | `max_probe_duration` elapsed | `timeout` |

Conditions are evaluated *after* each control step, so a stop overshoots its threshold by
at most one step. Condition 3 stops on the step after the command first equals `max_force`,
so `final_commanded_force` is exactly `max_force` and `duration` is
`ramp_duration + step_dt`. With the default configuration condition 4 is a backstop that a
correctly configured probe never reaches; it exists so a misconfigured ramp or an immovable
drawer cannot run forever.

### Fixed probe character — `ProbeControllerCfg`

These define what *kind* of probe this is, and are held fixed across a whole study. If a
caller could reshape the ramp per episode, probe histories would no longer be comparable.

| Field | Default | Meaning |
|---|---|---|
| `ramp_duration` | `1.0` | time from `initial_force` to `max_force` (s) |
| `ramp_shape` | `"linear"` | `linear`, `smoothstep` or `cosine` |
| `hold_after_max_force` | `0.0` | hold `max_force` this long before stopping (s) |
| `max_probe_duration` | `2.5` | hard time budget (s) |
| `settle_steps` | `30` | zero-force pose-hold steps before the ramp |

### `ProbeResult`

| Field | Shape | Meaning |
|---|---|---|
| `termination_reason` | `list[TerminationReason]` | one per environment |
| `duration` | `(num_envs,)` | simulated time at which that environment stopped (s) |
| `final_displacement` | `(num_envs,)` | drawer opening at the stop instant (m) |
| `final_velocity` | `(num_envs,)` | drawer opening speed at the stop instant (m/s) |
| `final_commanded_force` | `(num_envs,)` | pull-axis command at the stop instant (N) |
| `peak_measured_force` | `(num_envs,)` | largest measured pull-axis force (N) |
| `reached_target` | `(num_envs,)` bool | `final_displacement >= target_displacement` |
| `history` | `PullHistory` | the full time series |
| `parameters` | `dict` | task parameters, profile, config, safety, initial conditions |

`result.summary(env_index)` gives a JSON-friendly scalar summary of one environment.

### Example

```python
result = system.probe.run(
    initial_force=2.0, max_force=10.0, target_displacement=0.005, max_velocity=0.05
)
print(result.summary(0))
# {'termination_reason': 'displacement_reached', 'duration': 0.533,
#  'final_displacement': 0.00529, 'final_velocity': 0.0310,
#  'final_commanded_force': 6.133, 'peak_measured_force': 4.730, 'reached_target': True}
```

---

## ExecutionPullController

The full-duration force-driven pull. Applies `F(t) = peak_force * phi(t / duration)` with a
fixed normalised shape, for the whole duration.

```python
ExecutionPullController(env, osc, reader, cfg=None, safety=None)

run(
    peak_force: float,   # plateau pull force (N)
    duration: float,     # total execution time (s)
) -> ExecutionResult
```

### Force profile

`phi` rises smoothly from 0 over the first `rise_fraction` of the duration, holds at 1,
and falls smoothly back to 0 over the last `fall_fraction`. The rise and fall use a C1
smoothstep, so the command never steps. `phi` depends on neither `peak_force` nor
`duration`, which is what makes runs at different forces comparable.

```
   F/F_peak
     1 |      ______________________
       |     /                      \
       |    /                        \
     0 |___/                          \___
       0   0.1                     0.9   1.0        t/T
```

### There is no goal input

`d_goal` is not an argument and no stop condition anywhere in this controller refers to
drawer displacement. The controller executes the commanded duration whatever the drawer
does. Deciding whether `abs(d(T) - d_goal) <= epsilon` is the caller's job:

```python
result = system.execution.run(peak_force=5.0, duration=2.0)
success = abs(result.final_displacement[0] - d_goal) <= epsilon
```

See `docs/DECISIONS.md` D004. The guarantee is enforced by
`tests/unit/test_execution_has_no_goal_feedback.py`.

### Early termination

The only permitted early stop is an absolute safety violation, and it is implemented in
`BasePullController`, not here. `SafetyLimits`:

| Field | Default | Trip condition |
|---|---|---|
| `max_commanded_force` | `60.0` | a profile peaking above this is **refused** (raises) |
| `max_drawer_velocity` | `1.0` | drawer speed exceeds it (m/s) |
| `max_tcp_speed` | `2.0` | TCP speed exceeds it (m/s) |
| `max_lateral_error` | `0.05` | TCP drifts this far off the pull axis (m) |
| `max_orientation_error_deg` | `30.0` | TCP rotates this far from the reference |
| `max_arm_joint_velocity` | `6.0` | any arm joint exceeds it (rad/s) |

### Fixed profile shape — `ExecutionControllerCfg`

| Field | Default | Meaning |
|---|---|---|
| `rise_fraction` | `0.1` | fraction of the duration spent rising |
| `fall_fraction` | `0.1` | fraction of the duration spent falling. **The paper uses 0.20**; see `probe_drawer.experiment_plan.RECOMMENDED_EXECUTION_CFG` and `docs/DECISIONS.md` D023 |
| `shape` | `"smoothstep"` | interpolation of rise and fall |
| `settle_steps` | `30` | zero-force pose-hold steps before the profile starts |
| `zero_force_cleanup_steps` | `2` | zero-force steps sent **after** `T`, excluded from the result |
| `post_execution_settle_steps` | `0` | further steps, also outside `T` |

### What happens after `T`

The result is snapshotted at `t = T`. Zero pull force is then commanded for the cleanup
steps, which appear nowhere in the history, the duration, `d(T)`, `v(T)` or the success
evaluation — they exist only so no residual command is left standing (`docs/DECISIONS.md`
D022).

### `ExecutionResult`

| Field | Shape | Meaning |
|---|---|---|
| `termination_reason` | `list[TerminationReason]` | `duration_completed` nominally |
| `duration` | `(num_envs,)` | simulated time actually executed (s) |
| `final_displacement` | `(num_envs,)` | drawer opening when the force returned to zero (m) |
| `final_velocity` | `(num_envs,)` | drawer opening speed at that instant (m/s) |
| `peak_commanded_force` | `(num_envs,)` | largest pull-axis command issued (N) |
| `peak_measured_force` | `(num_envs,)` | largest measured pull-axis force (N) |
| `safety_aborted` | `(num_envs,)` bool | whether a limit cut the episode short |
| `history` | `PullHistory` | the full time series |
| `parameters` | `dict` | task parameters, profile, config, safety, initial conditions |

There is no success field, by design.

### Example

```python
result = system.execution.run(peak_force=5.0, duration=2.0)
print(result.summary(0))
# {'termination_reason': 'duration_completed', 'duration': 2.0,
#  'final_displacement': 0.1403, 'final_velocity': 0.1199,
#  'peak_commanded_force': 5.0, 'peak_measured_force': 5.295, 'safety_aborted': False}
```

---

## PullHistory

`time` has shape `(T,)`; scalar per-environment signals `(T, num_envs)`; Cartesian
signals `(T, num_envs, 3)`; joint signals `(T, num_envs, 7)`.

25 channels. Rather than repeat them here, every one carries its unit, source, filtering and
deployability in `probe_drawer.observations.OBSERVATION_SPECS`, which is the authoritative
list and is asserted to match `PullHistory` field for field. The classes are:

| Class | Meaning | Channels |
|---|---|---|
| `DEPLOYABLE` | a real Franka pulling a real drawer could produce it | commanded force, drawer position / velocity / acceleration, TCP pose, velocity, pull-axis projections and accelerations, held-axis errors, arm joint state |
| `DIAGNOSTIC` | obtainable on hardware, not needed by the first model | `measured_force` (wrist wrench), `handle_contact_force_w`, `drawer_velocity_raw` |
| `SIM_ONLY_PRIVILEGED` | **never** a deployed-model input | `drawer_resistance_force`, `drawer_external_force` |

Distinctions worth repeating:

* `commanded_force` vs `measured_force` — what was asked for, versus what the wrist sensor
  reports. Never conflated (D006).
* `drawer_velocity` vs `drawer_velocity_raw` — the causally filtered finite difference every
  decision uses, versus PhysX's own aliased reading, kept for audit (D009).
* every derivative is **causal** and its filter window and lag are recorded with the episode
  (D025). A non-causal filter cannot run on a robot.
* `drawer_position` is displacement **relative to the start of the pull**, not the absolute
  joint coordinate.

```python
from probe_drawer.observations import DEFAULT_ACE_INPUT, validate_model_input

validate_model_input(DEFAULT_ACE_INPUT)          # passes
validate_model_input(["drawer_external_force"])  # raises: privileged channel
```

A controller keeps stepping until every environment has stopped and zeroes the command of
those that already have, so mask a single environment with
`history.active_steps(env_index)` before analysing it.

---

## DynamicsRandomizer

```python
from probe_drawer.envs import DynamicsRandomizer, preset

randomizer = DynamicsRandomizer(cfg=None, seed=None)

params  = randomizer.sample(num_envs)            # list[DynamicsParameters]
applied = randomizer.apply(env, params)          # AppliedDynamics, with PhysX readback
applied = randomizer.apply(env, preset("hard"))  # one preset broadcast to every environment
randomizer.get_current_params()                  # the privileged state xi, for logging
```

`DynamicsParameters` is `xi`, and it is exactly four dimensional (D015):

| Field | Unit | Simulator quantity |
|---|---|---|
| `drawer_mass` | kg | mass of the `drawer_top` body, with inertia scaled to match |
| `joint_static_friction` | N | static Coulomb effort of `drawer_top_joint` |
| `joint_dynamic_friction` | N | dynamic Coulomb effort of the same joint |
| `joint_damping` | N s/m | viscous damping of the joint drive |

**PhysX requires `mu_d <= mu_s`** and silently discards writes that violate it, so
`DynamicsParameters` refuses such a combination and `sample()` draws `mu_d` as a fraction of
`mu_s` (D016). `AppliedDynamics.consistent` compares against the **simulator's own**
readback, and `mirror_agrees` reports whether Isaac Lab's buffers agree with it.

Presets: `nominal`, `easy`, `medium`, `hard`, `sticky`, `slippery`. All provisional; the
paper's ranges are in `probe_drawer.experiment_plan`.

---

## Evaluation

```python
from probe_drawer.evaluation import OperatingRegionCfg, SuccessCriteria, assess_validity, evaluate_execution

validity = assess_validity(result, OperatingRegionCfg())          # is this usable evidence?
report = evaluate_execution(result, criteria)                     # did it succeed?
report.success            # (num_envs,) bool
report.verdicts[0].invalid_reasons
```

`success = |d(T) - d_goal| <= eps_d AND |v(T)| <= eps_v AND valid` (D020). Validity subsumes
safety: a safety-aborted episode is never valid. Neither function can reach the controllers
or the environment, which is what keeps `d_goal` out of the control loop (D004).

---

## Experiment plan

```python
from probe_drawer.experiment_plan import MAIN_TASK, RECOMMENDED_PROBE_TASK, TRAINING_XI_RANGES

MAIN_TASK.criteria                       # SuccessCriteria for the paper's task
RECOMMENDED_PROBE_TASK.as_kwargs()       # splat straight into ProbePullController.run
TRAINING_XI_RANGES.as_randomizer_cfg()   # DynamicsRandomizerCfg for sampling xi
```

Every value is the output of a sweep, with its provenance recorded. Nothing here is loaded
by the controllers: this is the *experiment* definition, which is a separate thing.

---

## Changing this API

If a public signature changes, this file changes in the same commit. A stale `API.md` is
treated as a defect.

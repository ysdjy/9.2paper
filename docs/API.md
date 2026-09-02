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
| `fall_fraction` | `0.1` | fraction of the duration spent falling |
| `shape` | `"smoothstep"` | interpolation of rise and fall |
| `settle_steps` | `30` | zero-force pose-hold steps before the profile starts |

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

| Signal | Provenance |
|---|---|
| `active` | whether that environment was still being driven at that step |
| `commanded_force` | what the controller asked for (N) |
| `measured_force` | pull-axis component of the **wrist joint reaction wrench** (N) |
| `drawer_position` | drawer opening **relative to the start of the pull** (m) |
| `drawer_velocity` | moving average of the position finite difference (m/s) |
| `drawer_velocity_raw` | PhysX's own `joint_vel`, logged for transparency only |
| `tcp_position`, `tcp_linear_velocity`, `tcp_angular_velocity` | TCP state, world frame |
| `tcp_pull_axis_position` | TCP travel along the pull axis since the reference (m) |
| `tcp_lateral_error` | TCP drift orthogonal to the pull axis (m) |
| `tcp_orientation_error` | TCP orientation drift from the reference (rad) |
| `handle_contact_force_w` | net contact force on the handle body (N), a grip-load witness |
| `joint_position`, `joint_velocity` | arm joint state |
| `joint_applied_effort` | effort PhysX was *asked* to apply — a command, not a measurement |

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

`DynamicsParameters` is `xi`: `drawer_mass` (kg, the `drawer_top` body),
`joint_friction` (Coulomb coefficient of `drawer_top_joint`, written to both the static and
the dynamic channel), `joint_damping` (N s/m, the joint drive), and `joint_stiffness`
(N/m, held at 0 — see D008).

`AppliedDynamics.consistent` is `True` only if every requested value read back out of PhysX.
Presets: `nominal`, `easy`, `medium`, `hard` — values and their calibration in
`docs/VALIDATION.md`.

---

## Changing this API

If a public signature changes, this file changes in the same commit. A stale `API.md` is
treated as a defect.

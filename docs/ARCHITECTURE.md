# Architecture

## Runtime structure

```
                     Isaac Lab (official, never modified)
                                    |
                    import + config inheritance
                                    v
                          ProbeDrawerEnvCfg
                    (official cabinet scene, force-controlled)
                                    |
                          ManagerBasedRLEnv
                                    |
                 +------------------+------------------+
                 |                                     |
        DrawerStateReader                      HybridPullOSC
   drawer position / velocity                pose_abs + wrench_abs
   TCP pose / velocity                       -> 14-element action
   measured pull force (wrist)               owns the pose reference
   arm joint state                           reports held-axis drift
                 |                                     |
                 +------------------+------------------+
                                    |
                          BasePullController
                  steps, applies the profile, evaluates
                  stop conditions, records the history
                                    |
                 +------------------+------------------+
                 |                                     |
        ProbePullController                 ExecutionPullController
        RampForceProfile                    TrapezoidForceProfile
        4 stop conditions                   no task stop condition
                 |                                     |
            ProbeResult                          ExecutionResult
                 |                                     |
                 +------------------+------------------+
                                    |
                            EpisodeLogger
                   metadata.json + trajectory.npz
```

Randomisation is a separate path that only ever writes to the environment:

```
DynamicsRandomizer.sample()  ->  DynamicsParameters (xi)
DynamicsRandomizer.apply(env, xi)  ->  drawer mass / joint friction / joint damping
                                       + readback verification
```

`PullSystem` is the assembly point. It builds the environment, the reader, the one OSC and
both controllers, and is the only place that knows how they are wired.

## The control split

The drawer's opening direction is one coordinate axis of the robot base frame (measured:
base `-x`). Isaac Lab's `OperationalSpaceController` splits the six task-space axes with
two selection masks, and this project uses exactly that:

| Axis | Mode |
|---|---|
| base `x` (the pull axis) | open-loop **force** control |
| base `y`, `z`, `Rx`, `Ry`, `Rz` | **pose** hold at the reference captured when the pull started |

There is one OSC implementation in this project, and it is Isaac Lab's. `HybridPullOSC`
only assembles commands, owns the pose reference and reports drift; the Probe and Execution
controllers own no robot control code at all. That is what makes the two APIs comparable:
they differ in *force profile* and *stop conditions*, and in nothing else.

## Who owns what

| Package | Responsibility |
|---|---|
| `envs/drawer_env_cfg.py` | the research environment: official cabinet scene, hybrid OSC action, handle contact sensor, grasped reset state, pinned contact friction |
| `envs/hybrid_pull_cfg.py` | the OSC gains and the pull-axis definition |
| `envs/initialization.py` | loading the recorded grasp configuration and deriving the balanced grip command |
| `envs/dynamics_randomization.py` | the hidden state `xi` and the only code that writes it into PhysX |
| `controllers/force_profiles.py` | `F(t)`, and nothing else |
| `controllers/hybrid_osc.py` | action assembly, pose reference, settling, drift diagnostics |
| `controllers/base_pull_controller.py` | the shared step loop, safety limits, history recording |
| `controllers/probe_pull_controller.py` | the standardised probe and its four stop conditions |
| `controllers/execution_pull_controller.py` | the full-duration execution; no task stop condition |
| `controllers/types.py` | `ProbeResult`, `ExecutionResult`, `PullHistory`, `TerminationReason` |
| `sensors/pull_axis.py` | the signed axis the drawer opens along |
| `sensors/drawer_state.py` | read-only access to drawer, TCP, joints and the measured pull force |
| `state_machines/pull_state_machine.py` | approach and grasp; used to *record* the grasp pose, not during experiments |
| `logging/episode_logger.py` | writing an episode to disk |
| `pull_system.py` | wiring |
| `utils/isaaclab_compat.py` | version introspection, path resolution, stdout flushing |

## Data flow of one experiment

```
env.reset()                     -> arm in the recorded grasped configuration, drawer closed
DynamicsRandomizer.apply(...)   -> xi written into PhysX, read back and verified
controller.run(...)
    HybridPullOSC.settle(n)     -> pose held, pull axis braked, system brought to rest
    for each control step:
        F  = profile.force(t)                       (zeroed for environments that already stopped)
        a  = HybridPullOSC.action(F)                (pose reference + pull-axis wrench)
        HybridPullOSC.step(a)                       (wrapped env, then reader.update())
        record the post-step state
        evaluate safety, then the task stop conditions
    -> ProbeResult / ExecutionResult
EpisodeLogger.save(id, result, xi)  -> metadata.json + trajectory.npz
```

## What is deliberately not here yet

ACE, PSP, SPC, VLM, RL policies and RMA baselines. The current scope is the physical and
control substrate plus its validation. Adding any of them should not require changing the
two public controller APIs.

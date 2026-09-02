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
                 +------------------+------------------+
                 |                                     |
          EpisodeLogger                    evaluation/  (offline, no control)
   metadata.json + trajectory.npz          assess_validity -> is this usable?
                                           evaluate_execution -> did it succeed?
                                                     |
                                            analysis/  (offline, pure)
                                    sweeps, Oracle landscape, probe features,
                                    force-channel and hidden-state audits
```

Note the direction of the arrows into `evaluation/` and `analysis/`. Nothing flows back:
neither package can reach the controllers or the environment, which is what keeps the goal
displacement out of the control loop.

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
| `sensors/causal_derivative.py` | the one differentiator every derived channel goes through |
| `sensors/drawer_state.py` | read-only access to drawer, TCP, joints and every force channel |
| `observations.py` | what each of the 25 logged channels is, and who may consume it |
| `evaluation/operating_region.py` | whether an episode is usable evidence at all |
| `evaluation/task_evaluator.py` | position, terminal velocity and validity -> success |
| `analysis/hidden_state_audit.py` | what else the simulator would let us hide |
| `analysis/force_channel_analysis.py` | the force channels against the equation of motion |
| `analysis/sweep.py` | the sweep record format and the queries the selection needs |
| `analysis/oracle.py` | the acceptance conditions and the task recommendation |
| `analysis/probe_features.py` | probe summary features and their correlation with the answer |
| `experiment_plan.py` | the parameters Phase 9 selected, each citing its sweep |
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

## Three data contracts

The project has more channels than any model will read, so three registries keep them
honest, and each is enforced by a unit test rather than by convention:

| Registry | Guarantees |
|---|---|
| `controllers.types.HISTORY_CHANNELS` | the dataclass, the `.npz` layout and the recorder describe the same 25 channels |
| `observations.OBSERVATION_SPECS` | every channel has a unit, a source, a filter description and a deployability class |
| `observations.DEFAULT_ACE_INPUT` | the model's inputs are deployable, causal, and include `commanded_force` |

## What is deliberately not here yet

ACE, PSP, SPC, VLM, RL policies and RMA baselines. The current scope is the physical and
control substrate plus its validation. Adding any of them should not require changing the
two public controller APIs.

---

## Phase 10 additions

Two packages joined, both of which own a boundary rather than a mechanism.

### `protocols/` — sequencing, and nothing else

```
script  ->  protocol  ->  controllers  ->  environment
```

A one-way dependency. `SequentialPullProtocol` decides *when* the probe, the gap and the
execution happen; the controllers still decide *what force* and the sensors still decide
*what was measured*. It applies no force, reads no dynamics and holds no state between
episodes.

It exists because the alternative was for every sweep script to re-implement the ordering,
and a script that gets the ordering subtly wrong — a stray reset, a settle left enabled —
produces data that looks correct. The refusal to run with `settle_steps != 0` lives here for
the same reason (D029).

| Must | Must never |
|---|---|
| call the controllers in order | generate a force profile |
| record what happened between them | read or write `xi` |
| refuse a configuration that would corrupt the protocol | know what `d_goal` is used for |

The protocol *is* handed `criteria` so it can call the evaluator, but it only passes them
through; it does not act on them, and the execution controller never sees them.

### `dataset/` — the simulator-to-model boundary

`schema.py` defines one training sample and the three nested identifiers; `splits.py` does
grouped splitting and asserts the result does not leak. Neither imports Isaac Lab, so the
package runs wherever the data has been copied to.

The deployability rule is *not* re-implemented here — `validate_probe_history` delegates to
`observations.validate_model_input`, so "deployable" has one definition in the project.

| Must | Must never |
|---|---|
| define the sample's fields and identifiers | read a simulator |
| refuse a leaking split level | drop invalid rows (that is the training script's visible decision) |
| refuse a reset-protocol row | define what "deployable" means |

### Where the new modules sit

```
scripts/build_sequential_oracle.py
        |
        v
protocols/sequential_pull_protocol.py ----> controllers/{probe,execution}_pull_controller.py
        |                                            |
        |                                            v
        |                                   controllers/hybrid_osc.py  (step, settle, coast)
        v
evaluation/task_evaluator.py  --->  analysis/sweep.py  --->  analysis/oracle.py
                                            |
                                            v
                                    dataset/{schema,splits}.py   (no simulator)
```

---

## Phase 11 additions

Three packages, and one addition to `protocols/` that breaks its own rule on purpose.

### `dataset/` gains sampling, storage and audit

```
sampling.py  --> what to record, and what the samplers may not see
storage.py   --> the normalised on-disk layout
audit.py     --> nine gates plus the distributions
schema.py    --> one sample, three nested identifiers
splits.py    --> grouped splitting, and the check that it holds
```

Still no Isaac Lab import anywhere in the package, so the whole data side runs on a machine
with no simulator.

| Must | Must never |
|---|---|
| decide the plan before the simulator starts | read an outcome while sampling |
| refuse a dangling reference at write time | modify a dataset while auditing it |
| survive the breakage the audit detects | drop invalid rows (the training script's visible call) |

### `models/` and `training/`

```
models/psp.py         --> PrivilegedEncoder, AdaptationContextEncoder, SuccessPredictor
models/baselines.py   --> baselines A-D and the fixed-force floor
training/dataloader.py --> dynamic padding, train-only normalisation
training/metrics.py   --> classification and selection metrics
training/trainer.py   --> teacher phase, student phase, checkpoints
```

The privileged/deployable boundary is **structural**, not conventional: `StudentModel` has no
parameter that could carry `xi`, and the test suite corrupts `batch.xi` and asserts the
student's output is bit-identical while the teacher's is not.

| Must | Must never |
|---|---|
| fit any statistic on the training split alone | start a simulator |
| use `lengths`/`mask` so padding is never consumed | pad or resample a history on disk |
| record the label distribution it trained on | resample the evaluation set |

### `evaluation/force_selection.py`

The search that turns a predicted landscape into one force. It lives here rather than in a
controller because the execution controller must never learn what a goal is (D004); a deployed
system runs the search between the probe and the pull, and so does this.

### `protocols/simulation_snapshot.py` — the deliberate exception

`protocols/` otherwise only sequences things. This module reaches into the simulator to freeze
and restore an instant, which is a real violation of that boundary and is why it carries the
longest docstring in the package. It exists because 32 counterfactual labels per probe cannot
be produced any other way, it is used **only** by dataset generation, and
`docs/COUNTERFACTUAL_BRANCHING.md` records exactly what it does and does not capture.

### The whole Phase 11 flow

```
scripts/generate_dataset.py
   |  protocols/{sequential_pull_protocol, simulation_snapshot}
   |  controllers/{probe,execution}_pull_controller
   v
outputs/dataset_v0/            <-- dataset/storage.py   (no simulator beyond this line)
   |
   +--> scripts/audit_dataset.py      --> dataset/audit.py + dataset/splits.py
   |
   +--> scripts/train_models.py       --> training/* + models/*
   |          |
   |          v
   |    outputs/training/run_v0/
   |          |
   +----------+--> scripts/evaluate_closed_loop.py   <-- back into Isaac Sim, once
                       |  evaluation/force_selection.py
                       v
                  closed_loop.json --> scripts/plot_phase11.py
```

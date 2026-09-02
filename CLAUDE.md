# CLAUDE.md — read this before touching anything

This is a long-running research repository, not a demo. Several agents work on it over
time. The rules below exist so that the next one can trust what is here.

---

## 0. Where things are

* Repository root: `/home/zbh/Downloads/IsaacLab/9.2paper` (its own git repository,
  remote `https://github.com/ysdjy/9.2paper.git`).
* It sits **inside** the Isaac Lab source checkout at `/home/zbh/Downloads/IsaacLab`, which
  is a *different* repository. `9.2paper/` is listed in that checkout's
  `.git/info/exclude`, so it never shows up in Isaac Lab's `git status`.
* Interpreter: `/home/zbh/anaconda3/envs/env_isaaclab/bin/python` (conda env `env_isaaclab`).
* Never hard-code either path in code. Use `probe_drawer.utils.project_root()` and
  `probe_drawer.utils.isaaclab_root()`.

## 1. Before you change anything

```bash
git status
git branch
git log --oneline -10
```

Then read, in this order:

1. `README.md` — what the project is and what currently works
2. `CLAUDE.md` — this file
3. `docs/ARCHITECTURE.md` — who owns what
4. `docs/API.md` — the two public controller APIs
5. `docs/DECISIONS.md` — questions that are already settled; do not relitigate them
6. `docs/SESSION_LOG.md` — what the previous session did and what it left open

## 2. Work on a branch

```
main
 └── agent/<task-name>        e.g. agent/probe-data-pipeline
```

Do not accumulate undocumented commits on `main`. If GitHub push access is not configured,
still commit locally and record the SHA in `docs/SESSION_LOG.md`.

## 3. Backups are git

Use commits, branches and tags. Do **not** create `foo_old.py`, `foo_new.py`,
`foo_final2.py`, `foo_backup.py`. If a large experiment snapshot genuinely cannot live in
git, put it under `backups/` and document it in `backups/README.md`.

## 4. Never modify Isaac Lab

This project *reuses* Isaac Lab by importing and by config inheritance:

```
Isaac Lab (official)  --import / config inheritance-->  probe_drawer
```

Editing anything under `/home/zbh/Downloads/IsaacLab/source/` is prohibited. If the
official API cannot do what you need: read the local source, check whether a wrapper or a
subclass suffices, make the smallest possible addition on our side, and record it in
`docs/DECISIONS.md` with what was missing and whether it can be reverted later.

## 5. Module boundaries

| Package | Owns | Must never |
|---|---|---|
| `envs/` | scene configuration, reset state, dynamics randomisation | control the robot |
| `controllers/` | force profiles, hybrid OSC, Probe and Execution | modify the environment or randomise anything |
| `sensors/` | read-only accessors, pull axis, causal differentiation | write to the simulation, or decide anything |
| `evaluation/` | validity and success labelling | touch the simulation or the controllers |
| `analysis/` | offline audits, sweeps, the Oracle, probe features | be imported by anything that runs control |
| `logging/` | writing episodes to disk | be imported by `controllers/` or `envs/` |
| `state_machines/` | approach-and-grasp | be used during a probe or an execution |
| `observations.py` | the channel registry | import Isaac Lab |
| `experiment_plan.py` | the selected parameters | be read by a controller |
| `pull_system.py` | wiring the above together | contain physics or control logic |

No circular imports. `envs`, `controllers` and `logging` must not import each other
except through `sensors` and `controllers.types`.

## 6. Hard rules

* **Public controller results are dataclasses**, `ProbeResult` and `ExecutionResult`. Never
  return an ad-hoc dict.
* **`d_goal` never enters the execution control loop** (D004). This is enforced by
  `tests/unit/test_execution_has_no_goal_feedback.py`; do not weaken that test.
* **`commanded_force` is never used as `measured_force`** (D006). If you add a new force
  signal, name its physical source in the docstring.
* **No magic numbers in scripts or controllers.** Every tunable lives in a dataclass under
  `envs/`, `controllers/` or `sensors/`, and is mirrored into `configs/` (see D011).
* **`xi` is exactly four dimensional** (D015) and **`mu_d <= mu_s`** is a PhysX requirement,
  not a preference (D016). Read dynamics back from `root_physx_view`, never from
  `Articulation.data`, which mirrors the request.
* **Never feed a `SIM_ONLY_PRIVILEGED` channel to a model.** Call
  `observations.validate_model_input()` wherever an observation vector is assembled (D017).
* **Every derivative must be causal** and must record its filter (D025). A non-causal filter
  cannot run on a robot.
* **Experimental parameters come from a sweep, not from judgement.** If you change one,
  change the scored rule in `analysis/oracle.py` and re-derive it, then update
  `experiment_plan.py` and `docs/EXPERIMENT_SPACE.md` together (D024).
* **No debug `print` in `src/`.** Scripts print reports; library code does not.
* **Type-hint and docstring every public class and method.**
* Modules under `controllers/`, `sensors/`, `envs/initialization.py`,
  `envs/dynamics_randomization.py` and `envs/hybrid_pull_cfg.py` must stay importable
  **without** the Isaac Sim application running (Isaac Lab types go under `TYPE_CHECKING`).
  `tests/unit/` depends on this.
* Every simulation step goes through `HybridPullOSC.step`. Calling `env.step` directly
  bypasses the drawer-velocity update and the video recorder.

## 7. Before you call a task done

1. delete leftover debugging code
2. look for logic you duplicated instead of reusing
3. check each file still has one job
4. check no function grew to do force generation *and* control *and* logging *and* plotting
5. check names
6. check type hints
7. `python -m pytest tests/unit -q`
8. `python -m pytest tests/integration -q` (this launches Isaac Sim; it must pass)
9. run at least one script end to end and **look at the physical numbers**, not just the exit code
10. update `docs/API.md` if any public signature changed
11. update `README.md` status if a phase completed
12. add an entry to `docs/SESSION_LOG.md`
13. read your own `git diff` file by file
14. commit, and push if you have access

## 8. Isaac Sim gotchas found the hard way

These cost real debugging time; do not rediscover them.

* `SimulationApp.close()` ends the process **without flushing Python's buffers**. Anything
  printed after the app launched is lost when stdout is a pipe or a file. Call
  `probe_drawer.utils.enable_unbuffered_stdout()` right after launching.
* Do not wrap a loop in `torch.inference_mode()` if the environment will be reset
  afterwards: Isaac Lab writes into buffers that then become inference tensors and the
  reset raises.
* `Articulation.data.joint_vel` for the drawer joint is unusable at the 60 Hz control rate
  (D009). Use `DrawerStateReader.drawer_velocity`.
* A `ContactSensor`'s `net_forces_w` does not include tangential friction, so it cannot
  measure a pull transmitted through a friction grip (D006).
* `gymnasium.wrappers.RecordVideo.reset()` **discards** the frames captured so far, and it
  only sees steps that go through the wrapper. `PullSystem` therefore steps the wrapped
  environment and starts/stops the recorder explicitly.
* This machine has ROS 2 Humble on `PYTHONPATH`; its pytest plugins break under Python
  3.11. `pyproject.toml` disables them via `addopts`.
* **One simulation context per process.** `PullSystem.build` cannot be called twice; a script
  that needs two configurations must reuse one system or run twice.
* `Articulation.data.joint_friction_coeff` and friends report what you *asked for*. After a
  write PhysX rejected, they disagree with the simulator. Always read back from
  `root_physx_view` (D016).
* The environment holds the last action it was given. After an episode ends, command zero
  explicitly, outside the recorded window (D022).

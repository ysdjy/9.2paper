# 9.2paper — Single-Probe Physical Adaptation for Force-Driven Drawer Pulling

**Single-probe physical adaptation for force-driven Franka drawer pulling under unknown dynamics.**

机器人面对一个视觉上可见、但动力学（质量 / 摩擦 / 阻尼）未知的抽屉。它先执行一次标准化的短时物理
Probe——用已知的递增力输入去"试探"抽屉，并记录完整的力—位移—速度响应；随后由模型根据这段 Probe 历史
预测：正式执行时应该用多大的峰值作用力 `F_peak`，才能在规定时间 `T` 内把抽屉拉到目标位移 `d_goal`。

本仓库目前实现的是这套研究的**物理与控制底座**：官方环境验证、力驱动的 Probe / Execution 两个公开控制器、
隐藏动力学随机化，以及它们的完整验证。**尚未**包含 ACE / PSP / SPC / VLM / RL policy。

---

## 1. Current status

```
[x] Phase 0  Project / Git / documentation bootstrap
[x] Phase 1  Local Isaac Sim + Isaac Lab inspection
[x] Phase 2  Official Franka drawer environment validated (IK-Abs, full approach->grasp->open)
[x] Phase 3  Shared hybrid OSC integrated (1 DOF force + 5 DOF pose hold, official OSC reused)
[x] Phase 4  ProbePullController
[x] Phase 5  ExecutionPullController
[x] Phase 6  Controller validation (force ramp, all four stop conditions, profile invariance,
             duration, held-axis stability, safety abort)
[x] Phase 7  DynamicsRandomizer (drawer mass / joint friction / joint damping)
[x] Phase 8  Dynamics physical validation (easy/medium/hard clearly separated)
[ ]          Training-data generation
[ ]          ACE
[ ]          PSP
[ ]          SPC
```

Detailed, dated results with observed values: [docs/VALIDATION.md](docs/VALIDATION.md).

---

## 2. Environment

Read off this machine, not from documentation:

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS, Linux 5.15.0-1097-realtime (x86_64) |
| CPU | Intel Core i9-14900K |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, driver 580.126.09 |
| Isaac Sim | 5.1.0.0 (pip, `isaacsim==5.1.0.0`) |
| Isaac Lab | 2.3.0 (`VERSION`), extension `isaaclab` 0.49.0 |
| Isaac Lab root | `/home/zbh/Downloads/IsaacLab` (source checkout, never modified by this project) |
| Python | 3.11.15 (conda env `env_isaaclab`) |
| PyTorch | 2.7.1+cu128 |
| CUDA | 12.8 |

The interpreter every command below uses is
`/home/zbh/anaconda3/envs/env_isaaclab/bin/python`; activate the environment with
`conda activate env_isaaclab` and plain `python` works.

Regenerate this table's machine-read part with:

```bash
python scripts/inspect_isaaclab.py
```

---

## 3. Install

```bash
conda activate env_isaaclab
cd /home/zbh/Downloads/IsaacLab/9.2paper
python -m pip install -e . --no-deps
```

`--no-deps` is deliberate: `isaacsim`, `isaaclab` and `torch` are supplied by the conda
environment and must not be resolved by pip.

---

## 4. Quick start

Every command below has been run on this machine; none is aspirational.

```bash
# Phase 1 -- what is installed, and which drawer environments are registered
python scripts/inspect_isaaclab.py

# Phase 2 -- official IK-Abs drawer environment, full approach -> grasp -> open
python scripts/run_official_drawer.py --num_envs 4 --headless
# ...and (re)record the grasped arm configuration the research environment resets into
python scripts/run_official_drawer.py --num_envs 1 --headless --deterministic-init --export-grasp-pose

# Phase 4/6 -- one standardised probe
python scripts/test_probe_pull.py --headless --preset medium --experiment-id probe_displacement_stop

# Phase 5/6 -- one full-duration force-driven execution (d_goal is evaluated after the run)
python scripts/test_execution_pull.py --headless --preset medium --peak-force 5.0 --duration 2.0 --d-goal 0.15

# Phase 7/8 -- easy / medium / hard side by side in one simulation
python scripts/test_dynamics_randomization.py --headless

# Plots (no Isaac Sim needed)
python scripts/visualize_response.py --all --profile-invariance

# Tests
python -m pytest tests/unit -q          # 80 tests, ~1 s, no Isaac Sim
python -m pytest tests/integration -q   # 44 tests, ~60 s, launches Isaac Sim once
```

Add `--video` to `run_official_drawer.py`, `test_probe_pull.py` or `test_execution_pull.py`
to write an MP4 into `outputs/videos/`.

### Minimal API usage

```python
from probe_drawer.envs import DynamicsRandomizer, preset
from probe_drawer.pull_system import PullSystem, PullSystemCfg

system = PullSystem.build(PullSystemCfg(num_envs=1))
randomizer = DynamicsRandomizer()

system.reset()
xi = randomizer.apply(system.env, preset("medium"))       # the hidden dynamics

probe = system.probe.run(initial_force=2.0, max_force=10.0,
                         target_displacement=0.005, max_velocity=0.05)

system.reset()
randomizer.apply(system.env, preset("medium"))
execution = system.execution.run(peak_force=5.0, duration=2.0)

success = abs(execution.final_displacement[0] - 0.15) <= 0.02   # evaluation, not control
```

---

## 5. Project structure

```
9.2paper/
├── configs/            generated snapshots of the dataclass defaults + the recorded grasp pose
├── src/probe_drawer/
│   ├── envs/           environment configuration, research initialisation, dynamics randomisation
│   ├── controllers/    force profiles, shared hybrid OSC, Probe and Execution controllers
│   ├── state_machines/ approach-and-grasp state machine (used to record the grasp pose)
│   ├── sensors/        read-only accessors: drawer state, TCP state, measured pull force
│   ├── logging/        per-episode JSON metadata + NPZ trajectory
│   ├── utils/          version introspection, path resolution
│   └── pull_system.py  wires environment + reader + OSC + both controllers together
├── scripts/            one runnable entry point per validation step
├── tests/unit/         no Isaac Sim; force profiles, config validation, snapshot drift, D004 audit
├── tests/integration/  launches Isaac Sim once; physical behaviour of both controllers
├── docs/               architecture, API, official baseline, validation, decisions, session log
└── outputs/            logs, plots, videos (git-ignored)
```

Module boundaries are strict: `controllers/` never modifies the environment, `envs/` never
controls the robot, `logging/` only writes, and nothing imports Isaac Lab's source tree
except through its public packages. Isaac Lab itself is never edited.

---

## 6. Documentation

| Document | What it is for |
|---|---|
| [docs/API.md](docs/API.md) | the two public controllers, their inputs, units and return values |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the pieces fit together and who owns what |
| [docs/OFFICIAL_BASELINE.md](docs/OFFICIAL_BASELINE.md) | the official Isaac Lab environment as installed here |
| [docs/VALIDATION.md](docs/VALIDATION.md) | every test, its command, and the value actually observed |
| [docs/DECISIONS.md](docs/DECISIONS.md) | design decisions that are settled, and why |
| [docs/SESSION_LOG.md](docs/SESSION_LOG.md) | what each work session changed |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | released state |
| [CLAUDE.md](CLAUDE.md) | **read first** if you are an agent working on this repository |

---

## 7. Validation status

All of Phases 0-8 pass. 80 unit tests and 44 integration tests pass. Highlights, with the
values actually measured (full table in [docs/VALIDATION.md](docs/VALIDATION.md)):

* the official drawer opens to 308.7 mm; the drawer's travel direction measures exactly
  `(-1, 0, 0)` in the robot base frame, 0.000° from the configured pull axis;
* all four probe stop conditions fire as specified — displacement (5.29 mm at 0.533 s),
  velocity (5.05 mm/s at 0.100 s), max force (exactly 10.000 N at 1.017 s), timeout (0.500 s);
* the execution force profile is invariant in `F_peak`, and the commanded duration is
  executed to within one control step;
* the five held task-space axes drift at most 0.66 mm and 0.41° during a full execution;
* the same `F_peak = 5 N`, `T = 2 s` gives `d(T) =` 326.1 / 141.7 / 59.6 mm on
  easy / medium / hard — consecutive ratios 2.30 and 2.38.

## 8. Known limitations

See the end of [docs/VALIDATION.md](docs/VALIDATION.md) for the current list.

# 9.2paper — Single-Probe Physical Adaptation for Force-Driven Drawer Pulling

**Single-probe physical adaptation for force-driven Franka drawer pulling under unknown dynamics.**

机器人面对一个视觉上可见、但动力学（质量 / 摩擦 / 阻尼）未知的抽屉。它先执行一次标准化的短时物理
Probe——用已知的递增力输入去"试探"抽屉，并记录完整的力—位移—速度响应；随后由模型根据这段 Probe 历史
预测：正式执行时应该用多大的峰值作用力 `F_peak`，才能在规定时间 `T` 内把抽屉拉到目标位移 `d_goal`。

本仓库已实现完整的**物理底座 + 数据管线 + 适应模型**：官方环境验证、力驱动的 Probe / Execution 两个公开
控制器、隐藏动力学随机化、连续 Sequential Protocol、counterfactual branching、无泄漏数据集、ACE、PSP、
privileged teacher、direct-regression baselines，以及 Isaac Sim 物理闭环评估。**尚未**包含 SPC / VLM /
RL policy；RMA² baseline 由 `baselines/rma2/` 独立负责，主方法不依赖它。

论文正式设定 **Setting V1** 已于 Phase 13 冻结：Probe 为固定预算激励（3.5 N，0.3 s，对所有 xi 完全相同，
不依赖 `d_goal`，跑完整个 profile），任务为 `d_goal = 100 mm` / `T_goal = 1.5 s`，只搜索 `F_peak`；
`T_goal` 是任务条件而非适应参数。成功判据拆成 `reach_success`（主）与 `stable_success`（次）两个指标。
详见 [docs/PROBE_V1.md](docs/PROBE_V1.md) 与 `docs/DECISIONS.md` D044–D046。

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
[x] Phase 9A Repository baseline re-validation
[x] Phase 9B Hidden state fixed at four dimensions [m, mu_s, mu_d, b]
[x] Phase 9C Hidden-state capability audit (15 candidates probed)
[x] Phase 9D Observation expansion: 25 channels, causal derivatives, deployability registry
[x] Phase 9E Force-channel audit
[x] Phase 9F Execution snapshot at T + zero-force cleanup
[x] Phase 9G TaskEvaluator: position + terminal velocity + validity
[x] Phase 9H Execution sweep (coarse 495 rows, fine 5 x 4536 rows)
[x] Phase 9I Valid operating region, thresholds anchored to measurements
[x] Phase 9J Oracle success landscape
[x] Phase 9K Main-task parameter selection
[x] Phase 9L Probe re-calibration
[x] Phase 10A Task precision refined: eps_d 15 -> 7.5 mm, eps_v 0.08 -> 0.03 m/s
[x] Phase 10B Sequential protocol: probe -> inference gap -> execution, no reset
[x] Phase 10C Sequential Oracle (5616 rows, 97.2 % valid) -- the authoritative ground truth
[x] Phase 10D Reset vs sequential comparison: required force x0.80 median, ranking preserved
[x] Phase 10E Formal dataset schema with leak-free grouped splits
[x] Phase 11A Repository baseline re-validation
[x] Phase 11B Counterfactual post-Probe branching: validated, two bugs found
[x] Phase 11C Sobol xi sampling, label-independent force strata, normalised storage
[x] Phase 11D/E Smoke + pilot datasets, nine-gate audit
[x] Phase 11F Dataset v0: 512 xi x 3 probes x 32 candidates
[x] Phase 11G DataLoader with dynamic padding, train-only normalisation
[x] Phase 11H/I/J Baselines, privileged teacher, ACE + PSP
[x] Phase 11K/L Offline comparison, ablation, closed-loop Isaac Sim deployment
[x] Phase 12   2-D (F,T) landscape, goal-distance feasibility, probe redesign -- all
               measured, all frozen OUT of the paper setting (see docs/DECISIONS.md D044-D046)
[x] Phase 13A  Structure audit: two public controllers, experiments demoted, docs reconciled
[x] Phase 13B  Setting V1 frozen: fixed-budget probe 3.5 N / 0.3 s, d_goal 100 mm, T_goal 1.5 s
[x] Phase 13C  reach_success / stable_success split; Dataset v1 schema; task-conditioned PSP
[x] Phase 13D  Dataset v1 pilot (96 xi x 3 probes x 32 candidates), nine-gate audit passed
[ ] Phase 13E  Full Dataset v1 -- deliberately NOT started; gated on review of the pilot
[ ]            SPC
[ ]            VLM
```

**The physics question is answered.** At the Phase 10/11 task, different hidden states need
peak forces spanning **0.20-4.30 N -- a 21.5x range** -- with success bands a median 0.20 N wide.
One force cannot serve every drawer, and a standardised probe's response correlates with the
force each drawer needs at **|rho| = 0.97**. Details: [docs/ORACLE_LANDSCAPE.md](docs/ORACLE_LANDSCAPE.md).

**Setting V1 (the paper's setting) is frozen.** A standardised fixed-budget probe -- 3.5 N
through a smoothstep trapezoid over 0.3 s, identical for every hidden state, run to completion
-- then a 100 mm goal in 1.5 s with only `F_peak` searched. At this operating point the
required force spans **0.70-5.40 N, a 7.7x range**, and the probe's nine deployable features
recover it at leave-one-out RMSE **0.333 N** on a target sd of 1.411 N. Derivation:
[docs/PROBE_V1.md](docs/PROBE_V1.md), decisions D044-D046.

**Success is now two numbers, not one.** `reach_success` (position + validity) is primary;
`stable_success` adds the terminal velocity and is secondary. At Setting V1's operating point
they are **24/24 and 0/24** -- reaching 100 mm inside 1.5 s leaves the drawer moving at
0.05-0.08 m/s where `eps_v` is 0.03. Setting V1 therefore poses a *reaching* task, not a
*placement* task; that is reported as a limitation rather than tuned away.

On the Dataset v1 pilot (96 hidden states, 9,216 rows, all nine audit gates passed), one seed
of the full chain gives **ACE + PSP 77.8 %** selection success on held-out drawers against a
privileged teacher's **86.7 %**, the best scalar baseline's **73.3 %**, and a single fixed
force's **13.3 %**.

Phase 10 replaced that reset with the real thing — probe, a fixed inference gap, then the
execution, with nothing reset or written in between — and re-selected the task against it. The
position tolerance halved to **7.5 mm** and the terminal-velocity tolerance fell to
**0.03 m/s**, at a *higher* coverage (**0.972**, 105 of 108 hidden states) than the looser
Phase 9 task achieved. Across those hidden states the required force spans **0.20-4.30 N**, a
**21.5x** range. The reset turned out to matter: it overstates the required force by a median
factor of **1.25** (per-state ratios 0.32-1.02), though it preserves the *ranking* of hidden
states (**+0.95**). A single scalar probe feature still tracks the answer at **|rho| = 0.91**,
but not tightly enough to replace a model. Details:
[docs/SEQUENTIAL_PROTOCOL.md](docs/SEQUENTIAL_PROTOCOL.md).

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

# Phase 9C -- what else the simulator would let us hide, probed live
python scripts/audit_hidden_states.py --headless

# Phase 9E -- every force channel against the drawer's equation of motion
python scripts/audit_force_channels.py --headless

# Phase 9H -- sweep (xi, F_peak, T). Coarse first, then a fine grid at one ramp-down
python scripts/sweep_execution_space.py --headless --stage coarse
python scripts/sweep_execution_space.py --headless --stage fine --fall-fraction 0.20 \
    --output outputs/logs/sweep_fine_fall020.json

# Phase 9J/9K -- Oracle landscape and the task recommendation (no Isaac Sim)
python scripts/build_oracle_landscape.py

# Phase 9L -- calibrate the standardised probe against the selected task
python scripts/calibrate_probe.py --headless

# Phase 9M -- the figures that carry the argument
python scripts/plot_experiment_space.py                       # no Isaac Sim
python scripts/plot_probe_identifiability.py --headless

# Phase 10B -- validate the sequential protocol and re-derive the inference gap length
python scripts/validate_sequential_protocol.py --headless --transition-steps 0 2 4 8 12 --repeats 6

# Phase 10C -- the authoritative Oracle: probe -> gap -> execution, no reset (~2.5 h)
python scripts/build_sequential_oracle.py --headless
python scripts/build_sequential_oracle.py --headless --fall-fraction 0.30 \
    --output outputs/logs/sequential_oracle_fall030.json

# Phase 10A -- re-select the task against the sequential Oracle (no Isaac Sim)
python scripts/refine_task_space.py

# Phase 10D -- what the reset was hiding, and probe features vs the sequential answer
python scripts/compare_reset_vs_sequential.py                 # no Isaac Sim

# Phase 10 figures A-G (no Isaac Sim)
python scripts/plot_phase10.py

# Phase 11B -- is one probe allowed to answer many candidate forces? (the gate)
python scripts/validate_branching.py --headless --preset medium --drift-force 2.5 \
    --branch-forces 2.5 --bias-force 2.5 --branch-repeats 2

# Phase 11D -- a smoke dataset first; it must pass the audit before the full run
python scripts/generate_dataset.py --headless --num-xi 8 --repeats 2 --candidates 6 \
    --num_envs 8 --dataset-version smoke --output outputs/dataset_smoke
python scripts/audit_dataset.py --dataset outputs/dataset_smoke        # no Isaac Sim

# Phase 11F -- Dataset v0 (about 40 min)
python scripts/generate_dataset.py --headless --num-xi 512 --repeats 3 --candidates 32 \
    --num_envs 32 --dataset-version v0 --output outputs/dataset_v0
python scripts/audit_dataset.py --dataset outputs/dataset_v0           # no Isaac Sim

# Phase 11H-K -- baselines, teacher, student, ablation (no Isaac Sim)
python scripts/train_models.py --dataset outputs/dataset_v0 --seeds 0 1 2 --ablation

# Phase 11L -- closed-loop deployment on unseen hidden states
python scripts/evaluate_closed_loop.py --headless --run outputs/training/run_XXXX \
    --dataset outputs/dataset_v0

# Phase 11 figures A-I (no Isaac Sim)
python scripts/plot_phase11.py --dataset outputs/dataset_v0 --run outputs/training/run_XXXX

# Plots (no Isaac Sim needed)
python scripts/visualize_response.py --all --profile-invariance

# Tests
python -m pytest tests/unit -q          # 371 tests, ~5 s, no Isaac Sim
python -m pytest tests/integration -q   # 69 tests, ~106 s, launches Isaac Sim once
```

Add `--video` to `run_official_drawer.py`, `test_probe_pull.py` or `test_execution_pull.py`
to write an MP4 into `outputs/videos/`.

### Minimal API usage

```python
from probe_drawer.envs import DynamicsRandomizer
from probe_drawer.evaluation import evaluate_execution
from probe_drawer.experiment_plan import MAIN_TASK, RECOMMENDED_EXECUTION_CFG, RECOMMENDED_PROBE_CFG, RECOMMENDED_PROBE_TASK, TRAINING_XI_RANGES
from probe_drawer.pull_system import PullSystem, PullSystemCfg

system = PullSystem.build(
    PullSystemCfg(num_envs=1, probe=RECOMMENDED_PROBE_CFG, execution=RECOMMENDED_EXECUTION_CFG)
)
randomizer = DynamicsRandomizer(TRAINING_XI_RANGES.as_randomizer_cfg(), seed=0)

system.reset()
xi = randomizer.apply(system.env, randomizer.sample(system.env.num_envs))   # the hidden dynamics
probe = system.probe.run(**RECOMMENDED_PROBE_TASK.as_kwargs())             # what the robot may see

system.reset()
randomizer.apply(system.env, xi.requested)
execution = system.execution.run(peak_force=2.5, duration=MAIN_TASK.duration)

report = evaluate_execution(execution, MAIN_TASK.criteria)   # evaluation, never control
report.success[0]
```

---

## 5. Project structure

```
9.2paper/
├── configs/            generated snapshots of the dataclass defaults + the recorded grasp pose
├── src/probe_drawer/
│   ├── envs/           environment configuration, research initialisation, dynamics randomisation
│   ├── controllers/    force profiles, shared hybrid OSC, Probe and Execution controllers
│   ├── evaluation/     validity mask and success labelling -- outside the controllers by design
│   ├── analysis/       offline: hidden-state audit, force channels, sweeps, Oracle, probe features
│   ├── state_machines/ approach-and-grasp state machine (used to record the grasp pose)
│   ├── sensors/        read-only accessors, pull axis, causal differentiation
│   ├── logging/        per-episode JSON metadata + NPZ trajectory
│   ├── utils/          version introspection, path resolution
│   ├── observations.py what every logged channel is, and who may consume it
│   ├── experiment_plan.py  the parameters Phase 9 selected, with their provenance
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
| [docs/HIDDEN_STATE_AUDIT.md](docs/HIDDEN_STATE_AUDIT.md) | every hidden-state candidate the simulator offers, and why four were chosen |
| [docs/FORCE_CHANNEL_AUDIT.md](docs/FORCE_CHANNEL_AUDIT.md) | what each "pull force" actually measures |
| [docs/EXPERIMENT_SPACE.md](docs/EXPERIMENT_SPACE.md) | the valid operating region and the selected parameters |
| [docs/ORACLE_LANDSCAPE.md](docs/ORACLE_LANDSCAPE.md) | the Oracle success landscape and how the task was chosen |
| [docs/SEQUENTIAL_PROTOCOL.md](docs/SEQUENTIAL_PROTOCOL.md) | the protocol the paper runs: probe, gap, execution, no reset |
| [docs/DATASET_SCHEMA.md](docs/DATASET_SCHEMA.md) | one training sample, and how to split without leaking |
| [docs/COUNTERFACTUAL_BRANCHING.md](docs/COUNTERFACTUAL_BRANCHING.md) | whether one probe may answer many candidate forces, and the evidence |
| [docs/DATASET_V0.md](docs/DATASET_V0.md) | what Dataset v0 contains, and its audit |
| [docs/TRAINING_V0.md](docs/TRAINING_V0.md) | the first training round: baselines, teacher, ACE + PSP |
| [docs/DECISIONS.md](docs/DECISIONS.md) | design decisions that are settled, and why |
| [docs/SESSION_LOG.md](docs/SESSION_LOG.md) | what each work session changed |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | released state |
| [CLAUDE.md](CLAUDE.md) | **read first** if you are an agent working on this repository |

---

## 7. Selected experiment parameters

All measured, never guessed. Authoritative in `probe_drawer.experiment_plan`, mirrored in
`configs/experiment_plan.yaml`, reasoning in [docs/EXPERIMENT_SPACE.md](docs/EXPERIMENT_SPACE.md).

| | Value |
|---|---|
| `T_goal` | 1.5 s |
| `d_goal` | 40 mm, measured from *before* the probe (D027) |
| `epsilon_d` / `epsilon_v` | 7.5 mm / 0.03 m/s |
| `F_peak` range | 0.15 - 4.5 N (the union of every hidden state's success band) |
| execution ramp-down | 35 % of the duration |
| inference gap | 8 control steps (133 ms) of zero pull force |
| probe | 1.0 -> 6.0 N over 1.0 s, stop at 3 mm or 0.08 m/s, budget 1.5 s |
| training `xi` | m 4-12 kg, mu_s 0.5-3.0 N, mu_d/mu_s 0.3-1.0, b 2-10 N s/m |
| OOD `xi` | m 2-18 kg, mu_s 0.25-4.5 N, mu_d/mu_s 0.15-1.0, b 1-16 N s/m |

## 8. Validation status

All of Phases 0-10 pass. 371 unit tests pass; 84 integration tests are defined. Highlights, with the
values actually measured (full table in [docs/VALIDATION.md](docs/VALIDATION.md)):

* the official drawer opens to 308.7 mm; the drawer's travel direction measures exactly
  `(-1, 0, 0)` in the robot base frame, 0.000° from the configured pull axis;
* all four probe stop conditions fire as specified — displacement (5.29 mm at 0.533 s),
  velocity (5.05 mm/s at 0.100 s), max force (exactly 10.000 N at 1.017 s), timeout (0.500 s);
* the execution force profile is invariant in `F_peak`, and the commanded duration is
  executed to within one control step;
* the five held task-space axes drift at most 0.66 mm and 0.41° during a full execution;
* the same `F_peak = 5 N`, `T = 2 s` gives `d(T) =` 326.1 / 141.7 / 59.6 mm on
  easy / medium / hard — consecutive ratios 2.30 and 2.38;
* the drawer's internal resistance channel matches `-(mu_d + b*v)` to within 0.0099 N, and
  the derived delivered force agrees with the wrist sensor to within 0.134 N;
* 105 of 108 hidden states have a succeeding peak force at the Phase 10 task, spanning
  0.20-4.30 N (the Phase 9 reset Oracle's 106/108 over 1.00-4.50 N is superseded --
  see `docs/SEQUENTIAL_PROTOCOL.md` for why the reset overstated the force).

## 9. Known limitations

See the end of [docs/VALIDATION.md](docs/VALIDATION.md) for the current list.

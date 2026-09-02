# Official Isaac Lab baseline, as installed on this machine

Everything below was read off this installation (Isaac Lab 2.3.0 / extension 0.49.0,
Isaac Sim 5.1.0.0), not from online documentation. Regenerate the machine-read part with
`python scripts/inspect_isaaclab.py`, which writes
`outputs/logs/isaaclab_inspection.json`.

Isaac Lab root: `/home/zbh/Downloads/IsaacLab` (source checkout).

## Registered drawer environments

All four expected IDs plus the direct-workflow one are present:

| Environment ID | Registered | Entry point |
|---|---|---|
| `Isaac-Open-Drawer-Franka-v0` | yes | `isaaclab.envs:ManagerBasedRLEnv` |
| `Isaac-Open-Drawer-Franka-Play-v0` | yes | `isaaclab.envs:ManagerBasedRLEnv` |
| `Isaac-Open-Drawer-Franka-IK-Abs-v0` | yes | `isaaclab.envs:ManagerBasedRLEnv` |
| `Isaac-Open-Drawer-Franka-IK-Rel-v0` | yes | `isaaclab.envs:ManagerBasedRLEnv` |
| `Isaac-Franka-Cabinet-Direct-v0` | yes | direct workflow, not used by this project |

## Source files (real paths on this machine)

| What | Path |
|---|---|
| registration | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/__init__.py` |
| base env cfg | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/cabinet_env_cfg.py` |
| joint-position cfg | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/joint_pos_env_cfg.py` |
| IK-absolute cfg | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/ik_abs_env_cfg.py` |
| state machine script | `scripts/environments/state_machine/open_cabinet_sm.py` |
| operational-space controller | `source/isaaclab/isaaclab/controllers/operational_space.py` (+ `operational_space_cfg.py`) |
| OSC action term | `source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py`, class `OperationalSpaceControllerAction` |
| OSC action cfg | `source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py`, class `OperationalSpaceControllerActionCfg` |
| Franka asset cfgs | `source/isaaclab_assets/isaaclab_assets/robots/franka.py` |
| official OSC example | `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/osc_env_cfg.py` |

## `Isaac-Open-Drawer-Franka-IK-Abs-v0` in detail

| | |
|---|---|
| Environment config | `ik_abs_env_cfg.FrankaCabinetEnvCfg` (inherits `joint_pos_env_cfg.FrankaCabinetEnvCfg` -> `cabinet_env_cfg.CabinetEnvCfg`) |
| Robot | `FRANKA_PANDA_HIGH_PD_CFG` (arm stiffness 400, damping 80, gravity disabled) |
| Robot USD | `{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd` |
| Cabinet USD | `{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd` |
| Cabinet pose | position `(0.8, 0, 0.4)`, quaternion `(0, 0, 0, 1)` (w,x,y,z), i.e. rotated 180° about z |
| Arm action | `DifferentialInverseKinematicsActionCfg`, joints `panda_joint.*`, body `panda_hand`, offset `(0, 0, 0.107)`, controller `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")` |
| Gripper action | `BinaryJointPositionActionCfg`, joints `panda_finger.*`, open `0.04`, close `0.0` |
| Action space | 8 = position(3) + quaternion(4) + gripper(1) |
| Simulation | `sim.dt = 1/60`, `decimation = 1`, so `step_dt = 0.016667 s`; `episode_length_s = 8.0` |

### Cabinet articulation, as instantiated

Bodies (9) and their masses in kg:

| Body | Mass |
|---|---|
| `sektion` | 69.749 |
| `drawer_bottom` | 43.277 |
| `drawer_top` | 5.175 |
| `door_left_link` | 3.876 |
| `door_right_link` | 3.876 |
| `drawer_handle_bottom` | 0.1486 |
| `drawer_handle_top` | 0.1486 |
| `door_left_nob_link` | 0.01456 |
| `door_right_nob_link` | 0.01450 |

Actuated joints (4): `drawer_bottom_joint`, `drawer_top_joint` (both prismatic, limits
`[0, 0.4] m`), `door_left_joint`, `door_right_joint`. Defaults as configured: joint
stiffness 10 N/m, damping 1.0 N s/m for the drawers (2.5 for the doors), joint friction 0,
armature 0.

**The drawer this project studies** is `drawer_top_joint`, whose moving assembly is
`drawer_top` plus the rigidly attached `drawer_handle_top`, i.e. 5.324 kg by default.

### Frames

| Sensor | Source prim | Target | Offset |
|---|---|---|---|
| `cabinet_frame` | `{ENV_REGEX_NS}/Cabinet/sektion` | `drawer_handle_top` | pos `(0.305, 0, 0.01)`, rot `(0.5, 0.5, -0.5, -0.5)` — aligns the handle frame with the end-effector convention |
| `ee_frame` | `{ENV_REGEX_NS}/Robot/panda_link0` | `ee_tcp` (`panda_hand`) | pos `(0, 0, 0.1034)` |
| | | `tool_leftfinger`, `tool_rightfinger` | pos `(0, 0, 0.046)` |

Note the two different TCP offsets in the official configuration: the IK action term uses
`0.107` while the `ee_frame` sensor's `ee_tcp` uses `0.1034`. This project uses `0.1034`
throughout, so that the pose the OSC controls is the same point the sensors report.

## Official state machine, `open_cabinet_sm.py`

Runs indefinitely (`while simulation_app.is_running()`). Verified on this machine:

```bash
./isaaclab.sh -p scripts/environments/state_machine/open_cabinet_sm.py --num_envs 8 --headless
```

ran for 240 s with no errors and was then terminated by us; the abort message in the log is
the SIGTERM landing during shutdown, not a failure. The script prints nothing, so the
drawer trajectory was measured separately (see below).

Implemented as a `warp` kernel, `infer_state_machine`. Its actual states on this version —
note the names differ from the ones in the original task description:

| State | Desired TCP target | Gripper | Advances after |
|---|---|---|---|
| `REST` | current TCP pose | open | 0.5 s |
| `APPROACH_INFRONT_HANDLE` | `handle_pose + (-0.1, 0, 0)` | open | waypoint reached, then 1.25 s |
| `APPROACH_HANDLE` | `handle_pose` | open | waypoint reached, then 1.0 s |
| `GRASP_HANDLE` | `handle_pose + (0.025, 0, 0)` | close | 1.0 s |
| `OPEN_DRAWER` | `handle_pose + (-0.015, 0, 0)`, recomputed each step | close | 3.0 s |
| `RELEASE_HANDLE` | current TCP pose | close | terminal |

The offsets are added to the handle *position* without being rotated
(`wp.transform_multiply` with an identity quaternion), so they are base-frame offsets, and
the orientation target is the handle frame's orientation. `position_threshold = 0.01 m`.
Because `OPEN_DRAWER` re-derives its target from the handle each step, it drags the handle
open at a bounded rate rather than commanding an absolute goal.

## What this project measured on the official environment

`scripts/run_official_drawer.py` runs the same environment through the same waypoints using
this project's own plain-`torch` state machine (`DrawerGraspStateMachine`), which adds the
`SETTLE`/`READY` phases the research initialisation needs and is importable as a library.
Results, 4 environments, 9 s, headless:

| Quantity | Observed |
|---|---|
| phase sequence | `REST -> APPROACH_INFRONT_HANDLE -> APPROACH_HANDLE -> CLOSE_GRIPPER -> SETTLE -> OPEN_DRAWER` |
| grasp complete at | t = 3.983 s |
| final drawer displacement | 0.30855, 0.30842, 0.30916, 0.30792 m (saturating below the 0.4 m limit) |
| measured drawer travel direction (world) | `(-1.0, 0.0, 0.0)` in all environments |
| angle from the configured pull axis `-x` | 0.000° |

So the drawer opens along the robot base `-x` axis, exactly axis-aligned. That is what
allows a single coordinate axis to be force-controlled (`PullAxis(index=0, sign=-1)`), and
`scripts/run_official_drawer.py` refuses to export a grasp pose if this ever stops holding.

## Differences from the original task description

| Description said | This installation has |
|---|---|
| states `REST / APPROACH / APPROACH_HANDLE / GRASP / OPEN_DRAWER / RELEASE` | `REST / APPROACH_INFRONT_HANDLE / APPROACH_HANDLE / GRASP_HANDLE / OPEN_DRAWER / RELEASE_HANDLE` |
| a `FrameTransformer` for the handle | `cabinet_frame`, targeting `drawer_handle_top` with an EE-aligned offset |
| "drawer joint" | `drawer_top_joint`; `drawer_bottom_joint` also exists and is untouched |

In every case the installed version was taken as authoritative.

# RMA² official code: reproduction and source analysis

What the official RMA² implementation actually does, read from its source rather than from
the paper, and how far it was possible to run it on this machine.

The official code is **cloned, not vendored**: `baseline/rma2_direct/third_party/` is
git-ignored, so this
document plus the recorded commit hashes are what makes the reproduction repeatable. Nothing
in `src/probe_drawer/` depends on it.

Companion document: [RMA2_TO_DRAWER_MAPPING.md](RMA2_TO_DRAWER_MAPPING.md).

Status legend: ✅ done · ⚠️ partial · ❌ not done.

---

## 1. Repository information

| | |
|---|---|
| repository | `https://github.com/yichao-liang/rma4rma` |
| commit | `2f938f6518709ac8cbda05c294b7765c6d16630d` (2026-04-19, `main`) |
| paper | Liang, Ellis, Henriques, *Rapid Motor Adaptation for Robotic Manipulator Arms*, arXiv:2312.04670 |
| local path | `baseline/rma2_direct/third_party/rma4rma` (git-ignored) |
| size | 4 745 lines of Python across 18 files |
| submodule `ManiSkill2` | `https://github.com/yichao-liang/ManiSkill2`, branch `rma2`, commit `49c3093` (2024-03-16) |
| submodule `stable-baselines3` | `https://github.com/yichao-liang/stable-baselines3`, branch `rma2`, commit `6f0069a` (2024-03-16) |

### 1.1 The documented clone command does not work — first fix

`README.md` says `git clone --recurse-submodules https://github.com/yichao-liang/rma4rma`,
and `.gitmodules` declares both forks. But **no gitlink entries are committed**:

```bash
$ git -C third_party/rma4rma ls-files -s | grep -c 160000
0
$ git -C third_party/rma4rma submodule status      # prints nothing
```

So `--recurse-submodules` and `git submodule update --init --recursive` both succeed and
create nothing, and `environment.yml`'s `pip: [./ManiSkill2, ./stable-baselines3]` then points
at two directories that do not exist. The repository cannot be installed as documented.

**Fix applied** (recorded here rather than committed into `third_party/`, which is kept
pristine):

```bash
cd third_party/rma4rma
git clone -b rma2 https://github.com/yichao-liang/ManiSkill2.git ManiSkill2
git clone -b rma2 https://github.com/yichao-liang/stable-baselines3.git stable-baselines3
```

Both forks exist and are reachable, so this is a packaging defect, not a missing dependency.

## 2. Environment

Isolated from this project's Isaac Lab environment, per the commission's §11. The Isaac Lab
environment `env_isaaclab` was not touched.

| | RMA² env (`rma2`) | this project (`env_isaaclab`) |
|---|---|---|
| Python | 3.11 | 3.11.15 |
| PyTorch | see §2.1 | 2.7.1+cu128 |
| simulator | SAPIEN 2.2.2 (pinned by `ManiSkill2/requirements.txt`) | Isaac Sim 5.1.0.0 |
| ManiSkill2 | 0.5.3 (fork) | — |
| SB3 | fork of 2.x | — |
| gymnasium | `>=0.28.1,<0.30` | — |
| numpy | `<1.24` | — |

Host: Ubuntu 22.04.5, Linux 5.15.0-1097-realtime, Intel i9-14900K, **NVIDIA RTX 5080 16 GB,
driver 580.126.09**.

### 2.1 The GPU is the binding constraint, and it is architectural

The RTX 5080 is Blackwell, compute capability **sm_120**. PyTorch only gained sm_120 support
in the 2.7 CUDA-12.8 builds. The forks were written in March 2024 against PyTorch 2.1-era
wheels, which contain no sm_120 kernels and fail at the first CUDA op with
`no kernel image is available for execution on the device`. The install therefore pins

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

which the SB3 fork's `torch>=1.13` allows. This is a deviation from the official
`environment.yml` (which pins nothing) and it is the *only* way to get a working GPU on this
host. It is recorded as a risk: a 2026 PyTorch running 2024 SB3 code is not the combination
the authors tested.

## 3. Installation — the commands actually run

```bash
conda create -n rma2 python=3.11 -y
PY=~/anaconda3/envs/rma2/bin/python
$PY -m pip install --upgrade pip setuptools wheel
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install baseline/rma2_direct/third_party/rma4rma/ManiSkill2
$PY -m pip install baseline/rma2_direct/third_party/rma4rma/stable-baselines3
$PY -m pip install -e baseline/rma2_direct/third_party/rma4rma
```

Committed, with the patches applied in the right order, as
`baseline/rma2_direct/patches/rma4rma/install_rma2.sh`.

**Result: ✅**, with two version deviations that the official `environment.yml` does not pin
and that this host forces.

| Package | Installed | Note |
|---|---|---|
| torch | **2.11.0+cu128** | §2.1. `torch.cuda.get_device_capability()` returns `(12, 0)`; a GPU matmul runs. |
| sapien | 2.2.2 | as pinned |
| mani_skill2 | 0.5.3 (fork) | `mani_skill2.__version__` does not exist; read with `importlib.metadata`. |
| stable_baselines3 | 2.1.0 (fork) | as pinned |
| gymnasium | 0.29.1 | inside the fork's `>=0.28.1,<0.30` |
| **numpy** | **1.23.5** | see below |

**numpy had to be pinned down, and two more packages with it — second fix.** `pip` resolved
numpy 2.4.6 because nothing in the install chain enforces ManiSkill2's `numpy<1.24` ahead of
torch's requirement. SAPIEN 2.2.2 is a C extension built against the numpy 1 ABI, and the
result is not an import error but a **segmentation fault inside `env.step`** — the environment
constructs and resets perfectly and then the process dies. Fixed with

```bash
pip install "numpy<1.24" "pandas==2.0.3" "matplotlib==3.7.5"
```

pandas and matplotlib come along because the versions pip had installed are numpy-2 builds and
fail to import against numpy 1.23 (`ImportError: C extension: None not built`), which SB3 hits
on import.

## 4. Which official task, and why

**TurnFaucet-v1**, per the commission's §12: it is the only one of the four official tasks
that is articulated manipulation, so its interaction structure — break a joint free, then
drive it against friction and damping — is the closest of the four to drawer pulling.

Its assets are 60 PartNet-Mobility faucets, which are a separate download
(`python -m mani_skill2.utils.download_asset partnet_mobility_faucet`), unlike PickCube which
needs none.

**Result: ⚠️ — TurnFaucet is not runnable on this host, and the fallback is
`PegInsertionSide-v1`.**

`gym.make("TurnFaucet-v1", ...)` raises
`FileNotFoundError: 'data/partnet_mobility/dataset'`, and the download fails:

```
$ python -m mani_skill2.utils.download_asset partnet_mobility_faucet -y
Downloading https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/5046.zip
urllib.error.URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING]>
```

This network reaches the internet through an HTTP proxy (`curl -I` returns
`HTTP/1.1 200 Connection established`, the proxy's `CONNECT` reply) but the TLS tunnel to
`storage1.ucsd.edu` is closed before any bytes arrive — `curl` reports HTTP code `000`, 0 bytes.
The failure is the network, not the code. It rules out **TurnFaucet** (60 PartNet-Mobility
faucets), **PickSingleYCB** (78 YCB objects) and **PickSingleEGAD** (2281 EGAD objects), which
are three of the four official tasks.

Of the two remaining asset-free candidates:

* **`PickCube-v1`** — which is the launcher's own **default** `-e` — cannot be constructed by
  the launcher at all. `config_envs` always passes `inc_obs_noise_in_priv`, `test_eval` and
  `auto_dr` (`config.py:304-312`, `:345-354`), but `PickCubeRMA.__init__`
  (`tasks/pick_cube.py:23-30`) accepts none of them and forwards them to ManiSkill2's
  `BaseEnv`, which raises
  `TypeError: BaseEnv.__init__() got an unexpected keyword argument 'inc_obs_noise_in_priv'`.
  `PickCubeRMA` also references `init_step` before assignment whenever `obs_noise=True` and
  `randomized_training=False` (`tasks/pick_cube.py:43`, `:61`).
* **`PegInsertionSide-v1`** — procedurally generated peg and box, no downloads, and its
  constructor takes the full kwarg set (`tasks/peg_insertion.py:110-120`). **This is the task
  used.**

It is the weaker choice scientifically — an insertion is less like a drawer pull than a faucet
turn — but the reproduction's purpose is to verify the *pipeline and the implementation*, and
every part of RMA² this project cares about (privileged encoder, latent, temporal-CNN adapter,
distillation loss, two-stage schedule, deployment path) is task-independent code.

## 5. Environment test

**Result: ✅** after the numpy fix and patch 0001 (§20.3).

```
obs keys: {'agent_state': (32,), 'object1_state': (6,), 'object1_type_id': (1,),
           'object1_id': (1,), 'obj1_priv_info': (7,), 'goal_info': (14,)}
20 steps ok; reward sum 0.9761 | term False trunc False
info: ['elapsed_steps', 'peg_head_pos_at_hole', 'success']
action (7,)
```

Two things the source analysis predicted are confirmed by measurement:

* `obj1_priv_info` is **7**-dimensional for PegInsertionSide, matching §9.2 exactly;
* `agent_state` is **32** and the action is **7**, so `proprio_dim = 39` — which is the number
  hard-coded throughout `ProprioCNN` (§12).

## 6. Policy training (Stage 1)

**Result: ✅**

```bash
python -m rma4rma.train -e PegInsertionSide-v1 -n 4 -bs 64 -rs 64 \
    --randomized_training --ext_disturbance --obs_noise \
    --total_timesteps 768 --max_episode_steps 50 --seed 0 --log_dir <scratch>/rma2_logs
```

Deliberately tiny (§20.6). Three PPO iterations completed:

```
| rollout/ ep_len_mean | 50        |
| rollout/ ep_rew_mean | 0.5981123 |
| time/ fps            | 16        |
| time/ iterations     | 3         |
| time/ total_timesteps| 768       |
Eval num_timesteps=256, episode_reward=4.03 +/- 10.90
```

Rollouts collect, the PPO update runs, the eval callback fires, and
`ckpt/best_model.zip` plus `ckpt/model_latest.zip` are written. No conclusion about
*learning* is available at 768 steps and none is claimed.

### 6.1 The checkpoint confirms the architecture read in §10–§12

Reading `policy.pth` out of `best_model.zip`:

| Tensor | Shape | Confirms |
|---|---|---|
| `priv_enc.mlp.0/3/6.weight` | `(128, 71)`, `(128, 128)`, **`(67, 128)`** | `d_z = 67` for PegInsertion — 7 privileged + 64 embedding = 71 in, `71 − 4` out (§10) |
| `obj_id_emb.weight` / `obj_type_emb.weight` | `(80, 32)` / `(50, 32)` | the identity embeddings, 64 of the 71 input dims |
| `mlp_extractor.policy_net.0.weight` | `(512, 119)` | `119 = 67 (z) + 32 (agent) + 6 (object) + 14 (goal)` — the concatenation in `models.py:120-125` |
| `action_net.weight` | `(7, 128)` | policy head `[512, 256, 128] → 7` |
| `adapt_tconv.fc.weight` | **`(71, 78)`** | `78 = 39 × 2`, the `ProprioCNN` flatten (§11) — **and an output width of 71, not 67** |

That last row is a bug, and §7 is where it surfaces.

## 7. Adaptation training (Stage 2)

**Result: ✅, after finding and fixing two more defects.** Both were predicted from the source
and then observed.

### 7.1 The adapter is built four dimensions too wide for PegInsertionSide — third fix

`ActorCriticPolicyRMA.__init__` computes the privileged width with **its own rule**
(`policy.py:45-49`): `4 + 3 + 4`, plus one for TurnFaucet. There is **no PegInsertionSide
branch**, while `FeaturesExtractorRMA` (`models.py:41-48`) correctly gives PegInsertionSide 7.
So the adapter is sized from 75 inputs (`d_z = 71`) and the encoder from 71 (`d_z = 67`).

Running stage 2 unpatched:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x123 and 119x512)
```

`123 = 71 + 32 + 6 + 14` against the policy's `119 = 67 + 32 + 6 + 14`. **Stage-2 adaptation
training cannot run at all on PegInsertionSide as published.** The other three tasks are
unaffected: the two rules agree for PickCube, PickSingle* and TurnFaucet. Patch
`0002` mirrors `models.py`'s per-task rule into `policy.py`.

### 7.2 The adapter loop hard-codes 50 environments — fourth fix

Predicted from `adaptation.py:102` and then observed. After 49 clean adapter steps
(loss 0.578 → 0.430) the first episode ended and:

```
File ".../rma4rma/algo/adaptation.py", line 102, in learn
    n_succ = sum([infos[i]["success"] for i in range(50)])
IndexError: list index out of range
```

`range(50)` is the paper's environment count written as a literal. Any reduced-scale
reproduction crashes at its first `done`. Patch `0003` sums over `infos` instead, in both
places it appears.

### 7.3 With all three patches, stage 2 runs

```
Agent Steps: 0000k | FPS:   0.0 | Current Loss: 0.57815 | Succ. Rate: 0.00000
Agent Steps: 0000k | FPS:   9.0 | Current Loss: 0.57132 | Succ. Rate: 0.00000
...
Agent Steps: 0003k | FPS: 166.1 | Current Loss: 0.00001 | Succ. Rate: 0.00000
```

3157 adapter steps, no error, and the L² latent loss falls from **0.578 to ~1e-5**.

**This is not evidence that the adapter works, and it must not be reported as such.**
PegInsertion's randomisation is curriculum-annealed from step 0 to `2e6`
(`tasks/peg_insertion.py:61`), so at ~3 000 steps the ranges are roughly 0.15 % of their full
width — there is almost nothing for `z_priv` to encode, and the adapter is learning a near
constant. What is verified is that the **loop is correct**: the freeze holds, the teacher and
student shapes agree, the MSE is computed and back-propagated, the optimiser steps, and the
loss decreases monotonically.

## 8. Evaluation

**Result: ✅**

The 100-episode evaluation at the end of `train.py` ran to completion and wrote
`eval_results_finegrained.csv`:

```
log_name,expert_adapt,inc_obs_noise_in_priv,only_DR,without_adapt_module,sys_iden,
    success_rate,mean_ep_len,n_eval_eps,env_name,model_id,adapt_loss
PPO,False,False,False,False,False,0.000,200.0,100,PegInsertionSide,002_master_chef_can
```

Success rate 0.000 at a mean episode length of exactly 200.0 — which is the expected result
for 768 training steps on peg insertion, and which also demonstrates §18's point concretely:
`success` here is `ep_len < 200`, so "no episode terminated early" and "no episode succeeded"
are the same statement by construction.

Note the `model_id` column reads `002_master_chef_can` — the `--eval_model_id` default
(`config.py:213`), a YCB object, recorded even for a PegInsertion run.

---

# Source analysis

Everything below is read from the source at commit `2f938f6` and is independent of whether
the code ran. File references are `path:line`.

## 9. Privileged information

Assembled per task into `obs_dict["obj1_priv_info"]`, then concatenated with two identity
embeddings before the encoder (`algo/models.py:104-110`).

### 9.1 TurnFaucet — `tasks/turn_faucet.py:410-418`

| Variable | Dim | Source | Physical meaning | Available on a real robot? |
|---|---:|---|---|---|
| `obj_ang` | 4 | `target_link` centre-of-mass pose quaternion | handle orientation | with pose estimation |
| `angle_dist` | 1 | `target_angle − current_angle` | **task progress**, not physics | with pose estimation |
| `target_joint_axis` | 3 | faucet joint axis | geometry | with a model of the object |
| `obj_density` | 1 | `density / 8e3`, randomised ×0.5–5 | inertia | **no** |
| `obj_friction` | 1 | randomised ×0.5–1.1 | joint friction | **no** |
| `limpulse` | 1 | `‖contact impulse finger1 ↔ target_link‖₂` | grip interaction | with a force sensor, approximately |
| `rimpulse` | 1 | same, finger2 | grip interaction | with a force sensor, approximately |
| **total** | **12** | | | |

`+ 32` object-id embedding `+ 32` object-type embedding = **76** encoder inputs
(`models.py:43-49`).

With `--inc_obs_noise_in_priv`, a further 19 dims are appended (`models.py:52`,
`turn_faucet.py:426-431`): `proprio_noise` (9), `pos_noise` (3), `rot_noise` (4),
`disturb_force` (3) — i.e. the agent is told the realisation of its own observation noise and
the external disturbance.

### 9.2 The other three tasks

| Task | `priv_info_dict` | Dim | Source |
|---|---|---:|---|
| PickSingleYCB / EGAD / PickCube | `obj_ang`(4), `bbox_size`(3), `obj_density`(1), `obj_friction`(1), `limpulse`(1), `rimpulse`(1) | 11 | `tasks/pick_single.py:188-195` |
| PegInsertionSide | `bbox_size`(3), `obj_density`(1), `obj_friction`(1), `limpulse`(1), `rimpulse`(1) | 7 | `tasks/peg_insertion.py:385-391` |

### 9.3 Three observations that matter for our mapping

1. **Only two of the twelve are true hidden physics** (`obj_density`, `obj_friction`).
   The rest is pose, geometry, task progress and contact impulse. "Privileged environment
   information" in RMA² is much broader than "the hidden dynamics parameters".
2. **`angle_dist` is task progress**, so `z_priv` carries a goal-relative signal, not only a
   system-identification signal. The adapter is therefore also learning to estimate progress.
3. **The identity embeddings dominate the input width** — 64 of 76 dims. RMA² leans heavily on
   learned per-instance geometry, which is exactly the part with no analogue for a single
   drawer.

### 9.4 Normalisation

`obj_density` is normalised by `/8e3` (`turn_faucet.py:279`). Nothing else is. There is no
running observation normaliser and no `VecNormalize` on the observations (`config.py` wraps
with `VecMonitor` only). The encoder's own `LayerNorm` is what keeps the scale in hand.

### 9.5 Randomisation is curriculum-scheduled, not fixed

`tasks/turn_faucet.py:60-84`: scale, density, friction, disturbance force and observation
noise all ramp linearly from no randomisation to full range between step `1e6` and `2e6`
(`linear_schedule`, `algo/misc.py`). Ranges for TurnFaucet (`turn_faucet.py:34-47`):

| Parameter | Training range (multiplier) |
|---|---|
| `obj_scale` | 0.7 – 1.2 |
| `obj_density` | 0.5 – 5.0 |
| `obj_friction` | 0.5 – 1.1 |
| disturbance force | 0 – 2, decay 0.8 |
| proprio noise | ±0.005 |
| object position noise | ±0.005 m |
| object rotation noise | ±10° |

At evaluation (`test_eval`) the ranges are widened by `l_scl, h_scl = 0.8, 1.2`
(`turn_faucet.py:24-25`) — i.e. the official evaluation is mildly out-of-distribution by
construction.

## 10. Environment encoder

`FeaturesExtractorRMA`, `algo/models.py:15-147`. It is an SB3 `BaseFeaturesExtractor`, i.e. a
*layer of the policy*, not a separate model.

```python
self.priv_enc = MLP(units=[128, 128, priv_env_out_dim], input_size=priv_enc_in_dim)  # :87
```

`MLP` (`models.py:150-163`) is `Linear → LayerNorm → ELU` per entry in `units`, **including
the last**. So `z_priv` is LayerNorm-ed and ELU-ed — bounded below at −1, roughly zero-mean
per sample. This is load-bearing for the distillation (see §12).

**Latent width.** `priv_env_out_dim = priv_enc_in_dim − 4` (`models.py:50`):

| Task | encoder input | `d_z` |
|---|---:|---:|
| TurnFaucet | 76 | **72** |
| PickSingle* / PickCube | 75 | **71** |
| PegInsertionSide | 71 | **67** |

with `+15` on each when `--inc_obs_noise_in_priv` (`models.py:54`).

**Paper vs code (§19).** The README describes this as distilling privileged information "into
a low-dimensional embedding `z_t`". At `d_z = 72` from a 76-dim input, the encoder removes
four dimensions. It is a *learned re-representation*, not a bottleneck. Any reading of RMA²
that treats `z` as a compact physics code is not supported by this code.

**Forward path** (`models.py:93-147`): `z` (either `e_gt` or the adapter's `pred_e`) is
concatenated with `agent_state`, `object1_state` and `goal_info` and returned as the SB3
feature vector; the policy MLP is `[512, 256, 128]` for both actor and critic
(`train.py:48`).

**Training.** No optimiser, no loss, no scheduler of its own — it is inside `policy.parameters()`
and is updated purely by PPO's gradient. There is no auxiliary system-identification loss.

**`--sys_iden`** (`models.py:58-59`, `:112-113`) bypasses the encoder entirely and feeds the
raw privileged vector to the policy. That is RMA²'s own explicit-system-identification
ablation, and it is directly reusable as the shape of our Explicit SysID baseline.

## 11. Adaptation network

`AdaptationNet`, `algo/models.py:166-194`; `ProprioCNN`, `models.py:248-293`.

```
prop (N, 50, 39)
  ├ channel_transform: Linear(39,39) → LN → ReLU → Linear(39,39) → LN → ReLU
  ├ permute → (N, 39, 50)
  └ temporal_aggregation:
        Conv1d(39,39,k=9,s=2) → LayerNorm((39,21)) → ReLU
        Conv1d(39,39,k=7,s=2) → LayerNorm((39, 8)) → ReLU
        Conv1d(39,39,k=5,s=1) → LayerNorm((39, 4)) → ReLU
        Conv1d(39,39,k=3,s=1) → LayerNorm((39, 2)) → ReLU
     → flatten → (N, 78)
fc  : Linear(78 + camera_param_dim + depth_dim → d_z)
relu
fc2 : Linear(d_z → d_z)                                   → z_hat
```

**The stale comments in `ProprioCNN` are wrong and the code is right.** `models.py:262-271`
annotates the intermediate lengths as `21, 17, 13, 11`; the actual arithmetic from
`calc_activation_shape_1d` gives `21, 8, 4, 2`, which is what makes `self.fc`'s `39 * 2 = 78`
correct (`models.py:181`). Anyone re-deriving the architecture from the comments will build a
different network.

**Output has no LayerNorm** while the teacher's does (§10). The distillation therefore asks an
unnormalised output to match a LayerNorm-ed, ELU-ed target. Preserved deliberately in our
mapping rather than "fixed".

**Depth branch** (`models.py:173-179`, `197-236`): `DepthCNN`, three `Conv2d(3×3)+BN+ReLU+MaxPool`
blocks `1→32→64→128` on a 32×32 image, `fc(128·4·4 → 256) → fc(256 → 64)`, plus a 41-dim
camera parameter vector. Enabled by `--use_depth_adaptation`.

**Two defects worth recording.** `self.perc_cnn = DepthCNN(...)` is constructed unconditionally
(`models.py:179`), so with `use_depth=False` a `Linear(256, 0)` is created and never used. And
`ActorCriticPolicyRMA.test_eval` calls the adapter with only `prop` and `perc`
(`policy.py:80-85`) while `AdaptationNet.forward` unpacks `x["cparam"]` unconditionally
(`models.py:186`) — so `--without_adapt_module` raises `KeyError: 'cparam'`.

## 12. History format

`ActorCriticPolicyRMA`, `algo/policy.py:36-44`, `:126-153`, `:321-340`.

```
prop_buffer : (n_envs, 50, proprio_dim)
proprio_dim = observation_space["agent_state"].shape[0] + action_space.shape[0]
```

For the Panda with `pd_ee_delta_pose`: `agent_state` = proprioception (qpos 9 + qvel 9) +
`base_pose` (7) + `tcp_pose` (7) = 32, action = 7, so `proprio_dim = 39` — which is why 39 is
hard-coded throughout `ProprioCNN`. **The architecture is hard-coded to this one robot and
control mode**, despite `--robot` offering `xarm7` and `xmate3_robotiq`.

Buffer mechanics:

* per step, `state_action_vec = cat(agent_state, prev_actions)` is appended and the oldest
  row dropped (`policy.py:138-144`) — front-padded with zeros, newest last;
* the adapter reads `prop_buffer[:, -50:]` **detached** (`policy.py:149`), so no gradient
  flows into the history;
* on episode end the buffer rows for the finished environments are zeroed
  (`policy.py:326-329`), called from `adaptation.py:99`;
* `prev_actions` is reset per finished environment (`policy.py:313-319`).

The `--use_prop_history_base` ablation instead reconstructs the history *inside the rollout
buffer* (`algo/buffer.py:274-317`), with the episode length hard-coded to 50 in three places.

## 13. Latent

* dimension: 72 (TurnFaucet) / 71 / 67, see §10;
* teacher range: LayerNorm + ELU ⇒ bounded below by −1, unbounded above, per-sample zero mean;
* student range: unbounded (no output normalisation);
* use: concatenated into the policy's feature vector in place of the ground-truth encoding
  (`models.py:116-125`).

## 14. Loss

`algo/adaptation.py:81`:

```python
loss = ((e - e_gt.detach()) ** 2).mean()
```

Plain MSE, teacher detached. No cosine term, no normalisation, no weighting. Identical
expression in the no-update evaluation path (`adaptation.py:185`).

## 15. Training stage 1 — policy

`train.py:181-186` → `PPORMA.learn` (`algo/ppo.py`).

* privileged `e` → `priv_enc` → `z_priv` → concatenated → PPO policy → action;
* `extract_features(..., adapt_trn=False)` ⇒ `use_pred_e = False` (`policy.py:98-108`), so
  Stage 1 always uses the **ground-truth** encoding;
* the encoder is trained **jointly** with the policy by PPO's gradient, with no loss of its
  own;
* PPO settings (`train.py:85-101`): `gamma = 0.85`, `n_epochs = 10`, `target_kl = 0.05`,
  `clip_range = 0.2` annealed to 0.05 by step `1e7`, `lr = 3e-4` (constant unless
  `--lr_schedule`), `n_steps = 2000`, `batch_size = 5000`, 50 parallel environments,
  `max_episode_steps = 50` for training against the registered 200 for evaluation;
* the rollout buffer is `DictRolloutBufferRMA` (`ppo.py:92`);
* `--auto_dr` adds Automatic Domain Randomisation: a 500-episode success queue widens or
  narrows one randomly chosen parameter's range (`ppo.py:268-315`).

## 16. Training stage 2 — adapter

`train.py:166-179` → `ProprioAdapt.learn` (`adaptation.py:58-145`).

1. every parameter whose name does not contain `adapt_tconv` gets `requires_grad = False`
   (`adaptation.py:47-52`); optimiser is `Adam(adapt_params, lr=1e-4)`;
2. `actions, _, _, e, e_gt = self.policy(obs_tensor, adapt_trn=True)` — one forward pass
   produces both the prediction and the teacher and **the action the environment then takes**;
3. `adapt_trn=True` forces `use_pred_e = True` (`policy.py:99-100`), so **the policy acts on
   the adapter's output**. The history the adapter learns from is the one its own predictions
   generated — on-policy distillation, not offline regression on teacher rollouts. This is the
   single most important implementation detail in the file;
4. MSE step, then `env.step(clipped_actions)`;
5. buffers reset for finished environments; checkpoint every `1e4` steps and on best success
   rate;
6. loop bound `while n_steps <= 1e6` (`adaptation.py:73`) — 1 M adapter steps × 50
   environments.

**A hard-coded 50 breaks any smaller run.** `adaptation.py:102` computes
`sum([infos[i]["success"] for i in range(50)])` regardless of `env.num_envs`, so adapter
training with `-n` below 50 raises `IndexError`. Any reduced-scale reproduction must keep
`-n 50` or patch this line.

## 17. Deployment

`policy.test_eval` (`policy.py:70-85`) then `policy.predict` (`policy.py:211-291`).

* `test_mode = True` ⇒ `use_pred_e = True` (`policy.py:99`), so `z_priv` is never computed;
* the adapter reads the same 50-step proprioception buffer, maintained inside `predict`
  (`policy.py:239-267`);
* everything is inside `th.no_grad()` — **pure forward adaptation, no online gradient**;
* ablations flip this back: `--expert_adapt` and `--only_DR` set `use_pred_e = False`
  (`policy.py:101-102`), i.e. `--expert_adapt` is the privileged-oracle upper bound;
* `--without_adapt_module` freezes `pred_e` at the first timestep's prediction
  (`policy.py:79-85`, `:104-107`) — a *one-shot* adaptation ablation, conceptually the closest
  thing in RMA² to this paper's one-probe setting. It is also the code path broken by the
  missing `cparam` key (§11).

## 18. Evaluation

`algo/evaluate_policy.py`, invoked from `train.py:193-213` with `n_eval_episodes = 100`,
`deterministic=True`, on a single-environment eval env.

**Success is inferred from episode length**: `success = np.array(ep_lens) < 200`
(`train.py:210`), 200 being the registered `max_episode_steps` for these envs
(`turn_faucet.py:20`). So "success" means "terminated before truncation". `EvalCallbackRMA`
(`callbacks.py:61-151`) does read `info["is_success"]` properly during training, so the two
success numbers in a run are computed two different ways.

Results append to `logs/eval_results_finegrained.csv` (`train.py:215-240`).

## 19. Paper versus code

| Claim | What the code does |
|---|---|
| "distills into a **low-dimensional** embedding" | `d_z = 72` from a 76-dim input (`models.py:50`). Four dimensions removed. |
| Adaptation from "proprioception history and a depth image" | Correct, and `agent_state` also contains `base_pose` and `tcp_pose`, which are not proprioception. |
| Privileged information = object physics | Only 2 of 12 dims are randomised physics; the rest is pose, geometry, task progress and contact impulse (§9.3). |
| Two-phase training with the policy frozen in phase 2 | Correct and strictly enforced (`adaptation.py:47-52`). |
| Adapter trained by L² regression on `z` | Correct (`adaptation.py:81`), and additionally **on-policy** — the paper's Fig. 2 does not make this explicit. |
| Domain randomisation over a fixed range | Ranges are curriculum-annealed between 1e6 and 2e6 steps, and *widened* at evaluation (§9.5). |
| Table 1 headline results | Not reproducible at this scale; see §20. |

## 20. Problems encountered

1. **Submodules declared but never committed** — §1.1. Blocks installation as documented.
   Fixed by cloning the forks explicitly.
2. **Blackwell GPU vs 2024-era pins** — §2.1. Fixed by installing the cu128 PyTorch build,
   which the forks' `torch>=1.13` permits, at the cost of running 2024 code on a 2026 stack.
3. **`n_envs` hard-coded to 50 in the adapter loop** (`adaptation.py:102`) — a reduced-scale
   adapter run crashes. Not fixed; `-n 50` is used instead.
4. **`--without_adapt_module` raises `KeyError: 'cparam'`** (§11). Not fixed; the ablation is
   not needed for this project.
5. **`ProprioCNN`'s shape comments are wrong** (§11). No code change needed, but they mislead.
6. **Compute.** The paper's numbers were produced on an A100 with 50 parallel environments;
   Stage 1 anneals its randomisation curriculum out to 2e6 steps and its clip range to 1e7,
   and Stage 2 runs to 1e6 adapter steps × 50 envs. On this host that is days per task, per
   seed. The commission's §14 and §56 explicitly scope this out: what is verified here is
   pipeline correctness, not benchmark numbers.

7. **numpy 2 segfaults SAPIEN** — §3. The dependency chain does not enforce ManiSkill2's own
   `numpy<1.24`, and the symptom is a core dump inside `env.step`, not an import error.
8. **The asset server is unreachable** through this network's proxy — §4. Rules out three of
   the four official tasks.
9. **`PickCube-v1`, the launcher's default task, cannot be constructed by the launcher** —
   §4.
10. **The adapter is sized by a different rule than the encoder** — §7.1. Fatal for
    PegInsertionSide stage 2.
11. **`range(50)` hard-coded in the adapter loop** — §7.2, predicted then observed.

## 21. Fixes applied

Four fixes were needed to get from `git clone` to a running two-stage pipeline. Every one is
in `baseline/rma2_direct/patches/rma4rma/`, which is committed even though `third_party/` is
not, so the
reproduction is repeatable.

| # | Problem | Fix | Patch |
|---|---|---|---|
| 1 | submodules declared but never committed (§1.1) | `git clone -b rma2` both forks explicitly | — (a command, not a diff) |
| 2 | no sm_120 kernels for the RTX 5080 (§2.1) | `pip install torch --index-url .../cu128` | — |
| 3 | numpy 2 segfaults SAPIEN 2.2.2 (§3) | `pip install "numpy<1.24" "pandas==2.0.3" "matplotlib==3.7.5"` | — |
| 4 | `joint.set_drive_trget` typo breaks every `env.step` | one-word correction | `0001-maniskill2-fix-set_drive_target-typo.patch` |
| 5 | adapter latent 4 wider than the encoder's on PegInsertionSide (§7.1) | mirror `models.py`'s per-task rule into `policy.py` | `0002-rma4rma-fix-adapter-latent-width-for-peginsertion.patch` |
| 6 | `range(50)` hard-coded in the adapter loop (§7.2) | sum over `infos` | `0003-rma4rma-unhardcode-n_envs-in-adapter-loop.patch` |

Fix 4 deserves emphasis: `mani_skill2/agents/controllers/pd_joint_pos.py:48` calls
`joint.set_drive_trget(...)`, and `git log -S` shows the typo was introduced by the fork's own
**HEAD** commit `49c3093 "add new robot"`. `PDJointPosController.set_drive_targets` is reached
by every PD-joint-position-derived controller, including the `pd_ee_delta_pose` mode all four
RMA² tasks train with, so **the published `rma2` branch cannot execute a single environment
step**. This is not a version-drift problem; it is broken as published.

Nothing else was modified. `git -C third_party/rma4rma diff` shows only patch 0002 and 0003;
`git -C third_party/rma4rma/ManiSkill2 diff` shows only patch 0001.

## 22. Remaining issues

1. **TurnFaucet was not reproduced.** It is the task closest to a drawer and it needs assets
   this network cannot fetch (§4). If the faucet assets can be obtained another way, the
   TurnFaucet run is worth redoing — it is the only official task whose privileged vector
   contains an *articulated joint* term (`target_joint_axis`, `angle_dist`).
2. **No claim is made about learning.** 768 policy steps and 3157 adapter steps verify the
   pipeline. The paper's setting is 50 environments, a randomisation curriculum out to 2e6
   steps, a clip-range schedule out to 1e7, and 1e6 adapter steps — days per task per seed on
   this host, against an A100 in the paper (§20.6). Reproducing Table 1 was explicitly out of
   scope for this round.
3. **The adapter loss falling to 1e-5 is an artefact of the curriculum**, not a result (§7.3).
   A meaningful adapter number needs training past the point where randomisation is actually
   on, i.e. ≫ 2e6 environment steps.
4. **`--without_adapt_module` is still broken** (`KeyError: 'cparam'`, §11). Not patched —
   this project does not need that ablation, and patching unused paths would enlarge the
   deviation from the official code for no benefit.
5. **`PickCube-v1` is still broken** under the launcher (§4). Not patched, same reasoning.
6. **Two success definitions coexist** in one run (§18) and were not reconciled.

## 23. What this means for our baseline

The transferable core of RMA² is small and clean, and it is exactly the part this project
needs: **an encoder over privileged physics trained through a downstream task loss, a frozen
teacher, and a temporal-CNN student distilled onto its latent by plain MSE.** Everything
around it — PPO, the reward, depth, the identity embeddings, the on-policy distillation loop —
is scaffolding for the problem RMA² solved and not for this one.

The design that follows from this reading is in
[RMA2_TO_DRAWER_MAPPING.md](RMA2_TO_DRAWER_MAPPING.md).

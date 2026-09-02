# Patches to the official RMA² dependencies

Applied to `third_party/` clones, which are git-ignored, so the patches live here instead.
Each one is the **smallest change that makes the official code run on this machine**, and
each is justified in [../../docs/RMA2_REPRODUCTION_REPORT.md](../../docs/RMA2_REPRODUCTION_REPORT.md) §21.

Nothing in `src/probe_drawer/` depends on any of this. These exist so the reproduction can be
repeated, not because the project builds on the patched code.

## Applying

```bash
cd third_party/rma4rma/ManiSkill2
git apply ../../../patches/rma4rma/0001-maniskill2-fix-set_drive_target-typo.patch
pip install --no-deps .
```

## The patches

| File | Target | What it fixes |
|---|---|---|
| `0001-maniskill2-fix-set_drive_target-typo.patch` | `yichao-liang/ManiSkill2@49c3093`, `mani_skill2/agents/controllers/pd_joint_pos.py:48` | `joint.set_drive_trget(...)` → `joint.set_drive_target(...)`. The typo was introduced by the fork's own HEAD commit ("add new robot"), so **every** `env.step` under any PD-joint-position-based controller — including the `pd_ee_delta_pose` mode all four RMA² tasks train with — raises `AttributeError`. The branch as published cannot run. |
| `0002-rma4rma-fix-adapter-latent-width-for-peginsertion.patch` | `yichao-liang/rma4rma@2f938f6`, `src/rma4rma/algo/policy.py:45` | `ActorCriticPolicyRMA` computes the privileged-vector width with its own rule (`4+3+4`, `+1` for TurnFaucet) that has **no PegInsertionSide branch**, while `FeaturesExtractorRMA` (`algo/models.py:41-48`) correctly gives PegInsertionSide 7. So the adapter is built to output 71 while the environment encoder's latent is 67, and stage-2 adaptation training dies at the first forward pass with `mat1 and mat2 shapes cannot be multiplied (N x 123 and 119 x 512)`. Observed, then fixed by mirroring `models.py`'s per-task rule. The other three tasks are unaffected — the two rules agree there. |
| `0003-rma4rma-unhardcode-n_envs-in-adapter-loop.patch` | `yichao-liang/rma4rma@2f938f6`, `src/rma4rma/algo/adaptation.py:102` and `:204` | `n_succ = sum([infos[i]["success"] for i in range(50)])` hard-codes the paper's 50 parallel environments, so stage-2 adapter training raises `IndexError: list index out of range` at the first episode end for any other `-n`. Observed at `-n 4` after 49 successful adapter steps. Replaced with a sum over `infos`, which is one entry per environment. |

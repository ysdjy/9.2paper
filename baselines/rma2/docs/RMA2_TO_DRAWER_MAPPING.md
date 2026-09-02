# RMA² → Drawer: what transfers, what does not, and what the baseline becomes

How the official RMA² method (`baselines/rma2/third_party/rma4rma`, commit `2f938f6`) maps onto this
project's Franka + drawer + one-probe + parameterised-pull setting, and what the
**RMA²-inspired Direct Adaptation** baseline is defined to be.

Companion document: [RMA2_REPRODUCTION_REPORT.md](RMA2_REPRODUCTION_REPORT.md), which is
where every claim about the official code is sourced to a file and a line.

Nothing in this document has been implemented. It is the design contract the next phase must
satisfy. §3 records the one decision that is **not ours to make** — the dimensionality of the
skill parameter — and §19 records the risks, every one of them measured from the Oracle
already on disk rather than assumed
(`python baselines/rma2/scripts/audit_adaptation_premise.py`).

---

## 1. The original RMA² pipeline

Two stages, then deployment. Source locations in the reproduction report §10–§17.

**Stage 1 — policy training.** A privileged vector `e` (object pose, bounding box or joint
axis, density, friction, the two fingertip contact impulses, plus two 32-dim learned
embeddings of object *identity* and *category*) is compressed by an MLP into `z_priv`, which
is concatenated with the agent state, the object state and the goal and fed to a PPO
actor–critic. The encoder has **no loss of its own**: it is a layer of the features
extractor and is trained only by PPO's gradient.

```
e  ──►  priv_enc (MLP 128-128-d_z)  ──►  z_priv  ──┐
                                                    ├─►  π (512-256-128)  ──►  a ∈ R⁷
agent_state, object_state, goal  ─────────────────  ┘
```

**Stage 2 — adapter training.** The encoder and the policy are frozen. A temporal CNN reads
the last 50 steps of `(agent_state, previous_action)` — optionally with a depth image — and
regresses `z_priv`:

```
(s, a)_{t-49..t}  ──►  ProprioCNN (4 × Conv1d) ──► fc ──► z_hat
L = mean( (z_hat − stopgrad(z_priv))² )
```

The rollout is driven by the policy **acting on `z_hat`**, so the history distribution is the
adapter's own — an on-policy distillation, not a replay of the teacher's trajectories.

**Deployment.** `z_priv` is never computed. `z_hat` replaces it at every control step. No
gradient is taken online.

## 2. What this paper is actually asking

A different question, and the difference is the whole reason the baseline has to be adapted
rather than ported.

RMA² asks: *given an unknown object, what low-level action should I take at this control
step?* It adapts a **policy**, continuously, over a 200-step episode.

This paper asks: *given one short probe of an unknown drawer, what parameters should the
already-correct pull skill be run with?* It adapts a **parameter vector**, once, from a
single probe. The geometry of the pull is a solved problem handled by a deterministic
controller (`ExecutionPullController`); no low-level policy is learned, and no reward is
defined.

Concretely, the setting is fixed by `probe_drawer.experiment_plan` and
[SEQUENTIAL_PROTOCOL.md](SEQUENTIAL_PROTOCOL.md):

| | |
|---|---|
| hidden state `ξ` | `[mass, μ_s, μ_d, b]`, 4-D, `TRAINING_XI_RANGES` |
| probe | 1 → 6 N over 1 s, stop at 3 mm / 0.08 m/s / 6 N / 1.5 s |
| inference gap | 8 control steps (133 ms) of zero pull force, no reset, no braking |
| execution | `F(t) = F_peak · φ(t/T)`, `T = 1.5 s`, ramp-down 35 % |
| task | `d_goal = 40 mm` from *before* the probe, `ε_d = 7.5 mm`, `ε_v = 0.03 m/s` |
| skill parameter | `p = [F_peak, T]` — **two-dimensional, decided in §3**. The Oracle on disk covers `F_peak` at `T = 1.5 s` only and must be re-swept. |

## 3. The skill parameter is `p = [F_peak, T]`

**Decided (2026-09-02, by the project owner). Recorded as D034.**

The commission specified `p = [F_max, v_cmd]`. This repository has no `v_cmd`: the pull axis
is force-controlled throughout — `HybridPullOSC` commands a wrench on it and holds the other
five axes in pose — and `grep -r "vcmd\|v_cmd" src/` returns nothing. What the execution
controller accepts is

```python
ExecutionPullController.run(peak_force: float | Sequence[float], duration: float)
```

so `[F_peak, T]` is available **with no new control code**, while `[F_max, v_cmd]` would
require a new velocity-tracking or admittance mode, a re-run of all of Phase 6's controller
validation, a new Oracle and a new task selection — a new benchmark, not a new baseline.

`[F_peak, T]` was chosen over keeping `p` one-dimensional for a reason that is measured rather
than aesthetic, and it is set out in full in §19.4: **in one dimension the success set is an
interval and its midpoint succeeds for 104 of 105 hidden states**, so a single-output
regressor cannot be beaten by a success-landscape model on the grounds of multi-modality —
there is none. In two dimensions the success set is a *region*, which can be curved or
disconnected, and the mean of two succeeding parameter pairs need not succeed. The paper's
central claim needs the second dimension to be testable at all.

### 3.1 What this costs, and what has to happen first

Nothing in the controllers, the protocol, the evaluator or the dataset schema changes. Three
things do:

1. **The Oracle must be re-swept over a `(F_peak, T)` grid.** The Phase 9 coarse sweep already
   reached `T = 3 s` without the simulation misbehaving, so the region is known to be
   physically valid; `SweepRecord` and `SweepDataset` already carry `duration` as a first-class
   axis and `SweepDataset.surface()` already returns an `(F_peak, T)` surface. Cost is roughly
   the current 2.5 h multiplied by the number of `T` values.
2. **`MAIN_TASK` must be re-selected against the 2-D landscape**, by the same scored rule in
   `analysis/oracle.py` that selected the current one (D024). `T` stops being a task constant
   and becomes a chosen parameter, so `MainTask.duration` becomes a *range* and the task
   definition is `(d_goal, ε_d, ε_v)` alone.
3. **`rma2/adaptation_premise.py` must be generalised from bands to regions** and re-run.
   Its 1-D vocabulary — `success_forces`, `low`/`high`, `centre`, `interior_failures` — becomes
   a set of grid cells, a centroid, and a connected-component count. The connected-component
   count is the number the paper actually wants: it is the direct measurement of whether the
   success region is multi-modal, and therefore of whether Ours has anything to offer over
   Direct Regression. §19.4 is the 1-D answer; this is the question it could not ask.

Until those three are done, everything downstream of them — the dataset, Stage A, Stage B — is
blocked, because the oracle target `p_oracle` does not exist in two dimensions yet.

## 4. Module mapping

| Official RMA² | Source | Drawer baseline | Action | Why |
|---|---|---|---|---|
| Domain randomisation, curriculum-scheduled | `tasks/turn_faucet.py:23` `set_randomization` | `envs/DynamicsRandomizer` over `ξ` | **keep**, drop the curriculum | There is no RL curriculum to stabilise here — the dataset is generated once, offline. |
| Privileged vector `e` (physics + identity embeddings) | `tasks/*.py` `priv_info_dict` | `ξ = [m, μ_s, μ_d, b]` | **keep, shrink** | D015 fixes `ξ` at four dimensions. There is one drawer, so identity/category embeddings have nothing to index. |
| Object id / type `nn.Embedding(80/50, 32)` | `models.py:90` | — | **delete** | RMA² uses them as a geometry proxy across 78 YCB / 60 faucet instances. One drawer, one geometry (§20 of the report). |
| `priv_enc = MLP([128,128,d_z])` | `models.py:87` | `PrivilegedEncoder`, same MLP class | **keep** | Directly transferable; only the input and output widths change (§7). |
| `z_priv`, LayerNorm+ELU, `d_z = d_in − 4` | `models.py:150-163`, `:50` | `z_priv ∈ R^8` | **keep the block, replace the width rule** | `d_in − 4` gives 0 for a 4-D `ξ`. §7. |
| PPO actor–critic `[512,256,128]` → `a ∈ R⁷` | `train.py:48`, `policy.py` | `ParameterHead` MLP → `p` | **replace** | This paper does not learn low-level actions. §9. |
| PPO reward, GAE, clipping, ADR | `ppo.py` | — | **delete** | No reward is defined; the supervision is the Oracle label. §61 of the commission forbids inventing one for this baseline alone. |
| `AdaptationNet` = `ProprioCNN` + 2 fc | `models.py:166-194`, `:248-293` | `ProbeAdaptationEncoder` | **keep, re-widthed** | The Conv1d stack transfers *unchanged* — see §8, this is a real fidelity win. |
| History = last 50 `(agent_state, prev_action)` | `policy.py:36-44`, `:142` | probe history, 90 steps × 8 channels | **keep the shape, change the content** | §8. |
| `DepthCNN` | `models.py:197` | — | **delete** | No vision in this baseline; §43 of the commission defers the VLM entirely. |
| `L = ((e − e_gt.detach())²).mean()` | `adaptation.py:81` | identical | **keep** | §11. |
| Freeze everything but `adapt_tconv`, Adam 1e-4 | `adaptation.py:47-53` | identical | **keep** | §11. |
| On-policy distillation (policy acts on `z_hat`) | `policy.py:126-157` | — | **delete, and say so** | §11 — the probe is open-loop and independent of `z`, so there is no induced distribution to drift. |
| Deployment: `z_hat` replaces `z_priv`, no online gradient | `policy.py:99-116` | identical | **keep** | §12. |
| ManiSkill2 / SAPIEN | — | Isaac Lab drawer | **replace** | §36 of the commission. The official tasks are for verifying the official code only. |

## 5. Inputs

Every tensor the RMA²-style baseline may read, and from where. `TrainingSample` fields are
from `dataset/schema.py`; the deployability class is from `observations.py` (D017).

| Tensor | Shape | Source field | Class | Used by |
|---|---|---|---|---|
| `xi` | `(4,)` | `TrainingSample.xi` | `SIM_ONLY_PRIVILEGED` | **teacher only**, Stage A |
| `probe_history` | `(90, 8)` | `TrainingSample.probe_history` | `DEPLOYABLE` | student, Stage B/deployment |
| `post_probe_state` | `(2,)` | `TrainingSample.post_probe_state` | `DEPLOYABLE` | both branches |
| `goal` | `(2,)` | `goal_displacement`, `duration` | task-given | both branches |

The 8 history channels are `observations.DEFAULT_ACE_INPUT` (7 channels) plus `active`, the
padding mask, which is `DEPLOYABLE` and must be an input rather than an implicit convention so
that a zero-padded tail is distinguishable from a stationary drawer.

`xi` reaching the student is a silent, fatal bug of exactly the kind D017 exists to prevent.
The assembly point must call `observations.validate_model_input()` and
`schema.validate_probe_history()`, and a unit test must assert that the student module's
`forward` signature cannot accept `xi` at all.

## 6. Outputs

```
p* = [F_peak, T]     ∈ R²
```

Each component is squashed into `P_safe` with a `sigmoid` affine map — onto
`MAIN_TASK.peak_force_range` for `F_peak` and onto the swept duration range for `T` — rather
than clipped, so the gradient never vanishes at the bound and no number is hard-coded in the
module. The two components have different units and very different scales, so the head
predicts in **normalised** parameter space and one shared `parameter_space.py` owns the
normalisation, the squash and the inverse. Both losses and both error metrics are reported in
normalised units *and* in newtons and seconds; a raw-unit MSE over `[F_peak, T]` would be
dominated by whichever axis happens to have the larger numbers.

## 7. Privileged encoder

```python
PrivilegedEncoder = MLP(units=[128, 128, d_z], input_size=4)   # models.py:150, verbatim
```

Reusing RMA²'s `MLP` — `Linear → LayerNorm → ELU` per layer, **including the output layer** —
matters and is not cosmetic: it is why `z_priv` is bounded and roughly zero-mean per sample,
which is what makes a plain MSE distillation well conditioned. Reimplementing it without the
output LayerNorm would change the distillation problem while claiming to reproduce it.

`ξ` must be normalised to zero mean and unit variance over `TRAINING_XI_RANGES` before the
encoder, with the statistics stored in the checkpoint (they are part of the model, and an OOD
`ξ` must be normalised with the *training* statistics or the OOD result is meaningless).

**`d_z = 8`, and RMA²'s width rule is not transferable.** RMA² sets `d_z = d_in − 4`
(`models.py:50`), which is 71 for PickSingle and 72 for TurnFaucet; with `d_in = 4` it gives
0. Eight is chosen as small enough to be a bottleneck over a 4-D `ξ` and large enough not to
be one; `{4, 8, 16}` is an ablation, not a tuning knob to be run against the test set.

## 8. Probe adaptation encoder

The official `ProprioCNN` (`models.py:248`) is
`Linear(C,C) → LN → ReLU → Linear(C,C) → LN → ReLU`, then four `Conv1d` layers with kernels
`(9, 7, 5, 3)` and strides `(2, 2, 1, 1)`, each followed by `LayerNorm` and `ReLU`, then a
flatten and two fully connected layers.

**The kernel stack transfers unchanged if the window is the probe budget.** RMA² uses a
50-step window; the natural probe window here is the probe's *budget*, 1.5 s = 90 control
steps at 60 Hz, so no probe is ever truncated. Propagating 90 through the official stack:

```
90 --k9,s2--> 41 --k7,s2--> 18 --k5,s1--> 14 --k3,s1--> 12        flatten = 8 × 12 = 96
```

versus RMA²'s `50 → 21 → 8 → 4 → 2`, flatten `39 × 2 = 78`. So the architecture is kept
verbatim and only the channel count (8 instead of 39) and the flatten width change. This is
the strongest available answer to §32 of the commission — no "minimal necessary modification"
to the temporal CNN is needed at all, provided the window is 90 and not the median probe
length of ~28 steps, at which the official stack is arithmetically invalid (the third
convolution would need a length-5 input and receives 2).

The median probe lasts 0.467 s ≈ 28 steps, so roughly two thirds of the window is zero
padding. That is what the `active` channel is for. **Left-aligned, tail-padded**, because the
probe's information is front-loaded (breakaway) and RMA²'s buffer semantics — front-padded,
newest last — would put every probe's informative segment at a different offset.

Output: `z_probe ∈ R^{d_z}`, `fc(96 → d_z) → ReLU → fc(d_z → d_z)`, as in `models.py:181-183`.
Note the official adapter's output carries **no** LayerNorm while the teacher's does; that
asymmetry is preserved deliberately (it is what the official code does) and recorded here so
it is not "fixed" by accident.

Ours' ACE may use a GRU. That is allowed and is the point: same inputs, same data, different
architecture (§14).

## 9. Parameter head

```
[ z (d_z) ; post_probe_state (2) ; goal (1) ]  →  128 → 128 → 64 → 2   (ELU, LayerNorm)
```

`goal` is `d_goal` alone now: `T` has moved from the task definition to the predicted
parameter (§3.1), so feeding it as a condition would be feeding the answer.

Shared by both branches, and by construction identical in Stage A and Stage B — the head is
trained in Stage A and **frozen** in Stage B, exactly as RMA² freezes the policy.

Capacity is deliberately comparable to Ours' PSP rather than minimal: §26 of the commission
forbids a strawman, and a 3-layer head over a 12-input vector is already far more capacity
than a 1-D problem needs.

## 10. Training Stage A — privileged direct adaptation

```
ξ → PrivilegedEncoder → z_priv
(z_priv, post_probe_state, goal) → ParameterHead → p̂
L_A = ‖p̂ − p_oracle‖²   in normalised parameter units
```

Both modules are trained end to end here, mirroring RMA² Stage 1 in which the encoder has no
loss of its own and is trained only through the downstream objective.

`p_oracle` must be the point of the success region that is **furthest from its boundary** —
the 2-D generalisation of the 1-D band centre, and computable as the arg-max of a distance
transform over the region's grid cells. Maximum margin is what a predictor with error should
aim at. Two consequences follow and both must be handled explicitly rather than discovered
later:

* **A disconnected region has more than one such point.** Which one is labelled `p_oracle`
  is then an arbitrary choice, and it is exactly the case in which a single-output regressor
  is being asked an ill-posed question. The number of hidden states in this situation must be
  counted and reported (§3.1 item 3) — it is the paper's evidence, not an inconvenience.
* **The choice is shared, without exception, with Direct Regression** (§51 of the
  commission). One implementation, in `methods/common/oracle_target.py`.

Stage A is the baseline's **ceiling**. If it cannot fit, nothing downstream is interpretable,
and §29 of the commission's checklist applies before any adapter is trained.

## 11. Training Stage B — latent distillation

```
τ_p → ProbeAdaptationEncoder → z_probe
L_B = mean( (z_probe − stopgrad(z_priv))² )        # adaptation.py:81, verbatim
```

Frozen: `PrivilegedEncoder`, `ParameterHead`. Trainable: the adapter only, Adam `lr = 1e-4`
(`adaptation.py:53`). Implemented by the same `requires_grad = False` sweep over named
parameters that the official code uses, so the freeze is auditable.

**One official mechanism has no analogue and is deliberately dropped.** RMA² collects the
history *while the policy acts on `z_hat`*, so the adapter is trained on its own induced
distribution. Here the probe is a fixed open-loop force ramp that does not depend on `z` at
all, so there is no distribution to drift and the distillation is offline over a fixed
dataset. This is a simplification of the official method and must be stated as one; it is
also the honest consequence of the setting, not a shortcut.

## 12. Deployment

```
τ_p → ProbeAdaptationEncoder → z_probe → ParameterHead → p* → ExecutionPullController
```

`ξ` is not read. No online gradient. Identical to RMA² deployment modulo the policy/head
substitution.

## 13. Optional Stage C — end-to-end fine-tuning

`L_C = L_B + λ‖p̂(z_probe) − p_oracle‖²`, adapter and head unfrozen. Permitted only after A
and B both succeed, and **the pure Stage-B numbers must be reported alongside**, because
Stage C is no longer RMA²-style: RMA² never lets a parameter-level loss reach the adapter.

## 14. Fair comparison

The three methods differ in exactly one place — how `p` is produced after the probe.

| | Direct Regression | RMA²-inspired | Ours |
|---|---|---|---|
| privileged teacher | no | **yes** | yes (for ACE) |
| latent distillation | no | **yes** | yes |
| what is predicted | `p*` | `p*` | `P(success \| z, p, goal)` |
| how `p` is chosen | the output | the output | SPC search over candidates |
| probe | identical | identical | identical |
| controller, task, split, seeds | identical | identical | identical |

Everything else is held by construction: one `SequentialPullProtocol`, one
`RECOMMENDED_PROBE_TASK`, one `MAIN_TASK`, one `SplitCfg(level="xi_id")`, one
`OperatingRegionCfg`, one candidate grid, one evaluation seed protocol.

## 15. Difference from Direct Regression

Direct Regression is RMA²-style **with the privileged branch deleted**:
`τ_p → Encoder → p*`, trained in one stage on `L = ‖p̂ − p_oracle‖²`.

To keep the comparison about the *mechanism* and not about capacity, its encoder must be the
same `ProbeAdaptationEncoder` and its head the same `ParameterHead`, with the same widths.
Then the only difference is that RMA²-style trains the head against a privileged latent first
and distils into it, and Direct Regression does not. **If the two are implemented with
different encoders, the comparison measures architecture and the paper cannot claim anything
about privileged distillation.** They get separate directories, configs and result files
(§33) precisely so that this cannot happen by accident.

## 16. Difference from Ours

RMA²-style commits to one `p` before it knows whether that `p` succeeds. Ours predicts
`P(success | z, p, goal)` over the candidate set and lets SPC choose — so it can express
"anything in `[1.4, 1.6]` works", "nothing works", or "two separated regions work", none of
which a single-output regressor can represent.

## 17. Dataset requirement

`docs/DATASET_SCHEMA.md` already specifies the row. What Stage A/B additionally need, and
which the schema's §6 leaves open:

* **`probe_history` must actually be stored.** The Oracle files on disk carry probe
  *features*, not histories (`SweepRecord.probe_features`), so they are Oracle evidence and
  cannot train an adapter. A generator that emits `TrainingSample` rows does not exist yet.
* **Window and rate fixed at 90 steps × 8 channels**, tail-padded, per §8.
* **Repeats per hidden state.** The protocol's intrinsic `d_total(T)` noise is ~1 mm against
  `ε_d = 7.5 mm` (D028), so a single probe per `ξ` gives a label that is usually but not
  always right. Repeats are what let the label become a probability, which Ours' PSP needs
  and which the RMA²-style band centre also benefits from.
* **Far more than 108 hidden states.** 108 is a grid built to map a landscape; a teacher
  trained on 108 points split 70/15/15 has ~75 training drawers. §47 of the commission's
  100–500 range is the minimum and the upper end should be the target.
* **The 3 unsolvable hidden states have no `p_oracle`** and must be excluded from the
  regression target while remaining in the success-landscape dataset — an asymmetry Ours can
  exploit legitimately and the RMA²-style baseline structurally cannot. Worth reporting as a
  finding rather than hiding by dropping them everywhere.

## 18. Implementation plan

Next round, in this order. Nothing here is written yet.

Everything below is **inside `baselines/rma2/`** unless marked otherwise. This baseline
owns its own code; it imports `probe_drawer` and never edits it (§14).

```
src/rma2/
    parameter_space.py                     P_safe, normalisation, the squash — one definition
    oracle_target.py                       max-margin point of the success region
    dataset.py                             TrainingSample -> tensors; validate_model_input at the boundary
    metrics.py                             §48 metrics, computed once
    privileged_encoder.py                  §7
    adaptation_encoder.py                  §8
    parameter_head.py                      §9
    trainer.py                             Stage A, Stage B, optional Stage C
    config.py                              dataclass, snapshotted into configs/ per D011
    adaptation_premise.py                  exists; needs the band -> region generalisation
configs/rma2.yaml                   config snapshot, drift-tested
scripts/train_rma2_privileged.py           Stage A
scripts/train_rma2_adapter.py              Stage B
scripts/eval_rma2.py                deployment + §48 metrics
tests/test_rma2_no_privileged_leak.py      the student cannot read xi
tests/test_rma2_shapes.py                  90x8 -> 96 -> d_z, on CPU, no Isaac Sim
```

Three things are **not** this baseline's to write, because every method needs them and they
must be identical across all of them (§14). They belong to the main project:

```
scripts/sweep_task_space_2d.py             Isaac Sim; re-sweep the Oracle over (F_peak, T)   [§3.1 item 1]
scripts/generate_probe_dataset.py          Isaac Sim; sequential protocol; emits TrainingSample rows
src/probe_drawer/experiment_plan.py        MainTask.duration becomes a range                 [§3.1 item 2]
```

If this baseline ends up needing a fourth such thing, it goes to the project too — a shared
component living inside one method's folder is how an unfair comparison starts.

Also to be extended, not written fresh:

```
src/rma2/adaptation_premise.py             bands -> regions; connected components  [§3.1 item 3]
src/probe_drawer/experiment_plan.py               MainTask.duration becomes a range       [§3.1 item 2]
```

Gate order (§44–§47): controller ✓ → randomisation ✓ → probe identifiability ✓ (§19.3) →
**2-D Oracle re-sweep** → **2-D task re-selection** → **premise audit re-run on the region** →
dataset generation → tiny overfit (10–50 `ξ`) → Stage A → adapter overfit → Stage B →
small-scale eval (100–500 `ξ`).

## 19. Risks, measured where possible

Every number below is produced by `python baselines/rma2/scripts/audit_adaptation_premise.py`
(module: `src/rma2/adaptation_premise.py`, report: the *project's*
`outputs/logs/adaptation_premise.json`),
run against `outputs/logs/sequential_oracle_fall035.json` at `MAIN_TASK`. All model estimates
are leave-one-out over the 105 solvable hidden states.

**These describe the *current, one-dimensional* Oracle**, swept over `F_peak` at a fixed
`T = 1.5 s`. §19.4 is the reason the task moved to two dimensions (§3), and §19.1–19.6 must
all be re-measured on the 2-D landscape before any of them is quoted in the paper. §19.1,
§19.2 and §19.3 are properties of the drawer and the probe and should survive; §19.4 and
§19.6 are properties of the *parameter space* and are precisely what the re-sweep is expected
to change.

**19.1 Adaptation is necessary, and this is solid.** The best single fixed force (0.70 N)
succeeds on **20/108 = 18.5 %** of hidden states; the next four best constants score 19, 19,
16, 16. Required forces span 0.25–4.30 N, a **17.2×** range. There is no universal parameter,
so the Fixed Conservative baseline is genuinely weak and the study is well posed.

**19.2 The required force is essentially one hidden dimension.** Correlation of the band
centre with each `ξ` dimension:

| `mass` | `μ_s` | `μ_d` | `b` |
|---|---|---|---|
| +0.023 | +0.785 | **+0.987** | +0.077 |

`μ_d` alone explains the answer; mass and damping are very nearly irrelevant to it. So
"4-D hidden-dynamics adaptation" is, at this operating point, close to a 1-D regression, and a
privileged encoder over `ξ` has little to compress. This is a weakness of the *task*, not of
the baseline, and the fix is a task whose answer depends on more of `ξ` — which a
two-dimensional parameter space (§3 option 2) would plausibly give, since `T` interacts with
mass and damping in a way `F_peak` alone does not.

**19.3 The probe identifies friction and nothing else.** Leave-one-out linear readout of each
`ξ` dimension from the nine probe features:

| dimension | R² | RMSE |
|---|---|---|
| `joint_static_friction` | **+0.946** | 0.214 N |
| `joint_dynamic_friction` | **+0.883** | 0.282 N |
| `drawer_mass` | +0.251 | 2.81 kg |
| `joint_damping` | **−0.107** | 3.44 N·s/m |

Damping is not identified at all — worse than predicting its mean — which confirms D032
quantitatively and extends it to mass. Since `μ_d` is what determines the answer (§19.2), the
probe happens to measure the one dimension that matters. That is a fortunate alignment, not a
designed one, and it will not survive a task that depends on mass or damping.

**19.4 There is no within-`ξ` multi-solution structure, so regression-to-the-mean cannot arise
the way §28/§49 of the commission anticipates.** The median hidden state has **3** succeeding
forces on a 0.05 N grid, the median band is **0.20 N** wide, only **5 of 105** bands have any
interior failure and the largest such gap is a single grid point, and the band midpoint
succeeds for **104 of 105** states. In a one-dimensional parameter space whose success set is
an interval, averaging is safe by construction; the hypothesised failure mode of direct
regression essentially does not exist here, and a paper claiming it would be claiming
something this data contradicts.

This is the most important finding of this round and it is what settled §3: in two dimensions
a success *region* can be curved or disconnected and the mean of two succeeding parameter
pairs need not succeed. In one dimension it always does. **Whether the 2-D regions actually
are multi-modal is now an open empirical question**, and answering it — by counting connected
components per hidden state on the re-swept landscape (§3.1 item 3) — is the first thing the
next round should do, before any model is trained. If the 2-D regions turn out to be convex
blobs, the same objection returns and the paper's framing has to change rather than the
measurement.

**19.5 The ambiguity that does exist is between drawers, not within one.** Standardising the
nine probe features and clustering hidden states by radius in that space — the radius standing
in for how finely an encoder resolves the probe:

| radius (z units) | mean cluster | median force spread | cluster mean fails |
|---|---|---|---|
| 0.25 | 1.0 | 0.00 N | 1/105 = 1.0 % |
| 0.5 | 1.1 | 0.00 N | 4/105 = 3.8 % |
| 1.0 | 2.7 | 0.22 N | **31/105 = 29.5 %** |

and each state's nearest probe-neighbour's force fails on it **57 %** of the time. So if the
encoder resolves the probe to better than ~0.5 z-units the problem is effectively
deterministic; if it only resolves to ~1 z-unit, averaging over the residual ambiguity fails a
third of the time. Which regime holds is an empirical question about the encoder, and it is
the quantity that decides whether a success-landscape model has anything to offer over a
regressor. **It should be re-measured in the learned latent space** once an encoder exists;
the same function does it.

**19.6 The task is precision-limited, and R² is a misleading metric here.** The band
half-width is **0.100 N** on a **1.50 N** median target — a **±7 %** tolerance. Consequently:

| source | readout | R² | RMSE | lands in the band |
|---|---|---|---|---|
| probe features | linear | +0.898 | 0.352 N | 0.333 |
| probe features | quadratic | +0.912 | 0.326 N | 0.333 |
| probe features | 3-NN | +0.874 | 0.392 N | 0.333 |
| true `ξ` | linear | +0.984 | 0.139 N | 0.771 |
| true `ξ` | quadratic | **+0.992** | **0.099 N** | **0.867** |
| true `ξ` | 3-NN | +0.917 | 0.317 N | 0.305 |

A readout explaining 90 % of the variance in the required force is inside the success band a
third of the time, because the residual (0.33 N) is three times the band half-width. **The
paper must report success rate, not R², and any parameter-error metric must be read against
0.10 N.**

These also bracket the experiment usefully: 0.185 (best constant) → ~0.33 (a simple readout of
the probe) → 0.867 (a simple readout of the true `ξ`, i.e. the Oracle-`ξ` baseline). That is a
wide, healthy dynamic range for the comparison to live in — and it means a learned method that
does not clear ~0.33 is not beating a quadratic fit on nine hand-made features.

Both bracket numbers are *lower* bounds at this dataset size: they are fitted on 105 hidden
states, and a model trained on 500 would do better on both ends.

**19.7 Dataset scale.** RMA² distils over ~10⁶ environment steps of on-policy history. Here
the teacher sees one `z_priv` per hidden state and the adapter one probe per episode. With 108
hidden states there are, at the `xi_id` split level, ~75 training drawers — small enough that a
four-layer temporal CNN over a 90 × 8 window will overfit trivially. Either the grid grows or
the adapter shrinks, and shrinking it away from the official architecture costs the fidelity
§8 just bought.

**19.8 Leakage.** `SplitCfg(level="xi_id")` and `assert_no_leakage` exist and are unit tested,
but have never been run on real data. The generator must build `xi_id` from the values **read
back from `root_physx_view`** (D016), not from the requested ones, or two rows that PhysX
collapsed onto the same drawer will land on both sides of the split.

**19.9 OOD is not yet buildable.** `OOD_XI_RANGES` is defined, but no OOD sweep exists on
disk, and the commission's §40 three-way ID / OOD-single / OOD-compositional evaluation needs
three generated files, not one.

**19.10 Several commissioned metrics have no referent in this design.** Of §48's list:
execution time is a constant (`T = 1.5 s`) by construction, there is no timeout (the execution
always runs the full duration), and "force violation" is the controller's absolute
`SafetyLimits` abort, which no valid Oracle row ever triggers. They should be dropped from the
metric table rather than reported as identically zero, and `v_cmd` prediction error does not
exist until §3 is decided.

## 20. What was deliberately not carried over, and why

Recorded so that a reviewer asking "why is this not RMA²?" gets an answer with a reason
attached rather than an omission.

| Dropped | Reason |
|---|---|
| PPO, reward, GAE, ADR | No low-level policy is learned. Inventing a reward for this baseline alone would violate the shared-benchmark rule. |
| Depth CNN, camera params | No vision in this round (§43 of the commission). |
| Object id / category embeddings | One drawer. They are RMA²'s geometry proxy across ~78 object instances and index nothing here. |
| On-policy distillation | The probe is open-loop and independent of `z`; there is no induced distribution to correct. §11. |
| `sys_iden` mode | Retained conceptually — it is exactly the Explicit SysID baseline (§50 item 4), and `FeaturesExtractorRMA.sys_iden` (`models.py:58`) shows the official way to do it: skip the encoder and feed raw `ξ`. Worth reusing there, not here. |
| Curriculum-scheduled randomisation | The dataset is generated offline; there is nothing to stabilise. |

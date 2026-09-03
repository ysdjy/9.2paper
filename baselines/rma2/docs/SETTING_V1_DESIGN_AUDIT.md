# Design audit: is RMA²-inspired Direct Adaptation still a valid baseline under Setting V1?

A review, not an implementation. Nothing here is built or trained, and the main method is
untouched.

Read against: main project `docs/DECISIONS.md` D044–D047, `docs/DATASET_V1.md`,
`docs/TRAINING_V1.md`, and this baseline's `README.md` §2 and
`RMA2_TO_DRAWER_MAPPING.md` §3, §8, §14–§19.

**Verdict: yes, and it is now a more useful baseline than it would have been under the 2-D
plan — but for the opposite reason to the one the old documents give.**

---

## 0. The old blocker is not merely outdated; it was refuted

`README.md` §2 and `RMA2_TO_DRAWER_MAPPING.md` §3 block Stage A on this argument:

> in one dimension the success set is an interval whose midpoint succeeds for 104 of 105
> hidden states, so a success-landscape model cannot beat a single-output regressor on the
> grounds of multi-modality — there is none.

That argument is sound and its conclusion is false. Setting V1 is one-dimensional again
(D044: only `F_peak` is adapted, `T_goal` is a task condition), and on it a success-landscape
model beats a direct regressor **by a wide margin, measured in physics**:

| | in-distribution, 88 unseen drawers | OOD, 64 drawers |
|---|---|---|
| ACE + PSP (landscape + search) | **91.3 %** | **47.4 %** |
| Direct GRU (point regressor, same encoder) | 81.4 % | 33.3 % |
| gap | **+9.9 pp** (5/5 slot permutations, +6.8 to +15.9) | **+14.1 pp** |

So the premise behind D034 — "no multi-modality ⇒ regression is as good" — does not hold
empirically. The 2-D detour was motivated by a mechanism that turned out not to be the
operative one.

**What is the operative one?** The task is *precision-limited*, which the old audit already
said in §19.6 and then did not follow through: the success band is 0.30 N median against a
0.64–5.51 N required force, so a regressor's error has to be smaller than a ±0.15 N half-band
to land. A landscape model does not emit a number that must be accurate; it scores a grid and
takes an argmax, which is robust to a calibration offset that would sink a regressor. §19.6's
own table showed a readout with `R² = 0.90` landing inside the band only a third of the time.

That reframes what this baseline tests, and the new framing is cleaner than the old one.

---

## 1. What the baseline should be, under Setting V1

### Inputs

| Tensor | Shape | Source | Class | Branch |
|---|---|---|---|---|
| `xi` | `(4,)` | `TrainingSample.xi` | **SIM_ONLY_PRIVILEGED** | teacher only, Stage A |
| `probe_history` | **`(18, 7)`** | `TrainingSample.probe_history` | DEPLOYABLE | student, Stage B + deployment |
| `post_probe_state` | `(2,)` | `TrainingSample.post_probe_state` | DEPLOYABLE | both |
| `task_condition` | `(2,)` | `goal_displacement`, `duration` | task-given | both |

Three changes from the mapping's §5, all consequences of the frozen probe:

* **18 × 7, not 90 × 8.** Setting V1's probe is a fixed-budget 0.3 s excitation, so every one
  of Dataset v1's 1 536 histories is exactly 18 steps (verified). The window is no longer a
  design choice.
* **The `active` padding-mask channel is no longer needed and should be dropped.** It existed
  to distinguish a zero-padded tail from a stationary drawer. There is no padding now. Keeping
  it would feed a constant column.
* **`task_condition` is `(d_goal, T_goal)` again**, not `d_goal` alone. §9 of the mapping
  removed `T` because D034 had made it a predicted parameter; D044 put it back in the task.
  Both are constant across Dataset v1 and therefore carry no information — they belong in the
  input for contract symmetry with PSP (which takes the same five conditioning inputs), not
  because they help.

### Latent

`z_priv`, `z_probe ∈ R^{d_z}`. **`d_z` should be 16, matching `PspCfg.z_dim`,** not the
mapping's 8. The mapping picked 8 before ACE's width was fixed; a baseline given half the
bottleneck of the method it is compared against is a capacity handicap, and §26 of the
commission forbids a strawman. `{8, 16, 32}` remains a legitimate ablation.

Keep RMA²'s `MLP` block verbatim (`Linear → LayerNorm → ELU`, including the output layer) for
the teacher, and keep the official asymmetry that the adapter's output carries no LayerNorm.
Those are the parts of RMA² that make the latent-MSE distillation well conditioned.

### Output

```
F_peak* ∈ R¹,  sigmoid-affine onto SETTING_V1_TASK.peak_force_range = [0.5, 6.5] N
```

One dimension, not two. The mapping's §6 normalised-parameter-space machinery
(`parameter_space.py`, per-axis scaling, dual-unit reporting) was there because `[F_peak, T]`
mixes newtons and seconds. With a scalar output it is unnecessary; a single squash is enough,
and the extra module should not be built.

### The one design decision that must be settled before coding

**The official temporal CNN no longer fits.** The mapping's §8 calls the kernel stack
transferring unchanged "the strongest available answer to §32 of the commission — no minimal
necessary modification is needed at all". That was true at a 90-step window. At 18 steps it is
arithmetically dead:

```
official (9,7,5,3) / strides (2,2,1,1):
   90 -> 41 -> 18 -> 14 -> 12    OK          (the old plan)
   18 ->  5 -> INVALID           k=7 needs >= 7, got 5   (Setting V1)
```

So a choice is forced, and it is a fairness choice rather than a fidelity one:

* **(a) Reuse ACE's `AdaptationContextEncoder` (GRU).** Then all three methods share one
  encoder — `GruForceRegressor` already uses that same class — and the *only* differences left
  are the two mechanisms being studied. This is the cleanest possible comparison and needs no
  change to any existing baseline or number.
* **(b) Keep a temporal CNN, re-kernelled** to something that fits 18 steps, e.g. `(5,3,3,3)`
  with unit strides giving `18 → 14 → 12 → 10 → 8`. Higher RMA² fidelity, but it reintroduces
  an architecture difference into a comparison meant to be about distillation.

**Recommendation: (a) as the headline configuration, (b) as a reported architecture ablation.**
That keeps the mechanism claim clean and still answers "does RMA²'s own encoder matter here?".
Either way §8 must be rewritten; it currently asserts something false.

---

## 2. What actually distinguishes the three methods now

| | Direct GRU | **RMA²-inspired** | ACE + PSP |
|---|---|---|---|
| encoder | `AdaptationContextEncoder` | *same*, under (a) | `AdaptationContextEncoder` |
| privileged teacher | no | **yes** | yes |
| distillation | none | **latent MSE**, `‖z_probe − sg(z_priv)‖²` | **logit**, weight 0.5 |
| latent MSE on ACE | — | — | **deliberately 0** (Phase 13 §G) |
| predicts | `F*` | `F*` | `P(reach_success \| z, F, task)` |
| chooses `F` by | its output | its output | argmax over a 0.05 N grid |
| training | one stage | **two stages, head frozen in B** | one stage, joint |

**This is the comparison's missing cell, and that is the strongest argument for building it.**
Right now ACE + PSP beats Direct GRU by +9.9 pp, and that gap confounds *two* changes:
privileged distillation, and landscape-plus-search versus a point output. Nothing in the
current results separates them. RMA²-inspired sits exactly in between and splits the gap into
two attributable halves:

```
Direct GRU  --(+ privileged latent distillation)-->  RMA2-inspired  --(+ landscape & search)-->  ACE + PSP
   81.4 %                                                  ?                                        91.3 %
```

That single number is worth more to the paper than a second baseline scoring somewhere in the
range. It is also the honest test of the paper's own claim: if RMA²-inspired lands near 91 %,
the credit belongs to distillation and the landscape formulation is decoration; if it lands
near 81 %, the landscape is doing the work.

A second, smaller difference worth stating rather than smoothing: **RMA² distils latents and
ACE distils logits**, and ACE sets `latent_weight = 0.0` on purpose. So "both use
distillation" is true only loosely. The mapping should say which kind.

---

## 3. Can it use Dataset v1 directly? Yes — with the target taken from the shared helper

Every requirement in mapping §17 is now met, and one is met differently than expected.

| §17 requirement | status under Dataset v1 |
|---|---|
| `probe_history` actually stored | ✅ 18 × 7 float32 per probe, all 1 536 |
| fixed window and rate | ✅ 18 steps at 60 Hz, constant by construction — no padding needed |
| repeats per hidden state | ✅ 3 independent probes per `xi` |
| far more than 108 hidden states | ✅ **512** |
| shared `xi_id` split | ✅ by construction, hashed group keys — no seed coordination |
| `xi` available to the teacher | ✅ stored, `SIM_ONLY_PRIVILEGED` |
| **`p_oracle`, the regression target** | ⚠️ **not stored — must be derived, and the shared derivation already exists** |

Dataset v1 stores per-candidate `reach_success` labels, not a per-probe required force. The
mapping asked for the *max-margin* point of the success region. Two reasons not to build that:

1. `probe_drawer.training.metrics.reference_force_per_probe` already exists, is **already the
   target used by Direct GRU, the ridge and the linear baseline**, and is defined for **all
   1 536 probes** (range 0.64–5.71 N, median 2.99) including the 17 with no positive. Using a
   different target would make this baseline's error metric incomparable with the regressors it
   is meant to sit beside — mapping §10's own rule ("the choice is shared, without exception").
2. A max-margin centre over 32 jittered candidates spanning 6 N is resolvable to ~0.19 N at
   best, against a 0.30 N band. The margin it would estimate is mostly grid noise.

**So: reuse `reference_force_per_probe`, add nothing.** Mapping §10's `oracle_target.py` and
§18's `scripts/generate_probe_dataset.py` and `sweep_task_space_2d.py` should all be struck —
the first is superseded, the other two are done.

No regeneration of Dataset v1 is needed, and none should happen.

---

## 4. Documents that must be corrected before implementation

Nothing in this list is a code change; all of it is stale design text that would mislead an
implementer. **Not yet applied — awaiting confirmation of this audit.**

### `baselines/rma2/README.md`
* **§1 status** — remove the `[ ] --- blocked ---` line.
* **§2 in full** — the 2-D blocker is void. Replace with §0 above: the blocker is gone because
  the setting refroze to 1-D, *and* the argument that motivated it was empirically refuted.
* **§2's three "not this baseline's to unblock" tasks** — all moot. The 2-D sweep happened
  (Phase 12), the task was re-selected (D044), and Setting V1 is 1-D.
* **the method table** — "Ours (ACE + PSP + SPC)" should be "ACE + PSP"; SPC is not
  implemented, and what ships is an argmax over a 0.05 N grid.
* **§6 provenance** — the "90 × 8 probe window, `d_z = 8`" row and the closing paragraph about
  the kernel stack transferring unchanged are both false under Setting V1.

### `baselines/rma2/docs/RMA2_TO_DRAWER_MAPPING.md`
* **§2 setting table** — every row stale: probe (now 3.5 N / 0.3 s fixed budget), `d_goal`
  (40 → 100 mm), skill parameter (2-D → 1-D).
* **§3 and §3.1** — D034 is superseded by D044. Keep as history, mark superseded.
* **§5** — inputs table: 90 × 8 → 18 × 7, drop `active`, restore `T_goal`.
* **§6** — `p ∈ R²` → `F_peak ∈ R¹`; drop `parameter_space.py`.
* **§8** — the central fidelity claim is arithmetically dead. Rewrite around the (a)/(b) choice.
* **§9** — head input is `[z ; post_probe_state (2) ; task_condition (2)]`; output width 1.
* **§10** — `p_oracle` → `reference_force_per_probe`; drop the max-margin construction.
* **§14** — fair-comparison table: `MAIN_TASK` → `SETTING_V1_TASK`, `RECOMMENDED_PROBE_TASK` →
  `SETTING_V1_PROBE`, and name which distillation each method uses.
* **§17** — mark satisfied, with the one exception in §3 above.
* **§18** — strike the three main-project tasks; they are done.
* **§19.1, §19.2, §19.6** — all measured on the 40 mm 1-D Oracle and must be re-stated at
  Setting V1 or marked as historical. The Setting V1 equivalents already exist in
  `docs/DATASET_V1.md` and `docs/TRAINING_V1.md`.
* **§19.4** — keep the measurement, correct the conclusion: no multi-modality, and the
  landscape model wins anyway, by precision (§0).

### Main project `docs/DECISIONS.md`
* **D034** — add a superseded-by-D044 note. It currently reads as current policy and directly
  contradicts D044. This is the only main-project change proposed, it is one sentence, and it
  touches no code. *Also awaiting confirmation.*

---

## 5. Recommendation on Stage A / Stage B

**Proceed — after the §4 corrections and after settling the encoder choice (§1).**

Reasons, in order of weight:

1. It fills the one genuinely missing cell in the comparison and turns a confounded +9.9 pp
   gap into two attributable numbers. No other planned experiment does that.
2. It is cheap. Dataset v1 is done, the split is shared by construction, the target helper
   exists, no Isaac Sim is needed for Stage A or Stage B, and deployment reuses
   `evaluate_closed_loop.py` — which already supports an OOD population, so the OOD comparison
   comes almost free.
3. Its own ceiling is checkable before any adapter is trained: Stage A is
   `xi → z_priv → head → F*`, and `TRAINING_V1.md` already reports that a privileged model
   reaches 98.1 % in physics. If Stage A cannot fit, the fault is the head or the target, not
   the idea, and that is diagnosable in minutes.

Two risks to name up front, neither blocking:

* **The gap it is meant to measure may be small.** If privileged distillation is worth little
  here, RMA²-inspired will land near Direct GRU and the interesting number will be a null.
  That is still a result and should be pre-committed to as one, so it does not get relitigated
  after the fact.
* **`µ_d` dominates the answer** (§19.2: correlation +0.987 at the old operating point; worth
  re-measuring at Setting V1). If it still holds, the privileged encoder has little to
  compress, which weakens the distillation mechanism for reasons that are the *task's* and
  should be reported as such rather than tuned around.

Suggested gate order, unchanged in spirit from mapping §18 minus the struck items:

```
re-measure §19.1/19.2/19.6 at Setting V1  ->  Stage A overfit (10-50 xi)  ->  Stage A
  ->  adapter overfit  ->  Stage B  ->  offline comparison  ->  closed loop (in-dist + OOD)
```

**Date.** 2026-09-03. No code written, no model trained, main method untouched.

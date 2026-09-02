# Training v0: baselines, a privileged teacher, and ACE + PSP

The first training round on Dataset v0. What was trained, in what order and why, what the
numbers were, and what they do and do not establish.

Code: `scripts/train_models.py`, `scripts/evaluate_closed_loop.py`,
`src/probe_drawer/{models,training}/`. Run: `outputs/training/run_v0/`.
Decisions: `docs/DECISIONS.md` D039. Dataset: `docs/DATASET_V0.md`.

---

## 1. What is being predicted, and by what

```
xi ------------> E_priv ---> z_priv --\
                                        >--- PSP(z, F_candidate, post_probe) -> P(success)
probe history -> ACE ------> z_ace ---/
```

The head is the same class with the same input contract for both, which is what makes their
landscapes comparable, but they are separate instances — a shared head would let the
teacher's gradients reshape what the student has to fit.

`z` is 16-dimensional and the hidden layers are 96 wide: 13 361 parameters for the teacher,
26 385 for the student. Small on purpose. The task's spread is driven almost entirely by
dynamic friction (`docs/ORACLE_LANDSCAPE.md`), so capacity is not the binding constraint, and
a large model would make the comparison against a scalar baseline harder to interpret rather
than easier.

`T_goal` and `d_goal` are fixed in this experiment, so they are stored in every row but not
fed to the network — a constant input contributes nothing but parameters.

## 2. Order, and why it is a gate

**The teacher first.** `E_priv + PSP` is *told* `xi`. It answers a prerequisite: is the
success landscape learnable at all from the four hidden values? If it were not, the data or
the task formulation would be wrong and a student's numbers would mean nothing. It is
therefore a gate, not a baseline.

It passes decisively. Test AUROC **0.9934 / 0.9940** across seeds, AUPRC 0.914 / 0.926, and it
selects a succeeding force for **92.0 % / 90.8 %** of feasible test probes.

**Then the student**, measured against the teacher as an upper bound.

## 3. The student's loss, and what it deliberately is not

The obvious loss is `||z_ace − z_priv||²`. It is wrong here and the reason is measured, not
stylistic: Phase 10 showed the probe barely responds to damping — `b` from 2 to 11 N·s/m
leaves the probe duration and the breakaway force essentially unchanged — so a teacher free to
encode `b` in `z_priv` would set the student a target it cannot observe. The same phase showed
`b` barely affects the required force, so encoding it would not even help.

So the student's objective is the task, plus optional **logit** distillation at weight 0.5:
match the teacher's success landscape, not the teacher's internal coordinates.
`latent_weight` exists, defaults to 0, and any run that raises it records that in its config
(D039).

Class imbalance — 6.5 % positives — is handled with `pos_weight = 14.25`, computed from the
training split as negatives/positives. Reweighting, not resampling: the evaluation set stays
the real distribution, because resampling the training set and then evaluating on a resampled
set would report a success rate no drawer has. Every run records the label distribution it saw
and the flag `resampled: false`.

## 4. How everything is scored

Candidate-level AUROC answers "can the model rank candidates". It is necessary and it is *not*
the task. The task is: choose one force, and have the drawer stop at the goal. So the metric
that decides anything is **selection**: scan a probe's candidates, take the best-scoring one,
and ask whether that candidate actually succeeded.

Rates are reported over all probes and over **feasible** probes only. The distinction matters:
a probe with no succeeding candidate cannot be selected correctly, so including it measures
the dataset's coverage rather than the model.

A force regressor names a force that was never executed, so it is scored by proximity — the
probe's nearest candidate is the one it effectively chose. That is what a deployed system
restricted to answerable forces would get.

## 5. Offline results

Test split, 88 hidden states / 8 448 rows, mean over 3 seeds. See
`outputs/training/run_v0/comparison.json` for the full table including validation and spread.

Everything, test split, mean ± sd over three seeds (the two closed-form fits have no seed):

| model | selection, all probes | selection, feasible only | force MAE | AUROC |
|---|---|---|---|---|
| fixed force (1.31 N) | 12.9 % | 13.6 % | 0.815 N | — |
| A linear (1 feature) | 30.0 % | 31.5 % | 0.318 N | — |
| **B ridge (9 features)** | 46.0 % | **48.2 %** | 0.196 N | — |
| C MLP (9 features) | 22.1 ± 2.4 % | 23.1 ± 2.5 % | 0.415 N | — |
| **D GRU (7-channel history)** | 75.4 ± 1.3 % | **79.0 ± 1.3 %** | 0.069 N | — |
| **teacher (privileged)** | 86.8 ± 0.8 % | **91.0 ± 0.8 %** | 0.068 N | 0.9924 |
| **ACE + PSP** | 83.4 ± 1.4 % | **87.4 ± 1.5 %** | 0.061 N | 0.9922 |

Three gaps, in decreasing size.

**The time series is worth +30.8 points** — D GRU's 79.0 % against B ridge's 48.2 %. Both
predict a single force; only the input differs. This is the largest effect in the table.

**Modelling the landscape is worth +8.4 points** — ACE + PSP's 87.4 % against D GRU's 79.0 %.
Identical seven channels through the identical encoder class; only the output differs.

**The probe costs 3.6 points against knowing `xi`** — 87.4 % against the teacher's 91.0 %.

The force MAE of about **0.06 N** is the number to hold against the task: the median success
band is 0.20 N, so a typical prediction lands comfortably inside it. For contrast, a
leave-one-out readout of the band centre from the nine scalar features achieves RMSE 0.352 N
and lands in the band 33 % of the time (`outputs/logs/adaptation_premise.json`), which is the
same story from a different direction.

**Baseline C is undertrained, not informative.** The MLP on summary features scores 23.1 %,
*below* the ridge on the same nine features. It is fitted full-batch for 30 epochs, which is
too few, and no attempt was made to tune it — the phase's budget went to the comparisons that
answer the question. It should be read as "not a working baseline in this run" rather than as
evidence about MLPs.

## 6. Closed-loop results — the number that counts

`scripts/evaluate_closed_loop.py`, 64 test hidden states, seed 0. Each drawer is probed once;
the post-probe state is snapshotted; each method's chosen force restores it and executes. So
every method faces an identical drawer **and an identical probe** — the reason the branching
validation had to come first. One probe per method would let a lucky probe decide the ranking.

All 88 test hidden states, seed 0:

| method | physical success | median `|d−goal|` | forces chosen |
|---|---|---|---|
| teacher (privileged, told `xi`) | **95.5 %** | 1.74 mm | 0.50–3.40 N |
| **ACE + PSP** | **93.2 %** | 2.22 mm | 0.45–3.40 N |
| D GRU (history → one force) | 79.5 % | 2.28 mm | 0.45–3.45 N |
| B ridge (9 scalar features) | 45.5 % | 7.21 mm | 0.70–3.30 N |
| A linear (one scalar feature) | 18.2 % | 14.99 mm | 0.95–2.60 N |
| fixed force (1.31 N) | 13.6 % | 29.97 mm | — |

No method produced an invalid episode. The ordering and the gaps reproduce the offline table,
which is the check that the offline metric measures the physical task. The learned models
score a few points *higher* in physics than offline because deployment searches a 0.05 N grid
while the offline metric can only pick among that probe's 32 training candidates.

**The probe history carries information the scalar features discard.** 93.2 % against the best
scalar baseline's 45.5 % — a factor of 2.05. The scalar baselines' failure is visible in the
last column: the linear fit chose forces spanning only 0.95–2.60 N against the teacher's
0.50–3.40 N. It *under-adapts*, because a linear fit on a visibly curved relationship shrinks
toward the mean, and a 0.20 N band does not forgive a magnitude error.

**The probe is very nearly sufficient.** ACE + PSP lands 2.3 points below a model that is
*given* the four hidden values.

**Predicting the landscape beats regressing one force, by 13.7 points on identical input.**
ACE + PSP and baseline D see exactly the same seven channels through the same encoder class;
only the output differs. This is the one comparison in which the gap is attributable to the
modelling rather than to the observation.

## 7. Honesty about the weakest baseline

Baseline A is a **linear** fit on one feature, and the relationship is strongly curved —
Phase 10 measured Spearman 0.910 against Pearson 0.841 for that feature, and the fit's
residual is plainly non-random. So 18.8 % understates what a scalar-feature approach can
achieve; the ridge and MLP baselines in the three-seed table are the fair comparison, and the
parallel premise audit's quadratic readout reaches 33 % in-band.

The correction is applied: the closed-loop table above reports the ridge, and the honest
headline is **93.2 % against 45.5 %**, a factor of 2.05 — not the 5.1× that comparing against
the linear fit would have suggested.

## 8. Ablation: how much of the probe is needed

Four encoder input sets, same architecture, same training, mean ± sd over three seeds on the
test split. Wrist force is deliberately absent (D018): it is recorded in the dataset as a
diagnostic and can be added without regenerating anything.

| channels | selection (feasible) | force MAE | AUROC |
|---|---|---|---|
| ACE-1 `[F_cmd, d]` | 85.3 ± 0.9 % | 0.069 N | 0.9881 |
| ACE-2 `[F_cmd, d, v]` | 86.1 ± 1.2 % | 0.068 N | 0.9912 |
| ACE-3 `[F_cmd, d, v, a]` | **87.8 ± 1.7 %** | 0.063 N | 0.9925 |
| ACE-4 (7 channels, + TCP) | 87.4 ± 1.5 % | 0.061 N | 0.9922 |

**The extra observations buy very little.** Commanded force and drawer position alone reach
85.3 %, and the full seven channels reach 87.4 % — a 2.1-point spread across the whole
ablation, against seed-to-seed standard deviations of 0.9 to 1.7 points. Adding velocity is
worth +0.8, adding acceleration +1.7, and adding the three TCP-axis channels is −0.4, i.e.
nothing distinguishable from noise.

Two readings, and the honest one is the second. Optimistically, `[F_cmd, d]` is nearly enough,
which is good news for a real robot with fewer sensors. Sceptically, the derived channels are
functions of `d` — velocity and acceleration are causal differences of it — so a GRU can in
principle compute them itself, and the ablation mostly confirms that rather than measuring new
information. The one channel that is genuinely independent, the wrist force, was excluded by
D018 and is the ablation worth running next.

## 9. What these numbers do not establish

`T` is fixed, so the adapted parameter is one-dimensional. Measured on the Phase 10 Oracle and
reproduced independently: the midpoint of a hidden state's succeeding force set also succeeds
for **104 of 105** solvable states. In one dimension the success set is essentially a
contiguous interval whose midpoint works, so a landscape model has **no structural advantage**
over a single-output regressor — there is no multi-modality to exploit.

The 13.7-point gap between ACE + PSP and baseline D is therefore a gap in *accuracy*, not
proof that the landscape must be modelled. A better regressor could in principle close it.
The question of whether landscape modelling is *necessary* needs a parameter space where
averaging two good answers can give a bad one, which is what D034's `p = [F_peak, T]` is for.

## 10. Reproducing

```bash
python scripts/train_models.py --dataset outputs/dataset_v0 --seeds 0 1 2 \
    --epochs 50 --baseline-epochs 30 --ablation --output outputs/training/run_v0
python scripts/evaluate_closed_loop.py --headless --run outputs/training/run_v0 \
    --dataset outputs/dataset_v0 --seed 0 --num-xi 64
python scripts/plot_phase11.py --dataset outputs/dataset_v0 --run outputs/training/run_v0
```

Every run directory records its training config, the fitted scaler, the label distribution it
saw, per-epoch history, the best checkpoint and the git commit.

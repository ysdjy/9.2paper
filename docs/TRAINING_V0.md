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

Teacher and student, per seed:

| model | seed | test AUROC | test AUPRC | selection (feasible) | force MAE |
|---|---|---|---|---|---|
| teacher (privileged) | 0 | 0.9934 | 0.914 | 92.0 % | 0.064 N |
| teacher (privileged) | 1 | 0.9940 | 0.926 | 90.8 % | 0.067 N |
| ACE + PSP | 0 | 0.9925 | 0.910 | 89.2 % | 0.061 N |
| ACE + PSP | 1 | 0.9921 | 0.899 | 85.7 % | 0.063 N |

The force MAE of about **0.06 N** is the number to hold against the task: the median success
band is 0.20 N wide, so the typical prediction lands comfortably inside it. For contrast, a
leave-one-out readout of the band centre from the nine *scalar* probe features achieves an
RMSE of 0.352 N and lands in the band 33 % of the time
(`outputs/logs/adaptation_premise.json`).

## 6. Closed-loop results — the number that counts

`scripts/evaluate_closed_loop.py`, 64 test hidden states, seed 0. Each drawer is probed once;
the post-probe state is snapshotted; each method's chosen force restores it and executes. So
every method faces an identical drawer **and an identical probe** — the reason the branching
validation had to come first. One probe per method would let a lucky probe decide the ranking.

| method | physical success | median `|d−goal|` | median `|v(T)|` | forces chosen |
|---|---|---|---|---|
| teacher (privileged, told `xi`) | **89.1 %** | 1.85 mm | 0.0244 | 0.50–3.35 N, sd 0.71 |
| **ACE + PSP** | **87.5 %** | 2.25 mm | 0.0221 | 0.45–3.30 N, sd 0.70 |
| D GRU (history → one force) | 81.2 % | 1.72 mm | 0.0247 | 0.45–3.40 N, sd 0.74 |
| A linear (one scalar feature) | 18.8 % | 14.28 mm | 0.0314 | 0.95–2.60 N, sd 0.45 |
| fixed force (1.31 N) | 14.1 % | 24.09 mm | 0.0035 | — |

No method produced an invalid episode.

Three findings, in order of importance.

**The probe history carries information the scalar features discard.** 87.5 % against 18.8 %.
The linear baseline's failure is visible in the last column: it chose forces spanning only
0.95–2.60 N with a standard deviation of 0.45 N, against the teacher's 0.50–3.35 N and 0.71 N.
It *under-adapts* — a linear fit on a visibly curved relationship shrinks toward the mean — so
it gets the ordering roughly right and the magnitude wrong, which a 0.20 N band does not
forgive.

**The probe is very nearly sufficient.** ACE + PSP lands 1.6 points below a model that is
*given* the four hidden values. Whatever the probe fails to reveal costs about 1.6 points of
task success at this operating point.

**Predicting the landscape beats regressing one force, by 6.3 points on identical input.**
ACE + PSP and baseline D see exactly the same seven channels through the same encoder class;
only the output differs. This is the one comparison in which the gap is attributable to the
modelling rather than to the observation.

## 7. Honesty about the weakest baseline

Baseline A is a **linear** fit on one feature, and the relationship is strongly curved —
Phase 10 measured Spearman 0.910 against Pearson 0.841 for that feature, and the fit's
residual is plainly non-random. So 18.8 % understates what a scalar-feature approach can
achieve; the ridge and MLP baselines in the three-seed table are the fair comparison, and the
parallel premise audit's quadratic readout reaches 33 % in-band.

The conclusion "the history helps" survives that correction — 87.5 % against 33 % is still a
factor of 2.6 — but the honest headline is *ACE + PSP against the best scalar readout*, not
against the worst.

## 8. Ablation: how much of the probe is needed

Four encoder input sets, same architecture, same training, seed-averaged on the test split.
`ACE-2` against `ACE-4` is the comparison that answers the question; the intermediate rungs
show where the value appears. Wrist force is deliberately absent (D018): it is recorded in the
dataset as a diagnostic and can be added without regenerating anything.

See `outputs/training/run_v0/comparison.json`, keys `ablation ACE-*`.

## 9. What these numbers do not establish

`T` is fixed, so the adapted parameter is one-dimensional. Measured on the Phase 10 Oracle and
reproduced independently: the midpoint of a hidden state's succeeding force set also succeeds
for **104 of 105** solvable states. In one dimension the success set is essentially a
contiguous interval whose midpoint works, so a landscape model has **no structural advantage**
over a single-output regressor — there is no multi-modality to exploit.

The 6.3-point gap between ACE + PSP and baseline D is therefore a gap in *accuracy*, not
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

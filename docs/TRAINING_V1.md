# Training v1: the main experiment

Three seeds on Dataset v1, offline and then back in physics. What the numbers are, which gaps
they establish, and which of them a single seed would have got wrong.

Code: `scripts/train_models.py`, `scripts/evaluate_closed_loop.py`.
Run: `outputs/training/v1/`. Dataset: [DATASET_V1.md](DATASET_V1.md).
Figures: `outputs/plots/v1/`.

---

## 1. What is predicted, and by what

```
xi ------------> E_priv ---> z_priv --\
                                        >--- PSP(z, F_cand, post_probe, d_goal, T_goal)
probe history -> ACE ------> z_ace ---/            |
                                                   +--> logit P(reach_success)   <- the task
                                                   +--> d_hat(T), v_hat(T)       <- auxiliary
```

The head is one class with one input contract for both, which is what makes teacher and student
comparable. `d_goal` and `T_goal` are **constant across Dataset v1**, so they carry no
information the network can use — they are in the input to make the contract "given *this* task,
would this force work?", not to help. That also means the direct-GRU baseline, which does not
receive them, is not disadvantaged by their absence.

Label: `reach_success`, resolved from the data rather than configured
([D046](DECISIONS.md#d046)). Rows outside the operating region are dropped before training —
30 982 / 5 562 / 7 634 rows survive of 34 464 / 6 240 / 8 448.

## 2. Offline: selection success on held-out drawers

Test split, 88 unseen hidden states, mean ± sd over seeds 0/1/2.

| model | test selection success | force MAE (N) | AUROC |
|---|---|---|---|
| fixed force (2.82 N) | 9.09 % | 0.864 | — |
| A linear, 1 feature (`final_velocity`) | 56.06 % | 0.193 | — |
| B ridge, 9 summary features | 62.12 % | 0.178 | — |
| C MLP, summary features | 42.30 ± 0.4 % | 0.271 | — |
| D GRU on the history, direct force | 70.96 ± 4.8 % | 0.141 | — |
| **ACE + PSP** | **83.33 ± 1.6 %** | **0.091** | 0.985 |
| teacher, privileged `xi` | 93.81 ± 0.7 % | 0.056 | 0.991 |

**Baseline A's feature is chosen, not inherited.** Phase 10 measured
`displacement_per_newton` as the strongest single feature *against the ramp probe*. Under the
fixed-budget probe the ranking is different — `final_velocity` leads at |ρ| = 0.969 against its
0.947 — and `duration` and `final_commanded_force` are constant by construction. Running
baseline A on the retired probe's winner scored it at 41.29 %; giving it the right one scores
56.06 %, so the hard-coded choice was understating the baseline by **15 points** and flattering
everything compared against it. `scripts/train_models.py` now selects the strongest feature on
its own training split and records it in the run.

## 3. Physical closed loop: probe, choose, pull

The number the paper reports. All 88 held-out hidden states, all three seeds deployed **in one
Isaac Sim session from the same probe snapshots**, so a difference between seeds is the seed and
not the drawer. 264 episodes per seeded method.

| method | reach | ± sd | per-seed range | stable | invalid | safety abort | \|d−goal\| med | p90 | \|v(T)\| med |
|---|---|---|---|---|---|---|---|---|---|
| teacher (privileged) | **98.1 %** | 0.5 | 97.7–98.9 | 0.0 % | 0.8 % | **0.0 %** | 1.83 mm | 4.27 | 0.084 |
| **ACE + PSP** | **91.3 %** | 1.4 | 89.8–93.2 | 0.0 % | 1.1 % | **0.0 %** | 2.28 mm | 6.53 | 0.085 |
| D GRU (history) | 81.4 % | 5.1 | 75.0–87.5 | 0.0 % | 1.1 % | **0.0 %** | 4.58 mm | 8.93 | 0.080 |
| B ridge (summary) | 59.1 % | — | — | 0.0 % | 1.1 % | **0.0 %** | 6.57 mm | 13.38 | 0.085 |
| A linear (1 feature) | 54.5 % | — | — | 0.0 % | 1.1 % | **0.0 %** | 7.30 mm | 18.03 | 0.085 |
| fixed force | 8.0 % | — | — | 0.0 % | 1.1 % | **0.0 %** | 32.60 mm | 74.86 | 0.074 |

Zero safety aborts in 1 320 physical episodes. Invalidity is ~1 % for every method including the
fixed force, so it is a property of the rig at this operating point rather than of any policy.

`stable_success` is 0.0 % for every method, teacher included. That is the task, not the models:
at `d_goal = 100 mm` and `T = 1.5 s` the drawer arrives at 0.076–0.087 m/s where `ε_v` is 0.03.
A privileged model that is *told* the hidden state cannot do better, which is the cleanest
possible evidence that this is a property of the setting.

## 4. The three gaps

**ACE + PSP against the best scalar baseline: +32.2 pp** (91.3 % vs 59.1 %). Not marginal and
not seed-dependent — the *worst* ACE seed is 89.8 %, thirty points above a ridge that has no
seed. Offline the same gap is +21.2 pp. The mechanism is visible in figure I: the strongest
single probe feature tracks the required force at |ρ| = 0.969, and that is still not enough,
because its residual is wider than the 0.30 N success band. A strong correlation misses when the
target is a narrow window.

**ACE + PSP against the privileged teacher: 6.8 pp** (91.3 % vs 98.1 %). The probe recovers most
of what knowing `[m, µ_s, µ_d, b]` exactly is worth, and the remaining gap is the honest cost of
inferring the hidden state instead of being told it. The teacher is close to saturated at 98.1 %,
so this is near the ceiling the setting allows.

**ACE + PSP against the direct GRU: +9.9 pp** (91.3 % vs 81.4 %). Same encoder, same inputs, same
post-probe state — the only difference is that one predicts a success landscape and the other
regresses a force. Two things make this the most interesting comparison:

* the ordering is **strict across seeds**: worst ACE seed 89.8 % > best GRU seed 87.5 %;
* the GRU is **3.6× more variable** (sd 5.1 vs 1.4, range 12.5 pp vs 3.4 pp). Predicting one
  number from a probe is a less stable thing to learn than predicting whether a given force will
  work.

A single seed would have put this gap anywhere between 2.3 pp and 18.2 pp. It needed three.

## 5. How precisely to read these numbers

Absolute closed-loop rates carry session history and should be read to the point, not the
decimal ([D047](DECISIONS.md#d047)). Two runs of identical code are bit-reproducible, but
changing anything that alters how many executions run in an earlier batch shifts later batches'
*probes*: batch 1 is exact (32/32 identical probe displacements), batches 2 and 3 are not,
because `system.reset()` does not clear everything PhysX carries. The largest shift observed was
5.7 pp.

This does not touch the comparison. Within one run there is one probe per batch and one
snapshot, so every method is answered on identical evidence — and every number in §3 comes from
a single run in which all six methods were measured. It does mean a 1–2 pp difference between
two methods is not a result.

## 6. What this does and does not establish

Established: at Setting V1, adaptation from a single standardised probe recovers most of the
privileged upper bound, and the landscape formulation beats direct force regression both in mean
and in variance.

Not established, and out of scope here: any cross-goal generalisation (`d_goal` and `T_goal` were
constant, so their place in the input is untested), placement rather than reaching, and anything
about real hardware.

## 7. Reproducing

```bash
python scripts/train_models.py --dataset outputs/dataset_v1 --seeds 0 1 2 \
    --epochs 60 --baseline-epochs 40 --device cuda --output outputs/training/v1
python scripts/evaluate_closed_loop.py --headless --run outputs/training/v1 \
    --dataset outputs/dataset_v1 --seeds 0 1 2 --num-xi 0 --num_envs 32
python scripts/plot_phase11.py --dataset outputs/dataset_v1 --run outputs/training/v1
```

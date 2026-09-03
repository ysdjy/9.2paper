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

* it survives every environment-slot permutation tested — 5 of 5, gap +6.8 to +15.9 pp (§6);
* the GRU is **~3× more variable** (per-cell sd 6.6 against ACE's 2.3 over 5 permutations × 3
  seeds). Predicting one number from a probe is a less stable thing to learn than predicting
  whether a given force will work.

A single seed would have put this gap anywhere between 2.3 pp and 18.2 pp. It needed three.

**One claim corrected.** An earlier version of this document said the ordering was "strict across
seeds", on the evidence that in the reported run the worst ACE seed (89.8 %) beat the best GRU
seed (87.5 %). That holds for that run and does **not** generalise: over 5 slot permutations × 3
seeds the distributions overlap, ACE's worst cell being 89.8 % and the GRU's best 92.0 %. The
accurate statement is in §6.

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

## 6. Robustness to the environment-slot permutation

D047 established that absolute rates depend on which of the 32 parallel environment slots each
drawer occupies, and that no warm-up removes it. Rather than leave that as a caveat it is
measured: five deterministic slot permutations, each evaluating **all** methods and all three
seeds in one run. Permutation 0 is the identity and reproduces §3 exactly, which is the
correctness check. Report: `outputs/logs/slot_robustness.json`.

The same drawer's probe displacement moves by a median of **0.154 mm** between permutations
(p90 0.54, max 6.64) against a 6.7 mm median — about 2 %.

| method | reach mean ± sd | min–max | median \|d−goal\| |
|---|---|---|---|
| teacher (privileged) | **98.5 ± 0.9 %** | 97.0–99.2 | 1.71 ± 0.14 mm |
| **ACE + PSP** | **93.6 ± 1.5 %** | 91.3–95.8 | 2.69 ± 0.28 mm |
| D GRU (history) | 81.9 ± 2.5 % | 78.4–86.0 | 4.55 ± 0.19 mm |
| B ridge (summary) | 61.4 ± 3.6 % | 55.7–65.9 | 6.18 ± 0.52 mm |
| A linear (1 feature) | 58.6 ± 4.2 % | 54.5–64.8 | 6.48 ± 0.51 mm |
| fixed force | 9.1 ± 1.4 % | 8.0–11.4 | 31.33 ± 0.85 mm |

The spread grows as the method gets weaker — 0.9 pp for the teacher, 4.2 pp for the
single-feature fit. A method that picks forces near the middle of the success band is insensitive
to a 2 % probe change; one that picks near the edge is not.

**The gaps, differenced within each run and then aggregated:**

| gap | mean ± sd | min–max |
|---|---|---|
| ACE + PSP − D GRU | **+11.7 ± 3.4 pp** | +6.8 to +15.9 |
| ACE + PSP − B ridge | **+32.3 ± 3.2 pp** | +29.2 to +38.3 |
| teacher − ACE + PSP | +4.9 ± 1.1 pp | +3.4 to +6.8 |

**`teacher > ACE + PSP > D GRU > ridge` holds in 5 of 5 permutations.**

**How strong is ACE over the direct GRU in the worst case?** Three answers at three levels of
strictness, all worth stating:

* seed-averaged, per permutation: ACE ahead in **5 of 5**, worst margin **+6.8 pp**;
* paired by permutation *and* seed: ACE ahead in **14 of 15** cells, the exception being
  permutation 2 seed 2 at −1.1 pp;
* worst ACE seed against best GRU seed *within* a permutation: separated in **4 of 5**;
  permutation 2 is the exception (ACE 90.9 % against GRU 92.0 %).

So the mean gap is large and the ordering is robust, but the two are not disjoint at the level of
one seed on one schedule — and that single overlap pits the GRU's best seed against ACE's
weakest, which is exactly where the GRU's threefold variance shows.

**The reported table in §3 is the conservative one.** Permutation 0 sits at the bottom of ACE's
range (91.3 % against a 93.6 % mean) and gives the narrowest ACE−GRU gap of the five (+9.9 pp
against a +11.7 pp mean). Nothing was selected for; it is simply the sorted order.

## 7. What this does and does not establish

Established: at Setting V1, adaptation from a single standardised probe recovers most of the
privileged upper bound, and the landscape formulation beats direct force regression both in mean
and in variance.

Not established, and out of scope here: any cross-goal generalisation (`d_goal` and `T_goal` were
constant, so their place in the input is untested), placement rather than reaching, and anything
about real hardware.

## 8. Reproducing

```bash
python scripts/train_models.py --dataset outputs/dataset_v1 --seeds 0 1 2 \
    --epochs 60 --baseline-epochs 40 --device cuda --output outputs/training/v1
python scripts/evaluate_closed_loop.py --headless --run outputs/training/v1 \
    --dataset outputs/dataset_v1 --seeds 0 1 2 --num-xi 0 --num_envs 32
python scripts/plot_phase11.py --dataset outputs/dataset_v1 --run outputs/training/v1

# the slot-permutation robustness report (section 6)
for k in 0 1 2 3 4; do
  python scripts/evaluate_closed_loop.py --headless --run outputs/training/v1 \
      --dataset outputs/dataset_v1 --seeds 0 1 2 --num-xi 0 --slot-permutation $k \
      --output outputs/logs/slot_perm$k.json
done
python scripts/report_slot_robustness.py outputs/logs/slot_perm*.json
```

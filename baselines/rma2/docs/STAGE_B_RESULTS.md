# Stage B: latent distillation

`probe_history (18×7) → adapter → z_probe ∈ R¹⁶ → [frozen ParameterHead] → F_peak*`

Objective: `‖z_probe − stopgrad(z_priv)‖²` and nothing else. Privileged encoder and parameter
head frozen, adapter only trainable, Adam at RMA²'s own 1e-4. No force loss, no Stage C, no CNN
variant.

Code: `src/rma2/{config,model,trainer}.py`. Training: `scripts/train_rma2_adapter.py`.
Deployment: `scripts/eval_rma2_closed_loop.py`. Stage A: [STAGE_A_RESULTS.md](STAGE_A_RESULTS.md).

---

## 1. The headline: latent distillation explains ~87 % of the gap

All five methods deployed **in one Isaac Sim session from shared probe snapshots**
([D047](../../../docs/DECISIONS.md#d047)), 88 in-distribution test hidden states, 3 seeds.

| method | input → output | reach | ± sd | median \|d−goal\| | \|F − ref\| |
|---|---|---|---|---|---|
| **RMA² Stage A** | `xi` → point | **97.3 %** | 1.4 | **1.46 mm** | 0.097 N |
| teacher | `xi` → landscape | 97.0 % | 2.1 | 1.60 mm | 0.098 N |
| ACE + PSP | probe → landscape | **93.9 %** | 1.1 | 3.00 mm | 0.122 N |
| **RMA² Stage B** | probe → latent → point | **92.0 %** | 2.5 | 3.21 mm | 0.129 N |
| Direct GRU | probe → point | 79.2 % | 5.1 | 4.57 mm | 0.156 N |

Per-seed Stage B: 88.6 / 93.2 / 94.3 %. Zero safety aborts throughout.

### The three requested gaps

| gap | value | what it isolates |
|---|---|---|
| **Stage A − Stage B** | **+5.3 pp** | the cost of *estimating* the latent from the probe instead of knowing it |
| **Stage B − Direct GRU** | **+12.9 pp** | what privileged latent distillation buys, output form held fixed |
| **ACE + PSP − Stage B** | **+1.9 pp** | what a success landscape and a search add *on top of* distillation |

### Answering the question

**Latent distillation explains about 87 % of ACE + PSP's advantage over Direct GRU.**

```
ACE + PSP − Direct GRU  =  +14.8 pp   (this session)
     ├─ latent distillation   +12.9 pp   = 87 %
     └─ landscape + search     +1.9 pp   = 13 %
```

That is a deflating answer for the landscape formulation and it is consistent with what
Stage A already showed: at the **privileged** level, landscape versus point is worth
**−0.4 pp** — nothing, twice measured (Stage A's own run gave +0.0 pp). The landscape becomes
worth something only when the latent is uncertain, and even then only **+1.9 pp**.

So the mechanism doing the work in ACE + PSP is *having a privileged teacher to imitate*, not
*predicting a distribution over candidate forces*. The two are separable, and this is the
separation.

## 2. Offline force error

Raw continuous prediction against the shared per-probe reference force, test split:

| model | force MAE |
|---|---|
| RMA² Stage A (privileged) | 0.0829 ± 0.0022 N |
| **RMA² Stage B** (distilled) | **0.1189 ± 0.0050 N** |
| Direct GRU (no teacher) | 0.1574 ± 0.0117 N |

Distillation cuts the force error by 24 % against Direct GRU and leaves 43 % more error than
the privileged ceiling. The ordering matches the physics exactly, and the across-seed spread
shrinks with it (0.0117 → 0.0050 → 0.0022 N).

## 3. Diagnostics: the distillation worked

| | |
|---|---|
| validation latent MSE | **0.5992 → 0.0318**, a **94.7 %** reduction |
| per seed | 0.542→0.036, 0.841→0.033, 0.416→0.026 |
| convergence | val curve flat from epoch ~31 (0.0370–0.0389 band); best epochs 37–39 |
| selected on | **val latent MSE**, the objective actually optimised |
| frozen | `privileged.*`, `head.*` — reported by the freeze sweep and asserted in tests |
| trainable | `adapter.*` only, 15,056 parameters |

**Epoch selection is on the latent objective, not on force error**, deliberately. RMA² never
lets a parameter-level signal reach the adapter, so selecting on force MAE would have smuggled
Stage C's objective into Stage B under its own name. The force MAE is recorded per epoch as a
diagnostic and was not used to choose anything.

One official mechanism is absent and it is a consequence of the setting rather than a shortcut:
RMA² collects its history while the policy acts on the *estimated* latent, so its adapter trains
on its own induced distribution. Here the probe is a fixed open-loop excitation that does not
depend on the latent at all, so there is no distribution to drift and the distillation is
offline over a fixed dataset.

## 4. Reading these numbers against the main table

This session's absolute rates are not the main table's, by D047: five methods × three seeds is
15 branches per batch against the main run's 12, which shifts every batch after the first.
Direct GRU reads 79.2 % here against 81.4 % there, the teacher 97.0 % against 98.1 %, ACE + PSP
93.9 % against 91.3 %. **Every comparison above is within this one session and therefore exact;
none of the absolute values should be quoted against another run.**

## 5. Out of distribution: the landscape's advantage does **not** grow

The 64 genuine out-of-distribution states from [OOD_FEASIBILITY.md](../../../docs/OOD_FEASIBILITY.md),
with the strata that document fixed **before any model was evaluated**. Four methods, three
seeds, one session, shared probe snapshots. Nothing retrained.

| stratum | teacher | ACE + PSP | Stage B | D GRU | **ACE − Stage B** |
|---|---|---|---|---|---|
| all, n = 64 | 59.4 % | 47.4 % | 45.3 % | 33.9 % | **+2.1 ± 0.7** |
| oracle-feasible, n = 61 | 62.3 % | 49.7 % | 47.5 % | 35.5 % | **+2.2 ± 0.8** |
| responsive, n = 47 | 68.1 % | 61.0 % | 57.4 % | 44.0 % | **+3.5 ± 1.0** |
| no-breakaway, n = 17 | 35.3 % | 9.8 % | 11.8 % | 5.9 % | **−2.0 ± 2.8** |
| silent + feasible, n = 14 | 42.9 % | 11.9 % | 14.3 % | 7.1 % | **−2.4 ± 3.4** |

Median position error and force MAE track it: ACE 7.92 mm / 0.400 N against Stage B's 8.67 mm /
0.423 N on the full set, and 20.41 mm / 0.732 N against 19.39 mm / 0.664 N on silent+feasible —
where Stage B is marginally *better* on both.

`Stage B − D GRU` stays large everywhere: **+11.5 / +12.0 / +13.5 / +5.9 / +7.1 pp**. Latent
distillation is again the mechanism doing the work, exactly as in distribution.

### Answering the question

> Does a success landscape plus search show a clearer advantage over RMA²-style latent
> distillation with a point output when out of distribution, or when the probe is uninformative?

**No — the opposite, and this refutes a hypothesis stated in
[STAGE_A_RESULTS.md](STAGE_A_RESULTS.md) §3.**

The landscape's advantage is **largest where the probe is informative** (+3.5 pp on responsive,
positive on all three seeds under pairing) and **reverses where the probe is silent**
(−2.0 and −2.4 pp). Its size out of distribution (+2.1 pp) is indistinguishable from its size in
distribution (+1.9 pp). It does not grow with distribution shift and it does not grow with
information scarcity.

Read carefully, the silent-stratum reversal is better described as **no difference** than as
Stage B winning: paired per seed the gaps are −5.9 / 0.0 / 0.0 and −7.1 / 0.0 / 0.0, so one seed
favours Stage B and two are exactly tied. Either way there is no landscape advantage to find.

### Why, from the tracking correlation

`ρ(chosen force, required force)`, per stratum:

| stratum | teacher | ACE + PSP | Stage B | D GRU |
|---|---|---|---|---|
| responsive | +0.991 | +0.969 | +0.962 | +0.966 |
| no-breakaway | +0.931 | +0.385 | +0.323 | +0.340 |
| **silent + feasible** | **+0.876** | **+0.035** | **−0.075** | **+0.000** |

Where the probe informs, all three probe-based methods track the requirement almost as well as
the teacher. Where the probe is silent, **all three collapse to zero tracking together** while
the teacher keeps +0.876.

That is the explanation. A landscape can only marginalise over uncertainty it can *represent*;
with a silent probe the latent carries no information about which force this drawer needs, so a
distribution over candidate forces is exactly as uninformative as a single number. The output
form cannot recover a signal the input never contained.

### Where that leaves the comparison

Across every stratum, in and out of distribution, `ACE − Stage B` sits in the **1–3 pp** band —
the range agreed in advance as "report honestly and stop". So the landscape formulation is a
small, consistent, information-dependent improvement over an RMA²-style point output, and not a
mechanism that comes into its own under shift. The large effects in this project belong to the
privileged teacher (Stage B − GRU, +12 pp) and to the probe's own coverage (teacher − ACE,
+12 pp raw, +31 pp on silent states).

**Stopped here.** No tuning, and no search for a setting that would favour the landscape.

## 6. Reproducing

```bash
python baselines/rma2/scripts/train_rma2_privileged.py --seeds 0 1 2 --device cuda
python baselines/rma2/scripts/train_rma2_adapter.py    --seeds 0 1 2 --device cuda
python baselines/rma2/scripts/eval_rma2_closed_loop.py --headless --seeds 0 1 2 --num-xi 0
python -m pytest baselines/rma2/tests -q          # 65 tests, ~4 s, no Isaac Sim

# the out-of-distribution evaluation (section 5)
python baselines/rma2/scripts/eval_rma2_closed_loop.py --headless --seeds 0 1 2 --num-xi 0 \
    --methods teacher student stage_b gru --ood-report outputs/logs/ood_feasibility.json \
    --output baselines/rma2/checkpoints/stage_b/ood_closed_loop.json
python scripts/report_ood_evaluation.py \
    --report baselines/rma2/checkpoints/stage_b/ood_closed_loop.json \
    --output baselines/rma2/checkpoints/stage_b/ood_evaluation_summary.json
```

The joint five-method deployment report is written to
`checkpoints/stage_a/closed_loop.json` — it is the shared session, not a Stage-A-only artefact.

**Date.** 2026-09-03. No Stage C, no CNN ablation, no other ablation.

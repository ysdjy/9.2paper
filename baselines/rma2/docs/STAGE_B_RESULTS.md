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

## 5. Reproducing

```bash
python baselines/rma2/scripts/train_rma2_privileged.py --seeds 0 1 2 --device cuda
python baselines/rma2/scripts/train_rma2_adapter.py    --seeds 0 1 2 --device cuda
python baselines/rma2/scripts/eval_rma2_closed_loop.py --headless --seeds 0 1 2 --num-xi 0
python -m pytest baselines/rma2/tests -q          # 65 tests, ~4 s, no Isaac Sim
```

The joint five-method deployment report is written to
`checkpoints/stage_a/closed_loop.json` — it is the shared session, not a Stage-A-only artefact.

**Date.** 2026-09-03. No Stage C, no CNN ablation, no other ablation.

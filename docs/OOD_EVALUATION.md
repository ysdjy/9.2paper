# Out-of-distribution evaluation

The trained Setting V1 models, deployed unchanged on the 64 genuinely out-of-distribution
hidden states the feasibility pilot established ([OOD_FEASIBILITY.md](OOD_FEASIBILITY.md)).

Nothing was retrained. Setting V1, the probe, `OOD_XI_RANGES` and the action range are all as
frozen. Deployment: `scripts/evaluate_closed_loop.py --ood-report`. Report:
`scripts/report_ood_evaluation.py`. Data: `outputs/logs/ood_closed_loop.json`,
`ood_evaluation_summary.json`.

---

## 1. Why this is reported in strata

One OOD number mixes three situations the feasibility pilot had already separated, and the
stratification was fixed by that pilot **before** any model was evaluated:

* **3 of 64** states are not solvable inside the frozen 0.5–6.5 N range at all. Every method
  must fail on them and a low score there says nothing about adaptation.
* **47 of 64** are states the probe breaks away — the regime it was calibrated for.
* **17 of 64** are states the probe barely moves (0.22–0.83 mm against a 6.7 mm
  in-distribution median), every one with `µ_s` above the training maximum. **14 of those 17
  are still solvable.**

## 2. Results

Three seeds, 88 → 64 states, all methods in one deployment run. `± sd` is across seeds.

### Raw OOD, n = 64

| method | reach | ± sd | median \|d−goal\| | median F chosen |
|---|---|---|---|---|
| teacher (privileged) | **59.4 %** | 7.7 | 6.02 mm | 3.75 N |
| **ACE + PSP** | **47.4 %** | 3.7 | 7.97 mm | 4.35 N |
| D GRU (history) | 33.3 % | 3.2 | 10.72 mm | 4.32 N |
| B ridge (summary) | 20.3 % | — | 15.07 mm | 4.33 N |
| A linear (1 feature) | 18.8 % | — | 16.59 mm | 4.33 N |
| fixed force | 7.8 % | — | 72.46 mm | 2.82 N |

`teacher − ACE +12.0` · `ACE − GRU +14.1` · `ACE − ridge +27.1` pp

### Oracle-feasible, n = 61

| method | reach | ± sd | median \|d−goal\| |
|---|---|---|---|
| teacher (privileged) | **62.3 %** | 8.0 | 5.75 mm |
| **ACE + PSP** | **49.7 %** | 3.9 | 7.69 mm |
| D GRU (history) | 35.0 % | 3.4 | 10.19 mm |
| B ridge (summary) | 21.3 % | — | 14.67 mm |
| fixed force | 8.2 % | — | 70.42 mm |

`teacher − ACE +12.6` · `ACE − GRU +14.8` · `ACE − ridge +28.4` pp

Dropping the 3 infeasible states moves everything by 2–3 pp and changes nothing structural.

### Stratified by whether the probe broke away

| | responsive, n = 47 | no-breakaway, n = 17 | no-breakaway **and** feasible, n = 14 |
|---|---|---|---|
| teacher (privileged) | **68.1 %** | 35.3 % | 42.9 % |
| **ACE + PSP** | **60.3 %** | 11.8 % | 14.3 % |
| D GRU (history) | 42.6 % | 7.8 % | 9.5 % |
| B ridge (summary) | 23.4 % | 11.8 % | 14.3 % |
| A linear (1 feature) | 21.3 % | 11.8 % | 14.3 % |
| fixed force | 10.6 % | 0.0 % | 0.0 % |
| `teacher − ACE` | **+7.8 pp** | +23.5 pp | **+28.6 pp** |
| `ACE − GRU` | **+17.7 pp** | +3.9 pp | +4.8 pp |
| `ACE − ridge` | **+36.9 pp** | **0.0 pp** | **0.0 pp** |

On states the probe moves, ACE is within 7.8 pp of a model told the hidden state exactly and
36.9 pp ahead of the summary-feature ridge. On states the probe does not move, **its entire
advantage over the ridge disappears**, and the teacher's lead nearly quadruples.

## 3. Can ACE infer "needs more force" from "it barely moved"?

**Yes — and that is all it infers.** The direction is right and the magnitude is not.

It plainly detects the regime. Its median chosen force rises from **3.30 N on responsive states
to 5.70 N on silent ones, +2.40 N**, and only 19 % of its choices on the silent-and-feasible
subset are *under* the required force. It is not reading "nothing happened" as "this is easy".

What it loses is resolution. On the 14 silent-and-feasible states the Oracle requires
**3.95–5.95 N** (median 4.80), a genuine 2.00 N spread. ACE chooses within **5.35–5.95 N** — a
0.35 N spread — and the rank correlation between what it picks and what is needed falls from
**+0.965 to +0.089**:

| method | ρ(chosen, required) responsive | ρ silent | chosen-force spread, silent |
|---|---|---|---|
| teacher (privileged) | +0.989 | **+0.877** | 2.50 N |
| ACE + PSP | +0.965 | **+0.089** | 0.35 N |
| D GRU (history) | +0.965 | +0.134 | 0.40 N |
| B ridge (summary) | +0.898 | −0.254 | 0.25 N |
| *true requirement* | — | — | *2.00 N* |

So ACE collapses to a near-constant "large force" reply. It overshoots 26 of 42 episodes and
lands at a median 120.5 mm against a 100 mm goal, giving 14.3 %.

**The information exists; the probe does not carry it.** The teacher, reading `xi` directly,
keeps ρ = +0.877 and a 2.50 N spread on exactly these states, and reaches 42.9 %. The gap
between +0.877 and +0.089 is the measurement the silent probe failed to make.

Nothing is clipped by the action range here — no method's choice reached the 6.5 N ceiling on
this subset — so this is an inference limit, not an actuation limit.

## 4. The limitation, stated

**Setting V1's probe cannot identify drawers whose static friction is above roughly 3.3 N.** At
3.5 N commanded, of which 60–80 % reaches the drawer, such a drawer does not break away, and a
fixed-budget probe that produces no motion produces no information. Adaptation there degrades to
the prior: the right regime, the wrong amount.

This is a **probe** limitation, not a range or a model one, and the probe is frozen
([D044](DECISIONS.md#d044)). It is reported rather than worked around. It also means any
headline OOD number should be published beside the breakaway fraction — 47/64 here — because a
single figure blends a regime where ACE is near the privileged bound with one where it is no
better than a linear fit.

## 5. Reproducing

```bash
python scripts/evaluate_closed_loop.py --headless --run outputs/training/v1 \
    --dataset outputs/dataset_v1 --seeds 0 1 2 --num-xi 0 --num_envs 32 \
    --ood-report outputs/logs/ood_feasibility.json \
    --output outputs/logs/ood_closed_loop.json
python scripts/report_ood_evaluation.py
```

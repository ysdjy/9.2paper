# Stage A: privileged direct adaptation

`xi → PrivilegedEncoder → z_priv(16) → ParameterHead → F_peak*`

The RMA²-style baseline's ceiling, and the question it exists to answer: **knowing the hidden
state exactly, what can a point regressor do at all?**

Design: `docs/SETTING_V1_DESIGN_AUDIT.md`. Code: `src/rma2/{config,model,trainer}.py`.
Training: `scripts/train_rma2_privileged.py`. Deployment: `scripts/eval_rma2_closed_loop.py`.
Artefacts: `checkpoints/stage_a/`. No Stage B, no CNN ablation, no max-margin target.

---

## 1. The answer: point regression is not the limitation

| | reach success | ± sd | median \|d−goal\| | \|F − ref\| |
|---|---|---|---|---|
| **RMA² Stage A** — privileged, **point** | **95.8 %** | 0.5 | **1.55 mm** | 0.098 N |
| teacher — privileged, **landscape** | **95.8 %** | 0.5 | 1.72 mm | 0.099 N |
| Direct GRU — probe, point | 81.4 % | 7.0 | 4.52 mm | 0.172 N |

88 in-distribution test hidden states, 3 seeds, **all deployed in one Isaac Sim session from
shared probe snapshots** — required by [D047](../../../docs/DECISIONS.md#d047), since absolute
rates carry session history and a Stage A measured in its own run could not be differenced
against these. Zero safety aborts, zero invalid episodes for both privileged methods.

**Stage A ties the privileged teacher to the digit.** Given `xi`, emitting one force is as good
as scoring a grid and taking an argmax. It is also *slightly more precise* in position
(1.55 vs 1.72 mm median), which is what a continuous prediction snapped to a 0.05 N grid should
be against a grid argmax.

### It is at the task's ceiling, measured

All 88 test states are solvable in the dataset, but only **259 of 264 test probes (98.1 %)**
have any reaching candidate — probe-to-probe variation makes a few episodes unwinnable whatever
force is chosen. Stage A reaches 95.8 % (253/264), **2.3 pp off that ceiling**, and the teacher
is at exactly the same place. Neither privileged method has meaningful headroom left.

Stage A's 11 failing episodes fall on 5 states, 3 of which fail on all three seeds:

| `m` | `µ_s` | `µ_d` | `b` | seeds failed |
|---|---|---|---|---|
| 5.20 | 2.78 | 1.29 | 3.58 | 3/3 |
| 5.95 | 2.92 | 0.98 | 5.48 | 3/3 |
| 9.97 | 0.86 | 0.53 | 6.30 | 3/3 |
| 6.37 | 2.35 | 1.57 | 2.40 | 1/3 |
| 4.52 | 2.36 | 1.39 | 5.18 | 1/3 |

Two of the three consistent failures sit near the top of the training friction range
(`µ_s` 2.78, 2.92 against a 3.0 bound).

## 2. Offline force error

Raw continuous prediction against the shared per-probe reference force, test split:

| model | force MAE |
|---|---|
| **RMA² Stage A** (privileged) | **0.0829 ± 0.0022 N** |
| Direct GRU (probe) | 0.1574 ± 0.0117 N |
| ridge, 9 summary features | 0.2132 N |
| linear, best single feature | 0.2263 N |
| fixed force | 0.8734 N |

Privileged input roughly **halves** the force error against the best probe-based regressor.
Stage A's spread across seeds is also 5× tighter (0.0022 vs 0.0117 N).

Selected epochs were 11, 4 and 29 of 40 — chosen on validation force MAE, so the variation is
early-stopping noise rather than instability.

## 3. What this does to the paper's decomposition

Within this run, holding one factor fixed at a time:

```
                   point output        landscape + search
privileged xi        95.8 %       →         95.8 %          landscape buys +0.0 pp
probe only           81.4 %       →           ?             <- Stage B's question
                        ↑
              xi vs probe: +14.4 pp
```

Two conclusions, one firm and one now sharper:

**The bottleneck is the probe, not the output form.** Replacing `xi` with the probe costs
**14.4 pp** with the output form held fixed. Replacing a point output with a landscape and a
search costs **nothing** with the input held fixed.

**So the main table's +9.9 pp for ACE + PSP over Direct GRU cannot be "landscape beats point"
in general** — at the privileged level that difference is exactly zero. It has to come from one
of two things: the logit distillation ACE + PSP receives and Direct GRU does not, or a landscape
being worth something *specifically when the latent is uncertain*, because scoring a grid
marginalises over that uncertainty in a way a single output cannot. Those two are precisely what
Stage B separates.

**Stage A therefore strengthens the case for Stage B rather than settling it.**

> **Both legs have since been tested, and the second one was wrong.** Stage B attributed 87 % of
> the gap to distillation ([STAGE_B_RESULTS.md](STAGE_B_RESULTS.md) §1), and the
> out-of-distribution evaluation refuted the "landscape helps when the latent is uncertain"
> hypothesis directly: the landscape's advantage is *largest* where the probe is informative and
> reverses where it is silent (§5 there). The speculation below is kept as written; it did not
> survive measurement. Note the caveat:
ACE + PSP was not deployed in this session, so its 91.3 % from the main table is not
within-run-comparable to the numbers above. Direct GRU landed on 81.4 % in both, which is
reassuring, and the teacher moved 98.1 → 95.8 %, which is the D047 session sensitivity.

## 4. Diagnostics: nothing looks wrong

Stage A was checked against the failure modes that would have made it uninterpretable:

* **The target is learnable from `xi`.** Force MAE 0.083 N against a 0.30 N success band — well
  inside. The task is not asking for something `xi` cannot answer.
* **The head is not saturating.** The output is a sigmoid affine map onto [0.5, 6.5] N, verified
  to stay legal at `xi = ±10⁶` while keeping a live gradient; chosen forces span the range with
  a 3.10 N median against a 2.99 N reference median.
* **Capacity is not the constraint.** 47,921 parameters against the teacher's 13,747 and Direct
  GRU's 16,977. Stage A has *more* capacity than either, so its result is not a strawman's.
* **Training is stable.** Three seeds within 0.005 N of each other on test force MAE, and within
  1.1 pp of each other in physics.
* **No leakage in either direction.** Stage A reads `xi` legitimately; it structurally cannot
  read the probe (asserted by corrupting `history` and by running with no `history` field at
  all), and the `xi` normalisation lives in `state_dict` so a checkpoint cannot be deployed
  against the wrong statistics.

`stable_success` is 0.0 % for every method including both privileged ones, exactly as
in-distribution — the task is a reaching task at 100 mm / 1.5 s
([D046](../../../docs/DECISIONS.md#d046)), and this confirms it once more with a model that
knows the answer.

## 5. Reproducing

```bash
python baselines/rma2/scripts/train_rma2_privileged.py --dataset outputs/dataset_v1 \
    --seeds 0 1 2 --device cuda
python baselines/rma2/scripts/eval_rma2_closed_loop.py --headless --seeds 0 1 2 --num-xi 0
python -m pytest baselines/rma2/tests -q          # 45 tests, ~1 s, no Isaac Sim
```

**Date.** 2026-09-03. Stage B not started.

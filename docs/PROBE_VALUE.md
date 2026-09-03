# Is the active probe worth its budget?

The method's premise is that a short deliberate excitation reveals the force a drawer will
need. That premise had never been tested against the obvious alternatives at the same cost, so
this does: **do nothing** for the same 18 steps, or apply a **much weaker** generic force.

Script: `scripts/audit_probe_value.py`. Analysis: `probe_drawer.analysis.probe_value`.
Data: `outputs/logs/probe_value_audit.json`. No network trained, the probe not redesigned,
nothing tuned.

---

## 1. What is held identical

The three histories are the **same smoothstep trapezoid at three amplitudes**:

| history | amplitude | budget |
|---|---|---|
| frozen probe (Setting V1, [D044](DECISIONS.md#d044)) | **3.5 N** | 0.3 s |
| weak generic — a round number well below the probe, not tuned | **1.0 N** | 0.3 s |
| passive observation — the budget spent applying nothing | **0.0 N** | 0.3 s |

Everything else is shared: 18 steps, the seven deployable channels
(`[F_cmd, d, v, a, x_tcp, v_tcp, a_tcp]`), the nine-feature extractor, and the leave-one-out
ridge readout. Nothing varies but the amplitude, so a difference in identifiability is
attributable to the excitation rather than to the format.

64 in-distribution states, plain Sobol. Each history gets its **own** force sweep from its
**own** post-probe snapshot, because the force required depends on where the interaction left
the drawer.

## 2. Two targets, and why both are reported

**own** — each history predicts the force required from *its own* post-probe state. The
deployment-faithful question.

**common** — each history predicts the force the *frozen probe's* state requires. This isolates
knowledge of the hidden dynamics from knowledge of one's own starting point, and it is the
cleaner comparison because **all three then share one target with sd 1.0242 N**, so `R²` is
directly comparable. On the own-target the sds differ (1.0242 / 0.6937 / 0.7074 N), which is
exactly the confound [D043](DECISIONS.md#d043) records.

## 3. Results

### Common target — the clean comparison

| history | RMSE | R² | best feature | \|ρ\| | breakaway |
|---|---|---|---|---|---|
| **frozen probe (3.5 N)** | **0.2618 N** | **+0.935** | `final_velocity` | **0.970** | **100.0 %** |
| weak generic (1.0 N) | 0.8285 N | +0.346 | `final_displacement` | 0.613 | 10.9 % |
| passive (0.0 N) | 2.1061 N | **−3.229** | `peak_acceleration` | 0.571 | 1.6 % |

### Own target

| history | RMSE | R² | target sd | breakaway |
|---|---|---|---|---|
| frozen probe (3.5 N) | **0.2618 N** | **+0.935** | 1.0242 N | 100.0 % |
| weak generic (1.0 N) | 0.5587 N | +0.351 | 0.6937 N | 10.9 % |
| passive (0.0 N) | 1.3497 N | −2.641 | 0.7074 N | 1.6 % |

### The information difference

| against the frozen probe | own | common |
|---|---|---|
| weak generic (1.0 N) | **2.13× RMSE**, R² −0.583 | **3.17× RMSE**, R² −0.589 |
| passive (0.0 N) | **5.16× RMSE**, R² −3.575 | **8.05× RMSE**, R² −4.163 |

In units of the success half-band (0.20 N): frozen probe **1.31×**, weak generic **4.14×**,
passive **10.53×**.

## 4. Why: the excitation has to clear breakaway

| history | breakaway | post-probe displacement |
|---|---|---|
| 3.5 N | **100.0 %** | 1.12–13.1 mm (median 6.71) |
| 1.0 N | 10.9 % | 0.00–2.6 mm (median **0.02**) |
| 0.0 N | 1.6 % | 0.00–2.7 mm (median **0.00**) |

Static friction spans 0.5–3.0 N and only 60–80 % of a commanded force reaches the drawer, so
1.0 N delivers roughly 0.6–0.8 N — above `µ_s` for only the least frictional drawers. **A weak
excitation is not a weaker measurement, it is mostly no measurement**: nine tenths of its
histories are a drawer that did not move, which is the same signal the passive case gives.

The feature tables say the same thing. The passive history has **4 of 9 features constant**
(`breakaway_force`, `duration`, `final_commanded_force`, `displacement_per_newton`) and a
negative `R²` — a ridge on what remains does worse than predicting the mean. The frozen probe
has 3 of 9 constant, but by construction rather than by silence: a fixed budget makes `duration`
and `final_commanded_force` identical for everyone, and 3.5 N is high enough that every drawer
breaks away at the same early sample, saturating `breakaway_force`. Its information is in the
*response* — `final_velocity` alone reaches |ρ| = 0.970.

## 5. Answer

> Does the standardized active probe reveal the force requirement significantly better than no
> probe and than a weak generic action?

**Yes, decisively, on both targets.** Against an identical target the frozen probe's readout
error is **3.2× lower than a 1.0 N excitation and 8.0× lower than passive observation**, and it
is the only one of the three with a usable `R²` (+0.935 against +0.346 and −3.229). It is also
the only one that moves every drawer.

So the dedicated probe is doing real work, and it earns its place as the method's core. Two
honest qualifications:

* **The margin comes from clearing breakaway, not from being finely shaped.** The 1.0 N variant
  fails mainly because it does not break the drawer away; this audit does not show that 3.5 N is
  *optimal*, only that an amplitude which reliably moves the drawer is worth far more than one
  that does not. The frozen amplitude was selected separately ([PROBE_V1.md](PROBE_V1.md)).
* **Even the frozen probe's ridge readout is not precise enough on its own** — 0.262 N against a
  0.20 N half-band, so a linear readout still misses the band more often than not. That is the
  standing reason the method uses a learned encoder rather than this readout, and it is
  consistent with the earlier finding that a readout explaining 90 % of the variance lands
  inside the band about a third of the time.

**Stopped here** as agreed for a large margin: no probe redesign, no tuning, no amplitude
search.

## 6. Reproducing

```bash
python scripts/audit_probe_value.py --headless
```

The passive case uses `run_fixed_budget(peak_force=0.0)`. Zero is a documented, legal amplitude —
the null excitation — precisely so that all three histories run through one code path; only a
negative force is refused.

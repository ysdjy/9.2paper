# The Setting V1 probe

The standardised excitation the paper uses, what it measures, and what it gives up.
Decisions: [D044](DECISIONS.md#d044), [D045](DECISIONS.md#d045).

## 1. Definition

```
F(t) = F_probe · φ(t / H_probe),      t ∈ [0, H_probe]

F_probe = 3.5 N        H_probe = 0.3 s
φ  = smoothstep trapezoid, rise 10 %, hold 55 %, release 35 %
```

`φ` is the execution's own shape. Probe and execution therefore excite the drawer with one
curve at two amplitudes, so anything a model learns by comparing them is about amplitude
rather than about shape.

Call it with:

```python
from probe_drawer.experiment_plan import SETTING_V1_PROBE

result = system.probe.run_fixed_budget(**SETTING_V1_PROBE.as_kwargs())
```

## 2. What it does not do, and why each matters

| Not this | Why |
|---|---|
| Stop when the drawer has moved a set distance | The Phase 8–11 probe did, so its *duration* was a measurement — and its cost varied with what it was probing, 0.10 s to 0.93 s. Two hidden states were never excited the same way. |
| Read `d_goal` | The Phase 12 probe scaled its trigger to `α·d_goal`. That makes the probe task-dependent: change the task and the context the model trained on changes with it. |
| Wait for the drawer to stop | Whatever velocity the release leaves is the execution's initial condition. It is recorded and passed on, never zeroed ([D029](DECISIONS.md#d029)). |

One detail worth stating precisely: `φ` reaches zero at `t = H`, but a command is issued from
the time at the *start* of its control interval, so the last sampled command is one step
earlier — 0.24 N, 6.8 % of peak, held for 16.7 ms. That is a discretisation artefact of a
deliberately short probe, and it is left alone rather than fixed by shifting a sampling
convention the execution shares. The handover is unloaded because the 8-step inference gap
that follows commands exactly zero for 133 ms.
| End early on a task condition | Only the absolute safety limits may. Every healthy environment terminates with `duration_completed`, and **every history is the same length**. |

That last row is the quiet payoff. A fixed-length history means the response at any instant is
comparable across hidden states without first conditioning on when the probe happened to stop.

## 3. What it costs

The duration feature is gone. It was the single strongest identifying feature of the old probe
— Spearman **+0.932** with `µ_s`, **+0.852** with the required force ([D043](DECISIONS.md#d043))
— and under a fixed budget it is a constant. `extract_features` still reports it; it simply
carries no information now, which the readout reflects rather than hides.

What replaces it is the *response* at fixed times: how far this drawer got in 0.3 s under a
known 3.5 N, and how fast it was still going. Measured over 24 hidden states, that is enough
for a leave-one-out ridge readout of the required peak force at **RMSE 0.333 N on a target sd
of 1.411 N** (R² +0.944) — from nine deployable features, with no privileged input.

## 4. How the two numbers were chosen

Four gates, then one score, all fixed in `probe_drawer.analysis.fixed_probe_calibration`
before the first run:

1. **safe** — no hidden state trips a safety limit,
2. **responsive** — every hidden state breaks away (a constant identifies nothing),
3. **non-intrusive** — largest post-probe displacement ≤ 30 % of `d_goal`,
4. **task remains solvable** — ≥ 90 % of hidden states still have a force that reaches.

Score: lowest leave-one-out RMSE of the required peak force. Ties within 5 % go to the shorter
probe, because probe time is cost and nothing else.

The first candidate set (`H` = 0.4–0.6 s) was mis-centred — three of four failed gate 3, the
survivor passed at 0.2992. Widened downward **once**, to a 3×2 factorial:

| | H = 0.20 s | H = 0.30 s |
|---|---|---|
| **F = 3.5 N** | moved 22/24 ✗, d̃ 3.6 mm | **moved 24/24, d̃ 6.8 mm, RMSE 0.333 N ← selected** |
| **F = 4.5 N** | moved 24/24, d̃ 5.5 mm, RMSE 0.448 N | moved 24/24, d̃ 10.7 mm, RMSE 0.486 N |
| **F = 5.5 N** | moved 24/24, d̃ 7.6 mm, RMSE 0.570 N | moved 24/24, d̃ 14.3 mm, RMSE 0.590 N |

The mechanism the factorial exposes: **budget sets intrusion, amplitude sets breakaway.**
Displacement roughly doubles with the budget at fixed amplitude, while breakaway is a force
threshold and barely moves with it. `F = 3.5 N, H = 0.3 s` is the one cell that gets both from
a single point — the longer plateau breaks away every drawer at the lower amplitude, so no
displacement is paid for the coverage.

**Margin.** On a second 24-state draw the ordering held: 0.363 N against the runner-up's
0.403 N, with the gap narrowing. Real but modest; both would serve. The rule picks this one,
and it is frozen.

## 5. Measured behaviour at the frozen point

Over 24 hidden states, `d_goal` = 0.10 m:

| Quantity | Value |
|---|---|
| Broke away | 24 / 24 |
| Safety aborts | 0 |
| Probe displacement | 0.9 – 13.0 mm (median 6.9), i.e. ≤ 13 % of the goal |
| Post-probe velocity | 0.000 – 0.041 m/s, inherited by the execution |
| Probe duration | 0.300 s, constant by construction |
| Required peak force afterwards | 0.70 – 5.40 N, median 2.80 — a **7.7×** spread |

That last row is the evidence that adaptation is needed at all: no single fixed execution force
serves this range.

## 6. Reproducing

```bash
python scripts/calibrate_fixed_probe.py --headless                       # first candidate set
python scripts/calibrate_fixed_probe.py --headless \
    --candidates "3.5,0.20 3.5,0.30 4.5,0.20 4.5,0.30 5.5,0.20 5.5,0.30" \
    --output outputs/logs/fixed_probe_calibration_short.json             # the 3×2 factorial
python scripts/calibrate_fixed_probe.py --headless \
    --candidates "3.5,0.30 4.5,0.20 4.5,0.30" --seed 20260913 \
    --output outputs/logs/fixed_probe_calibration_reseed.json            # re-seed check
```

The two superseded probes remain runnable: the ramp probe as `ProbePullController.run`, and the
Phase 12 response-triggered probe in `probe_drawer.experimental.response_probe`, compared by
`scripts/compare_probes.py`.

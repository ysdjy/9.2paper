# How far can this drawer be pulled, and what stops it

The task has used `d_goal = 40 mm` since Phase 9, chosen by a sweep that never looked past
100 mm, while the drawer's travel is 400 mm. "Long pulls are probably a problem" was an
assumption. This is the measurement that replaces it.

Sweeps: `scripts/sweep_goal_distance.py` (5 632 episodes with posture diagnostics),
`scripts/sweep_parameter_space_2d.py` at three ramp-down fractions.
Analysis: `scripts/analyze_goal_distance.py`, `src/probe_drawer/analysis/goal_distance.py`.
Reports: `outputs/logs/goal_distance_{sweep,feasibility}.json`.

---

## 1. The answer

**Three different constraints bind at three different distances, and none of them is what
"long pulls are hard" would suggest.**

| range | what binds | evidence |
|---|---|---|
| 40–60 mm | **nothing** | 94–100 % of hidden states feasible, validity 100 %, drift < 0.6 mm |
| 100–250 mm | **the terminal-velocity condition** | every state can *reach* 100 mm validly (32/32) and 29/32 can reach 150 mm; almost none can **stop** there within `ε_v = 0.03 m/s` |
| from ~190 mm | **the arm's joint range** | joint margin crosses 10 % at 187–225 mm, 89 % of episodes near a limit, validity 90.9 % → 67.4 % |
| from ~350 mm | **the drawer's end stop** | `near_drawer_limit` is 0 % for every goal up to 300 mm and 100 % at 350 and 390 mm |

So the mid-range is bounded by the *task definition*, the upper range by the *robot*, and only
the last 50 mm by the *cabinet*.

## 2. The posture measurement

5 632 episodes, `F ∈ [0.5, 8.0] N`, `T ∈ [0.5, 3.0] s`, 32 hidden states, reaching 3.4 to
374.3 mm — 93.6 % of the drawer's travel. Binned by how far the drawer actually went:

| range (mm) | joint margin med / worst | near limit | limiting joint | manipulability | pull-axis transmission | Jacobian cond | lateral drift | valid |
|---|---|---|---|---|---|---|---|---|
| 0–37 | 0.122 / 0.118 | 0 % | 5 | 0.1117 | 0.2648 | 3.1 | 0.37 mm | 100 % |
| 37–75 | 0.143 / 0.134 | 0 % | 5 | 0.1006 | 0.2673 | 3.1 | 0.51 mm | 100 % |
| 75–112 | 0.156 / 0.144 | 0 % | 3 | 0.0904 | 0.2683 | 3.1 | 0.57 mm | 99.6 % |
| 112–150 | 0.137 / 0.126 | 0 % | 3 | 0.0807 | 0.2699 | 3.1 | 0.67 mm | 98.5 % |
| 150–187 | 0.122 / 0.104 | 0 % | 3 | 0.0717 | 0.2734 | 3.2 | 0.91 mm | 90.9 % |
| **187–225** | **0.085 / 0.068** | **89 %** | 1 | 0.0640 | 0.2813 | 3.2 | 1.62 mm | **67.4 %** |
| 225–262 | 0.053 / 0.029 | 100 % | 1 | 0.0589 | 0.2925 | 3.2 | 3.13 mm | 42.4 % |
| 262–299 | 0.020 / **−0.000** | 100 % | 1 | 0.0549 | 0.3098 | 3.2 | 16.72 mm | 22.6 % |
| 299–337 | 0.000 / −0.000 | 100 % | 1 | 0.0563 | 0.3466 | 3.1 | 20.47 mm | 7.1 % |
| 337–374 | 0.000 / −0.000 | 99 % | 1 | 0.0606 | 0.3811 | 2.9 | 41.75 mm | 0.0 % |

**It is a joint-range problem, not a singularity problem.** Manipulability halves (0.112 →
0.055) but the **velocity transmission along the pull axis actually rises** (0.265 → 0.381)
and the Jacobian's condition number stays flat at ~3.1. The arm does not lose the ability to
move the way the task needs; it runs out of configuration space. The limiting joint migrates
5 → 3 → 1 as the pull lengthens, which is a kinematic progression rather than a numerical
one.

**The drift is a consequence, not a cause.** Lateral drift is under 1 mm out to 187 mm and
then explodes — 1.62, 3.13, 16.72, 41.75 mm — in lockstep with the joint margin reaching
zero. An arm pinned against a joint stop cannot hold the five pose-controlled degrees of
freedom, so the drift is the symptom by which the joint limit announces itself.

**The joint margin is a function of displacement, not of the path.** Within every bin the
median and worst margins are within 0.02 of each other, across forces from 0.5 to 8 N and
durations from 0.5 to 3 s. That is what makes it a property of the task's geometry: the
gripper holds the handle, so the drawer's position determines the arm's configuration.

## 3. Why the mid-range fails, and what fixes it

At `fall_fraction = 0.35` on the fine grid (40 448 episodes, `F` at 0.05 N, `T` at 0.10 s):

| `d_goal` | feasible | can *reach* it validly, ignoring `ε_v` |
|---|---|---|
| 40 mm | **100.0 %** (32/32) | 32/32 |
| 60 mm | 93.8 % (30/32) | 32/32 |
| 100 mm | 21.9 % (7/32) | **32/32** |
| 150 mm | 0.0 % | **29/32** |
| 200 mm | 0.0 % | 28/32 |
| 250 mm | 0.0 % | 21/32 |

Every hidden state can travel 100 mm and stay inside the operating region. Nearly none can
*stop* there. The binding constraint is `|v(T)| ≤ 0.03 m/s`, and what sets it is the
ramp-down: 35 % of `T` is not enough deceleration for a drawer moving fast enough to have
covered 150 mm.

So the ramp-down was tested directly:

| `d_goal` | fall = 0.35 | fall = 0.50 | fall = 0.65 |
|---|---|---|---|
| 40 mm | 100.0 % | 93.8 % | 100.0 % |
| 60 mm | 93.8 % | 93.8 % | 100.0 % |
| **100 mm** | 21.9 % | 50.0 % | **75.0 %** |
| 150 mm | 0.0 % | 18.8 % | 12.5 % |
| 200 mm | 0.0 % | 6.2 % | 18.8 % |
| 250 mm | 0.0 % | 0.0 % | 0.0 % |
| 300 mm | 34.4 % | 50.0 % | 50.0 % |

The 0.50 and 0.65 sweeps use a *coarser* force grid (0.25 N against 0.05 N) and 16 hidden
states rather than 32, so they **understate** feasibility relative to the 0.35 column. Even
so, 100 mm goes from 22 % to 75 %.

**100 mm is therefore a viable task, conditional on a longer ramp-down.** 150–250 mm is not,
at any ramp-down tested.

## 4. The 300 mm anomaly, and a gap it exposes

300 mm shows 34–50 % feasibility, above 250 mm's zero. It should not be believed, for two
reasons.

**It is end-stop assisted.** At 300 mm the drawer is 75 % through its travel and decelerating
toward a stop it is about to hit. A small terminal velocity there is not a controlled
placement.

**The validity check does not look at joint limits.** `OperatingRegionCfg` checks mechanical
margin, peak velocity, lateral drift, orientation drift and minimum displacement — it has no
joint-margin term. So an episode can pass while the arm sits at a joint stop, which is exactly
what §2 measures at 300 mm (margin 0.000). Those 300 mm "successes" are valid by the current
definition and unsafe on evidence the definition does not cover.

This is recorded rather than silently patched. Adding a joint-margin term to the operating
region would change nothing for the 40 mm task — margins there are a comfortable 0.12 — but it
would alter what "valid" means for every dataset already generated, and that is a decision to
take deliberately (D042).

## 5. The summary table

| `d_goal` | feasible | control stability | posture risk | drawer-limit risk | recommended |
|---|---|---|---|---|---|
| **40 mm** | **100 %** | 100 % | none (margin 0.12) | none | **yes — the incumbent** |
| **60 mm** | **94 %** | 100 % | none (0.14) | none | **yes** |
| 100 mm | 22 % at fall 0.35, **75 % at fall 0.65** | 99.5 % | none (0.16) | none | **yes, with fall ≥ 0.5** |
| 150 mm | ≤ 19 % | 98 % | none yet (0.12) | none | no — cannot stop |
| 200 mm | ≤ 19 % | 66 % | **89 % near a joint limit** | none | no |
| 250 mm | 0 % | 34 % | 100 % near limit | none | no |
| 300 mm | 34–50 % | 25 % | margin **0.000** | none | no — end-stop assisted, arm at its limit |
| 350 mm | 0 % | 0 % | at limit | **100 %** | no |
| 390 mm | 0 % | 0 % | at limit | **100 %** | no |

## 6. What this means for the task

**40 mm stays.** It is not too short in any measurable sense: 100 % of hidden states are
feasible, and the required force still spans 0.20–4.30 N — a 21.5× range that discriminates
hidden states as strongly as any longer distance would.

**60 mm and 100 mm are the live alternatives**, 100 mm needing `fall_fraction ≥ 0.5`. A longer
goal is attractive for the paper — it is visually a real drawer-opening rather than a nudge —
and 100 mm costs a ramp-down change, which is a change to the profile and therefore a change
this phase deliberately did not make while `T` was the variable under test.

**Nothing above 150 mm is a candidate**, and the reason is the robot, not the cabinet. If a
long pull is wanted, the fix is to reposition the base or re-grasp partway, not to relax the
task.

## 7. Reproducing

```bash
python scripts/sweep_goal_distance.py --headless --num-xi 32 --num_envs 16 \
    --force-low 0.5 --force-high 8.0 --force-step 0.5 \
    --duration-low 0.5 --duration-high 3.0 --duration-step 0.25
python scripts/analyze_goal_distance.py --dataset outputs/logs/goal_distance_sweep.json

python scripts/sweep_parameter_space_2d.py --headless --stage fine --num-xi 16 --num_envs 16 \
    --fall-fraction 0.65 --force-low 0.5 --force-high 6.0 --force-step 0.25 \
    --duration-low 1.0 --duration-high 3.0 --duration-step 0.25 \
    --output outputs/logs/landscape_2d_fall065.json
```

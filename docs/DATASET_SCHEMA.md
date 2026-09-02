# The formal dataset schema

What one training sample is, what a model may read from it, and how to split it without
leaking. Implementation: `src/probe_drawer/dataset/`. Decision: `docs/DECISIONS.md` D031.

Nothing in this document has been generated yet. This is the contract the generation phase
must satisfy; the Oracle datasets on disk (`outputs/logs/sequential_oracle_*.json`) are
`SweepRecord` rows, which carry probe *features* but not probe *histories*, and are therefore
Oracle evidence rather than training samples.

---

## 1. One sample is one question

> Given what this probe measured, would this candidate peak force land the drawer on the
> goal?

```
(xi, probe_history, probe_summary, post_probe_state,
 candidate_peak_force, T_goal, d_goal, d_total(T), v(T), success, validity)
```

A sample is **not** "one episode". One probe is expensive and is naturally paired with many
candidate forces, so a probe episode produces many samples that share evidence. §4 is the
consequence.

## 2. Fields

| Field | Type | Deployability | Meaning |
|---|---|---|---|
| `candidate_id` | `str` | — | This row. Unique. |
| `probe_id` | `str` | — | The probe episode the evidence came from. |
| `xi_id` | `str` | — | The hidden state. |
| `xi` | `dict` | **SIM_ONLY_PRIVILEGED** | `{mass, static_friction, dynamic_friction, damping}`. |
| `probe_history` | `dict[str, list]` | DEPLOYABLE only | The probe's recorded time series, `{channel: values}`. |
| `probe_summary` | `dict` | DEPLOYABLE | Scalar features, from `analysis/probe_features.py`. |
| `post_probe_state` | `dict` | DEPLOYABLE | `{displacement, velocity}` at the moment the execution starts. |
| `candidate_peak_force` | `float` | input | `F_peak` this row asks about (N). |
| `duration` | `float` | input | `T_goal` (s). |
| `goal_displacement` | `float` | input | `d_goal` (m), from **before** the probe. |
| `final_total_displacement` | `float` | **label** | `d_total(T)` (m). |
| `final_velocity` | `float` | **label** | `v(T)` (m/s). |
| `success` | `bool` | **label** | Position **and** terminal velocity both within tolerance. |
| `valid` | `bool` | — | Whether the episode stayed inside the operating region. |
| `invalid_reasons` | `list[str]` | — | Why not, if not. |
| `protocol` | `str` | — | Always `"sequential"`. A reset row is not a training sample and the constructor refuses one. |

### `xi` is present and is not an input

It is recorded so that an upper-bound oracle and a per-dimension error analysis are possible
at all. `model_input_fields()` is the authoritative list of what a model may read, and `xi`
and `xi_id` are not in it. `validate_probe_history()` applies the same rule to the history's
channels, delegating to `observations.validate_model_input()` so there is one definition of
"deployable" in the project (D017).

### `post_probe_state` is deployable

It is state, not a privileged reading: a robot knows how far it has already pulled the handle
and roughly how fast the handle is moving. It is recorded separately from `probe_history`
because it is the *initial condition* of the thing being predicted, not evidence about the
drawer.

### `valid` rows are kept in the file

Invalid rows are dropped by the training script, not by the loader and not by the splitter,
so that how many were dropped and why is visible in the run that dropped them. An invalid row
is evidence about the rig — an end-stop impact, a lost grip — not about the drawer.

## 3. Identifiers

All three are content-addressed (SHA-1 of the defining values, truncated to 48 bits), not
counters. Two datasets built in different runs or different batch orders therefore agree on
which rows describe the same thing, and merging them is well defined.

* `xi_id = f(mass, static_friction, dynamic_friction, damping)` — key order irrelevant.
* `probe_id = f(xi_id, episode_index, probe_task)`. The episode index is there because two
  probes of the *same* drawer with the *same* parameters are still different episodes; without
  it, repeats would collapse into one group. The probe task is there so a re-calibrated probe
  never shares an identifier with the old one.
* `candidate_id = f(probe_id, peak_force, duration, goal_displacement)`. The task is part of
  it: the same probe and force judged against a different `d_goal` is a different question.

## 4. Splitting: grouped, or not at all

The three identifiers nest — every candidate belongs to one probe, every probe to one hidden
state:

```
xi_id  ⊇  probe_id  ⊇  candidate_id
```

A **random split over rows** would put near-duplicates of a training row into the test set:
same drawer, same probe recording, same post-probe state, a neighbouring force. The reported
error would measure memorisation. `SPLIT_LEVELS` is therefore `("xi_id", "probe_id")` and
`SplitCfg(level="candidate_id")` raises.

| Level | Question it answers | Use |
|---|---|---|
| `xi_id` | Does this work on a drawer never seen? | **Default.** The question the paper asks. |
| `probe_id` | Does this work on a probe recording never seen? | The minimum admissible level. Weaker: two probes of one drawer can straddle the split. |

`assert_no_leakage(split)` checks every level at or below the level split on, and no higher —
a probe-level split makes no claim about hidden states, and the checker does not pretend
otherwise.

### The split is stable, not seeded

Groups are assigned by hashing the group key, not by shuffling with a seed. Adding hidden
states to the dataset later therefore does not move existing ones between subsets, so a model
trained on an earlier version can still be evaluated honestly on the later version's test
set. Changing `SplitCfg.salt` gives a different, still stable, partition — for a
repeated-splits study, not for retrying until the numbers improve.

Default fractions are 0.70 / 0.15 / 0.15 **of groups**, not of rows. Row counts will not
match those fractions exactly, because hidden states differ in how many valid candidates they
have.

## 5. Out-of-distribution evaluation

The OOD set is a separate file, not a split: it is sampled from `OOD_XI_RANGES`, which extends
one step beyond training on every axis (`experiment_plan.py`). It shares the schema and its
`xi_id` values are disjoint from the training file's by construction. Mechanical-limit hits
and numerical blow-ups are **not** OOD — they are `valid = False` and are dropped.

## 6. What the generation phase still has to decide

Recorded here so they are decided deliberately rather than by whichever script is written
first:

* **Which channels go into `probe_history`.** `DEFAULT_ACE_INPUT` is the current answer
  (7 channels), but the history's sampling rate and length are not yet fixed. Probes have
  different durations, so either the history is variable-length or it is padded, and the
  choice affects the model architecture.
* **How many candidate forces per probe, and drawn how.** The Oracle used a uniform grid
  because it was mapping a landscape. A training set does not need uniformity, and
  concentrating candidates near each hidden state's success band would spend the budget
  better — at the cost of a distribution that depends on the label.
* **How many probe repeats per hidden state.** The intrinsic `d_total(T)` noise is about 1 mm
  against `ε_d = 7.5 mm`, so a single probe per hidden state gives a label that is right most
  of the time but not always. Repeats would let the label be a probability.
* **Whether the 3 unsolvable hidden states are included.** They have no succeeding force at
  the selected task, so every candidate row is a negative. They are legitimate data and they
  are also 3/108 of a strongly imbalanced kind.

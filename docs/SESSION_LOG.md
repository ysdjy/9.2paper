# Session log

One entry per work session. Newest first.

---

## 2026-09-02 — `agent/rma2-baseline-isolation` — RMA² baseline isolation

**Agent / task.** Claude Opus 5 — move the RMA² baseline to `baselines/rma2/` under the
project's standard baseline layout, confirm the main project is untouched, and report the
integration plan. Migration only; no model code was written.

**What changed.** `baseline/rma2_direct/` → **`baselines/rma2/`**, package `rma2_direct` →
`rma2`, plus the two directories the layout asks for and that now have real content:

* `configs/adaptation_premise.yaml` — the audit's tunables (sweep, report path, ambiguity
  radii, `k`), **read by its script** rather than left as a placeholder. The resolved settings
  are now recorded in the report, because the ambiguity radii change the answer a great deal
  and a number without them is misleading.
* `checkpoints/` — with the main project's own rule applied: weights git-ignored, `config.yaml`
  / `manifest.json` / `metrics.json` tracked, so a result traces back to a config, a split and
  a commit without the bulk.

**The main project's code was not touched.** Only three docs changed, and only where this move
would have left a dangling path: `API.md` (the `baselines/` pointer, plus a note distinguishing
it from `probe_drawer.models.baselines`, which is a different thing), `DECISIONS.md` (D034's
two file references), and this log's previous entry.

**The isolation invariant is checkable, and holds:**

```bash
grep -rn "baselines" src/probe_drawer/ --include="*.py"   # no import
```

**Tests.** `tests/unit` **371 passed**, unchanged by the move. `baselines/rma2/tests` **13
passed** (12 + one new: the report must record how the question was asked). The audit script
runs end to end and reproduces its previous numbers. The official RMA² clone was reinstalled
at the new path in the separate `rma2` env and still steps `PegInsertionSide-v1` with all
three patches applied.

**What this baseline may and may not take from the main project** is now written down in
`baselines/rma2/README.md` §4. Shared, by import: the environment, `DynamicsRandomizer`, the
probe and pull controllers, `SequentialPullProtocol`, `MAIN_TASK` and the success definition,
`dataset/` including the `xi_id` split, and `training/{dataloader,metrics}`. Owned here,
because they *are* the method: `PrivilegedEncoder`, the temporal-CNN `AdaptationEncoder`,
`ParameterHead`, and the Stage A/B trainer. The `xi_id` split needs no coordination — `SplitCfg`
assigns groups by hashing the key rather than by shuffling with a seed, so both methods get the
same drawers by construction.

**Still blocked, and not by this baseline.** Stage A cannot start until the three project-level
tasks in D034 are done: re-sweep the Oracle over `(F_peak, T)`, re-select `MAIN_TASK`, and
generalise the premise audit from bands to regions. The third produces the number that decides
whether the comparison is worth running — how many hidden states have a *disconnected* success
region. Everything in the design contract is written so the parameter space is a config value
rather than an architectural assumption, so 1-D → 2-D changes data, not code.

**Next.** Those three, in that order, as project work; then Stage A on 10–50 hidden states as a
tiny overfit test.

---

## 2026-09-02 — RMA² reproduction and baseline design (uncommitted; see "Git state")

**Agent / task.** Claude Opus 5 — reproduce the official RMA² implementation, understand it
from its source, and design the RMA²-inspired Direct Adaptation baseline for this project.
No baseline code was written this round; that was deliberate (the commission's §55).

**What was added.**

Everything is under **`baselines/rma2/`**, a self-contained folder that imports
`probe_drawer` and never modifies it. Nothing outside it was added.

* `docs/RMA2_REPRODUCTION_REPORT.md` — the official code, read line by line, and how far it
  ran here.
* `docs/RMA2_TO_DRAWER_MAPPING.md` — what transfers, what does not, and the design contract
  the implementation must satisfy.
* `src/rma2/adaptation_premise.py` + `scripts/audit_adaptation_premise.py` +
  `tests/test_adaptation_premise.py` (12 tests) — whether the adaptation problem is well
  posed, answered from the Oracle already on disk. Offline, no Isaac Sim.
* `patches/rma4rma/` — the four fixes the official code needed, plus the install script.
* `third_party/` (git-ignored) — the official `rma4rma` clone and its two forks.

The audit's report keeps its project-level path, `outputs/logs/adaptation_premise.json`, so
the citation in `docs/TRAINING_V0.md` is unaffected.

**The official RMA² code does not run as published.** Four separate defects, each observed
rather than inferred:

1. `.gitmodules` declares two submodules but **no gitlink is committed**, so
   `git clone --recurse-submodules` silently produces nothing and `environment.yml` then
   points at directories that do not exist.
2. `ManiSkill2@49c3093` (the fork's HEAD, "add new robot") calls `joint.set_drive_trget` —
   a typo — in `PDJointPosController.set_drive_targets`, so **every `env.step` raises
   `AttributeError`** under the `pd_ee_delta_pose` mode all four tasks train with.
3. `ActorCriticPolicyRMA` sizes the adapter with a different rule than
   `FeaturesExtractorRMA` sizes the encoder, and has no PegInsertionSide branch. The adapter
   comes out 71-wide against a 67-wide latent and stage 2 dies at the first forward pass:
   `mat1 and mat2 shapes cannot be multiplied (4x123 and 119x512)`.
4. `adaptation.py:102` hard-codes `range(50)`, so adapter training crashes at the first
   episode end for any `-n` other than 50.

With those patched, the full two-stage pipeline runs: PPO trains and checkpoints, the
100-episode evaluation completes, and the adapter's L² latent loss falls 0.578 → 1e-5 over
3157 steps. **That last number is an artefact, not a result** — PegInsertion's randomisation
is curriculum-annealed to 2e6 steps, so at 3k steps there is almost nothing for `z_priv` to
encode.

**TurnFaucet could not be run.** The ManiSkill2 asset server is unreachable through this
network's proxy (TLS closed, 0 bytes), which rules out TurnFaucet, PickSingleYCB and
PickSingleEGAD. `PickCube-v1` — the launcher's own default — cannot be constructed by the
launcher, because `PickCubeRMA.__init__` rejects three kwargs `config_envs` always passes.
`PegInsertionSide-v1` was used instead.

**What the source analysis found that matters for us.** `z_priv` is **72**-dimensional for
TurnFaucet (71 for PickSingle, 67 for PegInsertion) from a 76-dim input — RMA²'s "compact
embedding" removes four dimensions, and 64 of its 76 inputs are learned per-object identity
embeddings that have no analogue for one drawer. Only 2 of TurnFaucet's 12 privileged dims
are randomised physics. The adapter is distilled **on-policy** — the policy acts on `z_hat`
while the adapter learns — which the paper's figure does not make explicit and which has no
analogue here, because our probe is open-loop. The temporal CNN's kernel stack `(9,7,5,3)`
transfers to our probe **unchanged** if the window is the 1.5 s probe budget (90 steps at
60 Hz); at the median probe length of 28 steps it is arithmetically invalid.

**What was measured about our own task, before writing any model.**
`baselines/rma2/scripts/audit_adaptation_premise.py` on `sequential_oracle_fall035.json`
at `MAIN_TASK`:

* **Adaptation is necessary.** The best single fixed force (0.70 N) succeeds on 20/108 =
  **0.185** of hidden states. Required forces span 0.25–4.30 N.
* **The answer is an interval, not a set of modes.** Median band 0.20 N, median 3 succeeding
  forces, only 5/105 bands have an interior failure, and the band midpoint succeeds for
  **104/105**. So the regression-to-the-mean failure the commission anticipates **does not
  exist within a hidden state** — in 1-D with contiguous bands, averaging is safe by
  construction. This is the round's most important finding and it is an argument for a
  two-dimensional parameter space.
* **The probe identifies friction and nothing else.** Leave-one-out R² from nine probe
  features: `mu_s` +0.946, `mu_d` +0.883, `mass` +0.251, `damping` **−0.107**. D032's damping
  limitation is confirmed and extends to mass.
* **The required force is essentially `mu_d`** (corr +0.987; mass +0.023, damping +0.077).
* **The task is precision-limited.** Band half-width 0.100 N on a 1.50 N median target — 7 %.
  A leave-one-out readout of the probe explains 90 % of the variance in the required force and
  is inside the band **a third of the time**; from the true `xi` a quadratic reaches RMSE
  0.099 N and 0.867. So the experiment's dynamic range is 0.185 → ~0.33 → 0.867, and
  **success rate must be reported, not R²**.

**One decision was escalated and answered: `p = [F_peak, T]` (D034).** The commission
specified `p = [F_max, v_cmd]`, which this repository cannot express — the pull axis is
force-controlled throughout and there is no velocity command in `src/`. Of the three options
put to the owner, the two-dimensional `[F_peak, T]` was chosen: it needs **no new control
code** (`ExecutionPullController.run` already takes a duration, and `SweepRecord` already
carries `duration` as an axis), and it is the cheapest way to make the paper's central claim
*testable at all* — in one dimension the success set is an interval whose midpoint succeeds
104/105 times, so there is no multi-modality for a success-landscape model to exploit.

This blocks the next round on three things, in order: re-sweep the Oracle over a
`(F_peak, T)` grid; re-select `MAIN_TASK` against it with the existing scored rule (D024);
generalise `baselines/rma2/src/rma2/adaptation_premise.py` from bands to regions
and re-run it. That last one
produces the number the paper actually needs — the count of hidden states whose success region
is **disconnected**. If the 2-D regions turn out to be convex blobs, the framing has to change
rather than the measurement.

**A stale README table was corrected.** §7 still listed the Phase 9 task (`d_goal` 50 mm,
`eps_d` 15 mm, `eps_v` 0.08, `F_peak` 1–5 N, ramp-down 20 %) against `experiment_plan.py`'s
Phase 10 values. The `configs/experiment_plan.yaml` snapshot was correct, so no code or result
was affected. Test counts were also stale.

**A bug in this session's own new module, caught by its own test.**
`HiddenStateBand.contains` is interval membership, and using it for "does the midpoint
succeed?" made that question trivially true (it reported 105/105). Split into `contains`
(membership) and `succeeds_at` (snap to the nearest swept force, ask whether it succeeded);
the honest answer is 104/105. Every audit now uses `succeeds_at`.

**Tests.** `pytest tests/unit -q` → **245 passed**. Integration tests were not run: another
agent was holding the simulator (below).

**Git state.** **Nothing was committed.** A concurrent session was editing this working tree
throughout — `protocols/simulation_snapshot.py`, `scripts/validate_branching.py`,
`controllers/hybrid_osc.py`, `sensors/drawer_state.py`, `sensors/causal_derivative.py`,
`protocols/__init__.py` — so creating a branch or a commit would have moved shared git state
under a live session. This round's files are:

* new: everything under `baselines/rma2/`;
* modified: `README.md` (§7 table, test counts), `docs/API.md` (a pointer to `baseline/`),
  `docs/DECISIONS.md` (D034), `.gitignore` (`third_party/`, later moved into the baseline's
  own `.gitignore`).

That concurrent work is building state capture and restore — exactly the "candidate rollouts
from one shared post-probe state" capability this round listed as missing. The dataset-
generation plan in the mapping document should be revisited against it once it lands.

**Next.** The three blockers above (2-D Oracle, 2-D task selection, region-level premise
audit), then the probe dataset, then Stage A on 10–50 hidden states as a tiny overfit test.

---

## 2026-09-02 — `agent/phase11-dataset-v0` — Phase 11

**Agent / task.** Claude Opus 5 — generate the first real dataset, train the first models,
and put them back into the simulator on drawers they have never seen.

**The gate came first.** Dataset v0 wants 32 counterfactual labels per probe, which needs the
post-probe state captured and restored, and restoring a PhysX scene is not obviously sound.
`scripts/validate_branching.py` is the evidence: everything the snapshot writes comes back
bit-identical in float32, branch-to-branch spread is 23 µm where the execution barely moves
the drawer (20–30× better than re-probing) and 2.7–2.9 mm just past breakaway where fresh
episodes spread 2.1–8.0 mm, and the bias is negligible across seven runs. Full account in
`docs/COUNTERFACTUAL_BRANCHING.md`.

**Two bugs it caught**, both of which would have corrupted the dataset without failing
anything: a restore left the TCP pose stale by 34 mm — the execution reads its pose reference
from it, so every branch was hauling the arm back to where the previous one ended — and 24
branches of 1.5 s exceed the 30 s episode, so the environment would have auto-reset partway
through every candidate sweep.

**One finding changed the design.** Branching drifts systematically, 57 µm per branch at
medium/2.5 N, 1.3 mm across a full sweep. Drift matters more than its size: candidate forces
go to ordered strata, so a drift correlated with branch index would bias exactly the axis
being learned. The generator therefore shuffles the branch order deterministically and records
`branch_index` on every row; the audit checks the decorrelation instead of trusting it
(−0.0005, 0.11 σ over 1 536 probes).

**One number changed because it was measured.** The plan specified 24 candidates per probe. A
32-state pilot left 2 of 32 hidden states with no positive at all, and I first read those two
as physically infeasible — their displacement jumped from 6.6 mm straight to 140 mm between
adjacent candidates. Reading the force-sorted rows showed one of them reaching 40.1 mm at
2.61 N, inside the position tolerance, failing only on terminal velocity by 0.002 m/s, with
its neighbour at 2.42 N giving 31.7 mm at a compliant 0.023 m/s. The grid had missed a force
that works. Just past breakaway `dd/dF` reaches 100–130 mm/N, so 24 strata map to 18–24 mm
against a 15 mm success window. Raised to 32, the same pilot left zero states uncovered
(D038).

**Dataset v0.** 49 152 rows from 1 536 probes over 512 Sobol-sampled hidden states, 35 min,
29.5 MB, nine audit gates passed. 5 of 512 hidden states (0.98 %) have no positive in any of
three repeats.

**Results.** The privileged teacher passes its gate decisively (test AUROC 0.993–0.994,
selecting a succeeding force for 90.8–92.0 % of feasible probes), so training a student was
licensed. Closed-loop on 64 unseen drawers, all methods sharing one probe:

| method | physical success |
|---|---|
| teacher (privileged, told `xi`) | 89.1 % |
| ACE + PSP | 87.5 % |
| GRU regressing one force | 81.2 % |
| linear on one scalar feature | 18.8 % |
| best fixed force | 14.1 % |

So the probe history carries information the scalar features discard, the probe is very
nearly sufficient (1.6 points below being told `xi`), and predicting the landscape beats
regressing one force by 6.3 points on identical input.

**The thing the next session most needs to know.** Another session working in this repository
recorded D034 — `p = [F_peak, T]`, decided by the project owner — while this phase was
running, and my `git add -A` swept it in. I renumbered my branching decision to D040 and
verified D034's central measurement independently: of the 105 solvable hidden states, the
midpoint of the succeeding force set also succeeds for 104. So in the one-dimensional
`F_peak` parameterisation the success set is essentially a contiguous interval whose midpoint
works, a landscape model has no *structural* advantage over a single-output regressor, and the
6.3-point gap above is an accuracy gap rather than proof of necessity. Dataset v0 is the
one-dimensional baseline case, not the dataset D034 calls for.

**Also in the tree from that session, untouched by me:** `third_party/rma4rma` (139 MB,
gitignored), `patches/rma4rma/`, `docs/RMA2_{REPRODUCTION_REPORT,TO_DRAWER_MAPPING}.md`,
`analysis/adaptation_premise.py` and its script and tests.

**Tests.** 383 unit (~5 s) + 84 integration (297 s), all passing.

**Commit.** `05eb49d241ed21a0510d8338de15a7693c4abe62` on `agent/phase11-dataset-v0`; `main`
fast-forwarded to the same commit; tagged `v0.4.0-phase11`
(`7b4bd8417ecf488e8dd6dc3dee180d4a70bb91ef`).

**Push status.** Pushed. `main`, `agent/phase11-dataset-v0` and the tag all resolve on
`https://github.com/ysdjy/9.2paper.git`, confirmed with `git ls-remote`.

**Not done, deliberately.** No SPC, no VLM, no RMA baseline, no real robot, no hyperparameter
search, no dataset beyond the pilot scale the plan set.

**Left for whoever picks this up.** `.git` is 23 MB because several partial versions of
`dataset_v0/candidates.jsonl` were committed while generation was still writing. The
`.gitignore` now prevents it recurring. I did not rewrite that history on purpose: another
session is actively working in this repository, and rewriting commits it may have based work
on would be worse than the bloat.

---

## 2026-09-02 — `agent/phase10-sequential-refinement` — Phase 10

**Agent / task.** Claude Opus 5 — replace `probe -> reset -> execution` with a genuinely
sequential episode, tighten the task against the resulting landscape, and write the formal
dataset schema.

**What changed.**

* `protocols/sequential_pull_protocol.py`: one reset at the start and none after. Probe, a
  fixed 8-step zero-force gap, then the execution. Refuses a settling execution outright,
  because a settle would brake the pull axis and erase the post-probe velocity without saying
  so.
* `HybridPullOSC.coast()`: the gap. Zero pull force, five axes held, no braking, nothing
  written to the simulation.
* `dataset/`: one training sample, three nested content-addressed identifiers, hashed grouped
  splits, and `assert_no_leakage`. `SplitCfg(level="candidate_id")` raises.
* The task moved to `d_goal = 40 mm`, `eps_d = 7.5 mm`, `eps_v = 0.03 m/s`, `fall = 0.35`,
  `T = 1.5 s` unchanged.

**What was measured.**

* Sequential Oracle: 5616 rows at `fall = 0.35`, 97.2 % valid, 108 hidden states x 52 forces.
  Coverage **0.972**, required force **0.20-4.30 N** (21.5x), median band 0.20 N.
* Inference gap: 8 steps gives the most repeatable finished task (`d_total(T)` spread 0.90 mm,
  against 3.58 at 0 steps and 1.40 at 12). Velocity retained across the gap decays 1.000 ->
  0.539 -> 0.001 -> 0.000 over 0/2/4/8 steps, by physics.
* Reset vs sequential on the Phase 9 task: coverage 1.000 both ways, ranking preserved
  (**+0.95**), required force lower under the sequential protocol by a median factor of
  **0.80** — but bimodally, 0.32 to 1.02. The reset was a biased estimator, not a rescaled one.
* Probe feature vs sequential required force: `displacement_per_newton` at Spearman
  **-0.910** (Pearson -0.841), down from 0.969 against the reset Oracle. Strong, and *not*
  sufficient — residual spread reaches +/-0.3 N against a 0.20 N band.
* Figure F: required force is driven almost entirely by **dynamic friction**; mass and damping
  barely move the median. This is why the damping identifiability gap costs less than it looks.

**Two mistakes worth recording.**

* The `_vlow` supplement files were globbed as full datasets by `refine_task_space.py`, and
  since the tolerance curves were keyed by fall fraction, the last one loaded won. Figure D
  then showed coverage 0.06 where the selection said 0.972. Caught because the figure
  contradicted the selection, which is the argument for plotting things you have already
  computed. Selection itself was unaffected.
* Docs initially quoted gap-study numbers (0 and 2 steps) that were not in the report on disk —
  they were from an earlier run. Rather than soften the claim, the validation was re-run over
  all five gap lengths and every quoted number replaced with the observed one.

**Tests.** 233 unit (~1 s) + 69 integration (~106 s), all passing.

**Commit.** `089faed491626512d809e8c743b2110310a67259` on
`agent/phase10-sequential-refinement`; `main` fast-forwarded to the same commit; tagged
`v0.3.0-phase10` (`7df2e50e46ba884ca11ed91159e8b667a730ee3e`).

**Push status.** Pushed. `main`, `agent/phase10-sequential-refinement` and the tag all
resolve on `https://github.com/ysdjy/9.2paper.git`, confirmed with `git ls-remote`.

**Not done, deliberately.** No second probe for damping (D032). No networks trained. The
reset Oracle was kept, not deleted.

**Next.** Generate the formal training dataset against `docs/DATASET_SCHEMA.md` §6, which
lists the four decisions the generation phase still has to make (probe-history channels and
length, candidate sampling, repeats per hidden state, and whether the three unsolvable hidden
states are included).

---

## 2026-09-02 — `agent/phase9-oracle-audit` — Phase 9A through 9M

**Agent / task.** Claude Opus 5 — fix the hidden state at four dimensions, expand and
classify the observations, audit the force channels, correct the execution's post-`T`
behaviour, then sweep the experiment space and select every experimental parameter from the
data rather than by hand. No learning components; this phase establishes the physics.

**Modified.**

* `envs/dynamics_randomization.py` rewritten: `xi` is now
  `[m, mu_s, mu_d, b]`; readback comes from `root_physx_view`, not Isaac Lab's mirror.
* `sensors/`: new `causal_derivative.py`; `drawer_state.py` gained acceleration channels,
  TCP pull-axis velocity/acceleration, orientation, and the two privileged drawer force
  channels.
* `controllers/`: `PullHistory` 16 -> 25 channels with a generic recorder; `ExecutionResult`
  gained `peak_velocity`; the execution controller now snapshots at `T` and releases the
  pull force afterwards.
* New: `observations.py`, `experiment_plan.py`, `evaluation/` (2 modules), `analysis/`
  (5 modules).
* New scripts: `audit_hidden_states.py`, `audit_force_channels.py`,
  `sweep_execution_space.py`, `build_oracle_landscape.py`, `calibrate_probe.py`,
  `plot_experiment_space.py`, `plot_probe_identifiability.py`.
* New docs: `HIDDEN_STATE_AUDIT.md`, `FORCE_CHANNEL_AUDIT.md`, `EXPERIMENT_SPACE.md`,
  `ORACLE_LANDSCAPE.md`. Existing docs updated; 11 new decisions (D015-D025).
* New configs: `evaluation.yaml`, `experiment_plan.yaml`, both drift-tested.
* Isaac Lab's own source tree: **unmodified**.

**API introduced.**

```python
DynamicsParameters(drawer_mass, joint_static_friction, joint_dynamic_friction, joint_damping)
assess_validity(result, OperatingRegionCfg()) -> ValidityReport
evaluate_execution(result, SuccessCriteria(...)) -> EvaluationReport
observations.validate_model_input(channels)
experiment_plan.MAIN_TASK / RECOMMENDED_PROBE_TASK / TRAINING_XI_RANGES / OOD_XI_RANGES
```

`ProbePullController.run` and `ExecutionPullController.run` are unchanged, and `d_goal` still
does not appear anywhere near the execution controller.

**Verification.** 248 tests pass (196 unit, 52 integration), up from 124. 23 175 simulated
episodes across six sweeps. Full table in `docs/VALIDATION.md`.

**Headline results.**

* At the selected task the required peak force spans **1.00-4.50 N across the hidden-state
  grid — a 4.5x range — with success bands 0.50 N wide**, and 106 of 108 hidden states are
  achievable. One force cannot serve every drawer.
* A single standardised probe's best feature correlates with the required force at
  **|rho| = 0.969**, using deployable channels only.
* Selected: `T = 1.5 s`, `d_goal = 50 mm`, `eps_d = 15 mm`, `eps_v = 0.08 m/s`, execution
  ramp-down 20 %, probe 1.0 -> 6.0 N over 1.0 s stopping at 3 mm.

**Simulator findings that changed the design.**

1. **PhysX requires `mu_s >= mu_d` and silently discards violating writes**, while Isaac
   Lab's `data` buffers report the request. The previous readback check could have passed on
   a write that never landed (D016).
2. **`get_dof_projected_joint_forces` is the drawer's internal resistance**,
   `-(mu_d*sign(v) + b*v)`, verified to 0.0099 N. The joint reaction wrench is structurally
   zero along a prismatic joint's own axis, so it cannot give the drawer-axis force.
3. **The Phase 8 17-23 N wrist force is the mechanical end stop**: a deliberate episode
   reached 59.2 N, 9.87x the command, at 84.4 % of travel.
4. **A 10 % force ramp-down cannot bring the drawer to rest.** With the terminal-velocity
   requirement, the reachable-at-rest distance at `T = 1.5 s` was 49 mm at `fall = 0.10`
   against 65 mm at 0.20 (D023).
5. **The Phase 8 held-axis drift was the operating point, not the controller.** Drift stays
   under 1 mm across the whole force range at moderate displacement and speed.

**Commits.** `9b7af67` (Phases 9A-9G: hidden state, observations, audits, evaluator) and
`07c15dd` (Phases 9H-9M: sweeps, Oracle landscape, parameter selection) on
`agent/phase9-oracle-audit`, then `main`. Pushed to
`https://github.com/ysdjy/9.2paper.git`; the SHA of this log entry follows in the next
commit.

**Remaining issues.** See "Outstanding" in `docs/VALIDATION.md`. The one that matters most
for the next phase: **the calibrated probe does not identify damping** — sweeping `b` from 2
to 11 N s/m leaves the probe response unchanged, because the probe stops before the drawer
reaches speeds where viscous drag matters. The required force also depends weakly on `b`, so
the task stays predictable, but any claim that one probe identifies all four dimensions
would be false.

**Next.** Dataset generation at the selected operating point: sample `xi` from
`TRAINING_XI_RANGES`, run the calibrated probe, then the execution at a grid of `F_peak`, and
store paired `(probe history, F_peak, d(T), v(T), success, xi)`. The APIs need no change.

---

## 2026-09-02 — `agent/phase0-8-bootstrap` — Phases 0 through 8

**Agent / task.** Claude Opus 5 — bootstrap the project, validate the official Isaac Lab
drawer environment, build the shared hybrid OSC and the two public pull controllers, add
dynamics randomisation, and validate all of it physically.

**Modified.**

* Project created at `/home/zbh/Downloads/IsaacLab/9.2paper` (moved mid-session from
  `~/Documents/isaaclab/9.2paper` after the user corrected the intended parent directory;
  nothing needed editing because every path is derived from
  `probe_drawer.utils.project_root()`). Added `9.2paper/` to the Isaac Lab checkout's
  `.git/info/exclude` so the nested repository never appears in Isaac Lab's `git status`.
* New package `src/probe_drawer/` with `envs/`, `controllers/`, `state_machines/`,
  `sensors/`, `logging/`, `utils/` and `pull_system.py`.
* New scripts: `inspect_isaaclab.py`, `run_official_drawer.py`, `test_probe_pull.py`,
  `test_execution_pull.py`, `test_dynamics_randomization.py`, `visualize_response.py`.
* New tests: `tests/unit/` (80) and `tests/integration/` (44).
* New docs: `README.md`, `CLAUDE.md`, `docs/{ARCHITECTURE,API,OFFICIAL_BASELINE,VALIDATION,DECISIONS,SESSION_LOG,CHANGELOG}.md`,
  `backups/README.md`.
* New configuration: `configs/{controller,probe,execution,dynamics}.yaml` (generated
  snapshots, drift-tested) and `configs/grasp_pose.yaml` (recorded measurement).
* Isaac Lab's own source tree: **unmodified**.

**API introduced.**

```python
ProbePullController.run(initial_force, max_force, target_displacement, max_velocity) -> ProbeResult
ExecutionPullController.run(peak_force, duration) -> ExecutionResult
DynamicsRandomizer.sample(num_envs) / .apply(env, params) / .get_current_params()
PullSystem.build(PullSystemCfg) -> PullSystem
```

**Verification.** All of Phases 0-8 pass; 80 unit tests and 44 integration tests pass.
Headline measurements: official drawer opens to 308.7 mm with a travel direction 0.000°
from the configured pull axis; all four probe stop conditions demonstrated; execution
profile invariant in `F_peak` and duration accurate to one control step; `F_peak = 5 N`,
`T = 2 s` gives `d(T) =` 326.1 / 141.7 / 59.6 mm on easy / medium / hard. Full table with
commands and observed values in `docs/VALIDATION.md`.

**Physical problems found and fixed** (each is a decision entry):

1. The official gripper close command squeezed the handle with ~15 N and ~30 N because the
   hand is 3.1 mm off centre. That leaked a **0.68 N bias onto the pull axis** and opened
   the drawer 14.4 mm in 2 s with *zero* commanded force. Fixed with a per-finger command
   derived from the recorded contact equilibrium (D010).
2. PhysX's drawer `joint_vel` sampled at 60 Hz reported **-0.0076 m/s while the drawer was
   opening at +0.0073 m/s** (contact-chatter aliasing), and after the grip fix went
   identically zero. Replaced by a finite-difference estimate (D009).
3. A `ContactSensor` on the handle reads only normal load, so it measured 0.22-0.37 N
   whether the pull was 4 N or 12 N. Replaced by the wrist joint reaction wrench, validated
   against `m*a + f + c*v` (D006).
4. The pull axis is force-controlled and therefore has no damping of its own, so grasp
   momentum persisted into the probe. Added an initialisation-only velocity brake plus a
   discarded warm-up episode.
5. `HybridPullOSC` stepped the *unwrapped* environment, silently bypassing `RecordVideo`
   (videos were one frame long). All stepping now goes through `HybridPullOSC.step` on the
   wrapped environment (D014).
6. The probe's max-force stop fired one control step early (9.867 N instead of 10.000 N).
7. Writing the drawer's static and dynamic joint friction in two calls made PhysX reject the
   update; they are now written together.

**Commit.** `b85d069edb6ea6033181460b2dcc655da9dae44d` on `agent/phase0-8-bootstrap`; `main` points at the same commit; tagged
`v0.1.0-phase8`.

**Push status.** Pushed. `main`, `agent/phase0-8-bootstrap` and the tag `v0.1.0-phase8`
all resolve to `de31ea9011a2db6f99993cfc19218d5ee7953fae` on
`https://github.com/ysdjy/9.2paper.git`, confirmed with `git ls-remote`. (The first attempt
failed because no GitHub credential was configured on this machine; the user configured one
and the push then succeeded.)

**Remaining issues.** See "Known limitations" in `docs/VALIDATION.md`. The ones that would
most affect the next phase:

* the held axes drift up to 15 mm / 7.5° when the `easy` drawer approaches its end stop;
* a residual 0.25 N pull-axis bias and ~1.3 mm/s creep at zero command;
* only ~40 % of the commanded force reaches the drawer during acceleration, so the useful
  `F_peak` range for a 2 s execution is roughly 2-8 N.

**Next.** Training-data generation: sample `xi`, run probe then execution, and store paired
`(probe history, F_peak, T, d(T))` episodes. The controller APIs should not need to change.

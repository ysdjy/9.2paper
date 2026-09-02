# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is research software; versions mark validated states, not releases.

## [0.4.0] — 2026-09-02

Phase 11: the first real dataset, the first models, and the first time a trained model
chooses a force and the drawer actually moves to the goal.

### Added

* `probe_drawer.protocols.simulation_snapshot` — capture and restore an episode's state so one
  probe can answer many candidate forces. A dataset-generation device, never part of a
  deployment protocol. Validated in `docs/COUNTERFACTUAL_BRANCHING.md`.
* `probe_drawer.dataset` gains `sampling` (Sobol hidden states, label-independent force
  strata, deterministic branch shuffling), `storage` (normalised JSONL + per-probe NPZ) and
  `audit` (nine gates plus the distributions).
* `probe_drawer.models` — `PrivilegedEncoder`, `AdaptationContextEncoder`, `SuccessPredictor`,
  and baselines A–D plus the fixed-force floor.
* `probe_drawer.training` — dynamic-padding DataLoader, train-only `FeatureScaler`,
  classification and selection metrics, and the teacher/student loop.
* `probe_drawer.evaluation.force_selection` — grid search over a predicted landscape, run
  outside the controller.
* Scripts: `validate_branching`, `generate_dataset`, `audit_dataset`, `train_models`,
  `evaluate_closed_loop`, `plot_phase11`.
* Docs: `COUNTERFACTUAL_BRANCHING.md`, `DATASET_V0.md`, `TRAINING_V0.md`, decisions D035–D040.
* Tests: 383 unit (+115) and 84 integration (+15).

### Data and results

* **Dataset v0** — 49 152 candidate rows from 1 536 probes over 512 hidden states, 29.5 MB,
  nine audit gates passed, 0.98 % of hidden states with no positive.
* **Closed loop on 88 unseen drawers** — privileged teacher 95.5 %, ACE + PSP **93.2 %**,
  GRU regressing one force 79.5 %, ridge on nine scalar features 45.5 %, best fixed force
  13.6 %.

### Changed

* Candidates per probe raised from 24 to 32 after measuring that 24 left 6 % of hidden states
  with no positive — a grid-resolution miss, not infeasibility (D038).
* `DrawerStateReader` keeps its four derivative estimators in one registry, so resetting,
  describing and snapshotting them cannot drift apart.
* `MlpForceRegressor` owns its input standardisation as buffers.

### Fixed

* A restore left the TCP pose stale by 34 mm, so every branch inherited the previous branch's
  pose reference. Found by the branching validation.
* 24 branches of 1.5 s exceed the 30 s episode, so the environment would have auto-reset
  partway through every candidate sweep. `episode_length_buf` is now part of the snapshot.
* The audit crashed on a dangling probe reference, a missing history file and an incomplete
  hidden state instead of reporting them.
* `evaluate()` discarded the reliability curve, which would have left the calibration figure
  empty.

### Known limitations

* `T` is fixed, so the parameter is one-dimensional, and the midpoint of a hidden state's
  succeeding force set succeeds for 104 of 105 solvable states. A landscape model therefore has
  no *structural* advantage here; the measured gap is an accuracy gap. The project owner's
  D034 (`p = [F_peak, T]`) is what makes the question askable.
* Baseline C (MLP on summary features) is undertrained and scores below the ridge on the same
  features. Reported as not-a-working-baseline rather than as a result.
* The encoder ablation spans 2.1 points against 0.9–1.7 points of seed noise, and the channels
  it adds are functions of drawer position. The genuinely independent channel, wrist force, is
  excluded by D018 and is the next ablation.

---

## [0.3.0] — 2026-09-02

Phase 10: the probe and the execution become one continuous episode, the task is tightened
against the resulting landscape, and the training dataset gets a schema that cannot leak.

### Added

* `probe_drawer.protocols` — `SequentialPullProtocol` runs
  `INITIAL -> PROBE -> PROBE_END -> inference gap -> EXECUTION -> EVALUATE` with exactly one
  reset, at the start. It sequences the existing controllers and contains no physics. It
  refuses to run with a settling execution, because a settle would erase the post-probe
  velocity silently.
* `HybridPullOSC.coast(steps)` — hold the five motion axes, command zero pull force, do not
  brake. This is the inference gap; contrast `settle()`, which brakes at 200 N·s/m.
* `probe_drawer.dataset` — the formal training-sample schema, the three nested content-addressed
  identifiers (`xi_id`, `probe_id`, `candidate_id`), and grouped splitting with
  `assert_no_leakage()`. A per-row split is not offered.
* `analysis.sweep.force_grid(low, high, step)` — exact, mergeable force grids. The Phase 10
  dataset is three sweeps joined on exact force equality, which a drifting grid would break.
* `SweepRecord.from_sequential_episode`, and the `protocol` / `pre_execution_displacement` /
  `probe_displacement` / `probe_duration` / `probe_features` fields that make one Oracle
  analysis able to read both protocols.
* Scripts: `build_sequential_oracle.py`, `validate_sequential_protocol.py`,
  `refine_task_space.py`, `compare_reset_vs_sequential.py`, `plot_phase10.py` (figures A-G).
* Docs: `SEQUENTIAL_PROTOCOL.md`, `DATASET_SCHEMA.md`, decisions D026-D033.
* Tests: `tests/integration/test_sequential_protocol.py` (17) and
  `tests/unit/test_dataset_schema.py` (34). Suite is now 233 unit + 69 integration.

### Changed

* **The task.** `d_goal` 50 -> **40 mm**, `eps_d` 15 -> **7.5 mm**, `eps_v` 0.08 ->
  **0.03 m/s**, `fall_fraction` 0.20 -> **0.35**. `T` stays at 1.5 s. Coverage rose to 0.972
  despite the tighter tolerances, because the sequential protocol and the finer force grid
  (0.05 N against 0.25 N) resolve bands the Phase 9 grid could not express.
* **`d_goal` is measured from before the probe** (D027), so `d_total(T) = d_probe +
  d_coast + d_execution(T)`. `SweepRecord.final_displacement` carries that quantity for both
  protocols.
* `ExecutionPullController.run` takes a per-environment `peak_force` by scaling one shared
  unit-amplitude profile, so candidates in one episode differ only in amplitude.
* `assess_validity` and `evaluate_execution` take `pre_execution_displacement`.
* The ground truth is now `outputs/logs/sequential_oracle_fall035.json`. The Phase 9 reset
  datasets are kept, not superseded, and are used for the protocol comparison (D026).

### Fixed

* `refine_task_space.py` excluded supplementary datasets with `stem.endswith("_low")`, which
  does not match `..._vlow`. Two 540-row supplements were loaded as full datasets and, because
  the tolerance curves were keyed by fall fraction, the curves for 0.30 and 0.35 were computed
  from a 0.15-0.35 N supplement. The selection was unaffected; figure D was wrong, which is how
  it was found.
* `refine_task_space.py` printed the success band's *centre* under the label "required force",
  which disagreed with the `best_force` quoted everywhere else. Both are now printed, labelled.

### Known limitations

* The probe does not identify damping (D032). Recorded, not fixed; no second probe was added.
* Three of 108 hidden states have no succeeding force at the selected task.
* Intrinsic `d_total(T)` episode noise is about 1 mm, which is the floor under any position
  tolerance.

---

## [0.2.0] — 2026-09-02

Phase 9: the hidden state is fixed, the observations are classified, the force channels are
audited, and every experimental parameter is selected from a sweep.

### Added

* `probe_drawer.observations` — unit, source, filtering and deployability for all 25 logged
  channels, the first model's input vector, and `validate_model_input()` so a
  simulator-only channel cannot reach a deployable model.
* `probe_drawer.evaluation` — `assess_validity()` rejects operating points that are unusable
  as evidence; `evaluate_execution()` labels success as position **and** terminal velocity
  **and** validity.
* `probe_drawer.analysis` — the hidden-state capability audit, the force-channel analysis,
  the sweep record format, the Oracle acceptance rules, and probe summary features.
* `probe_drawer.experiment_plan` — the selected parameters, each citing the sweep it came
  from.
* `probe_drawer.sensors.CausalDerivative` — the one differentiator every derived channel
  goes through, causal by construction and self-describing.
* Seven scripts covering the two audits, the sweep, the Oracle landscape, the probe
  calibration and two figure sets.

### Changed

* `xi` is now exactly `[m, mu_s, mu_d, b]`. The merged `joint_friction` is gone; stiffness
  and per-DOF viscous friction became pinned non-`xi` configuration.
* Dynamics readback comes from `root_physx_view` rather than Isaac Lab's `data` mirror,
  which reports the *request* and can therefore hide a discarded write.
* `PullHistory` went from 16 to 25 channels, including causal accelerations and the two
  privileged drawer force channels.
* `ExecutionResult` gained `peak_velocity` and is snapshotted at `t = T`, with an explicit
  zero-force release afterwards that is excluded from the result.
* The execution profile's ramp-down is 20 % of the duration, chosen from a five-value sweep.
* The provisional 2 N / 10 N / 5 mm / 100 mm / 2 s / 5 N figures are labelled as such
  everywhere; none survived unchanged.

### Notable

* The physics question is answered: the required peak force spans 4.5x across the
  hidden-state grid, and a standardised probe predicts it at |rho| = 0.97.
* 248 tests pass, up from 124.

## [0.1.0] — 2026-09-02

First validated state: the physical and control substrate for single-probe adaptation,
Phases 0 through 8.

### Added

* `probe_drawer.envs.ProbeDrawerEnvCfg` — the official Isaac Lab Franka cabinet scene
  reconfigured for force-driven pulling: hybrid operational-space arm action, a drawer
  handle contact sensor, a grasped reset state with a balanced grip, and every remaining
  source of episode-to-episode randomness removed.
* `probe_drawer.controllers.ProbePullController` — the standardised probe: a configurable
  monotone force ramp with four stop conditions (displacement, velocity, max force,
  timeout) plus absolute safety limits, returning a full `ProbeResult` history.
* `probe_drawer.controllers.ExecutionPullController` — the full-duration execution:
  `F(t) = peak_force * phi(t/duration)` with a fixed smoothstep trapezoid, no goal
  feedback of any kind, returning an `ExecutionResult` that carries no notion of success.
* `probe_drawer.controllers.HybridPullOSC` — the one shared low-level controller, wrapping
  Isaac Lab's `OperationalSpaceController` for 1-DOF force plus 5-DOF pose hold.
* `probe_drawer.envs.DynamicsRandomizer` — samples and applies
  `xi = [drawer mass, joint friction, joint damping]`, with PhysX readback verification and
  four deterministic presets.
* `probe_drawer.sensors.DrawerStateReader` — read-only drawer, TCP, joint and measured
  pull-force access, with each signal's physical provenance documented.
* `probe_drawer.logging.EpisodeLogger` — per-episode `metadata.json` plus
  `trajectory.npz`, recording versions, git commit, controller parameters and `xi`.
* `probe_drawer.state_machines.DrawerGraspStateMachine` — approach and grasp, used to
  record the grasped arm configuration the research environment resets into.
* Six scripts covering inspection, official-baseline validation, probe, execution,
  dynamics and plotting.
* 80 unit tests (no Isaac Sim) and 44 integration tests (one shared Isaac Sim session).
* Full documentation set, including a decision log with 14 settled entries.

### Notable

* Isaac Lab's source tree is not modified anywhere.
* `d_goal` is not an input to the execution controller, and a unit test enforces it.
* `commanded_force` and `measured_force` are separate signals with separate provenance.

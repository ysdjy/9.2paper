# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is research software; versions mark validated states, not releases.

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

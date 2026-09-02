# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is research software; versions mark validated states, not releases.

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

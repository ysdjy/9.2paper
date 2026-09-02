# Session log

One entry per work session. Newest first.

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

**Commit.** See `git log` — recorded on `agent/phase0-8-bootstrap`, merged to `main`.

**Remaining issues.** See "Known limitations" in `docs/VALIDATION.md`. The ones that would
most affect the next phase:

* the held axes drift up to 15 mm / 7.5° when the `easy` drawer approaches its end stop;
* a residual 0.25 N pull-axis bias and ~1.3 mm/s creep at zero command;
* only ~40 % of the commanded force reaches the drawer during acceleration, so the useful
  `F_peak` range for a 2 s execution is roughly 2-8 N.

**Next.** Training-data generation: sample `xi`, run probe then execution, and store paired
`(probe history, F_peak, T, d(T))` episodes. The controller APIs should not need to change.

# Scripts

Every script here is a thin front end: argument parsing, then a call into
`src/probe_drawer/`. If you find yourself adding logic to a script, it belongs in the package
instead — that is what keeps two scripts from disagreeing about what "success" means.

Scripts are in one of three states. The distinction is about **what the repository currently
depends on**, not about age or quality: a historical script may have produced the most
important number in the project and still not be part of the pipeline.

---

## Active pipeline — Setting V1

These run the paper's experiment. They are the ones that must keep working.

| script | stage | Isaac Sim |
|---|---|---|
| `run_official_drawer.py` | record the grasp configuration (also `--cabinet-x-offset`) | yes |
| `generate_dataset.py` | build a dataset from probes and branched executions | yes |
| `audit_dataset.py` | nine gates plus the distributions; exits non-zero on failure | no |
| `train_models.py` | baselines, privileged teacher, ACE + PSP, ablations | no |
| `evaluate_closed_loop.py` | deploy on unseen hidden states, back in physics | yes |
| `plot_phase11.py` | dataset and training figures | no |

## Setting-defining evidence

Not run routinely, but these produced the numbers Setting V1 is frozen on. Keep them
runnable: if a frozen parameter is ever questioned, this is where the answer was measured.

| script | what it settled | doc |
|---|---|---|
| `sweep_goal_distance.py` + `analyze_goal_distance.py` | which goal distances the rig supports, and what bounds each | `docs/GOAL_DISTANCE.md` |
| `calibrate_probe.py` | the Phase 9 probe parameters | `docs/EXPERIMENT_SPACE.md` |
| `validate_branching.py` | whether one probe may answer many candidates | `docs/COUNTERFACTUAL_BRANCHING.md` |
| `validate_sequential_protocol.py` | the inference gap, and that the probe's state survives | `docs/SEQUENTIAL_PROTOCOL.md` |
| `refine_task_space.py` | the Phase 10 task selection | `docs/EXPERIMENT_SPACE.md` |
| `build_sequential_oracle.py` | the 1-D Oracle both datasets are judged against | `docs/ORACLE_LANDSCAPE.md` |

## Historical diagnostics

Answered a question, kept for the answer. **Not** part of Setting V1, and several drive
modules in `probe_drawer.experimental`.

| script | question it answered | outcome |
|---|---|---|
| `sweep_parameter_space_2d.py`, `analyze_landscape_2d.py`, `plot_phase12.py` | does a 2-D `(F, T)` action space buy structure? | real but moderate; `T` nearly degenerate for prediction. Frozen out of V1 (D045) |
| `compare_probes.py` | is a response-triggered probe better? | better on mass, `T` and `mu_d`; worse on `mu_s`; damping still unidentified (D044) |
| `analyze_damping_observability.py` | why is damping invisible — range, magnitude, or noise? | magnitude at probe speeds; the whole `b` range spans 0.75x the force noise floor |
| `analyze_probe_duration.py` | should a `min_probe_duration` exist? | no — duration is itself the strongest feature (D043) |
| `record_diagnostic_videos.py` | what does the motion actually look like? | 21 annotated clips + `index.csv` |
| `compare_reset_vs_sequential.py` | what did the Phase 9 reset hide? | required force overstated ~25 %, ranking preserved (D026) |
| `sweep_execution_space.py`, `build_oracle_landscape.py`, `plot_experiment_space.py`, `plot_phase10.py` | Phase 9/10 task selection | superseded by the sequential Oracle |
| `audit_force_channels.py`, `audit_hidden_states.py`, `plot_probe_identifiability.py` | which signals are real, which are privileged | `docs/FORCE_CHANNEL_AUDIT.md`, `docs/HIDDEN_STATE_AUDIT.md` |
| `inspect_isaaclab.py` | what is installed on this machine | `docs/OFFICIAL_BASELINE.md` |

## Manual checks

`test_probe_pull.py`, `test_execution_pull.py`, `test_dynamics_randomization.py`,
`visualize_response.py` — interactive sanity checks from Phases 4–8, kept because they are the
quickest way to see a controller behave. They are **not** pytest tests despite the names; the
automated suites are in `tests/`.

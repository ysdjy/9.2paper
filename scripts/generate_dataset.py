"""Phase 11C/11D/11F -- generate a dataset: probe once, ask 24 candidate forces.

Structure, and why it is this way:

* **Environments are the hidden-state axis.** Each parallel environment holds a different
  drawer, so one probe episode produces ``num_envs`` independent probes at once. This is the
  same arrangement Phase 10's Oracle used.
* **Each hidden state gets several independent probe repeats.** A repeat is a full episode --
  reset, probe, inference gap -- not a re-use of one probe, so the three repeats of a drawer
  differ by exactly the simulator noise a deployed robot would also face. That is what makes
  an empirical success *probability* measurable rather than assumed.
* **Candidates branch off one snapshot.** After the probe and the gap the state is captured,
  and each candidate force restores it before executing. So all candidates of a probe answer
  the same counterfactual question. Validated in ``docs/COUNTERFACTUAL_BRANCHING.md``; this
  script is not the place to re-litigate whether that is sound.
* **The branch order is shuffled.** Branching drifts slightly with sweep position, and
  executing candidates in force order would make that drift a function of force. Every row
  records its ``branch_index`` so the audit can verify the decorrelation.

The sampling plan is built before the simulator starts, so what will be recorded is decided
and inspectable up front rather than emerging from the loop.

Usage::

    # smoke dataset first -- 8 hidden states, and it must pass the audit
    python scripts/generate_dataset.py --headless --num-xi 8 --repeats 2 --candidates 6 \\
        --num_envs 8 --output outputs/dataset_smoke

    # Dataset v0
    python scripts/generate_dataset.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-xi", type=int, default=512, help="Hidden states to sample.")
parser.add_argument("--repeats", type=int, default=3, help="Independent probe episodes per hidden state.")
parser.add_argument("--candidates", type=int, default=24, help="Candidate forces per probe.")
parser.add_argument("--num_envs", type=int, default=32, help="Hidden states simulated in parallel.")
parser.add_argument("--seed", type=int, default=20260902, help="Dataset seed; fixes both samplers.")
parser.add_argument("--dataset-version", type=str, default="v0", help="Recorded in the manifest.")
parser.add_argument("--output", type=str, default=None, help="Dataset directory.")
parser.add_argument(
    "--jitter", type=float, default=0.4, help="Candidate jitter, as a fraction of a stratum's width."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.analysis.probe_features import extract_features  # noqa: E402
from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.dataset import (  # noqa: E402
    DatasetWriter,
    ForceSamplerCfg,
    ProbeRecord,
    XiSamplerCfg,
    branch_order,
    build_plan,
    candidate_id,
    probe_id,
    xi_id,
)
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, evaluate_execution  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    SEQUENTIAL_TRANSITION_STEPS,
    TRAINING_XI_RANGES,
)
from probe_drawer.observations import DEFAULT_ACE_INPUT  # noqa: E402
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)

#: Channels written into every probe history, and the model's input by default.
#:
#: ``DEFAULT_ACE_INPUT`` is the authoritative list (D018); it is used directly rather than
#: copied so the dataset cannot drift from the observation registry.
HISTORY_CHANNELS = DEFAULT_ACE_INPUT

#: Recorded alongside, excluded from the model's input. Kept so a later ablation -- wrist
#: force in particular -- needs no regeneration (D027 on rich logging, selective input).
DIAGNOSTIC_CHANNELS = (
    "time",
    "measured_force",
    "drawer_velocity_raw",
    "drawer_acceleration_raw",
    "tcp_lateral_error",
    "tcp_orientation_error",
)


def build_system(num_envs: int) -> PullSystem:
    """A system whose execution does not settle, as branching and the protocol require."""
    execution = ExecutionControllerCfg(
        rise_fraction=RECOMMENDED_EXECUTION_CFG.rise_fraction,
        fall_fraction=RECOMMENDED_EXECUTION_CFG.fall_fraction,
        shape=RECOMMENDED_EXECUTION_CFG.shape,
        settle_steps=0,
        zero_force_cleanup_steps=RECOMMENDED_EXECUTION_CFG.zero_force_cleanup_steps,
        post_execution_settle_steps=RECOMMENDED_EXECUTION_CFG.post_execution_settle_steps,
    )
    return PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=RECOMMENDED_PROBE_CFG,
            execution=execution,
        )
    )


def to_parameters(xi: dict, name: str) -> DynamicsParameters:
    """One sampled hidden state as the randomiser's own type."""
    return DynamicsParameters(
        name=name,
        drawer_mass=xi["mass"],
        joint_static_friction=xi["static_friction"],
        joint_dynamic_friction=xi["dynamic_friction"],
        joint_damping=xi["damping"],
    )


def probe_channels(probe, env_index: int, names: tuple[str, ...]) -> dict:
    """One environment's probe history, at its true length.

    ``PullHistory.channel(..., driven_only=True)`` trims to the steps that environment was
    actually driven for, which is what makes the length vary between probes -- a probe stops
    on a displacement threshold, so it takes as long as that drawer needs.

    ``time`` is not a per-environment channel: it is one clock shared by the batch. It is
    still recorded per probe, trimmed the same way, because the schema requires the timebase
    even though a fixed control rate makes it redundant with the sequence length (D037).
    """
    history = probe.history
    driven = history.active_steps(env_index)
    channels = {}
    for name in names:
        values = history.time[driven] if name == "time" else history.channel(name, env_index)
        channels[name] = np.asarray(values, dtype=np.float32)
    return channels


def batches(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    plan = build_plan(
        repeats=args_cli.repeats,
        xi_cfg=XiSamplerCfg(
            num_states=args_cli.num_xi,
            seed=args_cli.seed,
            mass=TRAINING_XI_RANGES.mass,
            static_friction=TRAINING_XI_RANGES.static_friction,
            dynamic_friction_ratio=TRAINING_XI_RANGES.dynamic_friction_ratio,
            damping=TRAINING_XI_RANGES.damping,
        ),
        force_cfg=ForceSamplerCfg(
            count=args_cli.candidates,
            force_range=MAIN_TASK.peak_force_range,
            jitter=args_cli.jitter,
            seed=args_cli.seed,
        ),
    )

    root = Path(args_cli.output) if args_cli.output else project_root() / "outputs" / f"dataset_{args_cli.dataset_version}"
    num_envs = min(args_cli.num_envs, len(plan.states))
    grouped = batches(list(enumerate(plan.states)), num_envs)
    region = OperatingRegionCfg()

    print("\n" + "=" * 78)
    print(f"[gen] dataset       : {args_cli.dataset_version} -> {root}")
    print(f"[gen] hidden states : {len(plan.states)} in {len(grouped)} batch(es) of <= {num_envs}")
    print(f"[gen] probes        : {plan.num_probes} ({plan.repeats} independent repeats each)")
    print(f"[gen] candidates    : {plan.num_candidates} ({plan.force_cfg.count} per probe, branched)")
    print(f"[gen] task          : T={MAIN_TASK.duration} s d_goal={MAIN_TASK.goal_displacement * 1000:g} mm "
          f"eps_d={MAIN_TASK.displacement_tolerance * 1000:g} mm eps_v={MAIN_TASK.velocity_tolerance:g} m/s")
    print(f"[gen] force range   : {MAIN_TASK.peak_force_range[0]}-{MAIN_TASK.peak_force_range[1]} N, "
          f"stratified, jitter {args_cli.jitter}, label-independent")
    print(f"[gen] history       : {list(HISTORY_CHANNELS)} at the raw control rate, true lengths")

    manifest = {
        "dataset_version": args_cli.dataset_version,
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "protocol": "sequential: reset -> probe -> inference gap -> branched executions",
        "counterfactual_labels": "post-probe snapshot restored before each candidate",
        "seed": args_cli.seed,
        "sampling": plan.as_dict(),
        "probe_task": RECOMMENDED_PROBE_TASK.as_dict(),
        "probe_cfg": RECOMMENDED_PROBE_CFG.as_dict(),
        "execution_cfg": RECOMMENDED_EXECUTION_CFG.as_dict(),
        "sequential_transition_steps": SEQUENTIAL_TRANSITION_STEPS,
        "main_task": MAIN_TASK.as_dict(),
        "operating_region": region.as_dict(),
        "history_channels": list(HISTORY_CHANNELS),
        "diagnostic_channels": list(DIAGNOSTIC_CHANNELS),
        "num_envs": num_envs,
        "environment": collect_environment_info().as_dict(),
    }

    system = build_system(num_envs)
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    probe_task = RECOMMENDED_PROBE_TASK.as_kwargs()
    criteria = MAIN_TASK.criteria

    positives = 0
    invalid = 0
    try:
        with DatasetWriter(root, manifest) as writer:
            for index, state in enumerate(plan.states):
                writer.add_hidden_state(xi_id(state), index, state, oracle_feasible=None)

            for batch_number, batch in enumerate(grouped, start=1):
                padded = batch + [batch[-1]] * (num_envs - len(batch))
                parameters = [to_parameters(state, f"xi{index:05d}") for index, state in padded]

                for repeat in range(plan.repeats):
                    randomizer.apply(system.env, parameters)
                    system.reset()

                    task_start = system.reader.drawer_position.clone()
                    probe = system.probe.run(**probe_task)
                    system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
                    pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
                    post_probe_velocity = system.reader.drawer_velocity.cpu().numpy().copy()
                    snapshot = capture_snapshot(system, label=f"batch {batch_number} repeat {repeat}")

                    # Probes first: the writer refuses a candidate whose probe is absent, so a
                    # dangling reference cannot be created even by a crash mid-batch.
                    probes = []
                    for env_index, (_, state) in enumerate(batch):
                        state_id = xi_id(state)
                        identifier = probe_id(state, repeat, RECOMMENDED_PROBE_TASK.as_dict())
                        writer.add_probe(
                            ProbeRecord(
                                probe_id=identifier,
                                xi_id=state_id,
                                repeat_index=repeat,
                                summary=extract_features(probe, env_index).as_dict(),
                                post_probe_state={
                                    "displacement": float(pre_execution[env_index]),
                                    "velocity": float(post_probe_velocity[env_index]),
                                },
                                history=probe_channels(probe, env_index, HISTORY_CHANNELS),
                                diagnostics=probe_channels(probe, env_index, DIAGNOSTIC_CHANNELS),
                            )
                        )
                        probes.append((identifier, state_id, plan.forces[state_id]))

                    # Shuffled per probe, so branch drift cannot align with force.
                    orders = [branch_order(identifier, plan.force_cfg.count) for identifier, _, _ in probes]

                    for position in range(plan.force_cfg.count):
                        restore_snapshot(system, snapshot)
                        forces = [probes[env][2][orders[env][position]] for env in range(len(batch))]
                        padded_forces = forces + [forces[-1]] * (num_envs - len(forces))
                        result = system.execution.run(peak_force=padded_forces, duration=MAIN_TASK.duration)
                        evaluation = evaluate_execution(
                            result, criteria, region, pre_execution_displacement=pre_execution
                        )

                        for env_index, (identifier, state_id, _) in enumerate(probes):
                            verdict = evaluation.verdicts[env_index]
                            force = forces[env_index]
                            writer.add_candidate(
                                {
                                    "candidate_id": candidate_id(
                                        identifier, force, MAIN_TASK.duration, MAIN_TASK.goal_displacement
                                    ),
                                    "probe_id": identifier,
                                    "xi_id": state_id,
                                    "branch_index": position,
                                    "candidate_peak_force": force,
                                    "duration": MAIN_TASK.duration,
                                    "goal_displacement": MAIN_TASK.goal_displacement,
                                    "final_total_displacement": verdict.total_displacement,
                                    "execution_displacement": verdict.execution_displacement,
                                    "pre_execution_displacement": verdict.pre_execution_displacement,
                                    "final_velocity": verdict.terminal_velocity,
                                    "success": bool(verdict.success),
                                    "valid": bool(verdict.valid),
                                    "invalid_reasons": [reason.value for reason in verdict.invalid_reasons],
                                }
                            )
                            positives += int(verdict.success)
                            invalid += int(not verdict.valid)

                    print(
                        f"[gen] batch {batch_number}/{len(grouped)} repeat {repeat + 1}/{plan.repeats} "
                        f"probe steps {[int(probe.history.active_steps(i).size) for i in range(min(4, len(batch)))]} "
                        f"positives so far {positives} invalid {invalid} "
                        f"({time.perf_counter() - started:.0f} s)"
                    )
    finally:
        system.close()

    elapsed = time.perf_counter() - started
    print("[gen]")
    print(f"[gen] wrote {plan.num_candidates} candidate rows from {plan.num_probes} probes in {elapsed:.0f} s")
    print(f"[gen] positives {positives} ({positives / max(plan.num_candidates, 1) * 100:.1f} %), invalid {invalid}")
    print(f"[gen] dataset : {root}")
    print("[gen] next    : python scripts/audit_dataset.py --dataset " f"{root}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

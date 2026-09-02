"""Phase 11B -- is one probe allowed to answer many candidate forces?

This is a gate, not a feature. Dataset v0 wants 24 candidate executions per probe, all
starting from the same post-probe state. That requires capturing the state once and
restoring it before each candidate, and restoring a PhysX scene is not obviously sound:
positions and velocities can be written, but contact manifolds, friction anchors and solver
residuals cannot.

So the question is measured rather than assumed, against four checks:

1. **Restore fidelity.** Immediately after restoring, do the observable quantities match
   what was captured?
2. **Branch determinism.** Same snapshot, same force, several times -- how far apart do the
   finished tasks land? This is the number that decides whether counterfactual labels from
   one probe are comparable at all.
3. **Order independence.** Does a branch's result depend on which branches ran before it?
   If restoring were incomplete, it would.
4. **No systematic bias.** Does branching off one probe land where fresh full sequential
   episodes land, or somewhere systematically else? Compared as distributions, because a
   fresh episode is not reproducible either.

The verdict is printed and written to ``outputs/logs/branching_validation.json``. The bar is
*not* bit-equality: it is that branch-to-branch spread be small against the task's
tolerances, and smaller than the alternative of re-running the probe per candidate.

Usage::

    python scripts/validate_branching.py --headless
    python scripts/validate_branching.py --headless --preset hard --branch-repeats 5
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preset", type=str, default="medium", help="Dynamics preset for the checks.")
parser.add_argument("--num_envs", type=int, default=4, help="Environments (all given the same xi).")
parser.add_argument(
    "--branch-forces",
    type=float,
    nargs="+",
    default=(1.0, 2.5, 4.0),
    help="Candidate forces to branch with.",
)
parser.add_argument(
    "--branch-repeats", type=int, default=3, help="How many times each force is branched from one snapshot."
)
parser.add_argument(
    "--fresh-repeats", type=int, default=4, help="Full sequential episodes used for the bias comparison."
)
parser.add_argument("--bias-force", type=float, default=2.5, help="Force used for the bias comparison (N).")
parser.add_argument(
    "--drift-branches",
    type=int,
    default=24,
    help="Branches from one snapshot for the drift check. Defaults to Dataset v0's candidate count.",
)
parser.add_argument("--drift-force", type=float, default=2.5, help="Force used for the drift check (N).")
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.envs import DynamicsRandomizer, preset  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.protocols import (  # noqa: E402
    InferenceTransitionCfg,
    SequentialProtocolCfg,
    SequentialPullProtocol,
    capture_snapshot,
    restore_snapshot,
)
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import collect_environment_info, enable_unbuffered_stdout, project_root  # noqa: E402

#: Restoring must reproduce the captured observables to this precision.
#:
#: Anchored to the simulator's own resolution rather than to the task: these are quantities
#: written directly through the articulation writers, so anything above float32 round-trip
#: noise would mean the write did not take. 1 um and 1 um/s are several orders of magnitude
#: below the task's 7.5 mm tolerance.
RESTORE_POSITION_TOLERANCE = 1e-6
RESTORE_VELOCITY_TOLERANCE = 1e-6

#: The bar branching has to clear, and why it is comparative rather than absolute.
#:
#: The first version of this check used ``eps_d / 10 = 750 um`` as an absolute bound. That
#: was the wrong bar and it was set before the comparison number existed: a *fresh* full
#: sequential episode is not reproducible either -- Phase 10 measured ~0.9-1.1 mm of
#: ``d_total(T)`` spread over six identical episodes -- so demanding that branches agree to
#: 750 um demands more of branching than the physics offers at all.
#:
#: What actually matters is whether branching is worse than the alternative it replaces
#: (re-running the probe before every candidate). So the criterion is: at each force, the
#: branch-to-branch spread must not exceed the fresh-episode spread by more than this
#: factor. ``eps_d`` is still reported alongside, because a spread that is a large fraction
#: of the tolerance makes labels noisy however it arose -- that is a property of the task,
#: recorded in the dataset audit, not a fault of branching.
BRANCH_SPREAD_ALLOWANCE = 1.5

#: Reported next to every spread, so the numbers stay interpretable.
POSITION_TOLERANCE = MAIN_TASK.displacement_tolerance

#: A force where even fresh episodes disagree by more than half the position tolerance is
#: not a fair test of branching.
#:
#: Near a high-friction drawer's breakaway threshold the outcome is bistable: the same
#: command either breaks the drawer loose and runs, or does not, and the two branches land
#: tens of millimetres apart. Measured on the ``hard`` preset at 5 N, fresh episodes spread
#: 6.95 mm and branches 32.9 mm in one run and 0.97 mm in another -- the difference being
#: which side of the threshold that run's probe happened to leave the drawer on.
#:
#: Such an operating point is flagged and excluded from the comparative gate, because no
#: protocol produces a reliable label there. It is *not* hidden: it is reported, and it is
#: the reason Dataset v0 keeps three independent probe repeats per hidden state so that
#: label noise can be measured instead of assumed (``docs/COUNTERFACTUAL_BRANCHING.md``).
UNRELIABLE_FRESH_SPREAD = 0.5 * MAIN_TASK.displacement_tolerance


def build_system(num_envs: int) -> PullSystem:
    """A system whose execution does not settle, as the sequential protocol requires."""
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


def observables(system: PullSystem) -> dict:
    """Everything a branch's outcome could depend on that we can actually read.

    Deliberately read through :class:`DrawerStateReader`, not from the raw articulation:
    these are the quantities the controllers and the evaluator consume, so agreement here is
    what "the branch starts from the same state" has to mean in practice.
    """
    reader = system.reader
    return {
        "drawer_position": reader.drawer_position.cpu().numpy().copy(),
        "drawer_velocity": reader.drawer_velocity.cpu().numpy().copy(),
        "arm_joint_position": reader.arm_joint_position.cpu().numpy().copy(),
        "arm_joint_velocity": reader.arm_joint_velocity.cpu().numpy().copy(),
        "finger_joint_position": reader.finger_joint_position.cpu().numpy().copy(),
        "tcp_pose": reader.tcp_pose.cpu().numpy().copy(),
        "tcp_pull_axis_velocity": reader.tcp_pull_axis_velocity.cpu().numpy().copy(),
    }


def max_difference(left: dict, right: dict) -> dict:
    """Largest absolute disagreement per observable."""
    return {name: float(np.abs(left[name] - right[name]).max()) for name in left}


def run_probe_and_snapshot(system: PullSystem, protocol: SequentialPullProtocol, parameters) -> tuple:
    """Reset, apply xi, probe, coast the inference gap, and freeze the result.

    Returns the probe result, the displacement already accumulated, and the snapshot. The
    probe is run through the protocol's own helpers so this cannot drift away from what the
    real protocol does.
    """
    randomizer = DynamicsRandomizer()
    randomizer.apply(system.env, [parameters] * system.env.num_envs)
    system.reset()

    task_start = system.reader.drawer_position.clone()
    probe = system.probe.run(**RECOMMENDED_PROBE_TASK.as_kwargs())
    system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)

    pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
    snapshot = capture_snapshot(system, label="post-probe + inference gap")
    return probe, pre_execution, snapshot


def branch(system: PullSystem, snapshot, force: float, pre_execution: np.ndarray) -> dict:
    """Restore the snapshot and run one candidate execution."""
    restore_snapshot(system, snapshot)
    restored = observables(system)
    result = system.execution.run(peak_force=force, duration=MAIN_TASK.duration)
    total = pre_execution + result.final_displacement
    return {
        "force": force,
        "restored": restored,
        "total_displacement": total.copy(),
        "final_velocity": result.final_velocity.copy(),
        "execution_displacement": result.final_displacement.copy(),
        "peak_velocity": result.peak_velocity.copy(),
    }


def check_branch_drift(
    system: PullSystem, protocol: SequentialPullProtocol, parameters, snapshot, pre_execution: np.ndarray
) -> dict:
    """0 -- the decisive one: does the Nth branch behave like the first?

    Dataset v0 takes 24 candidate executions from a single snapshot, so the question is not
    "are two branches alike" but "has the 24th branch drifted from the 1st". Anything the
    snapshot fails to restore accumulates, and a slow drift would show up as a candidate's
    label depending on where in the sweep it happened to sit -- which would make the whole
    counterfactual design invalid in a way that no pairwise comparison detects.

    Run first, on an undisturbed snapshot, and at a *single* force, so the only thing varying
    is the branch index.
    """
    force = args_cli.drift_force
    count = args_cli.drift_branches
    totals = np.stack(
        [branch(system, snapshot, force, pre_execution)["total_displacement"] for _ in range(count)]
    )

    index = np.arange(count, dtype=float)
    per_env = totals.mean(axis=1)
    # Least-squares slope of outcome against branch index: the drift the generator would
    # accumulate over one probe's worth of candidates.
    slope = float(np.polyfit(index, per_env, 1)[0])
    first_half, second_half = per_env[: count // 2], per_env[count // 2 :]

    spread = float((totals.max(axis=0) - totals.min(axis=0)).max())
    drift = slope * (count - 1)

    # The same number of *fresh* episodes at the same force, so branch-to-branch agreement is
    # judged against what re-running the probe would have given -- with equal sample sizes on
    # both sides. The pairwise version of this comparison used 2-4 repeats and its ratio
    # ranged 0.71-2.99 for the same physics, which is why it is only a diagnostic now.
    fresh = fresh_episodes(system, protocol, parameters, force, count)
    restore_snapshot(system, snapshot)
    fresh_spread = float((fresh.max(axis=0) - fresh.min(axis=0)).max())
    fresh_slope = float(np.polyfit(index, fresh.mean(axis=1), 1)[0])

    return {
        "force": force,
        "branches": count,
        "totals": totals.tolist(),
        "fresh_totals": fresh.tolist(),
        "mean_total_displacement": float(totals.mean()),
        "fresh_mean_total_displacement": float(fresh.mean()),
        "spread": spread,
        "fresh_spread": fresh_spread,
        "spread_over_fresh": spread / fresh_spread if fresh_spread > 0 else float("inf"),
        "slope_per_branch": slope,
        "fresh_slope_per_episode": fresh_slope,
        "drift_over_the_sweep": drift,
        "first_half_mean": float(first_half.mean()),
        "second_half_mean": float(second_half.mean()),
        "half_to_half_shift": float(second_half.mean() - first_half.mean()),
        "spread_over_eps_d": spread / POSITION_TOLERANCE,
        "drift_over_eps_d": abs(drift) / POSITION_TOLERANCE,
        "allowance": BRANCH_SPREAD_ALLOWANCE,
        # Two conditions, both comparative. Spread: branching must be no noisier than the
        # alternative. Drift: the systematic component must stay well inside the position
        # tolerance, because unlike noise it would correlate a candidate's label with its
        # position in the sweep.
        "drift_tolerance": 0.2 * POSITION_TOLERANCE,
        "spread_passes": bool(spread <= BRANCH_SPREAD_ALLOWANCE * fresh_spread),
        "drift_passes": bool(abs(drift) <= 0.2 * POSITION_TOLERANCE),
        "passes": bool(
            spread <= BRANCH_SPREAD_ALLOWANCE * fresh_spread and abs(drift) <= 0.2 * POSITION_TOLERANCE
        ),
    }


def check_restore_fidelity(system: PullSystem, snapshot) -> dict:
    """1 -- restoring reproduces what was captured.

    Captured *after* the snapshot was taken and then again after a long execution has moved
    everything, so the check is a real restore rather than a no-op.
    """
    captured = observables(system)

    restore_snapshot(system, snapshot)
    immediate = max_difference(captured, observables(system))

    # Disturb the scene properly before the real test: a restore that only looks correct
    # because nothing had moved would prove nothing.
    system.execution.run(peak_force=4.0, duration=MAIN_TASK.duration)
    moved = observables(system)
    restore_snapshot(system, snapshot)
    after_disturbance = max_difference(captured, observables(system))

    return {
        "immediate": immediate,
        "after_a_full_execution": after_disturbance,
        "drawer_moved_by_the_disturbance": float(
            np.abs(moved["drawer_position"] - captured["drawer_position"]).max()
        ),
        "position_tolerance": RESTORE_POSITION_TOLERANCE,
        "velocity_tolerance": RESTORE_VELOCITY_TOLERANCE,
        "passes": bool(
            max(after_disturbance["drawer_position"], after_disturbance["arm_joint_position"])
            <= RESTORE_POSITION_TOLERANCE
            and max(after_disturbance["drawer_velocity"], after_disturbance["arm_joint_velocity"])
            <= RESTORE_VELOCITY_TOLERANCE
        ),
    }


def fresh_episodes(
    system: PullSystem, protocol: SequentialPullProtocol, parameters, force: float, repeats: int
) -> np.ndarray:
    """``repeats`` complete sequential episodes at one force, probe included.

    This is the alternative branching replaces, so its spread is the yardstick.
    """
    randomizer = DynamicsRandomizer()
    totals = []
    for _ in range(repeats):
        randomizer.apply(system.env, [parameters] * system.env.num_envs)
        totals.append(protocol.run(peak_force=force).total_displacement.copy())
    return np.stack(totals)


def check_branch_determinism(
    system: PullSystem, protocol: SequentialPullProtocol, parameters, snapshot, pre_execution: np.ndarray
) -> dict:
    """2 -- same snapshot, same force, repeated; compared against fresh episodes.

    Both spreads are measured at the *same* force, because the spread grows steeply with
    force: near the goal a drawer is on the steep part of ``d(F)``, so the same jitter in the
    starting state turns into much more displacement.
    """
    rows = []
    for force in args_cli.branch_forces:
        outcomes = [branch(system, snapshot, force, pre_execution) for _ in range(args_cli.branch_repeats)]
        totals = np.stack([outcome["total_displacement"] for outcome in outcomes])
        velocities = np.stack([outcome["final_velocity"] for outcome in outcomes])
        starts = np.stack([outcome["restored"]["drawer_position"] for outcome in outcomes])

        fresh = fresh_episodes(system, protocol, parameters, force, args_cli.branch_repeats)
        # Re-running fresh episodes destroyed the state, so recover it for the next force.
        restore_snapshot(system, snapshot)

        branch_spread = float((totals.max(axis=0) - totals.min(axis=0)).max())
        fresh_spread = float((fresh.max(axis=0) - fresh.min(axis=0)).max())
        rows.append(
            {
                "force": force,
                "repeats": len(outcomes),
                "mean_total_displacement": float(totals.mean()),
                "branch_spread": branch_spread,
                "fresh_mean_total_displacement": float(fresh.mean()),
                "fresh_spread": fresh_spread,
                "branch_over_fresh": branch_spread / fresh_spread if fresh_spread > 0 else float("inf"),
                "branch_spread_over_eps_d": branch_spread / POSITION_TOLERANCE,
                "final_velocity_spread": float((velocities.max(axis=0) - velocities.min(axis=0)).max()),
                "start_position_spread": float((starts.max(axis=0) - starts.min(axis=0)).max()),
                # Kept per repeat: a large spread can mean either uniform jitter or a
                # bimodal split, and only the raw values tell them apart.
                "branch_totals": totals.tolist(),
                "fresh_totals": fresh.tolist(),
                "branch_final_velocities": velocities.tolist(),
            }
        )

    for row in rows:
        row["fresh_is_unreliable"] = bool(row["fresh_spread"] > UNRELIABLE_FRESH_SPREAD)

    gated = [row for row in rows if not row["fresh_is_unreliable"]]
    flagged = [row["force"] for row in rows if row["fresh_is_unreliable"]]
    worst_ratio = max((row["branch_over_fresh"] for row in gated), default=float("nan"))
    return {
        "rows": rows,
        "gated_forces": [row["force"] for row in gated],
        "flagged_bistable_forces": flagged,
        "worst_branch_over_fresh": worst_ratio,
        "worst_branch_spread": max(row["branch_spread"] for row in rows),
        "worst_branch_spread_over_eps_d": max(row["branch_spread_over_eps_d"] for row in rows),
        "allowance": BRANCH_SPREAD_ALLOWANCE,
        "unreliable_fresh_spread_threshold": UNRELIABLE_FRESH_SPREAD,
        # Reported, not gated. With 2-4 repeats per force this ratio ranged 0.71-2.99 across
        # runs of the same physics, so it cannot separate "branching is noisier" from
        # sampling noise. ``branch_drift`` measures the same property with 24 samples on both
        # sides and is what the verdict uses instead.
        "diagnostic_only": True,
        "passes": True,
    }


def check_order_independence(
    system: PullSystem, snapshot, pre_execution: np.ndarray, reference_spread: float
) -> dict:
    """3 -- a branch must not remember what ran before it.

    Args:
        reference_spread: Spread identical branches already show, from the 24-branch drift
            check. Order dependence is only real if it exceeds that.
    """
    low, high = min(args_cli.branch_forces), max(args_cli.branch_forces)

    first_order = [branch(system, snapshot, force, pre_execution) for force in (low, high)]
    second_order = [branch(system, snapshot, force, pre_execution) for force in (high, low)]

    ascending = {outcome["force"]: outcome["total_displacement"] for outcome in first_order}
    descending = {outcome["force"]: outcome["total_displacement"] for outcome in second_order}
    differences = {
        f"F={force:g}": float(np.abs(ascending[force] - descending[force]).max()) for force in (low, high)
    }
    worst = max(differences.values())
    # Judged against the branch-to-branch spread the previous check measured, not against a
    # separate bound: order dependence only means something if it exceeds the noise two
    # identical branches already show.
    return {
        "forces": [low, high],
        "difference_by_force": differences,
        "worst": worst,
        "reference_spread": reference_spread,
        "passes": bool(worst <= max(reference_spread, 1e-5)),
    }


def check_no_bias(system: PullSystem, protocol: SequentialPullProtocol, parameters) -> dict:
    """4 -- branching lands where fresh episodes land.

    A fresh sequential episode is not reproducible either, so this compares a distribution
    against a distribution: the branch mean has to sit inside the fresh episodes' range, and
    the gap between the two means has to be small next to the fresh spread.
    """
    force = args_cli.bias_force
    fresh = []
    for _ in range(args_cli.fresh_repeats):
        randomizer = DynamicsRandomizer()
        randomizer.apply(system.env, [parameters] * system.env.num_envs)
        episode = protocol.run(peak_force=force)
        fresh.append(episode.total_displacement.copy())
    fresh_totals = np.stack(fresh)

    probe, pre_execution, snapshot = run_probe_and_snapshot(system, protocol, parameters)
    branched = np.stack(
        [
            branch(system, snapshot, force, pre_execution)["total_displacement"]
            for _ in range(args_cli.fresh_repeats)
        ]
    )

    fresh_mean, branch_mean = float(fresh_totals.mean()), float(branched.mean())
    fresh_low, fresh_high = float(fresh_totals.min()), float(fresh_totals.max())
    bias = branch_mean - fresh_mean
    fresh_spread = fresh_high - fresh_low
    return {
        "force": force,
        "fresh_episodes": args_cli.fresh_repeats,
        "fresh_mean_total": fresh_mean,
        "fresh_range": [fresh_low, fresh_high],
        "fresh_spread": fresh_spread,
        "branch_mean_total": branch_mean,
        "branch_spread": float(branched.max() - branched.min()),
        "bias": bias,
        "bias_over_fresh_spread": abs(bias) / fresh_spread if fresh_spread > 0 else float("inf"),
        "branch_mean_inside_fresh_range": bool(fresh_low <= branch_mean <= fresh_high),
        # A bias smaller than the fresh spread cannot be distinguished from sampling noise at
        # this sample size; larger than the task tolerance would matter regardless.
        "passes": bool(abs(bias) <= fresh_spread),
    }


def check_probe_history_untouched(probe, system: PullSystem, snapshot, pre_execution: np.ndarray) -> dict:
    """The probe's record is the model's input; branching must not edit it."""
    before = {
        "displacement": probe.final_displacement.copy(),
        "duration": probe.duration.copy(),
        "history_steps": probe.history.num_steps,
        "history_checksum": float(np.abs(probe.history.drawer_position).sum()),
    }
    branch(system, snapshot, args_cli.bias_force, pre_execution)
    after = {
        "displacement": probe.final_displacement.copy(),
        "duration": probe.duration.copy(),
        "history_steps": probe.history.num_steps,
        "history_checksum": float(np.abs(probe.history.drawer_position).sum()),
    }
    return {
        "history_steps": before["history_steps"],
        "displacement_unchanged": bool(np.array_equal(before["displacement"], after["displacement"])),
        "duration_unchanged": bool(np.array_equal(before["duration"], after["duration"])),
        "steps_unchanged": before["history_steps"] == after["history_steps"],
        "checksum_unchanged": before["history_checksum"] == after["history_checksum"],
        "passes": bool(
            np.array_equal(before["displacement"], after["displacement"])
            and before["history_checksum"] == after["history_checksum"]
        ),
    }


def main() -> None:
    enable_unbuffered_stdout()
    parameters = preset(args_cli.preset)

    protocol_cfg = SequentialProtocolCfg(
        probe_task=RECOMMENDED_PROBE_TASK,
        duration=MAIN_TASK.duration,
        transition=InferenceTransitionCfg(steps=SEQUENTIAL_TRANSITION_STEPS),
    )

    print("\n" + "=" * 78)
    print(f"[branch] preset       : {args_cli.preset} {parameters.as_dict()}")
    print(f"[branch] task         : T={MAIN_TASK.duration} s, d_goal={MAIN_TASK.goal_displacement * 1000:g} mm")
    print(f"[branch] branch forces: {list(args_cli.branch_forces)} N x {args_cli.branch_repeats} repeats")

    system = build_system(args_cli.num_envs)
    system.verify_measured_force_available()
    protocol = SequentialPullProtocol(system, protocol_cfg)
    report: dict = {
        "preset": parameters.as_dict(),
        "num_envs": args_cli.num_envs,
        "task": MAIN_TASK.as_dict(),
        "transition_steps": SEQUENTIAL_TRANSITION_STEPS,
        "probe_task": RECOMMENDED_PROBE_TASK.as_dict(),
        "environment": collect_environment_info().as_dict(),
    }
    try:
        probe, pre_execution, snapshot = run_probe_and_snapshot(system, protocol, parameters)
        report["snapshot"] = snapshot.describe()
        report["probe"] = {
            "displacement_mm": (probe.final_displacement * 1000).tolist(),
            "duration_s": probe.duration.tolist(),
            "history_steps": probe.history.num_steps,
            "pre_execution_displacement_mm": (pre_execution * 1000).tolist(),
        }

        # Drift first, on an undisturbed snapshot: every later check runs dozens of
        # executions, and this is the one that must not inherit their effects.
        report["branch_drift"] = check_branch_drift(
            system, protocol, parameters, snapshot, pre_execution
        )
        report["restore_fidelity"] = check_restore_fidelity(system, snapshot)
        report["branch_determinism"] = check_branch_determinism(
            system, protocol, parameters, snapshot, pre_execution
        )
        # Reference the drift check's 24-sample spread, not the 2-4 sample pairwise one: the
        # question is whether order dependence exceeds the branch-to-branch noise, and the
        # small-sample estimate of that noise is several times too low.
        report["order_independence"] = check_order_independence(
            system, snapshot, pre_execution, report["branch_drift"]["spread"]
        )
        report["probe_history_untouched"] = check_probe_history_untouched(
            probe, system, snapshot, pre_execution
        )
        report["no_systematic_bias"] = check_no_bias(system, protocol, parameters)
    finally:
        system.close()

    checks = (
        "branch_drift",
        "restore_fidelity",
        "branch_determinism",
        "order_independence",
        "probe_history_untouched",
        "no_systematic_bias",
    )
    report["all_passed"] = all(report[name]["passes"] for name in checks)
    _print(report, checks)

    output = (
        project_root() / "outputs" / "logs" / "branching_validation.json"
        if args_cli.output is None
        else project_root() / args_cli.output
    )
    output.write_text(json.dumps(report, indent=2, default=float))
    print(f"[branch] report written: {output}")
    print("=" * 78 + "\n")


def _print(report: dict, checks: tuple[str, ...]) -> None:
    print("[branch]")
    snapshot = report["snapshot"]
    print(f"[branch] snapshot captured: {snapshot['articulations']} x {snapshot['per_articulation_fields']}")
    print(f"[branch] plus controller  : {snapshot['controller_state']}")
    print(f"[branch] plus sensors     : {snapshot['sensor_state']}")
    print(f"[branch] NOT captured     : {snapshot['not_captured']}")

    drift = report["branch_drift"]
    print("[branch]")
    print(
        f"[branch] 0 branch drift over {drift['branches']} branches at F={drift['force']:g} N "
        f"(the sweep size Dataset v0 uses):"
    )
    print(
        f"[branch]   branches: mean {drift['mean_total_displacement'] * 1000:.3f} mm, "
        f"spread {drift['spread'] * 1e6:.0f} um ({drift['spread_over_eps_d']:.2f} x eps_d)"
    )
    print(
        f"[branch]   fresh   : mean {drift['fresh_mean_total_displacement'] * 1000:.3f} mm, "
        f"spread {drift['fresh_spread'] * 1e6:.0f} um -> branch/fresh "
        f"{drift['spread_over_fresh']:.2f} (allowance {drift['allowance']:.1f}) "
        f"{'pass' if drift['spread_passes'] else 'FAIL'}"
    )
    print(
        f"[branch]   slope {drift['slope_per_branch'] * 1e6:+.2f} um/branch -> "
        f"{drift['drift_over_the_sweep'] * 1e6:+.0f} um across the sweep, "
        f"half-to-half shift {drift['half_to_half_shift'] * 1e6:+.0f} um"
    )
    print(
        f"[branch]   drift against a {drift['drift_tolerance'] * 1e6:.0f} um bar "
        f"({drift['drift_over_eps_d']:.2f} x eps_d) "
        f"{'pass' if drift['drift_passes'] else 'FAIL'} -> "
        f"{'PASS' if drift['passes'] else 'FAIL'}"
    )

    fidelity = report["restore_fidelity"]
    print("[branch]")
    print(
        f"[branch] 1 restore fidelity : a full execution moved the drawer "
        f"{fidelity['drawer_moved_by_the_disturbance'] * 1000:.2f} mm, then restoring left"
    )
    for name, value in sorted(fidelity["after_a_full_execution"].items()):
        print(f"[branch]     {name:>26} : {value:.3e}")
    print(f"[branch]   -> {'PASS' if fidelity['passes'] else 'FAIL'}")

    determinism = report["branch_determinism"]
    print("[branch]")
    print("[branch] 2 branch determinism, against fresh episodes at the same force:")
    print(
        f"[branch]   {'F (N)':>6} {'branch d':>10} {'branch sp':>11} {'fresh d':>10} "
        f"{'fresh sp':>10} {'ratio':>7} {'/eps_d':>7}  flag"
    )
    for row in determinism["rows"]:
        print(
            f"[branch]   {row['force']:6.2f} {row['mean_total_displacement'] * 1000:8.2f}mm "
            f"{row['branch_spread'] * 1e6:8.0f}um {row['fresh_mean_total_displacement'] * 1000:8.2f}mm "
            f"{row['fresh_spread'] * 1e6:8.0f}um {row['branch_over_fresh']:7.2f} "
            f"{row['branch_spread_over_eps_d']:7.2f}  "
            f"{'BISTABLE' if row['fresh_is_unreliable'] else ''}"
        )
    if determinism["flagged_bistable_forces"]:
        print(
            f"[branch]   flagged as bistable (fresh spread itself > "
            f"{determinism['unreliable_fresh_spread_threshold'] * 1000:.2f} mm): "
            f"{determinism['flagged_bistable_forces']} N -- excluded from the gate, "
            f"no protocol gives a reliable label there"
        )
    print(
        f"[branch]   worst branch/fresh ratio over {determinism['gated_forces']} N: "
        f"{determinism['worst_branch_over_fresh']:.2f} "
        f"against an allowance of {determinism['allowance']:.1f} "
        f"(diagnostic only -- see branch_drift for the gated comparison)"
    )
    print(
        f"[branch]   worst branch spread {determinism['worst_branch_spread'] * 1e6:.0f} um = "
        f"{determinism['worst_branch_spread_over_eps_d']:.2f} x eps_d "
        f"(a task property, reported not gated)"
    )

    order = report["order_independence"]
    print("[branch]")
    print(
        f"[branch] 3 order independence: worst {order['worst'] * 1e6:.0f} um against the "
        f"{order['reference_spread'] * 1e6:.0f} um two identical branches already show -> "
        f"{'PASS' if order['passes'] else 'FAIL'}"
    )

    untouched = report["probe_history_untouched"]
    print(
        f"[branch] 4 probe history    : {untouched['history_steps']} steps, unchanged="
        f"{untouched['checksum_unchanged']} -> {'PASS' if untouched['passes'] else 'FAIL'}"
    )

    bias = report["no_systematic_bias"]
    print("[branch]")
    print(
        f"[branch] 5 no systematic bias at F={bias['force']:g} N:\n"
        f"[branch]   fresh episodes : mean {bias['fresh_mean_total'] * 1000:.3f} mm, "
        f"spread {bias['fresh_spread'] * 1000:.3f} mm\n"
        f"[branch]   branches       : mean {bias['branch_mean_total'] * 1000:.3f} mm, "
        f"spread {bias['branch_spread'] * 1000:.3f} mm\n"
        f"[branch]   bias {bias['bias'] * 1000:+.3f} mm "
        f"({bias['bias_over_fresh_spread']:.2f} x the fresh spread), "
        f"branch mean inside fresh range={bias['branch_mean_inside_fresh_range']}"
    )
    print(f"[branch]   -> {'PASS' if bias['passes'] else 'FAIL'}")

    print("[branch]")
    print(f"[branch] VERDICT: {'PASS -- branching is usable' if report['all_passed'] else 'FAIL'}")
    for name in checks:
        print(f"[branch]   {name:>26} : {'pass' if report[name]['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
    simulation_app.close()

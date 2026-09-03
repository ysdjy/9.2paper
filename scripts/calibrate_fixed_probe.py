r"""Phase 13 §C -- pick the fixed-budget probe, once, then freeze it.

Setting V1's probe is a standardised excitation: one force profile, identical for every
hidden state, run for its whole budget, with only the safety limits able to end it early
(``docs/DECISIONS.md`` D044). Two numbers define it -- the amplitude ``F_probe`` and the
budget ``H`` -- and this script measures a handful of candidate pairs so that the choice is
data rather than judgement.

What it measures, per candidate
-------------------------------
For each candidate ``(F_probe, H)`` and each hidden state:

1. run the probe, recording whether the drawer broke away, how far it travelled, how fast it
   was still moving, and whether anything tripped a safety limit;
2. wait the fixed inference gap, then snapshot;
3. sweep ``F_peak`` from that snapshot, restoring between candidates, and take the **smallest
   force that achieves reach_success** as the force this drawer required.

The required force is re-derived per candidate on purpose. It is a property of the *state the
probe left behind*, not of the hidden state alone, so borrowing it from another candidate's
sweep -- or from Phase 11's 40 mm Oracle -- would score each probe against the wrong target.

The selection rule lives in ``probe_drawer.analysis.fixed_probe_calibration`` and was written
before this script was first run. This file gathers measurements and prints them; it does not
decide anything.

Deliberately small: the point is to choose and freeze, not to keep optimising the probe. The
default is 24 hidden states and four candidates.

Usage::

    python scripts/calibrate_fixed_probe.py --headless
    python scripts/calibrate_fixed_probe.py --headless --duration 2.0 \
        --candidates 4.5,0.5
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-xi", type=int, default=24, help="Hidden states to calibrate over (16-32).")
parser.add_argument("--num_envs", type=int, default=24, help="Hidden states in parallel.")
parser.add_argument(
    "--candidates",
    type=str,
    default="3.5,0.5 4.5,0.4 4.5,0.6 5.5,0.4",
    help=(
        "Space-separated 'F_probe,H' pairs in N and s. The defaults span the trade the rule "
        "has to resolve: amplitude has to reach the stiffest breakaway (mu_s up to 3 N, of "
        "which only 60-80 %% is transmitted), and the budget is what keeps the softest state "
        "from travelling a large part of the goal."
    ),
)
parser.add_argument("--goal", type=float, default=0.10, help="d_goal for Setting V1 (m).")
parser.add_argument("--duration", type=float, default=1.5, help="T_goal for Setting V1 (s).")
parser.add_argument("--force-low", type=float, default=0.5, help="Lowest candidate F_peak (N).")
parser.add_argument("--force-high", type=float, default=9.0, help="Highest candidate F_peak (N).")
parser.add_argument("--force-step", type=float, default=0.25, help="F_peak grid spacing (N).")
parser.add_argument("--seed", type=int, default=20260902)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.analysis.fixed_probe_calibration import (  # noqa: E402
    MAX_INTRUSION,
    MIN_REACH_COVERAGE,
    CandidateOutcome,
    FixedProbeCandidate,
    XiOutcome,
    score_candidate,
    select_candidate,
)
from probe_drawer.analysis.probe_features import (  # noqa: E402
    PROBE_FEATURES,
    assert_features_are_deployable,
    extract_features,
)
from probe_drawer.analysis.sweep import force_grid  # noqa: E402
from probe_drawer.dataset import branch_order  # noqa: E402
from probe_drawer.dataset.sampling import representative_hidden_states  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, SuccessCriteria, evaluate_execution  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)


def parse_candidates(text: str) -> list[FixedProbeCandidate]:
    """``"3.5,0.5 4.5,0.4"`` into candidates, rejecting anything malformed loudly."""
    candidates = []
    for token in text.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Candidate {token!r} is not 'F_probe,H'.")
        candidates.append(FixedProbeCandidate(peak_force=float(parts[0]), duration=float(parts[1])))
    if not candidates:
        raise ValueError("No candidates given.")
    return candidates


def batches(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def build_system(num_envs: int) -> PullSystem:
    """The frozen Setting V1 wiring: no execution settle, so the probe's state survives."""
    return PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=RECOMMENDED_PROBE_CFG,
            execution=RECOMMENDED_EXECUTION_CFG,
        )
    )


def required_forces(
    system: PullSystem,
    forces: tuple[float, ...],
    criteria: SuccessCriteria,
    region: OperatingRegionCfg,
    pre_execution: np.ndarray,
    snapshot,
    live: int,
    tag: str,
) -> tuple[list[float | None], list[dict]]:
    """Smallest ``F_peak`` reaching the goal, per environment, by sweeping from one snapshot.

    Returns the per-environment required force (``None`` where nothing reached) alongside the
    whole reach/stable table, because the table is what shows whether the band is a real band
    or a single grid cell.
    """
    reached: list[list[float]] = [[] for _ in range(live)]
    table: list[dict] = []
    # Shuffled so that any residual branch-order drift is uncorrelated with force.
    for position, index in enumerate(branch_order(tag, len(forces))):
        force = forces[index]
        restore_snapshot(system, snapshot)
        result = system.execution.run(peak_force=force, duration=args_cli.duration)
        report = evaluate_execution(result, criteria, region, pre_execution_displacement=pre_execution)
        row = {"peak_force": force, "order": position, "envs": []}
        for env in range(live):
            verdict = report.verdicts[env]
            row["envs"].append(
                {
                    "reach_success": verdict.reach_success,
                    "stable_success": verdict.stable_success,
                    "total_displacement": verdict.total_displacement,
                    "position_error": verdict.displacement_error,
                    "terminal_velocity": verdict.terminal_velocity,
                    "valid": verdict.valid,
                    "invalid_reasons": [reason.value for reason in verdict.invalid_reasons],
                }
            )
            if verdict.reach_success:
                reached[env].append(force)
        table.append(row)
    return [min(hits) if hits else None for hits in reached], table


def main() -> None:
    enable_unbuffered_stdout()
    assert_features_are_deployable()
    started = time.perf_counter()

    candidates = parse_candidates(args_cli.candidates)
    forces = force_grid(args_cli.force_low, args_cli.force_high, args_cli.force_step)
    states = representative_hidden_states(args_cli.num_xi, seed=args_cli.seed)
    num_envs = min(args_cli.num_envs, len(states))
    grouped = batches(states, num_envs)
    region = OperatingRegionCfg()
    criteria = SuccessCriteria(
        goal_displacement=args_cli.goal,
        displacement_tolerance=MAIN_TASK.displacement_tolerance,
        velocity_tolerance=MAIN_TASK.velocity_tolerance,
    )
    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / "fixed_probe_calibration.json"
    )

    print("\n" + "=" * 78)
    print(f"[cal] candidates : {', '.join(c.label for c in candidates)}")
    print(f"[cal] hidden     : {len(states)} states in {len(grouped)} batch(es) of <= {num_envs}")
    print(f"[cal] task       : d_goal={args_cli.goal * 1000:g} mm T_goal={args_cli.duration:g} s "
          f"eps_d={criteria.displacement_tolerance * 1000:g} mm eps_v={criteria.velocity_tolerance:g} m/s")
    print(f"[cal] F_peak     : {forces[0]:.2f} .. {forces[-1]:.2f} N step {args_cli.force_step:g} "
          f"({len(forces)} values)")
    print(f"[cal] gates      : safe, responsive, intrusion <= {MAX_INTRUSION:.0%} of d_goal, "
          f"reach coverage >= {MIN_REACH_COVERAGE:.0%}")
    print(f"[cal] episodes   : {len(candidates) * len(grouped) * (1 + len(forces))}")

    system = build_system(num_envs)
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    outcomes: list[CandidateOutcome] = []
    probe_tables: dict[str, list] = {}

    try:
        for candidate in candidates:
            rows: list[XiOutcome] = []
            tables: list[dict] = []
            for batch_number, batch in enumerate(grouped, start=1):
                padded = batch + [batch[-1]] * (num_envs - len(batch))
                parameters = [
                    DynamicsParameters(
                        drawer_mass=state["mass"],
                        joint_static_friction=state["static_friction"],
                        joint_dynamic_friction=state["dynamic_friction"],
                        joint_damping=state["damping"],
                        name=f"xi{index:03d}",
                    )
                    for index, state in enumerate(padded)
                ]
                randomizer.apply(system.env, parameters)
                system.reset()

                task_start = system.reader.drawer_position.clone()
                probe = system.probe.run_fixed_budget(
                    peak_force=candidate.peak_force, duration=candidate.duration
                )
                system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
                pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
                post_velocity = system.reader.drawer_velocity.cpu().numpy().copy()
                snapshot = capture_snapshot(system, label=f"{candidate.label} batch {batch_number}")

                needed, table = required_forces(
                    system,
                    forces,
                    criteria,
                    region,
                    pre_execution,
                    snapshot,
                    len(batch),
                    tag=f"{candidate.label}-batch{batch_number}",
                )
                tables.append({"batch": batch_number, "forces": table})

                for env in range(len(batch)):
                    features = extract_features(probe, env)
                    rows.append(
                        XiOutcome(
                            hidden_state=dict(batch[env]),
                            moved=features.moved,
                            post_probe_displacement=float(pre_execution[env]),
                            post_probe_velocity=float(post_velocity[env]),
                            safety_aborted=probe.termination_reason[env].value == "safety_abort",
                            features=features.as_vector(),
                            required_force=needed[env],
                        )
                    )
                print(
                    f"[cal] {candidate.label} batch {batch_number}/{len(grouped)}: "
                    f"probe d={pre_execution[: len(batch)].max() * 1000:.1f} mm max, "
                    f"solved {sum(f is not None for f in needed)}/{len(batch)} "
                    f"({time.perf_counter() - started:.0f} s)"
                )

            outcome = score_candidate(candidate, rows, args_cli.goal)
            outcomes.append(outcome)
            probe_tables[candidate.label] = tables
            metrics = outcome.metrics
            print(
                f"[cal] {candidate.label:<16} moved {metrics['moved_fraction']:.2f}  "
                f"intrusion {metrics['max_intrusion_fraction']:.2f}  "
                f"coverage {metrics['reach_coverage']:.2f}  "
                f"RMSE {outcome.score:.3f} N  "
                f"{'PASS' if outcome.passed else 'FAIL ' + ','.join(k for k, v in outcome.gates.items() if not v)}"
            )
    finally:
        system.close()

    winner = select_candidate(outcomes)
    payload = {
        "phase": "13C",
        "task": {"goal_displacement": args_cli.goal, "duration": args_cli.duration, **criteria.as_dict()},
        "force_grid": list(forces),
        "probe_cfg": RECOMMENDED_PROBE_CFG.as_dict(),
        "execution_cfg": RECOMMENDED_EXECUTION_CFG.as_dict(),
        "operating_region": region.as_dict(),
        "transition_steps": SEQUENTIAL_TRANSITION_STEPS,
        "hidden_state_seed": args_cli.seed,
        "feature_names": list(PROBE_FEATURES),
        "gates": {"max_intrusion": MAX_INTRUSION, "min_reach_coverage": MIN_REACH_COVERAGE},
        "candidates": [outcome.as_dict() for outcome in outcomes],
        "per_hidden_state": {
            outcome.candidate.label: [
                {
                    "hidden_state": row.hidden_state,
                    "moved": row.moved,
                    "post_probe_displacement": row.post_probe_displacement,
                    "post_probe_velocity": row.post_probe_velocity,
                    "safety_aborted": row.safety_aborted,
                    "features": dict(zip(PROBE_FEATURES, row.features, strict=True)),
                    "required_force": row.required_force,
                }
                for row in outcome.rows
            ]
            for outcome in outcomes
        },
        "force_tables": probe_tables,
        "selected": winner.candidate.as_dict() if winner else None,
        "git_commit": git_commit(),
        "environment": collect_environment_info().as_dict(),
        "elapsed_s": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    print("[cal]")
    for outcome in outcomes:
        metrics = outcome.metrics
        ratio = metrics["required_force_ratio"]
        span = (
            f"F_req {metrics['required_force_min']:.2f}-{metrics['required_force_max']:.2f} N ({ratio:.1f}x)"
            if ratio
            else "F_req none reached"
        )
        print(
            f"[cal] {outcome.candidate.label:<16} "
            f"post-probe d {metrics['median_post_probe_displacement'] * 1000:5.1f} / "
            f"{metrics['max_post_probe_displacement'] * 1000:5.1f} mm (median/max)  "
            f"v {metrics['max_post_probe_velocity']:.3f} m/s max  {span}"
        )
        print(
            f"[cal] {'':<16} readout R2 {outcome.readout['r2']:+.3f} "
            f"RMSE {outcome.readout['rmse']:.3f} N on sd {outcome.readout['target_sd']:.3f} N "
            f"(n={outcome.readout['n']})"
        )
    print("[cal]")
    if winner is None:
        print("[cal] SELECTED   : none -- no candidate passed all four gates.")
        print("[cal]              Widening the candidate set is a decision for a person.")
    else:
        print(f"[cal] SELECTED   : {winner.candidate.label} "
              f"(F_probe={winner.candidate.peak_force:g} N, H={winner.candidate.duration:g} s)")
    print(f"[cal] written    : {output}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

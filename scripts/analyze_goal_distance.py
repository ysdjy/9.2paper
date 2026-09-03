"""Which goal distances are usable, and what bounds the ones that are not.

Reads the sweep from ``scripts/sweep_goal_distance.py`` and answers, per candidate distance:
can it be reached, how safely, and when it cannot -- *why*. The attribution is the point:
"performance degrades past 250 mm" is not actionable, while "past 250 mm the wrist passes
within 8 % of a joint limit" says whether to shorten the task or move the base.

Four attributions, ordered by how fundamental the constraint is: ``drawer_limit`` (the
cabinet's end stop, unfixable), ``posture`` (the arm's reach or manipulability, fixable by
repositioning), ``control`` (held-axis drift or an OSC abort, fixable by tuning), and
``task`` / ``unreachable_in_swept_box`` (no constraint at all -- the swept parameters simply
did not land there).

No simulator. Usage::

    python scripts/analyze_goal_distance.py --dataset outputs/logs/goal_distance_sweep.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from probe_drawer.experimental.goal_distance import (
    CANDIDATE_GOALS,
    DRAWER_LIMIT_MARGIN,
    JOINT_LIMIT_MARGIN,
    LongPullRecord,
    feasibility_by_goal,
)
from probe_drawer.experiment_plan import MAIN_TASK
from probe_drawer.utils import git_commit, project_root


def posture_against_distance(records: list[LongPullRecord], bins: int = 10) -> list[dict]:
    """Joint margin, manipulability and drift as functions of how far the drawer went.

    This is what separates "the drawer ran out of travel" from "the arm ran out of posture":
    if the joint margin falls smoothly with displacement and crosses the threshold well before
    400 mm, the robot is the binding constraint, and the drawer's end stop is a red herring.
    """
    displacement = np.array([record.final_displacement for record in records])
    edges = np.linspace(0.0, displacement.max(), bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = [
            record
            for record, value in zip(records, displacement, strict=True)
            if low <= value < high or (high == edges[-1] and value == high)
        ]
        if not inside:
            continue
        rows.append(
            {
                "low_mm": float(low * 1000),
                "high_mm": float(high * 1000),
                "episodes": len(inside),
                "median_joint_margin": float(np.median([r.min_joint_margin for r in inside])),
                "worst_joint_margin": float(min(r.min_joint_margin for r in inside)),
                "near_joint_limit_fraction": float(np.mean([r.near_joint_limit for r in inside])),
                "limiting_joint_mode": Counter(r.limiting_joint for r in inside).most_common(1)[0][0],
                "median_manipulability": float(np.median([r.manipulability for r in inside])),
                "median_pull_axis_transmission": float(
                    np.median([r.pull_axis_transmission for r in inside])
                ),
                "median_jacobian_condition": float(np.median([r.jacobian_condition for r in inside])),
                "median_lateral_drift_mm": float(np.median([r.peak_lateral_drift for r in inside]) * 1000),
                "worst_lateral_drift_mm": float(max(r.peak_lateral_drift for r in inside) * 1000),
                "median_wrist_force": float(np.median([r.peak_measured_force for r in inside])),
                "worst_wrist_spike": float(max(r.wrist_force_spike for r in inside)),
                "valid_fraction": float(np.mean([r.valid for r in inside])),
                "invalid_reasons": dict(Counter(reason for r in inside for reason in r.invalid_reasons)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tolerance", type=float, default=None, help="eps_d (m). Defaults to the task's.")
    parser.add_argument("--velocity-tolerance", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.is_absolute():
        path = project_root() / path
    payload = json.loads(path.read_text())
    records = [LongPullRecord.from_dict(row) for row in payload["records"]]
    tolerance = args.tolerance if args.tolerance is not None else MAIN_TASK.displacement_tolerance
    velocity = (
        args.velocity_tolerance if args.velocity_tolerance is not None else MAIN_TASK.velocity_tolerance
    )

    feasibility = feasibility_by_goal(
        records, CANDIDATE_GOALS, tolerance=tolerance, velocity_tolerance=velocity
    )
    posture = posture_against_distance(records)
    report = {
        "dataset": str(path),
        "git_commit": git_commit(),
        "episodes": len(records),
        "hidden_states": payload["num_hidden_states"],
        "drawer_travel_limit": payload["drawer_travel_limit"],
        "tolerance": tolerance,
        "velocity_tolerance": velocity,
        "joint_limit_margin": JOINT_LIMIT_MARGIN,
        "drawer_limit_margin": DRAWER_LIMIT_MARGIN,
        "goal_distance_feasibility": [entry.as_dict() for entry in feasibility],
        "posture_against_distance": posture,
    }
    output = Path(args.output) if args.output else path.with_name("goal_distance_feasibility.json")
    output.write_text(json.dumps(report, indent=2, default=float))
    _print(report, feasibility, posture, records)
    print(f"[goal] report written: {output}")
    print("=" * 78 + "\n")


def _print(report: dict, feasibility: list, posture: list, records: list) -> None:
    print("\n" + "=" * 78)
    print(
        f"[goal] {report['episodes']} episodes over {report['hidden_states']} hidden states; "
        f"eps_d={report['tolerance'] * 1000:g} mm eps_v={report['velocity_tolerance']:g} m/s"
    )
    displacement = np.array([record.final_displacement for record in records])
    print(
        f"[goal] displacements reached: {displacement.min() * 1000:.1f} .. {displacement.max() * 1000:.1f} mm "
        f"({displacement.max() / report['drawer_travel_limit'] * 100:.1f} % of the 400 mm travel)"
    )

    print("[goal]")
    print("[goal] POSTURE AGAINST DISTANCE -- does the robot degrade before the drawer does?")
    print(
        f"[goal]   {'range (mm)':>14} {'n':>5} {'jmargin':>8} {'worst':>7} {'near%':>6} "
        f"{'joint':>5} {'manip':>7} {'pull-tx':>8} {'cond':>7} {'drift':>7} {'valid%':>7}"
    )
    for row in posture:
        print(
            f"[goal]   {row['low_mm']:6.0f}-{row['high_mm']:6.0f} {row['episodes']:5d} "
            f"{row['median_joint_margin']:8.3f} {row['worst_joint_margin']:7.3f} "
            f"{row['near_joint_limit_fraction'] * 100:5.0f}% {row['limiting_joint_mode']:5d} "
            f"{row['median_manipulability']:7.4f} {row['median_pull_axis_transmission']:8.4f} "
            f"{row['median_jacobian_condition']:7.1f} {row['median_lateral_drift_mm']:6.2f}mm "
            f"{row['valid_fraction'] * 100:6.1f}%"
        )

    print("[goal]")
    print("[goal] GOAL DISTANCE FEASIBILITY")
    print(
        f"[goal]   {'d_goal':>7} {'feasible':>9} {'stable':>7} {'joint@lim':>10} {'drawer@lim':>11} "
        f"{'drift med/worst (mm)':>21} {'wrist med/worst (N)':>20} {'rec':>4}"
    )
    for entry in feasibility:
        print(
            f"[goal]   {entry.goal * 1000:5.0f}mm {entry.feasible_fraction * 100:8.1f}% "
            f"{entry.stable_control_fraction * 100:6.1f}% "
            f"{entry.near_joint_limit_fraction * 100:9.0f}% {entry.near_drawer_limit_fraction * 100:10.0f}% "
            f"{entry.median_lateral_drift * 1000:9.2f} /{entry.worst_lateral_drift * 1000:9.2f} "
            f"{entry.median_wrist_force:9.2f} /{entry.worst_wrist_force:8.2f} "
            f"{'YES' if entry.recommended else 'no':>4}"
        )

    print("[goal]")
    print("[goal] why each distance is or is not feasible, per hidden state:")
    for entry in feasibility:
        print(f"[goal]   {entry.goal * 1000:5.0f} mm : {entry.attributions}  -- {entry.reason}")

    print("[goal]")
    recommended = [entry for entry in feasibility if entry.recommended]
    if recommended:
        print(
            f"[goal] RECOMMENDED d_goal range: "
            f"{min(e.goal for e in recommended) * 1000:.0f} - {max(e.goal for e in recommended) * 1000:.0f} mm"
        )
    else:
        print("[goal] RECOMMENDED d_goal range: none of the candidates met both bounds")


if __name__ == "__main__":
    main()

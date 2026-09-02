"""Phase 10 -- tighten the task's tolerances against the sequential Oracle.

Phase 9 settled on ``eps_d = 15 mm``, which is 30 % of a 50 mm goal and too loose to call the
pull accurate. That figure was set by the force grid's 0.25 N spacing, not by physics: near
breakaway ``dd/dF`` reaches about 40 mm/N, so a 0.25 N step *is* 10 mm of displacement and no
tighter tolerance could be resolved. This phase sweeps the force axis at 0.10 N and asks how
tight the tolerances can actually be.

Every candidate ``(ramp-down, T, d_goal, eps_d, eps_v)`` is scored by the same rule as
Phase 9 (:mod:`probe_drawer.analysis.oracle`), with one change of emphasis stated up front:
the recommendation is **not** the most discriminating candidate. Discrimination is the last
tie-breaker, after the task is achievable, precise, and comes to rest. Maximising it alone
was what produced the loose tolerance in the first place.

Priority order, applied as a lexicographic filter and recorded in the report:

1. coverage at least ``--min-coverage`` (default 0.95);
2. position tolerance at most ``--max-eps-d`` (default 7.5 mm);
3. terminal-velocity tolerance at most ``--max-eps-v`` (default 0.05 m/s);
4. the remaining acceptance conditions -- band width, contiguity, mechanical margin,
   grid resolution;
5. only then, the largest spread of required force.

If no candidate satisfies 1-4 the script says so and prints the Pareto front, rather than
relaxing a bound to produce an answer.

Usage::

    python scripts/refine_task_space.py
    python scripts/refine_task_space.py --max-eps-d 0.005 --max-eps-v 0.04
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from probe_drawer.analysis.oracle import ACCEPTANCE, TaskCandidate, score_candidate
from probe_drawer.analysis.sweep import SweepDataset
from probe_drawer.evaluation import SuccessCriteria
from probe_drawer.utils import project_root

#: Goals to try (m). The upper end is bounded by what a drawer can reach *and stop at*.
GOAL_CANDIDATES = (0.03, 0.04, 0.05, 0.06, 0.075)

#: Position tolerances to try (m). 15 mm is Phase 9's value, kept as a reference row.
DISPLACEMENT_TOLERANCES = (0.0025, 0.005, 0.0075, 0.010, 0.015)

#: Terminal-speed tolerances to try (m/s). 0.08 is Phase 9's value, kept as a reference row.
VELOCITY_TOLERANCES = (0.03, 0.04, 0.05, 0.08)


#: Matches only a merged Oracle dataset, e.g. ``sequential_oracle_fall035``.
#:
#: The low-force supplements (``..._low``, ``..._vlow``) were merged into the file they
#: supplement, so scoring them again would add a dataset that only spans 0.15-0.35 N under
#: the same ``fall=`` label -- and the tolerance curves, which are keyed by that label, would
#: silently end up describing the supplement instead of the real landscape.
MERGED_DATASET = re.compile(r"sequential_oracle_fall\d+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help="Sequential Oracle dataset(s). Defaults to outputs/logs/sequential_oracle_fall*.json.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.95, help="Requirement 1.")
    parser.add_argument("--max-eps-d", type=float, default=0.0075, help="Requirement 2 (m).")
    parser.add_argument("--max-eps-v", type=float, default=0.05, help="Requirement 3 (m/s).")
    parser.add_argument("--output", type=str, default=None, help="Where to write the report.")
    parser.add_argument("--top", type=int, default=14, help="How many ranked candidates to print.")
    args = parser.parse_args()

    paths = [Path(path) for path in args.dataset] if args.dataset else sorted(
        path
        for path in (project_root() / "outputs" / "logs").glob("sequential_oracle_fall*.json")
        if MERGED_DATASET.fullmatch(path.stem)
    )
    if not paths:
        parser.error("No sequential Oracle datasets found. Run scripts/build_sequential_oracle.py first.")

    scored: list[dict] = []
    summaries: list[dict] = []
    curves: dict[str, dict] = {}

    for path in paths:
        dataset = SweepDataset.load(path)
        fall_fraction = dataset.metadata.get("fall_fraction")
        label = f"fall={fall_fraction:g}"
        summaries.append(
            {
                "path": str(path),
                "label": label,
                "fall_fraction": fall_fraction,
                "protocol": dataset.metadata.get("protocol"),
                "transition_steps": dataset.metadata.get("transition_steps"),
                "rows": len(dataset),
                "valid_rows": len(dataset.valid_records),
                "validity_rate": dataset.validity_rate(),
                "hidden_states": len(dataset.xi_keys()),
                "forces": [dataset.forces()[0], dataset.forces()[-1], len(dataset.forces())],
                "durations": dataset.durations(),
                "invalid_reasons": dataset.invalid_reason_counts(),
            }
        )
        curves[label] = _tolerance_curves(dataset)

        for duration in dataset.durations():
            for goal in GOAL_CANDIDATES:
                for tolerance in DISPLACEMENT_TOLERANCES:
                    for velocity in VELOCITY_TOLERANCES:
                        candidate = TaskCandidate(duration, goal, tolerance, velocity)
                        score = score_candidate(dataset, candidate)
                        payload = score.as_dict()
                        payload.update(label=label, fall_fraction=fall_fraction, dataset=str(path))
                        payload["intervals"] = score.intervals
                        scored.append(payload)

    selected, front = _select(scored, args.min_coverage, args.max_eps_d, args.max_eps_v)
    report = {
        "requirements": {
            "min_coverage": args.min_coverage,
            "max_eps_d": args.max_eps_d,
            "max_eps_v": args.max_eps_v,
            "acceptance": _acceptance_dict(),
        },
        "datasets": summaries,
        "tolerance_curves": curves,
        "num_candidates": len(scored),
        "scores": [{key: value for key, value in row.items() if key != "intervals"} for row in scored],
        "selected": {key: value for key, value in selected.items() if key != "intervals"} if selected else None,
        "selected_intervals": selected.get("intervals") if selected else None,
        "pareto_front": [{key: value for key, value in row.items() if key != "intervals"} for row in front],
    }

    _print(report, scored, selected, front, args)
    output = Path(args.output) if args.output else project_root() / "outputs" / "logs" / "task_refinement.json"
    output.write_text(json.dumps(report, indent=2, default=float))
    print(f"[refine] report written : {output}")
    print("=" * 78 + "\n")


def _acceptance_dict() -> dict:
    return {
        "min_discrimination": ACCEPTANCE.min_discrimination,
        "min_relative_width": ACCEPTANCE.min_relative_width,
        "max_relative_width": ACCEPTANCE.max_relative_width,
        "max_travel_fraction": ACCEPTANCE.max_travel_fraction,
        "min_contiguous_fraction": ACCEPTANCE.min_contiguous_fraction,
        "max_tolerance_ratio": ACCEPTANCE.max_tolerance_ratio,
        "require_grid_resolved": ACCEPTANCE.require_grid_resolved,
    }


def _tolerance_curves(dataset: SweepDataset) -> dict:
    """Coverage against each tolerance separately, for the figures.

    One tolerance is varied while the other is held at the value being proposed, so the
    curves show what each bound costs on its own rather than a mixture of the two.
    """
    duration = dataset.durations()[0]
    keys = dataset.xi_keys()

    def coverage(goal: float, tolerance: float, velocity: float) -> float:
        criteria = SuccessCriteria(goal, tolerance, velocity)
        achieved = sum(
            1 for key in keys if any(row.succeeds(criteria) for row in dataset.select(xi_key=key, duration=duration))
        )
        return achieved / len(keys)

    return {
        "duration": duration,
        "goals": list(GOAL_CANDIDATES),
        "displacement_tolerances": list(DISPLACEMENT_TOLERANCES),
        "velocity_tolerances": list(VELOCITY_TOLERANCES),
        "coverage_vs_eps_d": {
            f"{goal:g}": [coverage(goal, tolerance, 0.05) for tolerance in DISPLACEMENT_TOLERANCES]
            for goal in GOAL_CANDIDATES
        },
        "coverage_vs_eps_v": {
            f"{goal:g}": [coverage(goal, 0.005, velocity) for velocity in VELOCITY_TOLERANCES]
            for goal in GOAL_CANDIDATES
        },
    }


def _select(
    scored: list[dict], min_coverage: float, max_eps_d: float, max_eps_v: float
) -> tuple[dict | None, list[dict]]:
    """Apply the priority order, and build a Pareto front for the report either way."""
    precise = [
        row
        for row in scored
        if row["candidate"]["displacement_tolerance"] <= max_eps_d + 1e-12
        and row["candidate"]["velocity_tolerance"] <= max_eps_v + 1e-12
    ]
    covered = [row for row in precise if row["coverage"] >= min_coverage]
    accepted = [row for row in covered if row["accepted"]]
    selected = max(accepted, key=lambda row: row["discrimination"]) if accepted else None

    # Pareto front over (coverage up, eps_d down, eps_v down): the trade-off to report when
    # the requirements cannot all be met at once.
    def dominates(left: dict, right: dict) -> bool:
        return (
            left["coverage"] >= right["coverage"]
            and left["candidate"]["displacement_tolerance"] <= right["candidate"]["displacement_tolerance"]
            and left["candidate"]["velocity_tolerance"] <= right["candidate"]["velocity_tolerance"]
            and (
                left["coverage"] > right["coverage"]
                or left["candidate"]["displacement_tolerance"] < right["candidate"]["displacement_tolerance"]
                or left["candidate"]["velocity_tolerance"] < right["candidate"]["velocity_tolerance"]
            )
        )

    front = [row for row in scored if row["accepted"] and not any(dominates(other, row) for other in scored)]
    front.sort(key=lambda row: (-row["coverage"], row["candidate"]["displacement_tolerance"]))
    return selected, front[:12]


def _print(report: dict, scored: list[dict], selected: dict | None, front: list[dict], args) -> None:
    print("\n" + "=" * 78)
    for summary in report["datasets"]:
        print(
            f"[refine] {summary['label']:<11}: {summary['rows']} rows, "
            f"{summary['validity_rate'] * 100:.1f} % valid, {summary['hidden_states']} hidden states, "
            f"F {summary['forces'][0]:.2f}-{summary['forces'][1]:.2f} N ({summary['forces'][2]} values), "
            f"gap={summary['transition_steps']} steps"
        )
    print(f"[refine] requirements  : {json.dumps(report['requirements'], default=float)}")
    print(f"[refine] candidates    : {report['num_candidates']} scored")
    print("[refine]")

    eligible = [
        row
        for row in scored
        if row["candidate"]["displacement_tolerance"] <= args.max_eps_d + 1e-12
        and row["candidate"]["velocity_tolerance"] <= args.max_eps_v + 1e-12
    ]
    ranked = sorted(eligible, key=lambda row: (-row["accepted"], -row["coverage"], -row["discrimination"]))
    print(
        f"[refine] {'profile':>11} {'T':>4} {'d_goal':>7} {'eps_d':>7} {'eps_v':>6} {'cover':>6} "
        f"{'discr':>6} {'width':>6} {'dF(N)':>6} {'contig':>6} {'ok':>3}"
    )
    for row in ranked[: args.top]:
        candidate = row["candidate"]
        print(
            f"[refine] {row['label']:>11} {candidate['duration']:4.1f} "
            f"{candidate['goal_displacement'] * 1000:7.1f} {candidate['displacement_tolerance'] * 1000:7.2f} "
            f"{candidate['velocity_tolerance']:6.3f} {row['coverage']:6.3f} {row['discrimination']:6.2f} "
            f"{row['median_relative_width']:6.2f} {row['median_absolute_width']:6.2f} "
            f"{row['contiguous_fraction']:6.2f} {'yes' if row['accepted'] else 'no':>3}"
        )

    if selected is None:
        print("[refine]")
        print("[refine] NO CANDIDATE MET ALL REQUIREMENTS. Pareto front (coverage / eps_d / eps_v):")
        for row in front:
            candidate = row["candidate"]
            print(
                f"[refine]   {row['label']:<11} d_goal={candidate['goal_displacement'] * 1000:5.1f} mm "
                f"eps_d={candidate['displacement_tolerance'] * 1000:5.2f} mm "
                f"eps_v={candidate['velocity_tolerance']:.3f} m/s coverage={row['coverage']:.3f} "
                f"discrimination={row['discrimination']:.2f}"
            )
        return

    candidate = selected["candidate"]
    print("[refine]")
    print(f"[refine] SELECTED      : {selected['label']}, {json.dumps(candidate)}")
    print(
        f"[refine]   coverage {selected['coverage']:.3f} "
        f"({selected['num_with_success']}/{selected['num_hidden_states']} hidden states)  "
        f"discrimination {selected['discrimination']:.2f}  "
        f"band {selected['median_absolute_width']:.2f} N ({selected['median_relative_width']:.2f} relative)  "
        f"contiguous {selected['contiguous_fraction']:.2f}  max travel {selected['max_travel_fraction']:.2f}"
    )
    _print_intervals(selected.get("intervals"))


def _print_intervals(intervals: list[dict] | None, count: int = 10) -> None:
    if not intervals:
        return
    achieved = sorted((row for row in intervals if row["any_success"]), key=lambda row: row["force_centre"])
    if not achieved:
        return
    step = max(1, len(achieved) // count)
    print("[refine]")
    print(f"[refine] success force bands, {len(achieved)}/{len(intervals)} hidden states:")
    print(
        f"[refine] {'m':>6} {'mu_s':>6} {'mu_d':>6} {'b':>6} | {'F_low':>6} {'F_high':>7} "
        f"{'F_best':>7} {'d_best':>8} {'v_best':>8}"
    )
    for row in achieved[::step][:count]:
        mass, static, dynamic, damping = row["xi"]
        print(
            f"[refine] {mass:6.1f} {static:6.2f} {dynamic:6.2f} {damping:6.1f} | "
            f"{row['force_low']:6.2f} {row['force_high']:7.2f} {row['best_force']:7.2f} "
            f"{row['best_displacement'] * 1000:8.1f} {row['best_velocity']:8.4f}"
        )
    # Two different quantities, both worth reporting. ``best_force`` is the grid point whose
    # displacement came closest to the goal, and is what the rest of the project means by
    # "the force this drawer needs". ``force_centre`` is the midpoint of the success band,
    # which is the more forgiving target for a predictor because it maximises the margin.
    bests = [row["best_force"] for row in achieved]
    centres = [row["force_centre"] for row in achieved]
    for label, values in (("closest to goal", bests), ("band centre    ", centres)):
        print(
            f"[refine] required force, {label}: {min(values):.2f} .. {max(values):.2f} N "
            f"(median {float(np.median(values)):.2f} N), a {max(values) / min(values):.1f}x range"
        )
    missing = [row for row in intervals if not row["any_success"]]
    if missing:
        print(f"[refine] no succeeding force for {len(missing)} hidden state(s), e.g. xi={missing[0]['xi']}")


if __name__ == "__main__":
    main()

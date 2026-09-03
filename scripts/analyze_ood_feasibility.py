"""Report on an out-of-distribution feasibility sweep.

Answers four questions and nothing else: what fraction of the out-of-distribution range is
solvable under Setting V1, where the failures sit in hidden-state space, what force they need,
and whether the task's own ``F_peak`` range is what is cutting them off.

It does not recommend changing the range. If the numbers say the range is unreasonable, that is
a decision for a person, and this prints the evidence for it.

Usage::

    python scripts/analyze_ood_feasibility.py
    python scripts/analyze_ood_feasibility.py --report outputs/logs/ood_feasibility.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_drawer.analysis.ood_feasibility import summarise_ood_feasibility
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root


def main() -> None:
    enable_unbuffered_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=str, default="outputs/logs/ood_feasibility.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.report)
    if not path.is_absolute():
        path = project_root() / path
    payload = json.loads(path.read_text())
    allowed = tuple(payload["task"]["peak_force_range"])
    summary = summarise_ood_feasibility(payload["rows"], allowed)

    counts = summary["counts"]
    print("\n" + "=" * 84)
    print(f"[ood] states     : {counts['states']} genuinely out-of-distribution")
    print(f"[ood] task       : d_goal={payload['task']['goal_displacement'] * 1000:g} mm "
          f"T={payload['task']['duration']:g} s, F_peak allowed {allowed[0]}-{allowed[1]} N")
    print(f"[ood] swept      : {payload['force_grid'][0]:.2f}-{payload['force_grid'][-1]:.2f} N "
          f"({len(payload['force_grid'])} values), deliberately past the task's ceiling")
    print("[ood]")
    print(f"[ood] SOLVABLE within the task's force range : {counts['solvable_within_task_range']}"
          f"/{counts['states']} = {counts['fraction_solvable_within_task_range'] * 100:.1f} %")
    print(f"[ood] solvable at some force, any magnitude  : {counts['solvable_any_force']}"
          f"/{counts['states']} = {counts['fraction_solvable_any_force'] * 100:.1f} %")
    print(f"[ood]   of those, needing more than allowed  : {counts['solvable_only_outside_task_range']}")
    print(f"[ood] unsolvable at any swept force          : {counts['unsolvable_at_any_force']}")

    required = summary["required_force"]
    band = summary["band_width"]
    if required:
        print("[ood]")
        print(f"[ood] required force : {required['min']:.2f} - {required['max']:.2f} N "
              f"(median {required['median']:.2f}, mean {required['mean']:.2f})")
        print(f"[ood] band width     : {band['min']:.2f} - {band['max']:.2f} N "
              f"(median {band['median']:.2f})")

    truncation = summary["truncation"]
    print("[ood]")
    print(f"[ood] TRUNCATION by F_peak in [{allowed[0]}, {allowed[1]}] N:")
    print(f"[ood]   states solvable only above the ceiling : "
          f"{truncation['states_needing_more_than_allowed']}")
    if truncation["required_force_above_ceiling"]:
        above = truncation["required_force_above_ceiling"]
        print(f"[ood]   they need {above['min']:.2f} - {above['max']:.2f} N "
              f"(median {above['median']:.2f})")
    print(f"[ood]   solvable states sitting at the ceiling  : "
          f"{truncation['solvable_states_at_the_ceiling']}")
    print(f"[ood]   solvable states sitting at the floor    : "
          f"{truncation['solvable_states_at_the_floor']}")

    print("[ood]")
    print(f"[ood] {'novel axis':>28} {'states':>7} {'failed':>7} {'rate':>7}")
    for axis, values in summary["novel_axis_rates"].items():
        print(f"[ood] {axis:>28} {values['states']:>7} {values['failed']:>7} "
              f"{values['failure_rate'] * 100:6.1f} %")

    failures = summary["failures"]
    if failures:
        print("[ood]")
        print(f"[ood] the {len(failures)} state(s) not solvable inside the task's range:")
        for row in failures:
            axes = row["sampled_axes"]
            print(
                f"[ood]   m={axes['mass']:5.2f} mu_s={axes['static_friction']:4.2f} "
                f"ratio={axes['dynamic_friction_ratio']:4.2f} b={axes['damping']:5.2f}  "
                f"novel {','.join(row['novel_axes'])}  moved={row['probe_moved']}  "
                f"closest {row['closest_position_error_mm']:6.1f} mm at {row['closest_force']:.2f} N  "
                f"needs {row['required_force'] if row['required_force'] else 'nothing swept'}"
            )

    safety = summary["safety"]
    print("[ood]")
    print(f"[ood] safety     : {safety['total_aborts']} aborts across "
          f"{safety['states_with_any_abort']} state(s); median invalid fraction "
          f"{safety['median_invalid_fraction'] * 100:.1f} % of swept forces")

    output = (
        Path(args.output)
        if args.output
        else project_root() / "outputs" / "logs" / "ood_feasibility_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"source": str(path), "git_commit": git_commit(), **summary}, indent=2)
    )
    print(f"[ood] written    : {output}")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()

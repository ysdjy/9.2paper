"""Turn D047's environment-slot sensitivity into an error bar.

D047 established that a deployment's absolute numbers depend on which of the parallel
environment slots each test drawer occupies, and that no warm-up removes it: the same hidden
state probed in a different slot measures about 0.17 mm differently, which a continuous force
predictor on a 0.05 N grid can turn into a different choice.

That is a caveat only while it is unquantified. This reads several deployment runs that differ
*only* in the slot permutation and reports what the assignment is worth: per method, per
pairwise gap, and whether the paper's ordering survives every one of them.

The gaps are differenced **within** each run before being aggregated. Differencing the
aggregates instead would hide the case that matters -- a gap staying put while both of its
terms move together.

Usage::

    for k in 0 1 2 3 4; do
      python scripts/evaluate_closed_loop.py --headless --run outputs/training/v1 \\
          --dataset outputs/dataset_v1 --seeds 0 1 2 --num-xi 0 --slot-permutation $k \\
          --output outputs/logs/slot_perm$k.json
    done
    python scripts/report_slot_robustness.py outputs/logs/slot_perm*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_drawer.analysis.closed_loop_determinism import EXPECTED_ORDERING, summarise_permutations
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root


def main() -> None:
    enable_unbuffered_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Deployment reports, one per slot permutation.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    summary = summarise_permutations(reports)

    print("\n" + "=" * 86)
    print(f"[slots] permutations : {summary['permutations']}")
    print(f"[slots] test states  : {reports[0]['num_test_states']}, identical in every run")
    print(f"[slots] seeds        : {reports[0].get('seeds')}")

    probe = summary["probe_displacement"]
    print(
        f"[slots] probe displacement, same drawer across permutations ({probe['pairs']} pairs): "
        f"median {probe['median_mm']:.4f} mm  p90 {probe['p90_mm']:.4f}  max {probe['max_mm']:.4f}"
    )

    print("[slots]")
    print(f"[slots] {'method':>22} {'reach mean':>11} {'sd':>6} {'min-max':>13} {'|d-goal| med':>14}")
    order = sorted(summary["methods"], key=lambda name: -summary["methods"][name]["reach_success_pp"]["mean"])
    for name in order:
        reach = summary["methods"][name]["reach_success_pp"]
        error = summary["methods"][name]["median_position_error_mm"]
        print(
            f"[slots] {name:>22} {reach['mean']:10.2f}% {reach['sd']:5.2f} "
            f"{reach['min']:5.1f}-{reach['max']:5.1f}% "
            f"{error['mean']:8.2f} +-{error['sd']:.2f} mm"
        )

    print("[slots]")
    print(f"[slots] {'pairwise gap':>22} {'mean':>9} {'sd':>6} {'min-max':>15}")
    for label, gap in summary["gaps"].items():
        print(
            f"[slots] {label:>22} {gap['mean']:+8.2f} {gap['sd']:5.2f} "
            f"{gap['min']:+6.1f} to {gap['max']:+5.1f}"
        )

    ordering = summary["ordering"]
    held = ordering["held_everywhere"]
    print("[slots]")
    print(f"[slots] claimed ordering: {' > '.join(ordering['claim'])}")
    print(
        f"[slots] held in {sum(ordering['held_per_permutation'])}/{len(ordering['held_per_permutation'])}"
        f" permutations -> {'HOLDS' if held else 'BROKEN'}"
    )

    worst = summary["gaps"].get("ACE + PSP - D GRU")
    if worst:
        print(
            f"[slots] worst permutation for ACE vs Direct GRU: {worst['min']:+.1f} pp "
            f"({'still ahead' if worst['min'] > 0 else 'BEHIND'})"
        )

    payload = {
        "expected_ordering": list(EXPECTED_ORDERING),
        "sources": [str(path) for path in args.reports],
        "git_commit": git_commit(),
        **summary,
    }
    output = (
        Path(args.output)
        if args.output
        else project_root() / "outputs" / "logs" / "slot_robustness.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"[slots] written : {output}")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()

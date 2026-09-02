"""Phase 9J/9K -- turn sweeps into the Oracle success landscape and a task recommendation.

Pure analysis: reads the datasets ``scripts/sweep_execution_space.py`` produced, applies the
success definition, and reports the per-hidden-state force bands. Needs no Isaac Sim.

Several datasets can be given at once, which is how the execution profile's ramp-down
fraction is chosen: it changes the physics of stopping, so it cannot be picked by taste. All
candidates from all datasets are scored together and the recommendation is global.

The recommendation is not a preference. Every candidate
``(fall_fraction, T, d_goal, eps_d, eps_v)`` is scored against the acceptance conditions in
:class:`~probe_drawer.analysis.oracle.AcceptanceThresholds`, and the accepted candidate with
the greatest spread of required force wins -- because that spread is what makes a probe
necessary at all. If nothing is accepted the script says so and reports which condition did
the eliminating, rather than returning a best-of-a-bad-bunch.

Usage::

    python scripts/build_oracle_landscape.py --dataset outputs/logs/sweep_fine_fall*.json
    python scripts/build_oracle_landscape.py --dataset outputs/logs/sweep_execution_fine.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_drawer.analysis.oracle import ACCEPTANCE, TaskCandidate, score_candidate
from probe_drawer.analysis.sweep import SweepDataset
from probe_drawer.utils import project_root

#: Goal distances to try (m). The upper end is set by what the sweep showed a drawer can
#: reach *and come to rest at*; asking for more is not a tuning question but a physical
#: impossibility for a low-resistance drawer.
GOAL_CANDIDATES = (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10)

#: Position tolerances to try (m).
DISPLACEMENT_TOLERANCES = (0.005, 0.0075, 0.01, 0.015)

#: Terminal-speed tolerances to try (m/s).
VELOCITY_TOLERANCES = (0.03, 0.05, 0.08)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help="Sweep dataset(s) to analyse. Defaults to outputs/logs/sweep_fine_fall*.json.",
    )
    parser.add_argument("--output", type=str, default=None, help="Where to write the landscape report.")
    parser.add_argument("--top", type=int, default=12, help="How many ranked candidates to print.")
    args = parser.parse_args()

    paths = [Path(path) for path in args.dataset] if args.dataset else sorted(
        (project_root() / "outputs" / "logs").glob("sweep_fine_fall*.json")
    )
    if not paths:
        parser.error("No sweep datasets found. Run scripts/sweep_execution_space.py first.")

    scored: list[dict] = []
    summaries: list[dict] = []
    best_by_dataset: dict[str, dict] = {}

    for path in paths:
        dataset = SweepDataset.load(path)
        fall_fraction = dataset.metadata.get("fall_fraction")
        label = f"fall={fall_fraction if fall_fraction is not None else 0.1:g}"
        summaries.append(
            {
                "path": str(path),
                "label": label,
                "fall_fraction": fall_fraction,
                "rows": len(dataset),
                "valid_rows": len(dataset.valid_records),
                "validity_rate": dataset.validity_rate(),
                "hidden_states": len(dataset.xi_keys()),
                "forces": dataset.forces(),
                "durations": dataset.durations(),
                "invalid_reasons": dataset.invalid_reason_counts(),
            }
        )

        rows: list[dict] = []
        for duration in dataset.durations():
            for goal in GOAL_CANDIDATES:
                for tolerance in DISPLACEMENT_TOLERANCES:
                    for velocity in VELOCITY_TOLERANCES:
                        score = score_candidate(dataset, TaskCandidate(duration, goal, tolerance, velocity))
                        payload = score.as_dict()
                        payload["label"] = label
                        payload["fall_fraction"] = fall_fraction
                        payload["dataset"] = str(path)
                        rows.append(payload)
                        if score.accepted:
                            payload["intervals"] = score.intervals
        scored.extend(rows)
        accepted_here = [row for row in rows if row["accepted"]]
        if accepted_here:
            best_by_dataset[label] = max(accepted_here, key=lambda row: row["discrimination"])

    accepted = [row for row in scored if row["accepted"]]
    recommended = max(accepted, key=lambda row: row["discrimination"]) if accepted else None

    report = {
        "thresholds": _thresholds_dict(),
        "datasets": summaries,
        "num_candidates": len(scored),
        "num_accepted": len(accepted),
        "scores": [{key: value for key, value in row.items() if key != "intervals"} for row in scored],
        "best_per_dataset": {label: {k: v for k, v in row.items() if k != "intervals"} for label, row in best_by_dataset.items()},
        "recommended": {k: v for k, v in recommended.items() if k != "intervals"} if recommended else None,
        "recommended_intervals": recommended.get("intervals") if recommended else None,
    }

    _print_report(report, recommended, args.top)

    output = Path(args.output) if args.output else project_root() / "outputs" / "logs" / "oracle_landscape.json"
    output.write_text(json.dumps(report, indent=2, default=float))
    print(f"[oracle] report written : {output}")
    print("=" * 78 + "\n")


def _thresholds_dict() -> dict:
    return {
        "min_coverage": ACCEPTANCE.min_coverage,
        "min_discrimination": ACCEPTANCE.min_discrimination,
        "min_relative_width": ACCEPTANCE.min_relative_width,
        "max_relative_width": ACCEPTANCE.max_relative_width,
        "max_travel_fraction": ACCEPTANCE.max_travel_fraction,
        "min_contiguous_fraction": ACCEPTANCE.min_contiguous_fraction,
        "max_tolerance_ratio": ACCEPTANCE.max_tolerance_ratio,
        "require_grid_resolved": ACCEPTANCE.require_grid_resolved,
    }


def _print_report(report: dict, recommended: dict | None, top: int) -> None:
    print("\n" + "=" * 78)
    for summary in report["datasets"]:
        print(
            f"[oracle] {summary['label']:<11}: {summary['rows']} rows, "
            f"{summary['valid_rows']} valid ({summary['validity_rate'] * 100:.1f} %), "
            f"{summary['hidden_states']} hidden states, T={summary['durations']}"
        )
    print(f"[oracle] acceptance     : {report['thresholds']}")
    print(f"[oracle] candidates     : {report['num_candidates']} scored, {report['num_accepted']} accepted")
    print("[oracle]")

    ranked = sorted(report["scores"], key=lambda row: (-row["accepted"], -row["discrimination"]))
    print(
        f"[oracle] {'profile':>11} {'T':>4} {'d_goal':>7} {'eps_d':>7} {'eps_v':>6} {'cover':>6} "
        f"{'discr':>6} {'width':>6} {'dF(N)':>6} {'contig':>6} {'ok':>3}"
    )
    for row in ranked[:top]:
        candidate = row["candidate"]
        print(
            f"[oracle] {row['label']:>11} {candidate['duration']:4.1f} {candidate['goal_displacement']:7.3f} "
            f"{candidate['displacement_tolerance']:7.4f} {candidate['velocity_tolerance']:6.3f} "
            f"{row['coverage']:6.2f} {row['discrimination']:6.2f} {row['median_relative_width']:6.2f} "
            f"{row['median_absolute_width']:6.2f} {row['contiguous_fraction']:6.2f} "
            f"{'yes' if row['accepted'] else 'no':>3}"
        )

    if report["best_per_dataset"]:
        print("[oracle]")
        print("[oracle] best accepted candidate per profile:")
        for label, row in report["best_per_dataset"].items():
            candidate = row["candidate"]
            print(
                f"[oracle]   {label:<11} T={candidate['duration']:.1f} d_goal={candidate['goal_displacement']:.3f} "
                f"eps_d={candidate['displacement_tolerance']:.4f} eps_v={candidate['velocity_tolerance']:.3f} "
                f"coverage={row['coverage']:.2f} discrimination={row['discrimination']:.2f}"
            )

    if recommended is None:
        print("[oracle]")
        print("[oracle] NO CANDIDATE ACCEPTED -- the experiment must change, not the bar.")
        counts: dict[str, int] = {}
        for row in report["scores"]:
            for failure in row["failures"]:
                counts[failure.split()[0]] = counts.get(failure.split()[0], 0) + 1
        print(f"[oracle] eliminated by  : {dict(sorted(counts.items(), key=lambda item: -item[1]))}")
        return

    print("[oracle]")
    print(f"[oracle] RECOMMENDED    : {recommended['label']}, {json.dumps(recommended['candidate'])}")
    print(
        f"[oracle]   coverage {recommended['coverage']:.2f}  discrimination {recommended['discrimination']:.2f}  "
        f"band {recommended['median_absolute_width']:.2f} N ({recommended['median_relative_width']:.2f} relative)  "
        f"contiguous {recommended['contiguous_fraction']:.2f}  "
        f"max travel {recommended['max_travel_fraction']:.2f}"
    )
    _print_representative_intervals(report["recommended_intervals"])


def _print_representative_intervals(intervals: list[dict] | None, count: int = 10) -> None:
    """Show hidden states across the whole range of required force."""
    if not intervals:
        return
    achieved = sorted((row for row in intervals if row["any_success"]), key=lambda row: row["force_centre"])
    if not achieved:
        return
    step = max(1, len(achieved) // count)
    print("[oracle]")
    print(f"[oracle] success force bands, {len(achieved)}/{len(intervals)} hidden states:")
    print(
        f"[oracle] {'m':>6} {'mu_s':>6} {'mu_d':>6} {'b':>6} | {'F_low':>6} {'F_high':>7} "
        f"{'F_best':>7} {'d_best':>8} {'v_best':>8}"
    )
    for row in achieved[::step][:count]:
        mass, static, dynamic, damping = row["xi"]
        print(
            f"[oracle] {mass:6.1f} {static:6.2f} {dynamic:6.2f} {damping:6.1f} | "
            f"{row['force_low']:6.2f} {row['force_high']:7.2f} {row['best_force']:7.2f} "
            f"{row['best_displacement'] * 1000:8.1f} {row['best_velocity']:8.4f}"
        )
    centres = [row["force_centre"] for row in achieved]
    print(
        f"[oracle] required force spans {min(centres):.2f} .. {max(centres):.2f} N "
        f"(median {float(np.median(centres)):.2f} N), i.e. a {max(centres) / min(centres):.1f}x range"
    )
    missing = [row for row in intervals if not row["any_success"]]
    if missing:
        print(f"[oracle] no succeeding force for {len(missing)} hidden state(s), e.g. xi={missing[0]['xi']}")


if __name__ == "__main__":
    main()

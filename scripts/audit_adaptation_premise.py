"""Is the adaptation problem well posed, and how hard is it? -- run before training anything.

Reads a finished sequential Oracle sweep and answers four questions that can each invalidate
the study on their own: whether a constant force would do, whether the answer is a point or a
set, whether the probe determines the answer, and what precision a predictor needs. No Isaac
Sim.

Usage::

    python scripts/audit_adaptation_premise.py
    python scripts/audit_adaptation_premise.py --sweep outputs/logs/sequential_oracle_fall030.json

Reasoning and the interpretation of every number:
``docs/RMA2_TO_DRAWER_MAPPING.md`` §19.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_drawer.analysis.adaptation_premise import audit
from probe_drawer.analysis.sweep import SweepDataset
from probe_drawer.experiment_plan import MAIN_TASK
from probe_drawer.utils import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sweep",
        type=str,
        default="outputs/logs/sequential_oracle_fall035.json",
        help="Sweep to audit, relative to the project root. Must be a sequential Oracle.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/logs/adaptation_premise.json",
        help="Where to write the report, relative to the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset = SweepDataset.load(root / args.sweep)

    report = audit(
        dataset=dataset,
        criteria=MAIN_TASK.criteria,
        duration=MAIN_TASK.duration,
        source=args.sweep,
    )
    structure, ambiguity, ident = report.structure, report.ambiguity, report.identifiability

    print(f"\n=== Adaptation premise: {args.sweep} ===")
    print(f"{len(dataset)} rows, {dataset.validity_rate():.3f} valid, "
          f"task d_goal={MAIN_TASK.goal_displacement * 1e3:.0f} mm "
          f"eps_d={MAIN_TASK.displacement_tolerance * 1e3:.1f} mm "
          f"eps_v={MAIN_TASK.velocity_tolerance:.2f} m/s T={MAIN_TASK.duration:.1f} s")

    print("\n-- 1. is adaptation necessary?")
    force, best = structure["required_force"], structure["best_constant_force"]
    print(f"  coverage                {structure['coverage']:.3f} "
          f"({structure['solvable']}/{structure['total_hidden_states']} hidden states solvable)")
    print(f"  required force          {force['min']:.2f} - {force['max']:.2f} N, "
          f"median {force['median']:.2f} N, {force['ratio']:.1f}x range")
    print(f"  best constant force     {best['force']:.2f} N succeeds on "
          f"{best['successes']}/{structure['total_hidden_states']} = {best['success_rate']:.3f}")
    print(f"  runners-up              {best['runners_up']}")

    print("\n-- 2. is the answer a point or a set?")
    width = structure["band_width"]
    print(f"  band width              median {width['median']:.2f} N "
          f"(min {width['min']:.2f}, max {width['max']:.2f})")
    print(f"  succeeding forces/state median {structure['succeeding_forces_per_state']['median']:.0f} "
          f"(min {structure['succeeding_forces_per_state']['min']}, "
          f"max {structure['succeeding_forces_per_state']['max']})")
    print(f"  non-contiguous bands    {structure['non_contiguous_bands']}/{structure['solvable']}, "
          f"largest interior gap {structure['largest_interior_gap']}")
    print(f"  band midpoint succeeds  {structure['midpoint_succeeds']}/{structure['solvable']}")

    print("\n-- 3. does the probe determine the answer?")
    nearest = ambiguity["nearest_neighbour"]
    print(f"  nearest probe neighbour's force misses this band {nearest['neighbour_misses_band']:.3f} "
          f"of the time (median gap {nearest['median_force_gap']:.2f} N, max {nearest['max_force_gap']:.2f} N)")
    print(f"  {'radius':>8} {'cluster':>9} {'spread':>9} {'mean misses band':>18}")
    for radius, row in ambiguity["radii"].items():
        print(f"  {radius:>8.2f} {row['mean_cluster_size']:>9.1f} "
              f"{row['median_force_spread']:>8.2f}N {row['cluster_mean_misses_band']:>17.3f}")

    print("\n-- 4. what precision does the task demand?")
    precision = ident["precision_required"]
    print(f"  band half-width {precision['median_half_width']:.3f} N on a "
          f"{precision['median_target']:.2f} N median target "
          f"({100 * precision['median_half_width'] / precision['median_target']:.0f} % relative)")
    print("\n  leave-one-out linear readout of xi from the probe features:")
    for name, scores in ident["xi_from_probe"].items():
        verdict = "identified" if scores["r2"] > 0.5 else "NOT identified"
        print(f"    {name:<24} R^2 {scores['r2']:>+7.3f}  RMSE {scores['rmse']:>7.3f}   {verdict}")
    print("\n  leave-one-out readouts of the band centre:")
    print(f"    {'source':<8} {'readout':<10} {'R^2':>8} {'RMSE (N)':>10} {'in band':>10}")
    for source, key in (("probe", "force_from_probe"), ("true xi", "force_from_xi")):
        for readout, scores in ident[key].items():
            print(f"    {source:<8} {readout:<10} {scores['r2']:>+8.3f} {scores['rmse']:>10.3f} "
                  f"{scores['in_band_rate']:>10.3f}")

    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.as_dict(), indent=2))
    print(f"\nreport -> {destination}")


if __name__ == "__main__":
    main()

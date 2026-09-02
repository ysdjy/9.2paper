"""Phase 9L -- choose the standardised probe from data, once the task is fixed.

The probe can only be calibrated after the execution operating point is known, because its
job is defined relative to that: apply a known increasing force, stop early, and produce a
history that says *which* peak force this drawer will need.

Each candidate probe configuration is run over the whole hidden-state grid and scored on
three things:

**Coverage** -- the fraction of hidden states whose drawer actually broke away. A probe that
tells you nothing about the stiffest drawers cannot identify them.

**Intrusion** -- the largest probe displacement as a fraction of the goal. A probe that
travels a large part of the way has performed the task rather than measured it.

**Predictive power** -- the rank correlation between each probe feature and the peak force
that hidden state actually needs, taken from the Oracle landscape. This is the number that
decides whether one probe can carry the information the adaptation model needs; it is
reported per feature so it is clear *what* the probe is measuring.

Features are computed from deployable channels only, and that is asserted rather than
assumed.

Usage::

    python scripts/calibrate_probe.py --headless
    python scripts/calibrate_probe.py --headless --duration 1.5 --goal 0.05
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--landscape",
    type=str,
    default=None,
    help="Oracle landscape report providing the per-xi required force. Defaults to outputs/logs/.",
)
parser.add_argument("--duration", type=float, default=None, help="Execution duration the probe serves (s).")
parser.add_argument("--goal", type=float, default=None, help="Goal displacement the probe serves (m).")
parser.add_argument("--max-envs", type=int, default=18, help="Hidden states per batch.")
parser.add_argument("--output", type=str, default=None, help="Where to write the calibration report.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.analysis.probe_features import (  # noqa: E402
    PROBE_FEATURES,
    assert_features_are_deployable,
    extract_features,
    rank_correlation,
)
from probe_drawer.analysis.sweep import xi_grid  # noqa: E402
from probe_drawer.controllers import ProbeControllerCfg  # noqa: E402
from probe_drawer.envs import DynamicsRandomizer  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, project_root  # noqa: E402

#: The hidden-state grid the Oracle sweep used, so features and required forces line up.
GRID = dict(
    masses=(4.0, 8.0, 12.0),
    static_frictions=(0.5, 1.25, 2.0, 3.0),
    friction_ratios=(0.3, 0.65, 1.0),
    dampings=(2.0, 6.0, 10.0),
)

#: Probe configurations to compare. ``initial_force`` sits below the weakest breakaway seen
#: in the sweep and ``max_force`` above the strongest, so the ramp brackets the whole grid;
#: the candidates vary how fast it gets there and how early it stops.
CANDIDATES = {
    "fast_ramp_5mm": dict(task=dict(initial_force=1.0, max_force=6.0, target_displacement=0.005, max_velocity=0.08),
                          cfg=ProbeControllerCfg(ramp_duration=0.5, max_probe_duration=1.0)),
    "medium_ramp_5mm": dict(task=dict(initial_force=1.0, max_force=6.0, target_displacement=0.005, max_velocity=0.08),
                            cfg=ProbeControllerCfg(ramp_duration=1.0, max_probe_duration=1.5)),
    "medium_ramp_3mm": dict(task=dict(initial_force=1.0, max_force=6.0, target_displacement=0.003, max_velocity=0.08),
                            cfg=ProbeControllerCfg(ramp_duration=1.0, max_probe_duration=1.5)),
    "medium_ramp_8mm": dict(task=dict(initial_force=1.0, max_force=6.0, target_displacement=0.008, max_velocity=0.08),
                            cfg=ProbeControllerCfg(ramp_duration=1.0, max_probe_duration=1.5)),
    "slow_ramp_5mm": dict(task=dict(initial_force=1.0, max_force=6.0, target_displacement=0.005, max_velocity=0.08),
                          cfg=ProbeControllerCfg(ramp_duration=1.5, max_probe_duration=2.0)),
    "low_velocity_cap": dict(task=dict(initial_force=1.0, max_force=6.0, target_displacement=0.005, max_velocity=0.03),
                             cfg=ProbeControllerCfg(ramp_duration=1.0, max_probe_duration=1.5)),
    "high_max_force": dict(task=dict(initial_force=1.0, max_force=8.0, target_displacement=0.005, max_velocity=0.08),
                           cfg=ProbeControllerCfg(ramp_duration=1.0, max_probe_duration=1.5)),
}


def required_forces(landscape_path: Path, duration: float | None, goal: float | None) -> tuple[dict, dict]:
    """Per-hidden-state required peak force from the recommended Oracle candidate."""
    report = json.loads(landscape_path.read_text())
    if report.get("recommended") is None:
        raise RuntimeError(f"{landscape_path} has no accepted candidate; rerun build_oracle_landscape.py.")
    candidate = report["recommended"]["candidate"]
    if duration is not None and duration != candidate["duration"]:
        raise RuntimeError(
            f"--duration {duration} does not match the recommended {candidate['duration']}; "
            "the probe is calibrated for one operating point."
        )
    if goal is not None and goal != candidate["goal_displacement"]:
        raise RuntimeError(f"--goal {goal} does not match the recommended {candidate['goal_displacement']}.")
    forces = {
        tuple(round(value, 6) for value in row["xi"]): row["best_force"]
        for row in report["recommended_intervals"]
        if row["any_success"]
    }
    return candidate, forces


def batches(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def main() -> None:
    enable_unbuffered_stdout()
    assert_features_are_deployable()

    landscape_path = (
        Path(args_cli.landscape) if args_cli.landscape else project_root() / "outputs" / "logs" / "oracle_landscape.json"
    )
    task, forces_by_xi = required_forces(landscape_path, args_cli.duration, args_cli.goal)

    hidden_states = xi_grid(**GRID)
    grouped = batches(hidden_states, args_cli.max_envs)
    batch_size = len(grouped[0])

    print("\n" + "=" * 78)
    print(f"[probe-cal] serving task  : {json.dumps(task)}")
    print(f"[probe-cal] hidden states : {len(hidden_states)}, {len(forces_by_xi)} with a known required force")
    print(f"[probe-cal] candidates    : {list(CANDIDATES)}")

    system = PullSystem.build(PullSystemCfg(num_envs=batch_size, device=args_cli.device))
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    results: dict[str, dict] = {}
    try:
        for name, spec in CANDIDATES.items():
            system.probe.cfg = spec["cfg"]
            rows: list[dict] = []
            for batch in grouped:
                padded = batch + [batch[-1]] * (batch_size - len(batch))
                system.reset()
                randomizer.apply(system.env, padded)
                result = system.probe.run(**spec["task"])
                for index, params in enumerate(batch):
                    features = extract_features(result, index)
                    key = tuple(round(value, 6) for value in params.as_vector())
                    rows.append(
                        {
                            "xi": list(key),
                            "required_force": forces_by_xi.get(key),
                            **features.as_dict(),
                        }
                    )
            results[name] = _score(name, spec, rows, task)
    finally:
        system.close()

    report = {
        "task": task,
        "landscape": str(landscape_path),
        "grid": {key: list(value) for key, value in GRID.items()},
        "candidates": results,
    }
    recommended = _recommend(results)
    report["recommended"] = recommended

    _print_results(results, recommended)
    output = Path(args_cli.output) if args_cli.output else project_root() / "outputs" / "logs" / "probe_calibration.json"
    output.write_text(json.dumps(report, indent=2, default=float))
    print(f"[probe-cal] report written: {output}")
    print("=" * 78 + "\n")


def _score(name: str, spec: dict, rows: list[dict], task: dict) -> dict:
    """Coverage, intrusion and per-feature predictive power for one probe configuration."""
    moved = [row for row in rows if row["moved"]]
    paired = [row for row in moved if row["required_force"] is not None]
    correlations = {
        feature: rank_correlation([row[feature] for row in paired], [row["required_force"] for row in paired])
        for feature in PROBE_FEATURES
    }
    finite = {key: value for key, value in correlations.items() if np.isfinite(value)}
    best_feature = max(finite, key=lambda key: abs(finite[key])) if finite else None
    displacements = [row["final_displacement"] for row in rows]

    return {
        "name": name,
        "task_parameters": spec["task"],
        "config": spec["cfg"].as_dict(),
        "coverage": len(moved) / len(rows) if rows else 0.0,
        "num_paired": len(paired),
        "max_displacement": max(displacements, default=0.0),
        "intrusion": max(displacements, default=0.0) / task["goal_displacement"],
        "median_duration": float(np.median([row["duration"] for row in rows])) if rows else 0.0,
        "termination_mix": _counts(row["termination_reason"] for row in rows),
        "correlations": correlations,
        "best_feature": best_feature,
        "best_correlation": abs(finite[best_feature]) if best_feature else float("nan"),
        "rows": rows,
    }


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _recommend(
    results: dict[str, dict],
    max_intrusion: float = 0.25,
    min_coverage: float = 0.95,
    correlation_margin: float = 0.02,
) -> dict | None:
    """The least intrusive configuration that is essentially as predictive as the best.

    ``min_coverage`` requires nearly every drawer to move at all, and ``max_intrusion``
    caps the probe's own travel as a fraction of the goal. Among the candidates that pass,
    predictive power is the primary criterion -- but the top few differ by under a
    percentage point of rank correlation, which is well inside the run-to-run spread of a
    contact-rich simulation. Treating those as tied and breaking the tie on *intrusion*
    picks the probe that disturbs the task least, which is what a sequential
    probe-then-execute protocol will need.
    """
    eligible = [
        row
        for row in results.values()
        if row["coverage"] >= min_coverage
        and row["intrusion"] <= max_intrusion
        and np.isfinite(row["best_correlation"])
    ]
    if not eligible:
        return None
    ceiling = max(row["best_correlation"] for row in eligible)
    tied = [row for row in eligible if row["best_correlation"] >= ceiling - correlation_margin]
    best = min(tied, key=lambda row: row["intrusion"])
    return {key: value for key, value in best.items() if key != "rows"} | {
        "acceptance": {
            "max_intrusion": max_intrusion,
            "min_coverage": min_coverage,
            "correlation_margin": correlation_margin,
            "correlation_ceiling": ceiling,
            "num_tied": len(tied),
        }
    }


def _print_results(results: dict[str, dict], recommended: dict | None) -> None:
    print("[probe-cal]")
    print(
        f"[probe-cal] {'candidate':>18} {'cover':>6} {'intrus':>7} {'dur':>6} "
        f"{'best feature':>28} {'|rho|':>6}"
    )
    for row in results.values():
        print(
            f"[probe-cal] {row['name']:>18} {row['coverage']:6.2f} {row['intrusion']:7.3f} "
            f"{row['median_duration']:6.3f} {str(row['best_feature']):>28} {row['best_correlation']:6.3f}"
        )

    print("[probe-cal]")
    print("[probe-cal] per-feature rank correlation with the required peak force:")
    names = list(results)
    print(f"[probe-cal] {'feature':>28} " + " ".join(f"{name[:14]:>15}" for name in names))
    for feature in PROBE_FEATURES:
        cells = " ".join(f"{results[name]['correlations'][feature]:15.3f}" for name in names)
        print(f"[probe-cal] {feature:>28} {cells}")

    if recommended is None:
        print("[probe-cal]")
        print("[probe-cal] NO CANDIDATE MET THE BAR -- widen the ramp or revisit the task.")
        return
    print("[probe-cal]")
    print(f"[probe-cal] RECOMMENDED   : {recommended['name']}")
    print(f"[probe-cal]   task        : {json.dumps(recommended['task_parameters'])}")
    print(f"[probe-cal]   config      : {json.dumps(recommended['config'])}")
    print(
        f"[probe-cal]   coverage {recommended['coverage']:.2f}, intrusion {recommended['intrusion']:.3f} "
        f"of the goal, median duration {recommended['median_duration']:.3f} s"
    )
    print(
        f"[probe-cal]   selected as the least intrusive of {recommended['acceptance']['num_tied']} candidate(s) "
        f"within {recommended['acceptance']['correlation_margin']:.2f} of the best |rho| = "
        f"{recommended['acceptance']['correlation_ceiling']:.3f}"
    )
    print(f"[probe-cal]   terminations: {recommended['termination_mix']}")


if __name__ == "__main__":
    main()
    simulation_app.close()

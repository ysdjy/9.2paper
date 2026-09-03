"""Is the standardised probe allowed to end after 0.1 s?

The probe stops when the drawer has moved 3 mm, and on a slippery drawer that can happen
almost immediately: Dataset v0's histories run from **6 steps (0.10 s) to 56 steps (0.93 s)**.
A 6-step recording is six samples of force and position, which is not obviously enough to
identify anything -- and the probe's whole purpose is identification.

This asks whether that matters, from data already on disk. No simulator, and **no change to
the probe**: the point is to decide whether a ``min_probe_duration`` is warranted before
touching a component that four phases of results depend on.

Three questions:

1. How are probe durations actually distributed, and how many are short?
2. Do short probes carry less information about the hidden state? Measured by how well each
   hidden dimension can be read out of the probe's summary features, short probes against
   long ones.
3. Do short probes carry less information about the *answer* -- the force the drawer needs?
   That is the one that decides, because identifying ``xi`` is not the goal.

Usage::

    python scripts/analyze_probe_duration.py --dataset outputs/dataset_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_drawer.analysis.probe_features import PROBE_FEATURES, rank_correlation
from probe_drawer.analysis.readout import RIDGE_PENALTY, leave_one_out
from probe_drawer.dataset import DatasetStore
from probe_drawer.dataset.schema import XI_DIMENSIONS
from probe_drawer.experiment_plan import MAIN_TASK
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root

#: Candidate floors to evaluate, in seconds. Each is scored by what it would cost and buy.
CANDIDATE_FLOORS = (0.20, 0.35, 0.50)

#: The readout now comes from ``probe_drawer.analysis.readout`` (Phase 13), which applies a
#: ridge penalty; this script's own copy did not. The Phase 12 conclusion is unaffected --
#: it rested on RMSE being flat while R-squared moved with the target's sd, and the penalty
#: changes neither -- but the numbers a re-run prints will differ slightly from
#: ``outputs/logs/probe_duration_analysis.json``, which is left as it was recorded.
#:
#: Below this a probe is called "short" for the split comparisons.
SHORT_PROBE = 0.20




def main() -> None:
    enable_unbuffered_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="outputs/dataset_v0")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_absolute():
        root = project_root() / root
    store = DatasetStore(root)
    probes = store.probes
    step_dt = 1.0 / 60.0

    durations = np.array([probe["num_steps"] * step_dt for probe in probes])
    summaries = {probe["probe_id"]: probe["summary"] for probe in probes}
    state_of = {probe["probe_id"]: probe["xi_id"] for probe in probes}
    xi_by_id = {row["xi_id"]: row["xi"] for row in store.hidden_states}

    # The answer each probe should support: the force that came closest to the goal, from the
    # candidates that probe was actually asked about. Defined per probe, so a short probe and
    # a long probe of the same drawer are scored against their own episodes.
    best_force: dict[str, float] = {}
    best_error: dict[str, float] = {}
    for row in store.candidates:
        error = abs(row["final_total_displacement"] - row["goal_displacement"])
        if row["probe_id"] not in best_error or error < best_error[row["probe_id"]]:
            best_error[row["probe_id"]] = error
            best_force[row["probe_id"]] = row["candidate_peak_force"]

    identifiers = [probe["probe_id"] for probe in probes]
    features = np.array([[summaries[key][name] for name in PROBE_FEATURES] for key in identifiers])
    targets = {
        name: np.array([xi_by_id[state_of[key]][name] for key in identifiers]) for name in XI_DIMENSIONS
    }
    targets["required_force"] = np.array([best_force[key] for key in identifiers])

    short = durations < SHORT_PROBE
    report = {
        "dataset": str(root),
        "git_commit": git_commit(),
        "probes": len(probes),
        "step_dt": step_dt,
        "duration": {
            "min": float(durations.min()),
            "max": float(durations.max()),
            "median": float(np.median(durations)),
            "mean": float(durations.mean()),
            "percentiles": {
                str(percentile): float(np.percentile(durations, percentile))
                for percentile in (1, 5, 10, 25, 50, 75, 90, 99)
            },
            "histogram_steps": {
                str(int(step)): int(count)
                for step, count in zip(*np.unique([p["num_steps"] for p in probes], return_counts=True), strict=True)
            },
        },
        "short_probe_threshold": SHORT_PROBE,
        "short_probes": int(short.sum()),
        "short_probe_fraction": float(short.mean()),
        "readouts": {},
        "candidate_floors": {},
    }

    # 2 and 3: what a probe supports, split by length.
    for name, target in targets.items():
        report["readouts"][name] = {
            "all": leave_one_out(features, target),
            "short_only": leave_one_out(features[short], target[short]) if short.sum() > 15 else None,
            "long_only": leave_one_out(features[~short], target[~short]),
            "spearman_duration_vs_target": rank_correlation(durations.tolist(), target.tolist()),
        }

    # The sharper form of the question: does identifiability degrade for the *shortest*
    # probes? A duration split at 0.2 s cannot answer it -- only five probes fall below --
    # so instead the probes are ranked by length and the readout is refitted on each tercile.
    # If the short tercile reads the required force as well as the long one, a duration floor
    # has nothing to buy.
    order = np.argsort(durations)
    terciles = np.array_split(order, 3)
    report["by_duration_tercile"] = [
        {
            "tercile": index,
            "duration_range": [float(durations[part].min()), float(durations[part].max())],
            "probes": len(part),
            "readouts": {
                name: leave_one_out(features[part], target[part])
                for name, target in targets.items()
            },
            # Recorded because it is the confound: R2 is scored against this, so a tercile
            # with a narrower target range shows a lower R2 at the same absolute error.
            "target_sd": {name: float(np.std(target[part])) for name, target in targets.items()},
            "target_range": {
                name: [float(target[part].min()), float(target[part].max())]
                for name, target in targets.items()
            },
        }
        for index, part in enumerate(terciles)
    ]

    # What each candidate floor would cost and buy.
    for floor in CANDIDATE_FLOORS:
        affected = durations < floor
        report["candidate_floors"][str(floor)] = {
            "probes_extended": int(affected.sum()),
            "fraction_extended": float(affected.mean()),
            "median_extension_s": float(np.median(floor - durations[affected])) if affected.any() else 0.0,
            "total_added_simulation_s": float((floor - durations[affected]).sum()) if affected.any() else 0.0,
            # What the drawers that would be extended have in common: if they are all one
            # corner of the xi box, a floor changes the sampling, not just the timing.
            "extended_median_xi": {
                name: float(np.median([xi_by_id[state_of[key]][name] for key, flag in zip(identifiers, affected, strict=True) if flag]))
                for name in XI_DIMENSIONS
            }
            if affected.any()
            else {},
            "unaffected_median_xi": {
                name: float(np.median([xi_by_id[state_of[key]][name] for key, flag in zip(identifiers, affected, strict=True) if not flag]))
                for name in XI_DIMENSIONS
            },
        }

    output = Path(args.output) if args.output else project_root() / "outputs" / "logs" / "probe_duration_analysis.json"
    output.write_text(json.dumps(report, indent=2, default=float))
    _print(report)
    print(f"[probe] report written: {output}")
    print("=" * 78 + "\n")


def _print(report: dict) -> None:
    duration = report["duration"]
    print("\n" + "=" * 78)
    print(f"[probe] {report['probes']} probe histories from {report['dataset']}")
    print(
        f"[probe] duration: {duration['min']:.3f} .. {duration['max']:.3f} s "
        f"(median {duration['median']:.3f}, mean {duration['mean']:.3f})"
    )
    print(
        "[probe] percentiles: "
        + ", ".join(f"p{key}={value:.3f}s" for key, value in duration["percentiles"].items())
    )
    print(
        f"[probe] shorter than {report['short_probe_threshold']:.2f} s: "
        f"{report['short_probes']} ({report['short_probe_fraction'] * 100:.1f} %)"
    )

    print("[probe]")
    print("[probe] what can be read out of the probe's summary features (leave-one-out linear):")
    print(f"[probe]   {'target':>16} {'all R2':>8} {'short R2':>9} {'long R2':>8} {'rho(dur,target)':>16}")
    for name, values in report["readouts"].items():
        short = values["short_only"]
        print(
            f"[probe]   {name:>16} {values['all']['r2']:8.3f} "
            f"{(short['r2'] if short else float('nan')):9.3f} {values['long_only']['r2']:8.3f} "
            f"{values['spearman_duration_vs_target']:+16.3f}"
        )

    print("[probe]")
    print("[probe] identifiability by duration tercile (does a short probe know less?):")
    print(
        f"[probe]   {'tercile':>8} {'duration (s)':>14} {'n':>5} "
        f"{'mu_s R2/RMSE':>14} {'mu_d R2/RMSE':>14} {'F* R2/RMSE':>14}"
    )
    for entry in report["by_duration_tercile"]:
        low, high = entry["duration_range"]
        cells = "".join(
            f"{entry['readouts'][name]['r2']:7.3f}/{entry['readouts'][name]['rmse']:6.3f}"
            for name in ("static_friction", "dynamic_friction", "required_force")
        )
        print(f"[probe]   {entry['tercile']:8d} {low:6.3f}-{high:6.3f} {entry['probes']:5d} {cells}")
    print(
        "[probe]   read the RMSE, not the R2: R2 = 1 - MSE/Var, and the terciles have different"
    )
    print(
        "[probe]   target variance (the long tercile's required force has sd 0.60 N against the"
    )
    print(
        "[probe]   short tercile's 0.42 N), so the same absolute error scores a higher R2 there."
    )

    print("[probe]")
    print("[probe] what a min_probe_duration would cost:")
    for floor, values in report["candidate_floors"].items():
        print(
            f"[probe]   {float(floor):.2f} s: extends {values['probes_extended']} probes "
            f"({values['fraction_extended'] * 100:.1f} %), median +{values['median_extension_s']:.3f} s, "
            f"{values['total_added_simulation_s']:.1f} s of extra simulation per dataset pass"
        )
        if values["extended_median_xi"]:
            print(
                "[probe]     the extended probes' median xi: "
                + ", ".join(f"{name}={value:.2f}" for name, value in values["extended_median_xi"].items())
            )
            print(
                "[probe]     everyone else's median xi:      "
                + ", ".join(f"{name}={value:.2f}" for name, value in values["unaffected_median_xi"].items())
            )


if __name__ == "__main__":
    main()

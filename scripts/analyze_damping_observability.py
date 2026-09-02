r"""Why is damping invisible: too narrow a range, too small a value, or below the noise?

Phase 10 established that the standardised probe cannot identify ``b``: sweeping it from 2 to
11 N*s/m leaves the probe's duration and breakaway force essentially unchanged. That is a
measurement, not an explanation, and three explanations are distinguishable:

**The range is too narrow.** 2-10 N*s/m is a 5x span. If the response were strong but the span
small, widening it would fix identifiability -- and would also mean the *sampled distribution*
is what is at fault, not the physics.

**The value is too small at the velocities involved.** The viscous force is ``b*v``, so at the
speeds the probe reaches it may simply be dwarfed by the Coulomb term ``mu_d``. Then the fix is
a faster probe, not a wider range or a longer probe.

**The signal is below the measurement floor.** Phase 8 measured a residual ~0.25 N bias on the
pull axis. A force difference smaller than that is unobservable however good the estimator.

The force budget (computed from data already on disk) separates them arithmetically, and the
simulation sweep below confirms it: damping is held as the *only* varying parameter over a
range far wider than the training distribution, at several probe speeds, and the coast fit's
recovered ``b/m`` is compared against the truth.

Usage::

    python scripts/analyze_damping_observability.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dampings",
    type=float,
    nargs="+",
    default=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0, 110.0, 150.0, 200.0),
    help="Damping values to sweep, far beyond the training range on purpose.",
)
parser.add_argument(
    "--trigger-fractions",
    type=float,
    nargs="+",
    default=(0.05, 0.10, 0.20),
    help="Probe alpha values. A larger trigger means a faster drawer when the force is released.",
)
parser.add_argument(
    "--max-velocities",
    type=float,
    nargs="+",
    default=(0.08, 0.30),
    help="Probe velocity ceilings. The default 0.08 is the current probe's; 0.30 lets it run.",
)
parser.add_argument("--mass", type=float, default=8.0)
parser.add_argument("--static-friction", type=float, default=1.75)
parser.add_argument("--dynamic-friction", type=float, default=1.00)
parser.add_argument("--dataset", type=str, default="outputs/dataset_v0")
parser.add_argument("--oracle", type=str, default="outputs/logs/landscape_2d_fine.json")
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

from probe_drawer.analysis.probe_features import rank_correlation  # noqa: E402
from probe_drawer.analysis.response_probe_features import extract_response_features  # noqa: E402
from probe_drawer.analysis.sweep import SweepDataset  # noqa: E402
from probe_drawer.controllers import ResponseProbeCfg, ResponseProbeController  # noqa: E402
from probe_drawer.dataset import DatasetStore  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.experiment_plan import MAIN_TASK, RECOMMENDED_PROBE_CFG  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)

#: Residual pull-axis force bias measured in Phase 8 (N). Any force difference smaller than
#: this is below the floor the wrist sensing can resolve, whatever the estimator.
FORCE_NOISE_FLOOR = 0.25


def force_budget() -> dict:
    r"""How large the viscous term is, next to the Coulomb term and the noise floor.

    Computed from the probe histories in Dataset v0 and the execution episodes in the 2-D
    Oracle, so it describes the velocities the system actually reaches rather than a
    hypothetical.
    """
    budget: dict = {"force_noise_floor": FORCE_NOISE_FLOOR}

    dataset_path = Path(args_cli.dataset)
    if not dataset_path.is_absolute():
        dataset_path = project_root() / dataset_path
    if dataset_path.exists():
        store = DatasetStore(dataset_path)
        xi = {row["xi_id"]: row["xi"] for row in store.hidden_states}
        peaks, dampings, frictions = [], [], []
        for probe in store.probes:
            history = store.probe_history(probe["probe_id"])
            state = xi[probe["xi_id"]]
            peaks.append(float(np.abs(history["drawer_velocity"]).max()))
            dampings.append(state["damping"])
            frictions.append(state["dynamic_friction"])
        budget["probe"] = _phase_budget(np.array(peaks), np.array(dampings), np.array(frictions))

    oracle_path = Path(args_cli.oracle)
    if not oracle_path.is_absolute():
        oracle_path = project_root() / oracle_path
    if oracle_path.exists():
        records = SweepDataset.load(oracle_path).records
        budget["execution"] = _phase_budget(
            np.array([row.peak_velocity for row in records]),
            np.array([row.xi["joint_damping"] for row in records]),
            np.array([row.xi["joint_dynamic_friction"] for row in records]),
        )
    return budget


def _phase_budget(velocity: np.ndarray, damping: np.ndarray, friction: np.ndarray) -> dict:
    viscous = damping * velocity
    crossover = friction / np.maximum(damping, 1e-9)
    median_velocity = float(np.median(velocity))
    # What the *whole* training range of b buys, at the velocity this phase actually reaches.
    span = (max(args_cli.dampings[:7]) - min(args_cli.dampings[:7])) * median_velocity
    return {
        "episodes": int(len(velocity)),
        "median_peak_velocity": median_velocity,
        "median_viscous_force": float(np.median(viscous)),
        "median_coulomb_force": float(np.median(friction)),
        "median_ratio": float(np.median(viscous / friction)),
        "p90_ratio": float(np.percentile(viscous / friction, 90)),
        "median_crossover_velocity": float(np.median(crossover)),
        "fraction_above_crossover": float(np.mean(velocity > crossover)),
        "training_range_force_span": float(span),
        "training_range_span_over_noise_floor": float(span / FORCE_NOISE_FLOOR),
        "observable": bool(span > FORCE_NOISE_FLOOR),
    }


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    budget = force_budget()
    dampings = list(args_cli.dampings)
    num_envs = len(dampings)

    print("\n" + "=" * 78)
    print(f"[damp] damping sweep : {dampings[0]} .. {dampings[-1]} N*s/m ({num_envs} values, one per env)")
    print(f"[damp] held fixed    : m={args_cli.mass} mu_s={args_cli.static_friction} "
          f"mu_d={args_cli.dynamic_friction}")
    print(f"[damp] probe alphas  : {list(args_cli.trigger_fractions)}")
    print(f"[damp] probe v_max   : {list(args_cli.max_velocities)} m/s")

    system = PullSystem.build(
        PullSystemCfg(num_envs=num_envs, device=args_cli.device, probe=RECOMMENDED_PROBE_CFG)
    )
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    parameters = [
        DynamicsParameters(
            drawer_mass=args_cli.mass,
            joint_static_friction=args_cli.static_friction,
            joint_dynamic_friction=args_cli.dynamic_friction,
            joint_damping=damping,
            name=f"b{damping:g}",
        )
        for damping in dampings
    ]

    configurations = []
    try:
        for max_velocity in args_cli.max_velocities:
            for alpha in args_cli.trigger_fractions:
                controller = ResponseProbeController(
                    system.env,
                    system.osc,
                    system.reader,
                    ResponseProbeCfg(
                        trigger_fraction=alpha,
                        release_fraction=0.20,
                        max_velocity=max_velocity,
                        max_total_displacement=max(0.020, 4.0 * alpha * MAIN_TASK.goal_displacement),
                        max_coast_duration=2.5,
                    ),
                )
                randomizer.apply(system.env, parameters)
                system.reset()
                result = controller.run(goal_displacement=MAIN_TASK.goal_displacement)
                features = [extract_response_features(result, index) for index in range(num_envs)]

                truth = np.array(dampings) / args_cli.mass
                recovered = np.array([entry.coast_damping_over_mass for entry in features])
                release_velocity = np.array([entry.release_velocity for entry in features])
                finite = np.isfinite(recovered)
                configurations.append(
                    {
                        "max_velocity": max_velocity,
                        "trigger_fraction": alpha,
                        "release_velocity_median": float(np.median(release_velocity)),
                        "coast_duration_median": float(np.median([e.coast_duration for e in features])),
                        "coast_fit_r2_median": float(np.nanmedian([e.coast_fit_r2 for e in features])),
                        "coasted_to_rest_fraction": float(np.mean([e.coasted_to_rest for e in features])),
                        "total_displacement_median": float(np.median([e.total_displacement for e in features])),
                        "recovered_b_over_m": recovered.tolist(),
                        "true_b_over_m": truth.tolist(),
                        "spearman_recovered_vs_true": rank_correlation(
                            truth[finite].tolist(), recovered[finite].tolist()
                        )
                        if int(finite.sum()) > 3
                        else float("nan"),
                        # Restricted to the training range: identifiability there is what the
                        # paper needs, and a correlation over a 200x range would flatter it.
                        "spearman_within_training_range": rank_correlation(
                            *_within(truth, recovered, finite, low=2.0 / args_cli.mass, high=10.0 / args_cli.mass)
                        ),
                        "finite_fits": int(finite.sum()),
                        "viscous_force_span_at_release": float(
                            (max(dampings[:7]) - min(dampings[:7])) * float(np.median(release_velocity))
                        ),
                    }
                )
                print(
                    f"[damp] v_max={max_velocity:.2f} alpha={alpha:.2f}: "
                    f"release v {np.median(release_velocity) * 1000:5.1f} mm/s, "
                    f"coast {np.median([e.coast_duration for e in features]):.3f}s, "
                    f"fit R2 {np.nanmedian([e.coast_fit_r2 for e in features]):+.3f}, "
                    f"rho(b) all {configurations[-1]['spearman_recovered_vs_true']:+.3f} "
                    f"in-range {configurations[-1]['spearman_within_training_range']:+.3f} "
                    f"({time.perf_counter() - started:.0f} s)"
                )
    finally:
        system.close()

    report = {
        "git_commit": git_commit(),
        "force_noise_floor": FORCE_NOISE_FLOOR,
        "held_fixed": {
            "mass": args_cli.mass,
            "static_friction": args_cli.static_friction,
            "dynamic_friction": args_cli.dynamic_friction,
        },
        "dampings": dampings,
        "force_budget": budget,
        "configurations": configurations,
        "environment": collect_environment_info().as_dict(),
    }
    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / "damping_observability.json"
    )
    output.write_text(json.dumps(report, indent=2, default=float))
    _print(report)
    print(f"[damp] report written: {output}")
    print("=" * 78 + "\n")


def _within(truth: np.ndarray, recovered: np.ndarray, finite: np.ndarray, low: float, high: float):
    keep = finite & (truth >= low - 1e-9) & (truth <= high + 1e-9)
    if int(keep.sum()) < 4:
        return [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]
    return truth[keep].tolist(), recovered[keep].tolist()


def _print(report: dict) -> None:
    print("[damp]")
    print("[damp] FORCE BUDGET -- how big is the viscous term, really?")
    print(
        f"[damp]   {'phase':>10} {'median |v|':>11} {'b*v':>8} {'mu_d':>8} {'ratio':>7} "
        f"{'v* (m/s)':>9} {'above v*':>9} {'b-range span':>13} {'/floor':>7}"
    )
    for name in ("probe", "execution"):
        entry = report["force_budget"].get(name)
        if not entry:
            continue
        print(
            f"[damp]   {name:>10} {entry['median_peak_velocity']:10.4f}  "
            f"{entry['median_viscous_force']:7.3f}N {entry['median_coulomb_force']:7.3f}N "
            f"{entry['median_ratio']:7.3f} {entry['median_crossover_velocity']:9.3f} "
            f"{entry['fraction_above_crossover'] * 100:8.1f}% "
            f"{entry['training_range_force_span']:12.3f}N {entry['training_range_span_over_noise_floor']:6.2f}x"
        )
    print(
        f"[damp]   the noise floor is {report['force_noise_floor']:.2f} N (Phase 8 residual bias). A "
        "b-range span below 1.00x is unobservable."
    )

    print("[damp]")
    print("[damp] RECOVERY OF b FROM THE COAST -- does a faster probe see damping?")
    print(
        f"[damp]   {'v_max':>6} {'alpha':>6} {'release v':>10} {'coast':>7} {'fit R2':>8} "
        f"{'rho all':>8} {'rho in-range':>13} {'span/floor':>11} {'probe d':>9}"
    )
    for entry in report["configurations"]:
        span = entry["viscous_force_span_at_release"]
        print(
            f"[damp]   {entry['max_velocity']:6.2f} {entry['trigger_fraction']:6.2f} "
            f"{entry['release_velocity_median'] * 1000:8.1f}mm/s {entry['coast_duration_median']:6.3f}s "
            f"{entry['coast_fit_r2_median']:+8.3f} {entry['spearman_recovered_vs_true']:+8.3f} "
            f"{entry['spearman_within_training_range']:+13.3f} "
            f"{span / report['force_noise_floor']:10.2f}x "
            f"{entry['total_displacement_median'] * 1000:7.1f}mm"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()

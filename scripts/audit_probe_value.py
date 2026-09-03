r"""Is the dedicated active probe worth it? Compare it against doing nothing and doing less.

The method's premise is that a short deliberate excitation reveals the force a drawer needs.
This tests that premise against the two obvious alternatives at the same cost:

    frozen probe        3.5 N over 0.3 s   (Setting V1, D044)
    weak generic        1.0 N over 0.3 s   (not tuned; a round number well below the probe)
    passive observation 0.0 N over 0.3 s   (the same budget, spent applying nothing)

All three are the **same smoothstep trapezoid at three amplitudes**, so they share the budget,
the 18-step length, the seven deployable channels, the feature extractor and the ridge readout.
Nothing varies but the amplitude, which is what makes a difference in identifiability
attributable to the excitation rather than to the format.

Each variant gets its **own** force sweep from its **own** post-probe snapshot, because the
force a drawer needs depends on where the interaction left it -- a passive observation starts
its execution from a closed drawer, the frozen probe from about 7 mm out. Two targets are
therefore scored: the force each history's own state requires (deployment-faithful), and the
force the frozen probe's state requires (which isolates knowledge of the hidden dynamics from
knowledge of one's own starting point).

No neural network is trained, the probe is not redesigned, and nothing is tuned.

Usage::

    python scripts/audit_probe_value.py --headless
    python scripts/audit_probe_value.py --headless --num-xi 64 --amplitudes 3.5 1.0 0.0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-xi", type=int, default=64, help="In-distribution states.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument(
    "--amplitudes",
    type=float,
    nargs="+",
    default=(3.5, 1.0, 0.0),
    help="Excitation amplitudes (N). The first must be the frozen probe -- it is the reference.",
)
parser.add_argument("--force-low", type=float, default=0.25)
parser.add_argument("--force-high", type=float, default=9.0)
parser.add_argument("--force-step", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=20260905)
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

from probe_drawer.analysis.probe_features import (  # noqa: E402
    PROBE_FEATURES,
    assert_features_are_deployable,
    extract_features,
)
from probe_drawer.analysis.probe_value import summarise_probe_value  # noqa: E402
from probe_drawer.analysis.sweep import force_grid  # noqa: E402
from probe_drawer.dataset import XiSamplerCfg, branch_order  # noqa: E402
from probe_drawer.dataset.sampling import sample_hidden_states  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, assess_validity  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    RECOMMENDED_EXECUTION_CFG,
    SEQUENTIAL_TRANSITION_STEPS,
    SETTING_V1_PROBE,
    SETTING_V1_PROBE_CFG,
    SETTING_V1_TASK,
    TRAINING_XI_RANGES,
)
from probe_drawer.observations import DEFAULT_ACE_INPUT  # noqa: E402
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)

NAMES = {3.5: "frozen probe (3.5 N)", 1.0: "weak generic (1.0 N)", 0.0: "passive (0.0 N)"}


def label(amplitude: float) -> str:
    return NAMES.get(amplitude, f"excitation ({amplitude:g} N)")


def main() -> None:
    enable_unbuffered_stdout()
    assert_features_are_deployable()
    started = time.perf_counter()

    task = SETTING_V1_TASK
    region = OperatingRegionCfg()
    forces = force_grid(args_cli.force_low, args_cli.force_high, args_cli.force_step)
    amplitudes = list(args_cli.amplitudes)
    if amplitudes[0] != SETTING_V1_PROBE.peak_force:
        raise ValueError(
            f"the first amplitude must be the frozen probe's {SETTING_V1_PROBE.peak_force} N so "
            f"that it is the reference, got {amplitudes[0]}."
        )

    states = sample_hidden_states(
        XiSamplerCfg(
            num_states=args_cli.num_xi,
            seed=args_cli.seed,
            mass=TRAINING_XI_RANGES.mass,
            static_friction=TRAINING_XI_RANGES.static_friction,
            dynamic_friction_ratio=TRAINING_XI_RANGES.dynamic_friction_ratio,
            damping=TRAINING_XI_RANGES.damping,
        )
    )
    num_envs = min(args_cli.num_envs, len(states))
    batches = [states[start : start + num_envs] for start in range(0, len(states), num_envs)]

    print("\n" + "=" * 92)
    print(f"[value] states     : {len(states)} in-distribution, plain Sobol, seed {args_cli.seed}")
    print(f"[value] histories  : " + ", ".join(label(a) for a in amplitudes))
    print(f"[value] shared     : {SETTING_V1_PROBE.duration:g} s budget, "
          f"{len(DEFAULT_ACE_INPUT)} deployable channels, {len(PROBE_FEATURES)} features, ridge readout")
    print(f"[value] task       : d_goal={task.goal_displacement * 1000:g} mm T={task.duration:g} s")
    print(f"[value] F_peak     : {forces[0]:.2f}..{forces[-1]:.2f} N step {args_cli.force_step:g} "
          f"({len(forces)} values) per history")
    print(f"[value] episodes   : {len(amplitudes) * len(batches) * (1 + len(forces))}")

    system = PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=SETTING_V1_PROBE_CFG,
            execution=RECOMMENDED_EXECUTION_CFG,
        )
    )
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()

    records: dict[float, list[dict]] = {amplitude: [] for amplitude in amplitudes}
    try:
        for amplitude in amplitudes:
            for number, batch in enumerate(batches, start=1):
                padded = batch + [batch[-1]] * (num_envs - len(batch))
                randomizer.apply(
                    system.env,
                    [
                        DynamicsParameters(
                            name=f"xi{index:03d}",
                            drawer_mass=state["mass"],
                            joint_static_friction=state["static_friction"],
                            joint_dynamic_friction=state["dynamic_friction"],
                            joint_damping=state["damping"],
                        )
                        for index, state in enumerate(padded)
                    ],
                )
                system.reset()

                task_start = system.reader.drawer_position.clone()
                history = system.probe.run_fixed_budget(
                    peak_force=amplitude, duration=SETTING_V1_PROBE.duration
                )
                system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
                pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
                snapshot = capture_snapshot(system, label=f"value {amplitude:g}N batch {number}")

                reaching: list[list[float]] = [[] for _ in batch]
                for position, index in enumerate(
                    branch_order(f"probe-value-{amplitude:g}-{number}", len(forces))
                ):
                    force = forces[index]
                    restore_snapshot(system, snapshot)
                    result = system.execution.run(peak_force=force, duration=task.duration)
                    validity = assess_validity(
                        result, region, pre_execution_displacement=pre_execution
                    )
                    for env in range(len(batch)):
                        verdict = validity.verdicts[env]
                        achieved = verdict.metrics["final_displacement"]
                        if verdict.valid and abs(achieved - task.goal_displacement) <= task.displacement_tolerance:
                            reaching[env].append(force)
                    if position % 30 == 0:
                        print(f"[value] {label(amplitude)} batch {number}/{len(batches)} "
                              f"point {position + 1}/{len(forces)} "
                              f"({time.perf_counter() - started:.0f} s)")

                for env, state in enumerate(batch):
                    features = extract_features(history, env)
                    hits = sorted(reaching[env])
                    records[amplitude].append(
                        {
                            "hidden_state": dict(state),
                            "features": list(features.as_vector()),
                            "moved": bool(features.moved),
                            "post_probe_displacement": float(pre_execution[env]),
                            "required_force": hits[0] if hits else None,
                            "band_centre": (hits[0] + hits[-1]) / 2.0 if hits else None,
                            "num_reaching": len(hits),
                        }
                    )
    finally:
        system.close()

    reference = amplitudes[0]
    common = [row["band_centre"] for row in records[reference]]
    variants = []
    for amplitude in amplitudes:
        rows = records[amplitude]
        variants.append(
            {
                "name": label(amplitude),
                "amplitude": amplitude,
                "features": np.array([row["features"] for row in rows], dtype=float),
                "moved": [row["moved"] for row in rows],
                "own_target": [
                    float("nan") if row["band_centre"] is None else row["band_centre"] for row in rows
                ],
                "common_target": [float("nan") if value is None else value for value in common],
            }
        )
    summary = summarise_probe_value(variants)

    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / "probe_value_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "phase": "probe value audit",
                "task": task.as_dict(),
                "frozen_probe": SETTING_V1_PROBE.as_dict(),
                "amplitudes": amplitudes,
                "channels": list(DEFAULT_ACE_INPUT),
                "features": list(PROBE_FEATURES),
                "force_grid": list(forces),
                "seed": args_cli.seed,
                "git_commit": git_commit(),
                "environment": collect_environment_info().as_dict(),
                "elapsed_s": time.perf_counter() - started,
                "summary": summary,
                "records": {str(key): value for key, value in records.items()},
            },
            indent=2,
            default=float,
        )
    )

    print("[value]")
    for target in ("own", "common"):
        note = (
            "the force each history's own post-probe state requires"
            if target == "own"
            else "the force the frozen probe's state requires -- isolates hidden-dynamics knowledge"
        )
        print(f"[value] === target: {target} -- {note}")
        print(f"[value] {'history':>22} {'RMSE':>9} {'R2':>8} {'sd':>8} {'n':>4} "
              f"{'best feature':>26} {'|rho|':>7} {'breakaway':>10}")
        for amplitude in amplitudes:
            v = summary["per_variant"][label(amplitude)]
            r = v[target]
            rho = r["best_feature_abs_spearman"]
            print(f"[value] {label(amplitude):>22} {r['rmse']:8.4f}N {r['r2']:+7.3f} "
                  f"{r['target_sd']:7.4f}N {r['n']:>4} {str(r['best_feature']):>26} "
                  f"{('n/a' if rho is None else f'{rho:6.3f}')} {v['breakaway_fraction'] * 100:9.1f}%")
        print("[value]")

    print("[value] against the frozen probe:")
    for name, row in summary["comparison"].items():
        for target in ("own", "common"):
            c = row[target]
            ratio = c["rmse_ratio"]
            print(f"[value]   {name:>22} [{target:>6}] RMSE {c['rmse_probe']:.4f} -> "
                  f"{c['rmse_other']:.4f} N "
                  f"({'n/a' if ratio is None else f'{ratio:.2f}x'}), "
                  f"R2 drops {c['r2_drop']:+.3f}")
    print("[value]")
    for amplitude in amplitudes:
        v = summary["per_variant"][label(amplitude)]
        constant = v["constant_features"]
        print(f"[value] {label(amplitude):>22}: {len(constant)}/{len(PROBE_FEATURES)} features constant"
              + (f" -- {', '.join(constant)}" if constant else ""))
    print(f"[value]")
    print(f"[value] written    : {output}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

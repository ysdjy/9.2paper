r"""Would one global slope solve the multi-goal problem? An offline pilot, no training.

`docs/TASK_CONDITIONING.md` showed the goal moves the required force by 1.29x the success band,
so a single-goal force fails at a new goal -- and that the ratio is arithmetic
(`delta_goal / 2*eps_d`) rather than dynamics. The follow-up decides whether a task-conditioned
experiment is worth running: if the mapping is near-affine, **one global slope** may be all
there is to learn, and the honest baseline to beat is not a fixed force but

    F(g) = F_100 + k_global * (g - 0.10)

This fits `k_global` on a **calibration** half and applies it to a **held-out** half. The split
is a content-addressed permutation fixed before any physics is read.

Two allowances are made **in the baseline's favour**, because the question is whether a global
slope suffices rather than whether this is deployable: it is handed the *correct* `F_100` for
each held-out drawer from that drawer's own Oracle band centre, and the slope is fitted on the
same distribution it is tested on. So these numbers are an upper bound; if a global correction
still falls short here, no simpler correction will close the gap.

No Isaac Sim -- it consumes the sweep written by `scripts/audit_task_conditioning.py`.

Usage::

    python scripts/audit_task_conditioning.py --headless --num-xi 64 --num_envs 32 \
        --sampler sobol --seed 20260904 --output outputs/logs/affine_goal_sweep.json
    python scripts/audit_affine_goal_baseline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_drawer.analysis.affine_goal_baseline import evaluate_affine_goal_baseline
from probe_drawer.dataset import stable_permutation
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root


def main() -> None:
    enable_unbuffered_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=str, default="outputs/logs/affine_goal_sweep.json")
    parser.add_argument("--reference-goal", type=float, default=0.10)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260904,
        help="Keys the calibration/held-out permutation. Fixed before any physics is read.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.sweep)
    if not path.is_absolute():
        path = project_root() / path
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    task = payload["task"]
    goals = payload["summary"]["goals"]
    step = payload["summary"]["force_step"]
    tolerance = task["displacement_tolerance"]
    action_range = tuple(task["peak_force_range"])

    # Content-addressed, so the halves are the same on any machine and were decided by the
    # seed rather than by looking at the outcome.
    order = stable_permutation("affine-goal-split", args.split_seed, len(rows))
    half = len(rows) // 2
    calibration, held_out = sorted(order[:half]), sorted(order[half:])

    result = evaluate_affine_goal_baseline(
        rows,
        calibration,
        held_out,
        goals,
        args.reference_goal,
        tolerance,
        step,
        action_range,
    )

    print("\n" + "=" * 92)
    print(f"[affine] sweep      : {path.name}, {len(rows)} in-distribution states")
    print(f"[affine] split      : {len(calibration)} calibration / {len(held_out)} held out, "
          f"keyed on seed {args.split_seed}")
    print(f"[affine] goals      : {[f'{g * 1000:g} mm' for g in goals]}, "
          f"reference {args.reference_goal * 1000:g} mm, eps_d {tolerance * 1000:g} mm")
    print(f"[affine] given away : the correct F_100 per held-out drawer -- this is an upper bound")

    cal = result["calibration"]
    slope = cal["slope"]
    print("[affine]")
    print(f"[affine] k_global   : {result['k_global']:.2f} N/m   "
          f"(fitted on {cal['states_with_a_slope']}/{cal['states']} calibration drawers; "
          f"their slopes {slope['min']:.1f}-{slope['max']:.1f}, sd {slope['sd']:.1f})")
    print(f"[affine]              i.e. {result['k_global'] * 0.02:.3f} N per 20 mm")

    print("[affine]")
    print(f"[affine] {'goal':>8} {'affine reach':>14} {'oracle':>10} {'gap':>8} "
          f"{'|d-goal| med':>13} {'signed med':>12}")
    for goal in goals:
        v = result["held_out"][goal]
        err, abs_err = v["position_error_mm"], v["abs_position_error_mm"]
        print(f"[affine] {goal * 1000:6.0f}mm {v['reached']:>4}/{v['states_with_a_reference']:<4} "
              f"= {v['reach_rate'] * 100:5.1f}% {v['oracle_rate'] * 100:8.1f}% "
              f"{v['gap_to_oracle_pp']:+7.1f}pp {abs_err['median']:12.2f}mm "
              f"{err['median']:+11.2f}mm")

    err = result["slope_error"]
    print("[affine]")
    print("[affine] per-drawer slope against k_global, held-out only:")
    print(f"[affine]   true slope   : {err['held_out_slope']['min']:.1f} - "
          f"{err['held_out_slope']['max']:.1f} N/m (median {err['held_out_slope']['median']:.1f}, "
          f"sd {err['held_out_slope']['sd']:.1f})")
    print(f"[affine]   |k_i - k_g|  : median {err['abs_error']['median']:.2f} N/m, "
          f"max {err['abs_error']['max']:.2f}")
    if err["relative_error"]:
        print(f"[affine]   relative     : median {err['relative_error']['median'] * 100:.1f} %, "
              f"max {err['relative_error']['max'] * 100:.1f} %")

    off = [goal for goal in goals if goal != args.reference_goal]
    worst = min(result["held_out"][goal]["reach_rate"] for goal in off)
    print("[affine]")
    print(f"[affine] worst off-reference reach: {worst * 100:.1f} %")
    print(f"[affine] verdict    : " + (
        "a global slope is essentially sufficient -- a multi-goal study would mostly re-learn it"
        if worst > 0.90
        else "a global slope is NOT sufficient; the residual is hidden-state dependent"
    ))

    output = (
        Path(args.output)
        if args.output
        else project_root() / "outputs" / "logs" / "affine_goal_baseline.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "sweep": str(path),
                "split_seed": args.split_seed,
                "calibration_indices": calibration,
                "held_out_indices": held_out,
                "git_commit": git_commit(),
                **result,
            },
            indent=2,
            default=float,
        )
    )
    print(f"[affine] written    : {output}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()

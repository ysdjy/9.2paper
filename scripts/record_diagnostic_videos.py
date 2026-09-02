"""Record annotated videos of representative episodes, so the motion can be judged by eye.

Every number this project reports is a summary. A 41 mm lateral drift, a breakaway jump, an
arm sitting on a joint stop -- these are all single scalars in a table, and a table cannot
show whether the robot looked sane while producing them. This records the episodes behind the
numbers, with the telemetry burned into each frame.

Seven categories, chosen to span what the analyses flagged rather than to look good:

======  ==========================================================================
A       normal success -- moderate hidden dynamics, smooth motion
B       high static friction -- a long stationary period, then breakaway
C       high mass -- moving, but barely accelerating
D       high dynamic friction and damping -- sustained motion is hard
E       low friction -- fast, prone to overshoot
F       the 2-D failure corners: low/high force x short/long duration
G       long-distance goals, 100 to 390 mm, where posture and limits bite
======  ==========================================================================

Each video covers ``INITIAL -> probe -> inference gap -> execution -> final state`` with an
overlay of ``xi``, the phase, the commanded force, drawer displacement and velocity, the
execution parameters, and the verdict. An ``index.csv`` alongside lists every clip with the
diagnostics needed to pick which to watch.

One environment per video, deliberately: a 32-drawer render is unreadable.

Usage::

    python scripts/record_diagnostic_videos.py --headless
    python scripts/record_diagnostic_videos.py --headless --categories A B G
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--categories", type=str, nargs="+", default=list("ABCDEFG"))
parser.add_argument("--fps", type=int, default=30, help="Output frame rate; frames are captured to match.")
parser.add_argument("--output", type=str, default="outputs/videos/diagnostics")
parser.add_argument("--reach", type=str, default="outputs/logs/goal_distance_sweep.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Rendering needs a camera, which needs the app to not be fully headless-offscreen-less.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import (  # noqa: E402
    DRAWER_TRAVEL_LIMIT,
    OperatingRegionCfg,
    assess_validity,
    evaluate_execution,
)
from probe_drawer.evaluation.task_evaluator import SuccessCriteria  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root  # noqa: E402


@dataclass
class Case:
    """One video: a drawer, an execution, and why it is worth watching."""

    name: str
    category: str
    xi: dict
    peak_force: float
    duration: float
    goal: float = MAIN_TASK.goal_displacement
    note: str = ""
    metrics: dict = field(default_factory=dict)


def reach_lookup(path: Path) -> list[dict]:
    """Episodes from the goal-distance sweep, for picking parameters that reach a distance."""
    if not path.exists():
        return []
    return json.loads(path.read_text())["records"]


def build_cases(reach: list[dict]) -> list[Case]:
    """The representative set. Hidden states are named by what makes them interesting."""
    moderate = {"mass": 8.0, "static_friction": 1.75, "dynamic_friction": 1.00, "damping": 6.0}
    sticky = {"mass": 8.0, "static_friction": 3.00, "dynamic_friction": 0.60, "damping": 6.0}
    heavy = {"mass": 12.0, "static_friction": 0.80, "dynamic_friction": 0.50, "damping": 3.0}
    draggy = {"mass": 8.0, "static_friction": 2.80, "dynamic_friction": 2.70, "damping": 10.0}
    slippery = {"mass": 4.0, "static_friction": 0.50, "dynamic_friction": 0.20, "damping": 2.0}

    cases: list[Case] = [
        Case("A1_normal_success", "A", moderate, 1.90, 1.5, note="moderate dynamics, expected success"),
        Case("A2_normal_success_slow", "A", moderate, 1.60, 2.2, note="same drawer, gentler and longer"),
        Case("B1_high_static_breakaway", "B", sticky, 3.20, 1.5, note="stationary, then breakaway"),
        Case("B2_high_static_marginal", "B", sticky, 2.60, 2.0, note="near the breakaway threshold"),
        Case("C1_high_mass_slow_accel", "C", heavy, 1.30, 1.8, note="moving but barely accelerating"),
        Case("C2_high_mass_more_force", "C", heavy, 2.20, 1.2, note="heavier push, shorter time"),
        Case("D1_high_drag_damping", "D", draggy, 3.60, 1.8, note="sustained motion is hard"),
        Case("D2_high_drag_underpowered", "D", draggy, 2.40, 2.4, note="likely stalls short"),
        Case("E1_low_friction_fast", "E", slippery, 0.60, 1.5, note="easy drawer, overshoot risk"),
        Case("E2_low_friction_overshoot", "E", slippery, 1.60, 2.0, note="deliberate overshoot"),
        # F: the four corners of the 2-D box, on a drawer that succeeds somewhere inside it.
        Case("F1_lowF_shortT", "F", moderate, 0.50, 1.0, note="too little force, too little time"),
        Case("F2_highF_shortT", "F", moderate, 4.00, 1.0, note="fast and still moving at T"),
        Case("F3_lowF_longT", "F", moderate, 0.50, 2.5, note="too little force, plenty of time"),
        Case("F4_highF_longT", "F", moderate, 4.00, 2.5, note="overshoot toward the end stop"),
    ]

    # G: long distances, with parameters taken from episodes that actually reached them.
    for goal_mm in (100, 150, 200, 250, 300, 350, 390):
        goal = goal_mm / 1000.0
        candidates = [
            row
            for row in reach
            if abs(row["final_displacement"] - goal) <= 0.010
            and abs(row["xi"]["static_friction"] - moderate["static_friction"]) < 1.2
        ]
        if not candidates:
            candidates = sorted(reach, key=lambda row: abs(row["final_displacement"] - goal))[:1]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: abs(row["final_displacement"] - goal))
        cases.append(
            Case(
                f"G_goal{goal_mm}mm",
                "G",
                dict(best["xi"]),
                float(best["peak_force"]),
                float(best["duration"]),
                goal=goal,
                note=(
                    f"reaches {best['final_displacement'] * 1000:.0f} mm; "
                    f"joint margin {best['min_joint_margin']:.3f}, "
                    f"drift {best['peak_lateral_drift'] * 1000:.1f} mm"
                ),
                metrics={
                    "expected_displacement_mm": best["final_displacement"] * 1000,
                    "min_joint_margin": best["min_joint_margin"],
                    "peak_lateral_drift_mm": best["peak_lateral_drift"] * 1000,
                    "manipulability": best["manipulability"],
                },
            )
        )
    return cases


def stride_for(fps: int, step_dt: float) -> int:
    """How many control steps to skip between captured frames.

    Rendering is the expensive part of this script, so the capture rate is the output frame
    rate rather than the 60 Hz control rate.
    """
    return max(1, round(1.0 / (step_dt * max(fps, 1))))


class Annotator:
    """Draws the telemetry onto a frame.

    Deliberately plain: a translucent panel and monospaced lines. The video is a diagnostic,
    not a presentation, and anything more elaborate would be one more thing to maintain.
    """

    def __init__(self, case: Case) -> None:
        self.case = case
        self.header = (
            f"m={case.xi['mass']:.1f}  mu_s={case.xi['static_friction']:.2f}  "
            f"mu_d={case.xi['dynamic_friction']:.2f}  b={case.xi['damping']:.1f}"
        )

    def draw(self, frame: np.ndarray, lines: list[str]) -> np.ndarray:
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        rows = [self.case.name, self.header, *lines]
        height = 14 * len(rows) + 10
        draw.rectangle([0, 0, image.width, height], fill=(0, 0, 0, 165))
        for index, text in enumerate(rows):
            draw.text((8, 5 + 14 * index), text, fill=(255, 255, 255))
        return np.asarray(image)


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    directory = Path(args_cli.output)
    if not directory.is_absolute():
        directory = project_root() / directory
    directory.mkdir(parents=True, exist_ok=True)

    reach_path = Path(args_cli.reach)
    if not reach_path.is_absolute():
        reach_path = project_root() / reach_path
    cases = [case for case in build_cases(reach_lookup(reach_path)) if case.category in args_cli.categories]

    print("\n" + "=" * 78)
    print(f"[vid] cases     : {len(cases)} in categories {sorted(set(c.category for c in cases))}")
    print(f"[vid] output    : {directory}")

    execution_cfg = ExecutionControllerCfg(
        rise_fraction=RECOMMENDED_EXECUTION_CFG.rise_fraction,
        fall_fraction=RECOMMENDED_EXECUTION_CFG.fall_fraction,
        shape=RECOMMENDED_EXECUTION_CFG.shape,
        settle_steps=0,
        zero_force_cleanup_steps=RECOMMENDED_EXECUTION_CFG.zero_force_cleanup_steps,
        post_execution_settle_steps=RECOMMENDED_EXECUTION_CFG.post_execution_settle_steps,
    )
    # video_folder is what puts the environment into rgb_array render mode; the RecordVideo
    # wrapper it installs is never started, because frames are captured here instead so the
    # overlay can be drawn from the state that produced them.
    system = PullSystem.build(
        PullSystemCfg(
            num_envs=1,
            device=args_cli.device,
            probe=RECOMMENDED_PROBE_CFG,
            execution=execution_cfg,
            video_folder=directory / "_render",
        )
    )
    randomizer = DynamicsRandomizer()
    region = OperatingRegionCfg()
    rows: list[dict] = []

    try:
        for number, case in enumerate(cases, start=1):
          try:
            frames: list[np.ndarray] = []
            annotator = Annotator(case)
            randomizer.apply(
                system.env,
                [
                    DynamicsParameters(
                        drawer_mass=case.xi["mass"],
                        joint_static_friction=case.xi["static_friction"],
                        joint_dynamic_friction=case.xi["dynamic_friction"],
                        joint_damping=case.xi["damping"],
                        name=case.name,
                    )
                ],
            )
            system.reset()
            start_position = system.reader.drawer_position.clone()

            def capture(phase: str, force: float) -> None:
                frame = system.env.render()
                if frame is None:
                    return
                displacement = float((system.reader.drawer_position - start_position)[0])
                velocity = float(system.reader.drawer_velocity[0])
                frames.append(
                    annotator.draw(
                        np.asarray(frame),
                        [
                            f"phase={phase:<10} d_goal={case.goal * 1000:.0f}mm  "
                            f"F_peak={case.peak_force:.2f}N  T={case.duration:.2f}s",
                            f"F_cmd={force:6.2f}N   d={displacement * 1000:7.2f}mm   "
                            f"v={velocity:+7.4f}m/s   travel={displacement / DRAWER_TRAVEL_LIMIT * 100:5.1f}%",
                            case.note,
                        ],
                    )
                )

            capture("INITIAL", 0.0)

            def on_probe_step(step: int, elapsed: float, commanded) -> None:
                if step % stride_for(args_cli.fps, system.step_dt):
                    return
                frame = system.env.render()
                if frame is None:
                    return
                displacement = float((system.reader.drawer_position - start_position)[0])
                frames.append(
                    annotator.draw(
                        np.asarray(frame),
                        [
                            f"phase=PROBE      t={elapsed:.2f}s  d_goal={case.goal * 1000:.0f}mm",
                            f"F_cmd={float(commanded[0]):6.2f}N   d={displacement * 1000:7.2f}mm   "
                            f"v={float(system.reader.drawer_velocity[0]):+7.4f}m/s",
                            case.note,
                        ],
                    )
                )

            probe = system.probe.run(**RECOMMENDED_PROBE_TASK.as_kwargs(), on_step=on_probe_step)
            capture("PROBE_END", float(probe.final_commanded_force[0]))
            for _ in range(SEQUENTIAL_TRANSITION_STEPS):
                system.osc.coast(1)
                capture("GAP", 0.0)

            pre_execution = (system.reader.drawer_position - start_position).cpu().numpy().copy()

            # Every ``stride`` steps of the execution, render the *live* scene and annotate it
            # from the state that produced it. Going through the controller's ``on_step``
            # observer rather than reimplementing its loop is what keeps the video showing the
            # real execution instead of a re-labelled still.
            def on_execution_step(step: int, elapsed: float, commanded) -> None:
                if step % stride_for(args_cli.fps, system.step_dt):
                    return
                frame = system.env.render()
                if frame is None:
                    return
                displacement = float((system.reader.drawer_position - start_position)[0])
                frames.append(
                    annotator.draw(
                        np.asarray(frame),
                        [
                            f"phase=EXECUTION  t={elapsed:.2f}/{case.duration:.2f}s  "
                            f"d_goal={case.goal * 1000:.0f}mm  F_peak={case.peak_force:.2f}N",
                            f"F_cmd={float(commanded[0]):6.2f}N   d={displacement * 1000:7.2f}mm   "
                            f"v={float(system.reader.drawer_velocity[0]):+7.4f}m/s   "
                            f"travel={displacement / DRAWER_TRAVEL_LIMIT * 100:5.1f}%",
                            case.note,
                        ],
                    )
                )

            result = system.execution.run(
                peak_force=case.peak_force, duration=case.duration, on_step=on_execution_step
            )
            capture("FINAL", 0.0)

            criteria = SuccessCriteria(case.goal, MAIN_TASK.displacement_tolerance, MAIN_TASK.velocity_tolerance)
            evaluation = evaluate_execution(result, criteria, region, pre_execution_displacement=pre_execution)
            verdict = evaluation.verdicts[0]
            # The drift metrics live on the *validity* verdict; ExecutionVerdict carries the
            # task outcome only.
            validity = assess_validity(result, region, pre_execution_displacement=pre_execution).verdicts[0]
            for _ in range(args_cli.fps // 2):
                frames.append(
                    annotator.draw(
                        np.asarray(frames[-1]),
                        [
                            f"phase=VERDICT    d_goal={case.goal * 1000:.0f}mm",
                            f"d_total={verdict.total_displacement * 1000:.2f}mm  "
                            f"v(T)={verdict.terminal_velocity:+.4f}m/s  "
                            f"{'SUCCESS' if verdict.success else 'FAIL'}"
                            f"{'' if verdict.valid else '  INVALID: ' + ','.join(r.value for r in verdict.invalid_reasons)}",
                            case.note,
                        ],
                    )
                )

            path = directory / f"{case.name}.mp4"
            if frames:
                imageio.mimsave(path, frames, fps=args_cli.fps, macro_block_size=None)
            rows.append(
                {
                    "filename": path.name,
                    "category": case.category,
                    **{f"xi_{key}": round(value, 4) for key, value in case.xi.items()},
                    "d_goal_mm": case.goal * 1000,
                    "F_peak": case.peak_force,
                    "T": case.duration,
                    "probe_displacement_mm": round(float(pre_execution[0]) * 1000, 3),
                    "d_total_mm": round(verdict.total_displacement * 1000, 2),
                    "v_T": round(verdict.terminal_velocity, 5),
                    "success": verdict.success,
                    "valid": verdict.valid,
                    "invalid_reasons": "|".join(r.value for r in verdict.invalid_reasons),
                    "peak_velocity": round(float(result.peak_velocity[0]), 4),
                    "peak_wrist_force": round(float(result.peak_measured_force[0]), 3),
                    "peak_lateral_drift_mm": round(float(validity.metrics["peak_lateral_drift"]) * 1000, 3),
                    "peak_orientation_drift_deg": round(
                        float(validity.metrics["peak_orientation_drift_deg"]), 3
                    ),
                    "travel_fraction": round(verdict.total_displacement / DRAWER_TRAVEL_LIMIT, 3),
                    "note": case.note,
                    **{key: round(value, 4) for key, value in case.metrics.items()},
                }
            )
            print(
                f"[vid] {number:2d}/{len(cases)} {case.name:<28} {len(frames):4d} frames  "
                f"d={verdict.total_displacement * 1000:7.2f}mm  "
                f"{'ok ' if verdict.success else 'FAIL'} "
                f"{'' if verdict.valid else 'INVALID'} ({time.perf_counter() - started:.0f} s)"
            )
          except Exception as error:  # noqa: BLE001 - one bad case must not cost the other twenty
            print(f"[vid] {number:2d}/{len(cases)} {case.name:<28} FAILED: {type(error).__name__}: {error}")
    finally:
        system.close()

    index = directory / "index.csv"
    if rows:
        keys = sorted({key for row in rows for key in row})
        ordered = [key for key in ("filename", "category", "d_goal_mm", "F_peak", "T", "success", "valid") if key in keys]
        ordered += [key for key in keys if key not in ordered]
        with index.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(rows)
    (directory / "manifest.json").write_text(
        json.dumps({"git_commit": git_commit(), "fps": args_cli.fps, "cases": rows}, indent=2, default=float)
    )
    print(f"[vid] index written: {index}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

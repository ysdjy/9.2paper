"""Phase 12-pre -- how far can this drawer be pulled, and what breaks first?

The task has used ``d_goal = 40 mm`` since Phase 9, selected by a sweep that never looked past
100 mm. The drawer's travel is 400 mm. So most of its range has never been tested, and "long
pulls are probably a problem" has been an assumption.

This sweep replaces the assumption with a measurement, and it is built to *attribute* failure
rather than only detect it. Alongside the usual outcome it records, at ``T``:

* the arm's joint configuration and its smallest margin to a joint limit;
* the manipulability ``sqrt(det(J J^T))`` and, more usefully, the velocity transmission
  ``1/sqrt(u^T (J J^T)^-1 u)`` along the pull direction -- the determinant can stay healthy
  while the one direction that matters collapses;
* the Jacobian's condition number;
* the largest single-step jump in wrist force, which is what an end-stop impact looks like
  and what a peak force does not distinguish from a slow hard pull;
* the drawer's travel fraction, held-axis drift, terminal and peak velocity, and every
  validity reason.

Those let ``analysis/goal_distance.py`` say whether a distance is bounded by the *drawer*, the
*robot's posture*, the *controller*, or merely by the parameter box that was swept.

The parameter box is deliberately wider than the Phase 12 box: reaching 390 mm needs forces
and durations the 40 mm task never used, and a sweep that stopped at the task's range would
find infeasibility that is an artefact of its own bounds.

Usage::

    python scripts/sweep_goal_distance.py --headless
    python scripts/sweep_goal_distance.py --headless --num-xi 24 --force-high 8.0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-xi", type=int, default=16, help="Representative hidden states.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--force-low", type=float, default=0.5)
parser.add_argument("--force-high", type=float, default=8.0)
parser.add_argument("--force-step", type=float, default=0.5)
parser.add_argument("--duration-low", type=float, default=0.5)
parser.add_argument("--duration-high", type=float, default=3.0)
parser.add_argument("--duration-step", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=20260902)
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
import torch  # noqa: E402

from probe_drawer.analysis.goal_distance import LongPullRecord  # noqa: E402
from probe_drawer.analysis.landscape_2d import representative_hidden_states  # noqa: E402
from probe_drawer.analysis.sweep import force_grid  # noqa: E402
from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.dataset import branch_order  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import DRAWER_TRAVEL_LIMIT, OperatingRegionCfg, assess_validity  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)


def build_system(num_envs: int) -> PullSystem:
    execution = ExecutionControllerCfg(
        rise_fraction=RECOMMENDED_EXECUTION_CFG.rise_fraction,
        fall_fraction=RECOMMENDED_EXECUTION_CFG.fall_fraction,
        shape=RECOMMENDED_EXECUTION_CFG.shape,
        settle_steps=0,
        zero_force_cleanup_steps=RECOMMENDED_EXECUTION_CFG.zero_force_cleanup_steps,
        post_execution_settle_steps=RECOMMENDED_EXECUTION_CFG.post_execution_settle_steps,
    )
    return PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=RECOMMENDED_PROBE_CFG,
            execution=execution,
        )
    )


class PostureProbe:
    """Reads joint margins and Jacobian conditioning out of the live environment.

    The joint limits come from ``Articulation.data.joint_pos_limits`` and the Jacobian from
    the OSC action term, which computes it every step anyway. Both are reached through private
    attributes because Isaac Lab exposes no public accessor for a named action term; that is
    noted rather than worked around, and the class fails loudly if the shapes are not what it
    expects rather than silently reporting nonsense.
    """

    def __init__(self, system: PullSystem) -> None:
        self.system = system
        robot = system.env.scene["robot"]
        self.arm_ids = system.reader._arm_joint_ids  # noqa: SLF001 - no public accessor
        limits = robot.data.joint_pos_limits[:, self.arm_ids, :]
        self.lower = limits[..., 0]
        self.upper = limits[..., 1]
        self.span = (self.upper - self.lower).clamp_min(1e-6)

        terms = system.env.action_manager._terms  # noqa: SLF001 - no public accessor
        self.term = terms.get("arm_action")
        if self.term is None:
            raise RuntimeError(f"no 'arm_action' term; found {sorted(terms)}")
        self.direction = system.pull_axis.direction(system.env.device)

    def joint_margins(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Smallest fractional distance to a limit, per environment, and which joint."""
        values = torch.as_tensor(positions, device=self.lower.device, dtype=self.lower.dtype)
        fractional = torch.minimum(values - self.lower, self.upper - values) / self.span
        margin, joint = fractional.min(dim=1)
        return margin.cpu().numpy(), joint.cpu().numpy()

    def conditioning(self) -> dict:
        r"""Manipulability, pull-axis transmission and condition number, per environment.

        The pull-axis transmission is :math:`1/\sqrt{u^\top (JJ^\top)^{-1} u}` restricted to
        the translational block: how much TCP velocity along the pull direction a unit of
        joint velocity buys. It is the quantity that would actually degrade if the arm's
        posture became a problem for long pulls, and unlike the determinant it cannot be
        propped up by healthy motion in directions the task does not use.
        """
        jacobian = self.term.jacobian_b.detach()
        if jacobian.ndim != 3 or jacobian.shape[1] != 6:
            raise RuntimeError(f"unexpected Jacobian shape {tuple(jacobian.shape)}")
        linear = jacobian[:, :3, :].double()
        gram = linear @ linear.transpose(1, 2)

        determinant = torch.linalg.det(gram).clamp_min(0.0)
        manipulability = torch.sqrt(determinant)

        unit = self.direction.double().unsqueeze(0).expand(gram.shape[0], 3).unsqueeze(-1)
        # Solve rather than invert: the Gram matrix is near-singular exactly where this
        # measurement matters, and an explicit inverse would lose the answer there.
        try:
            solved = torch.linalg.solve(gram + 1e-12 * torch.eye(3, dtype=gram.dtype, device=gram.device), unit)
            quadratic = (unit.transpose(1, 2) @ solved).squeeze(-1).squeeze(-1).clamp_min(1e-18)
            transmission = 1.0 / torch.sqrt(quadratic)
        except RuntimeError:
            transmission = torch.full((gram.shape[0],), float("nan"), dtype=gram.dtype, device=gram.device)

        singular = torch.linalg.svdvals(linear)
        condition = singular[:, 0] / singular[:, -1].clamp_min(1e-12)
        return {
            "manipulability": manipulability.cpu().numpy(),
            "pull_axis_transmission": transmission.cpu().numpy(),
            "jacobian_condition": condition.cpu().numpy(),
        }


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    forces = force_grid(args_cli.force_low, args_cli.force_high, args_cli.force_step)
    durations = force_grid(args_cli.duration_low, args_cli.duration_high, args_cli.duration_step)
    grid = [(force, duration) for duration in durations for force in forces]
    states = representative_hidden_states(args_cli.num_xi, seed=args_cli.seed)
    num_envs = min(args_cli.num_envs, len(states))
    region = OperatingRegionCfg()

    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / "goal_distance_sweep.json"
    )

    print("\n" + "=" * 78)
    print(f"[reach] hidden states : {len(states)} (16 corners of the xi box first)")
    print(f"[reach] F (N)         : {forces[0]:.2f} .. {forces[-1]:.2f} step {args_cli.force_step} ({len(forces)})")
    print(f"[reach] T (s)         : {durations[0]:.2f} .. {durations[-1]:.2f} step {args_cli.duration_step} "
          f"({len(durations)})")
    print(f"[reach] episodes      : {len(grid) * len(states)}")
    print(f"[reach] drawer travel : {DRAWER_TRAVEL_LIMIT * 1000:.0f} mm (official cabinet asset)")
    print("[reach] recording     : joint margins, manipulability, pull-axis transmission, "
          "Jacobian condition, wrist spike")

    system = build_system(num_envs)
    system.verify_measured_force_available()
    posture = PostureProbe(system)
    randomizer = DynamicsRandomizer()
    records: list[LongPullRecord] = []
    probe_durations: list[float] = []

    try:
        for start in range(0, len(states), num_envs):
            batch = states[start : start + num_envs]
            padded = batch + [batch[-1]] * (num_envs - len(batch))
            parameters = [
                DynamicsParameters(
                    drawer_mass=state["mass"],
                    joint_static_friction=state["static_friction"],
                    joint_dynamic_friction=state["dynamic_friction"],
                    joint_damping=state["damping"],
                    name=f"xi{index:03d}",
                )
                for index, state in enumerate(padded)
            ]
            randomizer.apply(system.env, parameters)
            system.reset()

            task_start = system.reader.drawer_position.clone()
            probe = system.probe.run(**RECOMMENDED_PROBE_TASK.as_kwargs())
            system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
            pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
            probe_durations.extend(probe.duration[: len(batch)].tolist())
            snapshot = capture_snapshot(system, label=f"reach batch {start // num_envs}")

            order = branch_order(f"reach-{start}", len(grid))
            for position, index in enumerate(order):
                force, duration = grid[index]
                restore_snapshot(system, snapshot)
                result = system.execution.run(peak_force=force, duration=duration)
                validity = assess_validity(result, region, pre_execution_displacement=pre_execution)
                conditioning = posture.conditioning()
                history = result.history
                joints = history.joint_position[-1]
                margins, limiting = posture.joint_margins(joints)

                for env_index in range(len(batch)):
                    driven = history.active_steps(env_index)
                    wrist = history.measured_force[driven, env_index]
                    total = float(pre_execution[env_index] + result.final_displacement[env_index])
                    verdict = validity.verdicts[env_index]
                    records.append(
                        LongPullRecord(
                            xi=dict(padded[env_index]),
                            peak_force=float(force),
                            duration=float(duration),
                            final_displacement=total,
                            final_velocity=float(result.final_velocity[env_index]),
                            peak_velocity=float(result.peak_velocity[env_index]),
                            peak_measured_force=float(np.abs(wrist).max()) if wrist.size else float("nan"),
                            wrist_force_spike=float(np.abs(np.diff(wrist)).max()) if wrist.size > 1 else 0.0,
                            peak_lateral_drift=float(verdict.metrics["peak_lateral_drift"]),
                            peak_orientation_drift_deg=float(verdict.metrics["peak_orientation_drift_deg"]),
                            travel_fraction=total / DRAWER_TRAVEL_LIMIT,
                            joint_position=[float(value) for value in joints[env_index]],
                            min_joint_margin=float(margins[env_index]),
                            limiting_joint=int(limiting[env_index]),
                            manipulability=float(conditioning["manipulability"][env_index]),
                            pull_axis_transmission=float(conditioning["pull_axis_transmission"][env_index]),
                            jacobian_condition=float(conditioning["jacobian_condition"][env_index]),
                            safety_aborted=bool(result.safety_aborted[env_index]),
                            valid=bool(verdict.valid),
                            invalid_reasons=[reason.value for reason in verdict.reasons],
                        )
                    )
                if position % 30 == 0:
                    print(
                        f"[reach] batch {start // num_envs + 1} point {position + 1}/{len(grid)} "
                        f"F={force:.1f} T={duration:.2f} "
                        f"d={result.final_displacement[0] * 1000:6.1f} mm "
                        f"margin={margins[0]:.3f} manip={conditioning['manipulability'][0]:.4f} "
                        f"({time.perf_counter() - started:.0f} s)"
                    )
    finally:
        system.close()

    payload = {
        "git_commit": git_commit(),
        "drawer_travel_limit": DRAWER_TRAVEL_LIMIT,
        "forces": list(forces),
        "durations": list(durations),
        "num_hidden_states": len(states),
        "hidden_state_seed": args_cli.seed,
        "probe_task": RECOMMENDED_PROBE_TASK.as_dict(),
        "probe_durations": probe_durations,
        "operating_region": region.as_dict(),
        "environment": collect_environment_info().as_dict(),
        "records": [record.as_dict() for record in records],
    }
    output.write_text(json.dumps(payload, default=float))
    displacements = np.array([record.final_displacement for record in records])
    print("[reach]")
    print(f"[reach] episodes  : {len(records)} in {time.perf_counter() - started:.0f} s")
    print(
        f"[reach] reached   : {displacements.min() * 1000:.1f} .. {displacements.max() * 1000:.1f} mm "
        f"({displacements.max() / DRAWER_TRAVEL_LIMIT * 100:.1f} % of travel)"
    )
    print(f"[reach] valid     : {np.mean([r.valid for r in records]) * 100:.1f} %")
    print(f"[reach] written   : {output}")
    print(f"[reach] next      : python scripts/analyze_goal_distance.py --dataset {output}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()

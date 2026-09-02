"""Phase 9E -- audit every force channel against the drawer's equation of motion.

Four quantities claim to be "the pull force". This script measures all of them on the same
episodes and reports which one means what:

``commanded_force``
    What the controller asked the operational-space controller for. Deployable.
``measured_force``
    Pull-axis component of the wrist joint reaction wrench -- what a real Franka's
    force/torque sensor reports. Deployable in principle, diagnostic here.
``drawer_resistance_force``
    ``get_dof_projected_joint_forces`` on the drawer joint: its internal resistance.
    Simulator-only.
``drawer_external_force``
    ``m_total * a - resistance``, the axial force actually delivered. Simulator-only and
    privileged, because it needs the drawer's mass.

Two checks:

1. **Resistance identity.** With the hidden state known, the resistance channel must equal
   ``-(mu_d * sign(v) + b * v)`` while the drawer slides. This validates the channel.
2. **Delivered-force agreement.** ``drawer_external_force`` and ``measured_force`` must
   agree to within the hand-and-finger inertial term, since they bracket the same
   interaction from either side of the grasp.

It also reproduces the finding that made this audit necessary: a wrist force far above the
command (17-23 N against 5 N commanded) is the end-stop impact, not a control fault.

Usage::

    python scripts/audit_force_channels.py --headless
    python scripts/audit_force_channels.py --headless --skip-end-stop
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--peak-force", type=float, default=4.0, help="Execution plateau force for the audit (N).")
parser.add_argument("--duration", type=float, default=1.5, help="Execution duration for the audit (s).")
parser.add_argument(
    "--skip-end-stop", action="store_true", help="Skip the deliberate end-stop episode at the end."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402

from probe_drawer.analysis.force_channel_analysis import (  # noqa: E402
    AUDIT_CASES,
    END_STOP_CASE,
    analyse_end_stop_episode,
    analyse_force_channels,
)
from probe_drawer.envs import DynamicsRandomizer  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, project_root  # noqa: E402


def _hand_mass(system: PullSystem) -> float:
    """Mass the wrist sensor carries beyond the drawer: the hand plus both fingers (kg)."""
    robot = system.env.scene["robot"]
    masses = robot.root_physx_view.get_masses()[0]
    indices = [robot.find_bodies(name)[0][0] for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger")]
    return float(sum(masses[index] for index in indices))


def main() -> None:
    enable_unbuffered_stdout()

    system = PullSystem.build(PullSystemCfg(num_envs=len(AUDIT_CASES), device=args_cli.device))
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()

    parameters = [case.parameters for case in AUDIT_CASES]
    system.reset()
    applied = randomizer.apply(system.env, parameters)
    result = system.execution.run(peak_force=args_cli.peak_force, duration=args_cli.duration)
    report = analyse_force_channels(result, parameters, applied, hand_mass=_hand_mass(system))

    print("\n" + "=" * 78)
    print(f"[force-audit] execution        : peak_force={args_cli.peak_force} N duration={args_cli.duration} s")
    print(f"[force-audit] readback ok      : {applied.consistent} (mirror agrees: {applied.mirror_agrees})")
    print("[force-audit]")
    print("[force-audit] mean over the sliding window, per case (N):")
    header = (
        f"{'case':>22} {'F_cmd':>8} {'F_wrist':>9} {'F_resist':>9} {'F_extern':>9} "
        f"{'-(mu+bv)':>10} {'resid_res':>10} {'|ext-wrist|':>12}"
    )
    print("[force-audit] " + header)
    for row in report["cases"]:
        if "error" in row:
            print(f"[force-audit] {row['name']:>22} -- {row['error']}")
            continue
        print(
            f"[force-audit] {row['name']:>22} {row['commanded_force']:8.3f} {row['measured_force']:9.3f} "
            f"{row['drawer_resistance_force']:9.3f} {row['drawer_external_force']:9.3f} "
            f"{row['predicted_resistance']:10.3f} {row['resistance_residual']:10.4f} "
            f"{row['external_minus_wrist']:12.4f}"
        )
    print("[force-audit]")
    print(
        f"[force-audit] resistance identity : max mean-residual = {report['max_resistance_residual']:.4f} N "
        f"(tolerance {report['resistance_tolerance']:.3f}); worst per-step "
        f"{report['max_resistance_residual_per_step']:.4f} N -- see the module docstring"
    )
    print(
        f"[force-audit] delivered agreement  : max |external - wrist| = "
        f"{report['max_external_wrist_gap']:.4f} N  (hand+finger inertia bound "
        f"{report['hand_inertia_bound']:.4f} N)"
    )
    print(f"[force-audit] command share        : F_external / F_cmd = {report['command_share']}")
    print(f"[force-audit] resistance check     : {report['resistance_verdict']}")
    print(f"[force-audit] delivered check      : {report['delivered_verdict']}")

    if not args_cli.skip_end_stop:
        system.reset()
        randomizer.apply(system.env, END_STOP_CASE.parameters)
        end_stop = system.execution.run(peak_force=END_STOP_CASE.peak_force, duration=END_STOP_CASE.duration)
        report["end_stop"] = analyse_end_stop_episode(end_stop)
        print("[force-audit]")
        print(
            f"[force-audit] end-stop episode    : {END_STOP_CASE.name}, "
            f"peak_force={END_STOP_CASE.peak_force} N duration={END_STOP_CASE.duration} s"
        )
        for key, value in report["end_stop"].items():
            print(f"[force-audit]   {key:28s}: {value}")

    output = project_root() / "outputs" / "logs" / "force_channel_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=float))
    print(f"[force-audit] report written    : {output}")
    print("=" * 78 + "\n")

    system.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

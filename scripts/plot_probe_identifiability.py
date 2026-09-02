"""Phase 9L/9M -- does the probe response separate one hidden dimension at a time?

Two figures, both needing the simulator.

``probe_identifiability.png``
    One column per hidden dimension. Each column varies *that* dimension while the other
    three stay at their mid-grid values, and shows the probe's commanded force,
    displacement, velocity and acceleration. The expectation, stated before looking, is:
    static friction shows up in when the drawer breaks away, dynamic friction in the speed
    it settles to afterwards, damping in the velocity-dependent part of that, and mass in
    the acceleration transient. Whether that expectation survives contact with the data is
    the point of the figure.

``force_channel_comparison.png``
    All four force channels on one execution, plus the end-stop episode that explains a
    wrist force far above the command.

Usage::

    python scripts/plot_probe_identifiability.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--initial-force", type=float, default=1.0, help="Calibrated probe ramp start (N).")
parser.add_argument("--max-force", type=float, default=6.0, help="Calibrated probe ramp end (N).")
parser.add_argument("--target-displacement", type=float, default=0.003, help="Calibrated probe stop (m).")
parser.add_argument("--max-velocity", type=float, default=0.08, help="Calibrated probe velocity stop (m/s).")
parser.add_argument("--execution-force", type=float, default=4.0, help="Force for the force-channel figure (N).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probe_drawer.analysis.force_channel_analysis import END_STOP_CASE  # noqa: E402
from probe_drawer.analysis.probe_features import BREAKAWAY_SPEED, extract_features  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, project_root  # noqa: E402

#: Mid-grid baseline; each sweep varies one field away from it.
BASELINE = dict(drawer_mass=8.0, joint_static_friction=1.25, joint_dynamic_friction=0.8125, joint_damping=6.0)

#: One sweep per hidden dimension. ``joint_dynamic_friction`` is capped by the static value,
#: so the static sweep raises both together to keep every point writable by PhysX.
SWEEPS = {
    "drawer_mass  m [kg]": ("drawer_mass", (4.0, 6.0, 9.0, 12.0)),
    "static friction  $\\mu_s$ [N]": ("joint_static_friction", (0.5, 1.25, 2.0, 3.0)),
    "dynamic friction  $\\mu_d$ [N]": ("joint_dynamic_friction", (0.15, 0.5, 0.9, 1.25)),
    "damping  b [N s/m]": ("joint_damping", (2.0, 5.0, 8.0, 11.0)),
}


def build_cases() -> tuple[list[DynamicsParameters], list[tuple[str, int]]]:
    """One hidden state per (dimension, level), plus an index so the plot can group them."""
    cases: list[DynamicsParameters] = []
    index: list[tuple[str, int]] = []
    for title, (field, levels) in SWEEPS.items():
        for level in levels:
            values = dict(BASELINE)
            values[field] = level
            if field == "joint_static_friction":
                # Keep mu_d below the swept mu_s: PhysX discards a write with mu_d > mu_s.
                values["joint_dynamic_friction"] = min(BASELINE["joint_dynamic_friction"], level)
            cases.append(DynamicsParameters(**values, name=f"{field}={level:g}"))
            index.append((title, len(cases) - 1))
    return cases, index


def plot_identifiability(result, cases, index, path):
    """Four rows of probe signals, one column per hidden dimension."""
    rows = (
        ("commanded_force", "$F_{cmd}$ [N]", 1.0),
        ("drawer_position", "d [mm]", 1000.0),
        ("drawer_velocity", "v [m/s]", 1.0),
        ("drawer_acceleration", "a [m/s$^2$]", 1.0),
    )
    titles = list(SWEEPS)
    figure, axes = plt.subplots(len(rows), len(titles), figsize=(4.0 * len(titles), 9.5), constrained_layout=True)

    for column, title in enumerate(titles):
        field, levels = SWEEPS[title]
        members = [env for name, env in index if name == title]
        for row_index, (channel, ylabel, scale) in enumerate(rows):
            axis = axes[row_index, column]
            for env, level in zip(members, levels, strict=True):
                driven = result.history.active_steps(env)
                axis.plot(
                    result.history.time[driven],
                    result.history.channel(channel, env) * scale,
                    linewidth=1.3,
                    label=f"{level:g}",
                )
            if channel == "drawer_velocity":
                axis.axhline(BREAKAWAY_SPEED, color="k", linestyle=":", linewidth=0.8)
            axis.grid(alpha=0.3)
            axis.set_ylabel(ylabel if column == 0 else "")
            if row_index == len(rows) - 1:
                axis.set_xlabel("time since probe start [s]")
        axes[0, column].set_title(title, fontsize=10)
        axes[0, column].legend(fontsize=7, title="level", title_fontsize=7)

    baseline = ", ".join(f"{key.replace('joint_', '').replace('drawer_', '')}={value:g}" for key, value in BASELINE.items())
    figure.suptitle(f"Probe response, one hidden dimension at a time (baseline {baseline})", fontsize=11)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_force_channels(normal, end_stop, path):
    """All four force channels, on a clean execution and on an end-stop impact."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    for axis, result, title in (
        (axes[0], normal, f"clean execution, $F_{{peak}}$={args_cli.execution_force:g} N"),
        (axes[1], end_stop, f"end-stop impact, $F_{{peak}}$={END_STOP_CASE.peak_force:g} N"),
    ):
        driven = result.history.active_steps(0)
        time = result.history.time[driven]
        for channel, label in (
            ("commanded_force", "$F_{cmd}$ (deployable)"),
            ("measured_force", "$F_{wrist}$ (diagnostic)"),
            ("drawer_external_force", "$F_{drawer}$ (privileged)"),
            ("drawer_resistance_force", "$F_{resistance}$ (privileged)"),
        ):
            axis.plot(time, result.history.channel(channel, 0), linewidth=1.2, label=label)
        twin = axis.twinx()
        twin.plot(time, result.history.channel("drawer_position", 0) * 1000, "k--", linewidth=0.9, alpha=0.6)
        twin.set_ylabel("d [mm]", color="k")
        axis.set(xlabel="time since pull start [s]", ylabel="force [N]", title=title)
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.suptitle("Force channels compared; dashed line is drawer displacement", fontsize=11)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    enable_unbuffered_stdout()
    plots = project_root() / "outputs" / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    cases, index = build_cases()
    system = PullSystem.build(PullSystemCfg(num_envs=len(cases), device=args_cli.device))
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    written = []
    try:
        system.reset()
        randomizer.apply(system.env, cases)
        probe = system.probe.run(
            initial_force=args_cli.initial_force,
            max_force=args_cli.max_force,
            target_displacement=args_cli.target_displacement,
            max_velocity=args_cli.max_velocity,
        )
        written.append(plot_identifiability(probe, cases, index, plots / "probe_identifiability.png"))

        print("[plot] probe features per swept dimension:")
        for title in SWEEPS:
            field, levels = SWEEPS[title]
            members = [env for name, env in index if name == title]
            print(f"[plot]   {title}")
            for env, level in zip(members, levels, strict=True):
                features = extract_features(probe, env)
                print(
                    f"[plot]     {field}={level:<6g} breakaway {features.breakaway_time:.3f} s at "
                    f"{features.breakaway_force:.2f} N | duration {features.duration:.3f} s | "
                    f"mean speed {features.mean_speed_after_breakaway:.4f} m/s | "
                    f"peak a {features.peak_acceleration:.3f} m/s^2"
                )

        # Isaac Sim allows one simulation context per process, so the force-channel figure
        # reuses this system rather than building a second one: the same hidden state is
        # applied to every environment and only environment 0 is plotted.
        system.reset()
        randomizer.apply(system.env, DynamicsParameters(**BASELINE, name="baseline"))
        normal = system.execution.run(peak_force=args_cli.execution_force, duration=1.5)
        system.reset()
        randomizer.apply(system.env, END_STOP_CASE.parameters)
        end_stop = system.execution.run(peak_force=END_STOP_CASE.peak_force, duration=END_STOP_CASE.duration)
        written.append(plot_force_channels(normal, end_stop, plots / "force_channel_comparison.png"))
    finally:
        system.close()

    for path in written:
        print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

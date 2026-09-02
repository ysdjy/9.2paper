"""Plot the response curves of logged episodes. Does not need Isaac Sim.

Reads the ``metadata.json`` / ``trajectory.npz`` pairs written by
:class:`~probe_drawer.logging.EpisodeLogger` and writes PNGs to ``outputs/plots/``.
Curves are labelled with each environment's dynamics preset, so a multi-environment
episode (as produced by ``scripts/test_dynamics_randomization.py``) comes out as a
preset comparison without any extra bookkeeping.

Usage::

    python scripts/visualize_response.py --episode probe_default
    python scripts/visualize_response.py --episode dynamics_probe_presets dynamics_execution_presets
    python scripts/visualize_response.py --profile-invariance
    python scripts/visualize_response.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probe_drawer.controllers.force_profiles import TrapezoidForceProfile  # noqa: E402
from probe_drawer.logging import default_log_root  # noqa: E402
from probe_drawer.utils import project_root  # noqa: E402

#: Signals plotted for every episode: (array name, y-axis label, unit scale, file suffix).
SIGNALS: tuple[tuple[str, str, float, str], ...] = (
    ("commanded_force", "commanded pull force [N]", 1.0, "force"),
    ("measured_force", "measured pull force at wrist [N]", 1.0, "measured_force"),
    ("drawer_position", "drawer displacement [mm]", 1000.0, "displacement"),
    ("drawer_velocity", "drawer velocity [m/s]", 1.0, "velocity"),
    ("tcp_lateral_error", "TCP lateral drift [mm]", 1000.0, "lateral_error"),
)


def plots_dir() -> Path:
    directory = project_root() / "outputs" / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def env_labels(metadata: dict, num_envs: int) -> list[str]:
    """One legend label per environment, using its dynamics preset where known."""
    requested = metadata.get("dynamics_parameters", {}).get("requested", [])
    if len(requested) == num_envs:
        return [
            f"{p.get('name', f'env {i}')} "
            f"(m={p['drawer_mass']:g}, mu_s={p['joint_static_friction']:g}, "
            f"mu_d={p['joint_dynamic_friction']:g}, b={p['joint_damping']:g})"
            for i, p in enumerate(requested)
        ]
    return [f"env {i}" for i in range(num_envs)]


def plot_episode(episode: str, log_root: Path) -> list[Path]:
    """Write one PNG per signal for one logged episode."""
    directory = log_root / episode
    metadata = json.loads((directory / "metadata.json").read_text())
    arrays = np.load(directory / "trajectory.npz")

    time = arrays["time"]
    labels = env_labels(metadata, arrays["commanded_force"].shape[1])
    controller = metadata.get("controller", "unknown")

    written: list[Path] = []
    for name, ylabel, scale, suffix in SIGNALS:
        if name not in arrays:
            continue
        figure, axes = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        for env_index, label in enumerate(labels):
            axes.plot(time, arrays[name][:, env_index] * scale, label=label, linewidth=1.4)
        axes.set_xlabel("time since pull start [s]")
        axes.set_ylabel(ylabel)
        axes.set_title(f"{episode} -- {controller}")
        axes.grid(alpha=0.3)
        if len(labels) > 1 or suffix == "force":
            axes.legend(fontsize=8)
        path = plots_dir() / f"{episode}_{suffix}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def plot_profile_invariance(peak_forces: tuple[float, ...] = (5.0, 10.0, 15.0)) -> list[Path]:
    """Show that the execution force profile's shape does not depend on ``peak_force``.

    Two panels: the raw commands, which differ, and the same commands divided by their own
    peak force, which must coincide exactly.
    """
    duration = 2.0
    time = np.linspace(0.0, duration, 1001)

    figure, (raw_axes, normalised_axes) = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for peak in peak_forces:
        force = np.asarray(TrapezoidForceProfile(peak_force=peak, duration=duration).force(time))
        raw_axes.plot(time, force, label=f"F_peak = {peak:g} N", linewidth=1.4)
        normalised_axes.plot(time / duration, force / peak, label=f"F_peak = {peak:g} N", linewidth=1.4)

    raw_axes.set(xlabel="time [s]", ylabel="commanded force [N]", title="F(t)")
    normalised_axes.set(xlabel="t / T", ylabel="F(t) / F_peak", title="phi(t/T) -- curves must coincide")
    for axes in (raw_axes, normalised_axes):
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)

    path = plots_dir() / "execution_profile_invariance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=str, nargs="*", default=[], help="Episode directory name(s).")
    parser.add_argument("--all", action="store_true", help="Plot every episode found under outputs/logs/.")
    parser.add_argument(
        "--profile-invariance", action="store_true", help="Also plot the execution profile invariance figure."
    )
    parser.add_argument("--log-root", type=str, default=None, help="Override outputs/logs/.")
    args = parser.parse_args()

    log_root = Path(args.log_root) if args.log_root else default_log_root()
    episodes = list(args.episode)
    if args.all:
        episodes = sorted(p.name for p in log_root.iterdir() if (p / "trajectory.npz").is_file())
    if not episodes and not args.profile_invariance:
        parser.error("Nothing to plot: pass --episode, --all, or --profile-invariance.")

    written: list[Path] = []
    for episode in episodes:
        if not (log_root / episode / "trajectory.npz").is_file():
            raise FileNotFoundError(f"No trajectory at {log_root / episode / 'trajectory.npz'}.")
        written += plot_episode(episode, log_root)
    if args.profile_invariance:
        written += plot_profile_invariance()

    for path in written:
        print(f"[visualize] wrote {path}")


if __name__ == "__main__":
    main()

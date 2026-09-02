"""Compatibility and environment-introspection helpers.

This module is the single place where the project asks *"what is actually installed
on this machine?"*.  Nothing here may hard-code an absolute path: every location is
derived either from an installed package's ``__file__`` or from this file's own
position inside the repository.

The module is deliberately import-safe **without** Isaac Sim running: every
Isaac-Lab-dependent lookup is wrapped and degrades to ``None``.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "EnvironmentInfo",
    "collect_environment_info",
    "enable_unbuffered_stdout",
    "git_commit",
    "isaaclab_root",
    "package_version",
    "project_root",
]

# Environment IDs this project depends on or compares against.
OFFICIAL_DRAWER_ENV_IDS: tuple[str, ...] = (
    "Isaac-Open-Drawer-Franka-v0",
    "Isaac-Open-Drawer-Franka-Play-v0",
    "Isaac-Open-Drawer-Franka-IK-Abs-v0",
    "Isaac-Open-Drawer-Franka-IK-Rel-v0",
    "Isaac-Franka-Cabinet-Direct-v0",
)


def enable_unbuffered_stdout() -> None:
    """Make ``print`` line-buffered, so output survives Isaac Sim shutting down.

    ``SimulationApp.close()`` terminates the process without flushing Python's buffers, so
    anything a script printed after launching the app is otherwise lost when stdout is a
    pipe or a file.  Every script in ``scripts/`` calls this right after the app launch.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass


def project_root() -> Path:
    """Absolute path of the repository root (the directory holding ``pyproject.toml``)."""
    # src/probe_drawer/utils/isaaclab_compat.py -> repo root is 4 levels up.
    return Path(__file__).resolve().parents[3]


def package_version(name: str) -> str | None:
    """Return ``<pkg>.__version__`` if the package is importable, else ``None``."""
    try:
        module = importlib.import_module(name)
    except Exception:
        # Deliberately broad: this is best-effort introspection for a report. A package can
        # fail to import for many reasons other than being absent, and none of them should
        # stop an experiment from running.
        return None
    return str(getattr(module, "__version__", "unknown"))


def isaaclab_root() -> Path | None:
    """Root of the Isaac Lab *source checkout* that supplies the ``isaaclab`` package.

    Returns ``None`` when ``isaaclab`` is not importable.  When Isaac Lab is installed
    from a source checkout the returned path is the repository root; for a wheel install
    it is the site-packages directory that contains the package.
    """
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return None
    # <root>/source/isaaclab/isaaclab/__init__.py
    package_dir = Path(spec.origin).resolve().parent
    for candidate in package_dir.parents:
        if (candidate / "VERSION").is_file() and (candidate / "source").is_dir():
            return candidate
    return package_dir.parent


def git_commit(repo: Path | None = None, short: bool = False) -> str | None:
    """Current git commit SHA of ``repo`` (defaults to this project), or ``None``."""
    repo = repo or project_root()
    cmd = ["git", "-C", str(repo), "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        # No commits yet (fresh repository), or not a git repository at all.
        return None
    return out.stdout.strip() or None


def _isaacsim_version() -> str | None:
    """Isaac Sim version string, read from the installed ``isaacsim`` distribution."""
    try:
        from importlib.metadata import version

        return version("isaacsim")
    except Exception:
        # No distribution metadata (e.g. a source install); fall back to the module.
        return package_version("isaacsim")


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        # No torch, no CUDA, or no visible device: the report simply omits the GPU.
        pass
    return None


@dataclass
class EnvironmentInfo:
    """Snapshot of the software/hardware stack an experiment ran on."""

    os: str
    python: str
    python_executable: str
    torch: str | None
    cuda: str | None
    gpu: str | None
    isaacsim: str | None
    isaaclab: str | None
    isaaclab_root: str | None
    project_root: str
    project_git_commit: str | None
    registered_drawer_envs: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def collect_environment_info(check_gym_registry: bool = False) -> EnvironmentInfo:
    """Gather the version information recorded with every experiment.

    Args:
        check_gym_registry: Also probe the gymnasium registry for the official drawer
            environment IDs.  This requires ``isaaclab_tasks`` to be importable, which in
            turn requires the Isaac Sim app to have been launched first.
    """
    torch_version, cuda_version = None, None
    try:
        import torch

        torch_version, cuda_version = torch.__version__, torch.version.cuda
    except Exception:
        # Reported as None rather than raising: the caller wants a report, not a failure.
        pass

    registered: dict[str, bool] = {}
    if check_gym_registry:
        try:
            import gymnasium as gym

            import isaaclab_tasks  # noqa: F401  (import registers the environments)

            known = set(gym.registry.keys())
            registered = {env_id: env_id in known for env_id in OFFICIAL_DRAWER_ENV_IDS}
        except Exception:
            # `isaaclab_tasks` needs the Isaac Sim app to be running. An empty mapping means
            # "not probed", which the caller distinguishes from "probed and absent".
            registered = {}

    lab_root = isaaclab_root()
    return EnvironmentInfo(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        python_executable=sys.executable,
        torch=torch_version,
        cuda=cuda_version,
        gpu=_gpu_name(),
        isaacsim=_isaacsim_version(),
        isaaclab=package_version("isaaclab"),
        isaaclab_root=str(lab_root) if lab_root else None,
        project_root=str(project_root()),
        project_git_commit=git_commit(),
        registered_drawer_envs=registered,
    )

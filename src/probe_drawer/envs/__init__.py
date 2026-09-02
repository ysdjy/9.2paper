"""Environment configuration, research initialisation and dynamics randomisation.

:mod:`probe_drawer.envs.initialization`, :mod:`probe_drawer.envs.dynamics_randomization` and
:mod:`probe_drawer.envs.hybrid_pull_cfg` do not need the Isaac Sim application and are
exported eagerly.
:class:`~probe_drawer.envs.ProbeDrawerEnvCfg` is built out of Isaac Lab config classes and is
therefore resolved lazily (PEP 562), so importing this package does not require the Isaac Sim
application to be running.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .dynamics_randomization import (
    PRESETS,
    REFERENCE_DURATION,
    REFERENCE_PEAK_FORCE,
    AppliedDynamics,
    DynamicsParameters,
    DynamicsRandomizer,
    DynamicsRandomizerCfg,
    preset,
)
from .hybrid_pull_cfg import HybridPullControlCfg
from .initialization import GraspConfiguration, default_grasp_pose_path, load_grasp_configuration

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .drawer_env_cfg import ProbeDrawerEnvCfg

_LAZY: dict[str, str] = {
    "ProbeDrawerEnvCfg": "drawer_env_cfg",
}

__all__ = [
    "PRESETS",
    "REFERENCE_DURATION",
    "REFERENCE_PEAK_FORCE",
    "AppliedDynamics",
    "DynamicsParameters",
    "DynamicsRandomizer",
    "DynamicsRandomizerCfg",
    "GraspConfiguration",
    "HybridPullControlCfg",
    "ProbeDrawerEnvCfg",
    "default_grasp_pose_path",
    "load_grasp_configuration",
    "preset",
]


def __getattr__(name: str):
    """Resolve the Isaac-Lab-dependent configuration classes on first use."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

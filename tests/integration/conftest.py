"""Session fixtures for the integration tests.

Launching Isaac Sim costs roughly 40 s, so the whole integration suite shares one
application and one environment.  Every test resets the environment before it runs, so
tests stay independent despite sharing the simulation.

Run them with::

    python -m pytest tests/integration -q
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from probe_drawer.utils import enable_unbuffered_stdout

# ``SimulationApp.close()`` ends the process without flushing, so a block-buffered stdout
# (which is what a pipe or a redirect to a file gives) would lose pytest's own report.
enable_unbuffered_stdout()

#: Environments the shared system runs, one per dynamics preset.
PRESET_ORDER = ("easy", "medium", "hard")


#: The Isaac Sim application, shared by the whole session. Closed in ``pytest_unconfigure``
#: rather than in a fixture teardown: ``SimulationApp.close()`` terminates the process
#: without flushing, so closing it any earlier discards pytest's own report.
_APP: Any = None


@pytest.fixture(scope="session")
def preset_order() -> tuple[str, ...]:
    """Dynamics preset assigned to each environment of the shared system, in order."""
    return PRESET_ORDER


@pytest.fixture(scope="session")
def simulation_app() -> Any:
    """Launch Isaac Sim headless for the whole session."""
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher

        _APP = AppLauncher(headless=True).app
    return _APP


def pytest_unconfigure(config: pytest.Config) -> None:  # noqa: ARG001
    """Shut Isaac Sim down after pytest has finished writing its report."""
    sys.stdout.flush()
    sys.stderr.flush()
    global _APP, _SYSTEM
    if _SYSTEM is not None:
        _SYSTEM.close()
        _SYSTEM = None
    if _APP is not None:
        _APP.close()
        _APP = None


_SYSTEM: Any = None


@pytest.fixture(scope="session")
def pull_system(simulation_app: Any) -> Any:
    """One built :class:`~probe_drawer.pull_system.PullSystem` with three environments.

    Torn down in ``pytest_unconfigure`` alongside the application, for the same reason:
    closing the environment during fixture teardown ends the process before pytest has
    written its report.
    """
    global _SYSTEM
    if _SYSTEM is None:
        from probe_drawer.pull_system import PullSystem, PullSystemCfg

        _SYSTEM = PullSystem.build(PullSystemCfg(num_envs=len(PRESET_ORDER)))
        _SYSTEM.verify_measured_force_available()
    return _SYSTEM


@pytest.fixture
def randomizer() -> Any:
    from probe_drawer.envs import DynamicsRandomizer

    return DynamicsRandomizer(seed=0)


@pytest.fixture
def uniform_system(pull_system: Any, randomizer: Any):
    """Reset the shared system and give every environment the same dynamics preset.

    Returns a callable ``prepare(preset_name) -> AppliedDynamics`` so a test can pick the
    preset it needs and still get a freshly reset environment.
    """
    from probe_drawer.envs import preset

    def prepare(preset_name: str = "medium"):
        pull_system.reset()
        return randomizer.apply(pull_system.env, preset(preset_name))

    return prepare

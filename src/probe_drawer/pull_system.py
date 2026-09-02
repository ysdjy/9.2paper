"""Assembly of the pull system: one environment, one hybrid OSC, two controllers.

Every experiment script needs the same four objects wired together in the same order.
That wiring lives here once, so scripts stay short and cannot drift apart in how they
configure the environment.

::

    PullSystem
      env         ManagerBasedRLEnv (ProbeDrawerEnvCfg)
      reader      DrawerStateReader
      osc         HybridPullOSC            <- shared by both controllers
      probe       ProbePullController
      execution   ExecutionPullController

Importing this module imports Isaac Lab, so it must happen after the Isaac Sim application
has been launched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from probe_drawer.controllers import (
    ExecutionControllerCfg,
    ExecutionPullController,
    HybridPullOSC,
    HybridPullOSCCfg,
    ProbeControllerCfg,
    ProbePullController,
    SafetyLimits,
)
from probe_drawer.envs import HybridPullControlCfg, ProbeDrawerEnvCfg
from probe_drawer.sensors import DrawerStateCfg, DrawerStateReader, PullAxis

__all__ = ["ENV_ID", "PullSystem", "PullSystemCfg"]

#: Gym ID this project's environment is registered under.
ENV_ID = "Probe-Drawer-Franka-v0"

gym.register(
    id=ENV_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "probe_drawer.envs.drawer_env_cfg:ProbeDrawerEnvCfg"},
    disable_env_checker=True,
)


@dataclass
class PullSystemCfg:
    """Everything a script may vary about the pull system.

    Args:
        num_envs: Parallel environments. Controllers are vectorised, so a single call to
            ``probe.run`` or ``execution.run`` probes/executes all of them at once.
        device: Torch/PhysX device.
        episode_length_s: Environment time-out. Only a backstop -- the controllers manage
            their own durations.
        warm_up_steps: Settle steps run and discarded at build time, so the first episode
            starts from the same contact state as every later one.
        video_folder: When set, the environment is wrapped in ``RecordVideo`` and rendered.
            Recording is started and stopped explicitly via
            :meth:`PullSystem.start_recording` / :meth:`PullSystem.stop_recording`, not by a
            step trigger, so that the warm-up and the reset are not part of the clip.
        video_name_prefix: File name prefix for the recording.
        hybrid: Hybrid OSC gains and pull-axis definition.
        probe: Fixed probe character.
        execution: Fixed execution profile shape.
        safety: Absolute limits shared by both controllers.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    episode_length_s: float = 30.0

    #: Settle steps run and discarded when the system is built. See ``PullSystem._warm_up``.
    warm_up_steps: int = 60

    video_folder: Path | str | None = None
    video_name_prefix: str = "pull"

    hybrid: HybridPullControlCfg = field(default_factory=HybridPullControlCfg)
    probe: ProbeControllerCfg = field(default_factory=ProbeControllerCfg)
    execution: ExecutionControllerCfg = field(default_factory=ExecutionControllerCfg)
    safety: SafetyLimits = field(default_factory=SafetyLimits)


class PullSystem:
    """A running environment with both pull controllers attached.

    Construct with :meth:`build`; the initialiser takes already-built parts and is used by
    tests that want to substitute one of them.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        wrapped_env: gym.Env,
        reader: DrawerStateReader,
        osc: HybridPullOSC,
        probe: ProbePullController,
        execution: ExecutionPullController,
        cfg: PullSystemCfg,
    ) -> None:
        self.env = env
        self.wrapped_env = wrapped_env
        self.reader = reader
        self.osc = osc
        self.probe = probe
        self.execution = execution
        self.cfg = cfg

    @classmethod
    def build(cls, cfg: PullSystemCfg | None = None) -> PullSystem:
        """Create the environment and both controllers from ``cfg``."""
        cfg = cfg or PullSystemCfg()

        # `hybrid_pull` and `research_episode_length_s` are consumed by __post_init__, so
        # they must be passed to the constructor rather than assigned afterwards.
        env_cfg = ProbeDrawerEnvCfg(
            hybrid_pull=cfg.hybrid, research_episode_length_s=cfg.episode_length_s
        )
        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.sim.device = cfg.device

        render_mode = "rgb_array" if cfg.video_folder is not None else None
        wrapped_env = gym.make(ENV_ID, cfg=env_cfg, render_mode=render_mode)
        if cfg.video_folder is not None:
            wrapped_env = gym.wrappers.RecordVideo(
                wrapped_env,
                video_folder=str(cfg.video_folder),
                name_prefix=cfg.video_name_prefix,
                # Never trigger automatically: `RecordVideo.reset` discards the frames
                # captured so far, so an automatic trigger would lose everything recorded
                # before the caller's reset. PullSystem starts the recorder instead.
                step_trigger=lambda step: False,
                video_length=0,
                disable_logger=True,
            )
        env: ManagerBasedRLEnv = wrapped_env.unwrapped  # type: ignore[assignment]

        pull_axis: PullAxis = cfg.hybrid.pull_axis()
        reader = DrawerStateReader(env, pull_axis, DrawerStateCfg())
        # The OSC steps the *wrapped* environment, so wrappers see every step.
        osc = HybridPullOSC(env, reader, pull_axis, HybridPullOSCCfg(), stepper=wrapped_env)
        probe = ProbePullController(env, osc, reader, cfg.probe, cfg.safety)
        execution = ExecutionPullController(env, osc, reader, cfg.execution, cfg.safety)

        system = cls(env, wrapped_env, reader, osc, probe, execution, cfg)
        system._warm_up()
        return system

    def _warm_up(self) -> None:
        """Run and discard one settle, so the first real episode matches later ones.

        Measured: without this, the first probe after building the system terminated 7 %
        earlier than every subsequent identical probe, because PhysX had not yet reached a
        steady contact state at the grasp. Discarding one settle removes the discrepancy.
        """
        self.reset()
        self.osc.settle(self.cfg.warm_up_steps)
        self.reset()

    @property
    def pull_axis(self) -> PullAxis:
        return self.reader.pull_axis

    @property
    def step_dt(self) -> float:
        """Environment step time in seconds."""
        return float(self.env.step_dt)

    def reset(self) -> None:
        """Reset every environment to the recorded grasp configuration."""
        self.wrapped_env.reset()
        self.osc.reset()

    @property
    def is_recording_enabled(self) -> bool:
        """Whether this system was built with video recording."""
        return self.cfg.video_folder is not None

    def start_recording(self) -> None:
        """Begin a video clip. No-op unless the system was built with ``video_folder``."""
        if self.is_recording_enabled:
            self.wrapped_env.start_video_recorder()  # type: ignore[attr-defined]

    def stop_recording(self) -> None:
        """Finish and write the current video clip. No-op when recording is disabled."""
        if self.is_recording_enabled:
            self.wrapped_env.close_video_recorder()  # type: ignore[attr-defined]

    def close(self) -> None:
        self.stop_recording()
        self.wrapped_env.close()

    def verify_measured_force_available(self) -> None:
        """Fail loudly if the handle contact sensor is missing.

        Without it ``measured_force`` would silently be zero everywhere, which is exactly
        the kind of quiet degradation that would make probe histories useless.
        """
        if not self.reader.has_force_measurement:
            raise RuntimeError(
                "The environment has no handle contact sensor, so no measured pull force is "
                "available. Check ProbeDrawerEnvCfg._add_handle_contact_sensor."
            )

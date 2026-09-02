"""Task-space state machine that brings a Franka from rest to a firm grasp on the handle.

Scope of this module: *getting hold of the drawer handle*.  Everything that happens after
the grasp -- probing and force-driven execution -- belongs to
:mod:`probe_drawer.controllers` and is deliberately not modelled here.

The waypoints reproduce the semantics of Isaac Lab's own cabinet state machine
(``<isaaclab>/scripts/environments/state_machine/open_cabinet_sm.py``, class
``OpenDrawerSm``): approach a point 10 cm in front of the handle, move onto the handle,
close the gripper, then hold.  The offsets are cross-referenced in
``docs/OFFICIAL_BASELINE.md``.  This is a plain-``torch`` reimplementation rather than a
copy of the official ``warp`` kernel because (a) this project needs a ``SETTLE``/``READY``
phase the official machine does not have, and (b) we need the state machine importable as
a library, which the official script -- which launches Isaac Sim at import time -- is not.

The optional :attr:`GraspStateMachineCfg.open_drawer_duration` phase exists purely to
replicate the official *motion-driven* drawer opening for baseline validation.  It is not
part of the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

__all__ = ["DrawerGraspStateMachine", "GraspPhase", "GraspStateMachineCfg", "GripperCommand"]


class GraspPhase(IntEnum):
    """Sequential phases of the approach-and-grasp manoeuvre."""

    REST = 0
    APPROACH_INFRONT_HANDLE = 1
    APPROACH_HANDLE = 2
    CLOSE_GRIPPER = 3
    SETTLE = 4
    READY = 5
    OPEN_DRAWER = 6
    """Motion-driven opening. Only entered when ``replicate_official_open`` is set."""


class GripperCommand:
    """Binary gripper action values expected by ``BinaryJointPositionActionCfg``."""

    OPEN = 1.0
    CLOSE = -1.0


@dataclass
class GraspStateMachineCfg:
    """Timing and geometry of the approach-and-grasp manoeuvre.

    Offsets are expressed in the *robot base frame*, matching the official state machine,
    which adds its offsets to the handle position without rotating them.

    Args:
        rest_duration: Dwell at the reset pose before moving (s).
        approach_infront_duration: Dwell once the pre-grasp waypoint is reached (s).
        approach_handle_duration: Dwell once the handle pose is reached (s).
        close_gripper_duration: Time given to the fingers to close on the handle (s).
        settle_duration: Quiet time after closing, so contact forces damp out before a
            probe starts (s).
        approach_offset: Pre-grasp waypoint relative to the handle frame origin (m).
        grasp_offset: Grasp waypoint relative to the handle frame origin (m).
        position_threshold: Distance below which a waypoint counts as reached (m).
        replicate_official_open: Append the official motion-driven ``OPEN_DRAWER`` phase.
            Used by ``scripts/run_official_drawer.py`` only.
        open_drawer_offset: Target offset that drags the handle open, per the official
            machine's ``drawer_opening_rate`` (m).
        open_drawer_duration: Duration of the motion-driven opening phase (s).
    """

    rest_duration: float = 0.3
    approach_infront_duration: float = 1.25
    approach_handle_duration: float = 1.0
    close_gripper_duration: float = 1.0
    settle_duration: float = 0.4
    approach_offset: tuple[float, float, float] = (-0.10, 0.0, 0.0)
    grasp_offset: tuple[float, float, float] = (0.025, 0.0, 0.0)
    position_threshold: float = 0.01
    replicate_official_open: bool = False
    open_drawer_offset: tuple[float, float, float] = (-0.015, 0.0, 0.0)
    open_drawer_duration: float = 3.0

    def dwell_times(self) -> dict[GraspPhase, float]:
        """Dwell time required in each phase before advancing."""
        return {
            GraspPhase.REST: self.rest_duration,
            GraspPhase.APPROACH_INFRONT_HANDLE: self.approach_infront_duration,
            GraspPhase.APPROACH_HANDLE: self.approach_handle_duration,
            GraspPhase.CLOSE_GRIPPER: self.close_gripper_duration,
            GraspPhase.SETTLE: self.settle_duration,
            GraspPhase.OPEN_DRAWER: self.open_drawer_duration,
        }

    def total_duration(self) -> float:
        """Lower bound on the time needed to reach :attr:`GraspPhase.READY` (s).

        A lower bound rather than an exact figure: the two approach phases only start
        their dwell once the waypoint is actually reached.
        """
        dwell = self.dwell_times()
        phases = [
            GraspPhase.REST,
            GraspPhase.APPROACH_INFRONT_HANDLE,
            GraspPhase.APPROACH_HANDLE,
            GraspPhase.CLOSE_GRIPPER,
            GraspPhase.SETTLE,
        ]
        return sum(dwell[p] for p in phases)


class DrawerGraspStateMachine:
    """Vectorised approach-and-grasp state machine producing IK-absolute-pose actions.

    The machine consumes the current end-effector pose and the handle pose and emits the
    action vector expected by ``Isaac-Open-Drawer-Franka-IK-Abs-v0``:
    ``[x, y, z, qw, qx, qy, qz, gripper]``.

    Args:
        cfg: Timing and geometry.
        dt: Environment step time (``sim.dt * decimation``) in seconds.
        num_envs: Number of parallel environments.
        device: Torch device the environment runs on.
    """

    def __init__(self, cfg: GraspStateMachineCfg, dt: float, num_envs: int, device: torch.device | str) -> None:
        self.cfg = cfg
        self.dt = float(dt)
        self.num_envs = int(num_envs)
        self.device = device

        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._phase_time = torch.zeros(self.num_envs, device=device)

        dwell = cfg.dwell_times()
        self._dwell = torch.zeros(len(GraspPhase), device=device)
        for phase, seconds in dwell.items():
            self._dwell[int(phase)] = seconds

        self._approach_offset = torch.tensor(cfg.approach_offset, device=device).repeat(self.num_envs, 1)
        self._grasp_offset = torch.tensor(cfg.grasp_offset, device=device).repeat(self.num_envs, 1)
        self._open_offset = torch.tensor(cfg.open_drawer_offset, device=device).repeat(self.num_envs, 1)

        # Phases whose transition additionally requires the waypoint to be reached.
        self._position_gated = (GraspPhase.APPROACH_INFRONT_HANDLE, GraspPhase.APPROACH_HANDLE)

    @property
    def phase(self) -> torch.Tensor:
        """Current phase index per environment, shape ``(num_envs,)``."""
        return self._phase

    @property
    def phase_time(self) -> torch.Tensor:
        """Time spent in the current phase per environment (s), shape ``(num_envs,)``."""
        return self._phase_time

    def is_ready(self) -> torch.Tensor:
        """Whether the grasp is complete and settled, shape ``(num_envs,)``."""
        return self._phase >= int(GraspPhase.READY)

    def all_ready(self) -> bool:
        """Whether *every* environment has reached :attr:`GraspPhase.READY`."""
        return bool(self.is_ready().all())

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Return the given environments (all of them by default) to :attr:`GraspPhase.REST`."""
        if env_ids is None:
            env_ids = slice(None)
        self._phase[env_ids] = int(GraspPhase.REST)
        self._phase_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor) -> torch.Tensor:
        """Advance the machine one environment step and return the IK-absolute action.

        Args:
            ee_pose: Current TCP pose in the environment frame, shape ``(num_envs, 7)`` as
                ``[position, quaternion(w, x, y, z)]``.
            handle_pose: Handle frame pose in the environment frame, same layout.

        Returns:
            Action tensor of shape ``(num_envs, 8)``:
            ``[position(3), quaternion(4), gripper(1)]``.
        """
        target_pos, gripper = self._targets(ee_pose, handle_pose)
        target_quat = torch.where(
            (self._phase == int(GraspPhase.REST)).unsqueeze(-1), ee_pose[:, 3:7], handle_pose[:, 3:7]
        )

        self._advance(ee_pose, target_pos)
        return torch.cat([target_pos, target_quat, gripper.unsqueeze(-1)], dim=-1)

    def _targets(self, ee_pose: torch.Tensor, handle_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-phase TCP position target and gripper command."""
        handle_pos = handle_pose[:, 0:3]
        phase = self._phase

        target_pos = ee_pose[:, 0:3].clone()
        gripper = torch.full((self.num_envs,), GripperCommand.OPEN, device=self.device)

        infront = phase == int(GraspPhase.APPROACH_INFRONT_HANDLE)
        target_pos = torch.where(infront.unsqueeze(-1), handle_pos + self._approach_offset, target_pos)

        on_handle = phase == int(GraspPhase.APPROACH_HANDLE)
        target_pos = torch.where(on_handle.unsqueeze(-1), handle_pos, target_pos)

        # From CLOSE_GRIPPER onwards the TCP target sits slightly *into* the handle so the
        # fingers stay loaded, and the gripper is commanded closed.
        holding = phase >= int(GraspPhase.CLOSE_GRIPPER)
        target_pos = torch.where(holding.unsqueeze(-1), handle_pos + self._grasp_offset, target_pos)
        gripper = torch.where(holding, torch.full_like(gripper, GripperCommand.CLOSE), gripper)

        opening = phase == int(GraspPhase.OPEN_DRAWER)
        target_pos = torch.where(opening.unsqueeze(-1), handle_pos + self._open_offset, target_pos)

        return target_pos, gripper

    def _advance(self, ee_pose: torch.Tensor, target_pos: torch.Tensor) -> None:
        """Accumulate dwell time and move environments whose phase is complete."""
        self._phase_time += self.dt

        dwell_elapsed = self._phase_time >= self._dwell[self._phase]

        gate = torch.ones_like(dwell_elapsed)
        for phase in self._position_gated:
            in_phase = self._phase == int(phase)
            reached = torch.linalg.norm(ee_pose[:, 0:3] - target_pos, dim=-1) < self.cfg.position_threshold
            gate = torch.where(in_phase, reached, gate)

        terminal = int(GraspPhase.OPEN_DRAWER) if self.cfg.replicate_official_open else int(GraspPhase.READY)
        advancing = dwell_elapsed & gate & (self._phase < terminal)

        # SETTLE leads to READY, or to OPEN_DRAWER when replicating the official baseline.
        next_phase = self._phase + 1
        if self.cfg.replicate_official_open:
            leaving_settle = self._phase == int(GraspPhase.SETTLE)
            next_phase = torch.where(leaving_settle, torch.full_like(next_phase, int(GraspPhase.OPEN_DRAWER)), next_phase)

        self._phase = torch.where(advancing, next_phase, self._phase)
        self._phase_time = torch.where(advancing, torch.zeros_like(self._phase_time), self._phase_time)

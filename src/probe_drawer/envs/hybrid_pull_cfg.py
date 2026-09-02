"""Configuration of the shared hybrid operational-space controller.

Separated from :mod:`probe_drawer.envs.drawer_env_cfg` because the gains are the control
design, not the scene, and because keeping them here means they can be inspected and
unit-tested without the Isaac Sim application running.
"""

from __future__ import annotations

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.utils import configclass

from probe_drawer.sensors import PullAxis

__all__ = ["HybridPullControlCfg"]


@configclass
class HybridPullControlCfg:
    """Tunables of the shared hybrid operational-space controller.

    The pull axis is stored as an index/sign pair rather than a vector because the
    operational-space controller can only select *coordinate axes* of its task frame for
    force control.  The default ``-x`` is the value measured empirically for the official
    cabinet layout by ``scripts/run_official_drawer.py``; it is re-verified at runtime by
    :meth:`~probe_drawer.sensors.DrawerStateReader.verify_pull_axis`.

    Args:
        pull_axis_index: Base-frame axis the drawer travels along (0=x, 1=y, 2=z).
        pull_axis_sign: ``-1`` if the drawer opens along the negative axis.
        motion_stiffness: Task-space stiffness of the five held degrees of freedom
            (N/m for translation, N m/rad for rotation).
        motion_damping_ratio: Damping ratio of the held degrees of freedom.
        nullspace_stiffness: Stiffness of the joint-space null-space controller, which
            keeps the redundant arm away from its limits without disturbing the task.
        nullspace_damping_ratio: Damping ratio of the null-space controller.
        inertial_dynamics_decoupling: Multiply the desired task-space acceleration by the
            operational-space inertia, so the held axes behave uniformly regardless of arm
            configuration.
        gravity_compensation: Add the gravity vector to the joint efforts. Left off because
            the arm spawns with ``disable_gravity=True`` for effort control.
    """

    pull_axis_index: int = 0
    pull_axis_sign: float = -1.0

    motion_stiffness: float = 500.0
    motion_damping_ratio: float = 1.0
    nullspace_stiffness: float = 20.0
    nullspace_damping_ratio: float = 1.0
    inertial_dynamics_decoupling: bool = True
    partial_inertial_dynamics_decoupling: bool = False
    gravity_compensation: bool = False

    def pull_axis(self) -> PullAxis:
        """The pull axis as a :class:`~probe_drawer.sensors.PullAxis`."""
        return PullAxis(index=self.pull_axis_index, sign=self.pull_axis_sign)

    def to_osc_cfg(self) -> OperationalSpaceControllerCfg:
        """Build the Isaac Lab operational-space controller configuration.

        ``target_types`` carries both a pose and a wrench target; the two selection masks
        then split the six task-space axes into "one force-controlled axis" and "five
        pose-held axes".  ``contact_wrench_stiffness_task`` is left at ``None``, i.e. the
        force channel is **open loop** -- there is no force feedback term, which is what
        makes ``commanded_force`` a clean, known input for the probe.
        """
        axis = self.pull_axis()
        return OperationalSpaceControllerCfg(
            target_types=["pose_abs", "wrench_abs"],
            impedance_mode="fixed",
            motion_control_axes_task=axis.motion_control_axes(),
            contact_wrench_control_axes_task=axis.wrench_control_axes(),
            contact_wrench_stiffness_task=None,
            inertial_dynamics_decoupling=self.inertial_dynamics_decoupling,
            partial_inertial_dynamics_decoupling=self.partial_inertial_dynamics_decoupling,
            gravity_compensation=self.gravity_compensation,
            motion_stiffness_task=self.motion_stiffness,
            motion_damping_ratio_task=self.motion_damping_ratio,
            nullspace_control="position",
            nullspace_stiffness=self.nullspace_stiffness,
            nullspace_damping_ratio=self.nullspace_damping_ratio,
        )

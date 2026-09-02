"""What every logged channel is, and who is allowed to consume it.

The project logs far more than the adaptation model will read. That is deliberate --
"rich logging, selective model input" (``docs/DECISIONS.md`` D019) -- but it creates a
hazard: a future agent wiring up ACE could reach for a channel that only exists because
this is a simulator, and produce a model that cannot be deployed. This module removes the
ambiguity by naming, for every channel, where the number comes from and whether a real
robot could ever have it.

Three deployability classes (``docs/DECISIONS.md`` D017):

:attr:`Deployability.DEPLOYABLE`
    A real Franka pulling a real drawer could produce this. Safe as a model input.
:attr:`Deployability.DIAGNOSTIC`
    Obtainable on real hardware in principle -- a wrist force/torque sensor, joint
    torques -- but not required by the first ACE. Recorded for ablations.
:attr:`Deployability.SIM_ONLY_PRIVILEGED`
    Only a simulator can produce it, or producing it needs the hidden state itself.
    **Never** a deployed-model input. For verification, analysis and privileged teachers.

Note the split within "drawer state": drawer displacement and its derivatives are
DEPLOYABLE because a real drawer's opening is observable (a joint encoder, a fiducial, or
the robot's own kinematics while it holds the handle). The drawer's *internal resistance*
and *external axial force* are not: they come out of PhysX.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "DEFAULT_ACE_INPUT",
    "OBSERVATION_SPECS",
    "ChannelShape",
    "ChannelSpec",
    "Deployability",
    "channels_by_deployability",
    "validate_model_input",
]


class Deployability(str, Enum):
    """Whether a real robot could ever produce this channel."""

    DEPLOYABLE = "deployable"
    DIAGNOSTIC = "diagnostic"
    SIM_ONLY_PRIVILEGED = "sim_only_privileged"


class ChannelShape(str, Enum):
    """Trailing shape of a channel, after the ``(time, environment)`` axes."""

    SCALAR = "scalar"
    VEC3 = "vec3"
    QUAT = "quat"
    JOINTS = "joints"


@dataclass(frozen=True)
class ChannelSpec:
    """Metadata for one :class:`~probe_drawer.controllers.types.PullHistory` channel.

    Args:
        name: The attribute name on ``PullHistory``.
        unit: Physical unit, or ``"-"`` for dimensionless.
        shape: Trailing shape after ``(time, environment)``.
        source: Where the number comes from, concretely enough to audit.
        deployability: Whether a real robot could produce it.
        in_default_ace_input: Whether the first ACE reads it (see :data:`DEFAULT_ACE_INPUT`).
        filtering: Description of any filter applied, or ``None`` for a raw channel.
        notes: Anything a consumer must know before using it.
    """

    name: str
    unit: str
    shape: ChannelShape
    source: str
    deployability: Deployability
    in_default_ace_input: bool
    filtering: str | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "shape": self.shape.value,
            "source": self.source,
            "deployability": self.deployability.value,
            "in_default_ace_input": self.in_default_ace_input,
            "filtering": self.filtering,
            "notes": self.notes,
        }


_CAUSAL_MA = "causal moving average of the previous-step finite difference"

#: Every channel a :class:`~probe_drawer.controllers.types.PullHistory` carries.
OBSERVATION_SPECS: dict[str, ChannelSpec] = {
    spec.name: spec
    for spec in (
        # -- bookkeeping
        ChannelSpec(
            "active",
            "-",
            ChannelShape.SCALAR,
            "controller loop",
            Deployability.DEPLOYABLE,
            False,
            notes="Whether this environment was still being driven at this step. Mask with it "
            "before analysing one environment; the tail of an early-stopping environment is padding.",
        ),
        # -- action
        ChannelSpec(
            "commanded_force",
            "N",
            ChannelShape.SCALAR,
            "force profile -> hybrid OSC pull-axis wrench command",
            Deployability.DEPLOYABLE,
            True,
            notes="The robot's own action. A real robot always knows what it asked for, which is "
            "why this is a mandatory ACE input (docs/DECISIONS.md D018).",
        ),
        # -- drawer response
        ChannelSpec(
            "drawer_position",
            "m",
            ChannelShape.SCALAR,
            "cabinet drawer_top_joint position, minus its value at the start of the pull",
            Deployability.DEPLOYABLE,
            True,
            notes="Displacement relative to the pull start, not the absolute joint coordinate.",
        ),
        ChannelSpec(
            "drawer_velocity",
            "m/s",
            ChannelShape.SCALAR,
            "finite difference of drawer_position across control steps",
            Deployability.DEPLOYABLE,
            True,
            filtering=f"{_CAUSAL_MA}, window 2 steps",
            notes="Substituted for PhysX's own joint_vel, which is aliased at this control rate "
            "(docs/DECISIONS.md D009).",
        ),
        ChannelSpec(
            "drawer_velocity_raw",
            "m/s",
            ChannelShape.SCALAR,
            "Articulation.data.joint_vel of drawer_top_joint",
            Deployability.DIAGNOSTIC,
            False,
            notes="Logged for transparency only. Aliased by gripper contact chatter; do not "
            "build anything on it.",
        ),
        ChannelSpec(
            "drawer_acceleration",
            "m/s^2",
            ChannelShape.SCALAR,
            "finite difference of the filtered drawer_velocity",
            Deployability.DEPLOYABLE,
            True,
            filtering=f"{_CAUSAL_MA}, window 4 steps",
            notes="Second difference of position, so noisier than velocity. Causal, hence usable "
            "by a deployed policy.",
        ),
        ChannelSpec(
            "drawer_acceleration_raw",
            "m/s^2",
            ChannelShape.SCALAR,
            "unsmoothed second difference of drawer_position",
            Deployability.DIAGNOSTIC,
            False,
            notes="Kept so the effect of the filter is auditable.",
        ),
        # -- end-effector, pull-axis projections
        ChannelSpec(
            "tcp_pull_axis_position",
            "m",
            ChannelShape.SCALAR,
            "TCP travel along the pull axis since the pose reference was captured",
            Deployability.DEPLOYABLE,
            True,
        ),
        ChannelSpec(
            "tcp_pull_axis_velocity",
            "m/s",
            ChannelShape.SCALAR,
            "TCP linear velocity from PhysX, projected on the pull axis",
            Deployability.DEPLOYABLE,
            True,
            notes="Unfiltered: the robot's own end-effector velocity is reliable, unlike the "
            "drawer joint's.",
        ),
        ChannelSpec(
            "tcp_pull_axis_acceleration",
            "m/s^2",
            ChannelShape.SCALAR,
            "finite difference of tcp_pull_axis_velocity",
            Deployability.DEPLOYABLE,
            True,
            filtering=f"{_CAUSAL_MA}, window 4 steps",
        ),
        ChannelSpec(
            "tcp_pull_axis_acceleration_raw",
            "m/s^2",
            ChannelShape.SCALAR,
            "unsmoothed finite difference of tcp_pull_axis_velocity",
            Deployability.DIAGNOSTIC,
            False,
        ),
        # -- end-effector, full state
        ChannelSpec(
            "tcp_position",
            "m",
            ChannelShape.VEC3,
            "ee_frame FrameTransformer, environment frame",
            Deployability.DEPLOYABLE,
            False,
        ),
        ChannelSpec(
            "tcp_orientation",
            "-",
            ChannelShape.QUAT,
            "ee_frame FrameTransformer, environment frame, (w, x, y, z)",
            Deployability.DEPLOYABLE,
            False,
        ),
        ChannelSpec(
            "tcp_linear_velocity",
            "m/s",
            ChannelShape.VEC3,
            "panda_hand body velocity transported to the TCP offset, world frame",
            Deployability.DEPLOYABLE,
            False,
        ),
        ChannelSpec(
            "tcp_angular_velocity",
            "rad/s",
            ChannelShape.VEC3,
            "panda_hand angular velocity, world frame",
            Deployability.DEPLOYABLE,
            False,
        ),
        ChannelSpec(
            "tcp_lateral_error",
            "m",
            ChannelShape.SCALAR,
            "TCP drift orthogonal to the pull axis, from the pose reference",
            Deployability.DEPLOYABLE,
            False,
            notes="Hybrid-control quality metric, and part of the validity mask.",
        ),
        ChannelSpec(
            "tcp_orientation_error",
            "rad",
            ChannelShape.SCALAR,
            "TCP orientation drift from the pose reference",
            Deployability.DEPLOYABLE,
            False,
            notes="Hybrid-control quality metric, and part of the validity mask.",
        ),
        # -- arm joints
        ChannelSpec(
            "joint_position",
            "rad",
            ChannelShape.JOINTS,
            "Articulation.data.joint_pos, panda_joint1..7",
            Deployability.DEPLOYABLE,
            False,
        ),
        ChannelSpec(
            "joint_velocity",
            "rad/s",
            ChannelShape.JOINTS,
            "Articulation.data.joint_vel, panda_joint1..7",
            Deployability.DEPLOYABLE,
            False,
        ),
        ChannelSpec(
            "joint_acceleration",
            "rad/s^2",
            ChannelShape.JOINTS,
            "finite difference of joint_velocity",
            Deployability.DEPLOYABLE,
            False,
            filtering=f"{_CAUSAL_MA}, window 4 steps",
        ),
        ChannelSpec(
            "joint_applied_effort",
            "N m",
            ChannelShape.JOINTS,
            "Articulation.data.applied_torque, panda_joint1..7",
            Deployability.DEPLOYABLE,
            False,
            notes="A command after actuator clipping, not a torque-sensor reading.",
        ),
        # -- force channels
        ChannelSpec(
            "measured_force",
            "N",
            ChannelShape.SCALAR,
            "panda_link7 -> panda_hand joint reaction wrench, projected on the pull axis",
            Deployability.DIAGNOSTIC,
            False,
            notes="What a real wrist force/torque sensor measures. Includes the hand and finger "
            "inertial term (~0.2 N). Recorded for the ACE-5 ablation (docs/FORCE_CHANNEL_AUDIT.md).",
        ),
        ChannelSpec(
            "handle_contact_force_w",
            "N",
            ChannelShape.VEC3,
            "ContactSensor net_forces_w on drawer_handle_top, world frame",
            Deployability.DIAGNOSTIC,
            False,
            notes="Normal grip load only -- net contact force excludes tangential friction, so it "
            "cannot measure the pull. Witness of grip quality, nothing more (D006).",
        ),
        ChannelSpec(
            "drawer_resistance_force",
            "N",
            ChannelShape.SCALAR,
            "ArticulationView.get_dof_projected_joint_forces on drawer_top_joint",
            Deployability.SIM_ONLY_PRIVILEGED,
            False,
            notes="The drawer's internal resistance, measured as -(mu_dynamic*sign(v) + b*v). Exact, "
            "and the ground truth the force audit checks the other channels against.",
        ),
        ChannelSpec(
            "drawer_external_force",
            "N",
            ChannelShape.SCALAR,
            "m_total * drawer_acceleration - drawer_resistance_force",
            Deployability.SIM_ONLY_PRIVILEGED,
            False,
            notes="The axial force actually delivered to the drawer. Needs the drawer's mass, i.e. "
            "part of the hidden state, so it can never be a deployed-model input.",
        ),
    )
}

#: The first ACE's observation vector, in order. Every entry is DEPLOYABLE by construction,
#: and ``commanded_force`` is mandatory (``docs/DECISIONS.md`` D018).
DEFAULT_ACE_INPUT: tuple[str, ...] = (
    "commanded_force",
    "drawer_position",
    "drawer_velocity",
    "drawer_acceleration",
    "tcp_pull_axis_position",
    "tcp_pull_axis_velocity",
    "tcp_pull_axis_acceleration",
)


def channels_by_deployability(deployability: Deployability) -> tuple[str, ...]:
    """Channel names in one deployability class, in registry order."""
    return tuple(name for name, spec in OBSERVATION_SPECS.items() if spec.deployability is deployability)


def validate_model_input(channels: tuple[str, ...] | list[str]) -> None:
    """Raise if a proposed model input is unknown or not deployable.

    Call this wherever an observation vector is assembled for something that is meant to
    run on hardware. It is cheap, and it turns "somebody used a privileged channel" from a
    silent modelling error into an immediate failure.

    Raises:
        KeyError: If a channel is not in :data:`OBSERVATION_SPECS`.
        ValueError: If a channel is not :attr:`Deployability.DEPLOYABLE`.
    """
    unknown = [name for name in channels if name not in OBSERVATION_SPECS]
    if unknown:
        raise KeyError(f"Unknown observation channels: {unknown}. Known: {sorted(OBSERVATION_SPECS)}.")
    undeployable = {
        name: OBSERVATION_SPECS[name].deployability.value
        for name in channels
        if OBSERVATION_SPECS[name].deployability is not Deployability.DEPLOYABLE
    }
    if undeployable:
        raise ValueError(
            f"These channels cannot be inputs to a deployable model: {undeployable}. "
            "See docs/DECISIONS.md D017 and docs/FORCE_CHANNEL_AUDIT.md."
        )

"""What else this simulator would let us hide, and why the main paper does not use it.

The main paper's hidden state is four dimensional (``docs/DECISIONS.md`` D015). That is a
choice, not a limitation, and a choice is only defensible if the alternatives were actually
examined. This module enumerates every physical quantity of the cabinet, the drawer and the
robot that Isaac Sim 5.1 / Isaac Lab 2.3 expose, probes each one on the live simulation to
establish whether it can really be written and read back, and records what it would do to a
force-driven pull.

Each candidate carries a *claim* (what it is, what it would do, whether it belongs in the
paper) and a *probe* (a function that writes a value and reads it back). The claims are
reviewed prose; the writable/readable columns are measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - needs the Isaac Sim app at runtime
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "CANDIDATES",
    "AuditVerdict",
    "HiddenStateCandidate",
    "PaperRole",
    "run_hidden_state_audit",
]


class PaperRole(str, Enum):
    """What role a quantity plays in the research programme."""

    MAIN_PAPER_XI = "main_paper_xi"
    HELD_FIXED = "held_fixed"
    OOD_CANDIDATE = "ood_candidate"
    NOT_SUITABLE = "not_suitable"


@dataclass(frozen=True)
class HiddenStateCandidate:
    """One physical quantity, its simulator mapping, and its role.

    Args:
        name: Short identifier used in the audit table.
        simulator_api: The exact call that writes it, as installed here.
        target: What object the call acts on.
        physical_interpretation: What the number means physically.
        impact_on_pull: Expected effect on a force-driven drawer pull.
        deployment_visible: Whether a robot could observe it directly at deployment. A
            hidden state must be invisible; anything visible belongs in the observation.
        role: See :class:`PaperRole`.
        rationale: Why it has that role.
        probe: ``(env) -> (written, read_back, detail)``. ``None`` means not probed, with
            the reason in :attr:`rationale`.
    """

    name: str
    simulator_api: str
    target: str
    physical_interpretation: str
    impact_on_pull: str
    deployment_visible: bool
    role: PaperRole
    rationale: str
    probe: Callable[[Any], tuple[bool, bool, str]] | None = None


@dataclass
class AuditVerdict:
    """The measured half of one candidate's row."""

    name: str
    writable: bool | None
    readable_back: bool | None
    detail: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "writable": self.writable,
            "readable_back": self.readable_back,
            "detail": self.detail,
        }


def _cabinet(env: ManagerBasedRLEnv):
    return env.scene["cabinet"]


def _robot(env: ManagerBasedRLEnv):
    return env.scene["robot"]


def _drawer_joint(env: ManagerBasedRLEnv) -> int:
    return _cabinet(env).find_joints("drawer_top_joint")[0][0]


def _drawer_body(env: ManagerBasedRLEnv) -> int:
    return _cabinet(env).find_bodies("drawer_top")[0][0]


def _roundtrip(
    read: Callable[[], float], write: Callable[[float], None], probe_value: float, restore: float | None = None
) -> tuple[bool, bool, str]:
    """Write ``probe_value``, read it back, then restore the original value.

    Returns ``(written, read_back, detail)``. ``written`` is false only if the call raised;
    ``read_back`` is false when the simulator did not take the value -- which is the case
    that matters, because a silently discarded write is indistinguishable from a successful
    one unless the readback is checked.
    """
    original = read()
    try:
        write(probe_value)
    except Exception as error:  # noqa: BLE001 - the point of the probe is to find out
        return False, False, f"write raised {type(error).__name__}: {error}"
    observed = read()
    try:
        write(original if restore is None else restore)
    except Exception:  # noqa: BLE001 - restoration failure must not mask the result
        pass
    matched = abs(observed - probe_value) <= 1e-3 * max(1.0, abs(probe_value))
    return True, matched, f"wrote {probe_value:g}, read back {observed:g} (was {original:g})"


def _probe_body_mass(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    view = _cabinet(env).root_physx_view
    body = _drawer_body(env)
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        masses = view.get_masses()
        masses[ids, body] = value
        view.set_masses(masses, ids)

    return _roundtrip(lambda: float(view.get_masses()[0, body]), write, 9.75)


def _probe_body_inertia(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    view = _cabinet(env).root_physx_view
    body = _drawer_body(env)
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        inertias = view.get_inertias()
        inertias[ids, body, 0] = value
        view.set_inertias(inertias, ids)

    return _roundtrip(lambda: float(view.get_inertias()[0, body, 0]), write, 13.5)


def _probe_body_com(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    view = _cabinet(env).root_physx_view
    body = _drawer_body(env)
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        coms = view.get_coms()
        coms[ids, body, 0] = value
        view.set_coms(coms, ids)

    return _roundtrip(lambda: float(view.get_coms()[0, body, 0]), write, -0.05)


def _probe_friction_channel(column: int, probe_value: float) -> Callable[[Any], tuple[bool, bool, str]]:
    """Probe one column of ``get_dof_friction_properties`` (static/dynamic/viscous)."""

    def probe(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
        view = _cabinet(env).root_physx_view
        joint = _drawer_joint(env)
        ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

        def write(value: float) -> None:
            props = view.get_dof_friction_properties()
            # Static must stay >= dynamic or PhysX discards the whole write, so raise static
            # alongside any dynamic probe.
            props[ids, joint, column] = value
            if column == 1:
                props[ids, joint, 0] = max(value, float(props[0, joint, 0]))
            view.set_dof_friction_properties(props, ids)

        return _roundtrip(lambda: float(view.get_dof_friction_properties()[0, joint, column]), write, probe_value)

    return probe


def _probe_dof_scalar(getter: str, setter: str, probe_value: float) -> Callable[[Any], tuple[bool, bool, str]]:
    """Probe a per-DOF scalar exposed as ``get_dof_*``/``set_dof_*`` on the cabinet."""

    def probe(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
        view = _cabinet(env).root_physx_view
        joint = _drawer_joint(env)
        ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

        def write(value: float) -> None:
            values = getattr(view, getter)()
            values[ids, joint] = value
            getattr(view, setter)(values, ids)

        return _roundtrip(lambda: float(getattr(view, getter)()[0, joint]), write, probe_value)

    return probe


def _probe_dof_limit(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    view = _cabinet(env).root_physx_view
    joint = _drawer_joint(env)
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        limits = view.get_dof_limits()
        limits[ids, joint, 1] = value
        view.set_dof_limits(limits, ids)

    return _roundtrip(lambda: float(view.get_dof_limits()[0, joint, 1]), write, 0.3)


def _probe_contact_material(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    view = _cabinet(env).root_physx_view
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        properties = view.get_material_properties()
        properties[ids, :, 0] = value
        view.set_material_properties(properties, ids)

    return _roundtrip(lambda: float(view.get_material_properties()[0, 0, 0]), write, 0.9)


def _probe_restitution(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    view = _cabinet(env).root_physx_view
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        properties = view.get_material_properties()
        properties[ids, :, 2] = value
        view.set_material_properties(properties, ids)

    return _roundtrip(lambda: float(view.get_material_properties()[0, 0, 2]), write, 0.3)


def _probe_robot_actuator_stiffness(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    robot = _robot(env)
    view = robot.root_physx_view
    joint = robot.find_joints("panda_joint1")[0][0]
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        values = view.get_dof_stiffnesses()
        values[ids, joint] = value
        view.set_dof_stiffnesses(values, ids)

    return _roundtrip(lambda: float(view.get_dof_stiffnesses()[0, joint]), write, 25.0)


def _probe_gravity(env: ManagerBasedRLEnv) -> tuple[bool, bool, str]:
    """Gravity is a per-body flag on the articulation view, not a scalar."""
    view = _cabinet(env).root_physx_view
    body = _drawer_body(env)
    ids = torch.arange(env.num_envs, dtype=torch.long, device="cpu")

    def write(value: float) -> None:
        flags = view.get_disable_gravities()
        flags[ids, body] = int(value)
        view.set_disable_gravities(flags, ids)

    return _roundtrip(lambda: float(view.get_disable_gravities()[0, body]), write, 1.0, restore=0.0)


CANDIDATES: tuple[HiddenStateCandidate, ...] = (
    HiddenStateCandidate(
        "drawer_mass",
        "ArticulationView.set_masses (+ set_inertias to keep the density consistent)",
        "cabinet body drawer_top",
        "Mass of the moving drawer assembly.",
        "Sets the inertial term: how much of the applied force goes into acceleration.",
        deployment_visible=False,
        role=PaperRole.MAIN_PAPER_XI,
        rationale="A loaded drawer is heavier than an empty one, and nothing on the robot can see it.",
        probe=_probe_body_mass,
    ),
    HiddenStateCandidate(
        "joint_static_friction",
        "Articulation.write_joint_friction_coefficient_to_sim (static channel)",
        "cabinet joint drawer_top_joint",
        "Coulomb friction effort resisting the start of motion (N).",
        "Sets the breakaway force: below it the drawer does not move at all.",
        deployment_visible=False,
        role=PaperRole.MAIN_PAPER_XI,
        rationale="The single most task-relevant unknown: it decides whether a given force moves the drawer.",
        probe=_probe_friction_channel(0, 4.25),
    ),
    HiddenStateCandidate(
        "joint_dynamic_friction",
        "Articulation.write_joint_friction_coefficient_to_sim (dynamic channel)",
        "cabinet joint drawer_top_joint",
        "Coulomb friction effort resisting continued motion (N).",
        "Sets the velocity-independent drag once sliding, hence the steady-state speed.",
        deployment_visible=False,
        role=PaperRole.MAIN_PAPER_XI,
        rationale="Independent of the static value and separately observable after breakaway. "
        "PhysX requires dynamic <= static.",
        probe=_probe_friction_channel(1, 1.75),
    ),
    HiddenStateCandidate(
        "joint_damping",
        "Articulation.write_joint_damping_to_sim",
        "cabinet joint drawer_top_joint",
        "Viscous damping of the joint drive (N s/m).",
        "Velocity-proportional drag: caps the terminal speed for a given force.",
        deployment_visible=False,
        role=PaperRole.MAIN_PAPER_XI,
        rationale="Distinguishable from Coulomb friction precisely because its effect scales with speed.",
        probe=_probe_dof_scalar("get_dof_dampings", "set_dof_dampings", 7.5),
    ),
    HiddenStateCandidate(
        "joint_viscous_friction",
        "Articulation.write_joint_friction_coefficient_to_sim (viscous channel)",
        "cabinet joint drawer_top_joint",
        "A second velocity-proportional drag term, in the friction model rather than the drive.",
        "Physically indistinguishable from joint_damping in this one-degree-of-freedom system.",
        deployment_visible=False,
        role=PaperRole.HELD_FIXED,
        rationale="Adding it would make xi unidentifiable: two parameters with the same signature. "
        "Held at zero so 'damping' maps onto exactly one simulator quantity.",
        probe=_probe_friction_channel(2, 2.0),
    ),
    HiddenStateCandidate(
        "joint_stiffness",
        "Articulation.write_joint_stiffness_to_sim",
        "cabinet joint drawer_top_joint",
        "Drive stiffness, i.e. a spring pulling the drawer back towards closed (N/m).",
        "Adds a displacement-proportional restoring force, so the force needed grows with travel.",
        deployment_visible=False,
        role=PaperRole.OOD_CANDIDATE,
        rationale="Real and interesting -- a spring-loaded drawer -- but it is a fourth mechanism on "
        "top of xi and the official value of 10 N/m already contributes 3 N over full travel. "
        "Removed from the main paper (D008); a natural out-of-distribution axis later.",
        probe=_probe_dof_scalar("get_dof_stiffnesses", "set_dof_stiffnesses", 5.0),
    ),
    HiddenStateCandidate(
        "joint_armature",
        "Articulation.write_joint_armature_to_sim",
        "cabinet joint drawer_top_joint",
        "Added rotor inertia on the joint (kg for a prismatic DOF).",
        "Indistinguishable from drawer mass along a single translational DOF.",
        deployment_visible=False,
        role=PaperRole.HELD_FIXED,
        rationale="Degenerate with drawer_mass, so including both would make xi unidentifiable.",
        probe=_probe_dof_scalar("get_dof_armatures", "set_dof_armatures", 1.5),
    ),
    HiddenStateCandidate(
        "drawer_inertia_tensor",
        "ArticulationView.set_inertias",
        "cabinet body drawer_top",
        "Rotational inertia of the drawer body.",
        "Almost none: the drawer translates along one axis and does not rotate.",
        deployment_visible=False,
        role=PaperRole.NOT_SUITABLE,
        rationale="No observable effect on a purely translational pull, so it cannot be inferred and "
        "would only add nuisance dimensions.",
        probe=_probe_body_inertia,
    ),
    HiddenStateCandidate(
        "drawer_center_of_mass",
        "ArticulationView.set_coms",
        "cabinet body drawer_top",
        "Offset of the drawer's centre of mass from its body origin.",
        "Second order for an axial pull: it tilts the load on the rails rather than opposing motion.",
        deployment_visible=False,
        role=PaperRole.OOD_CANDIDATE,
        rationale="Physically meaningful for an unevenly loaded drawer, and it would perturb the rail "
        "contact, but its effect on the axial response is far weaker than xi's. Left for OOD.",
        probe=_probe_body_com,
    ),
    HiddenStateCandidate(
        "handle_contact_friction",
        "ArticulationView.set_material_properties (static/dynamic columns)",
        "every cabinet collision material, including the handle",
        "Coulomb friction of the grasp contact between fingers and handle.",
        "Decides whether the grip slips; it does not resist the drawer, it limits transmissible force.",
        deployment_visible=False,
        role=PaperRole.HELD_FIXED,
        rationale="Changes the grasp rather than the drawer, so it confounds the study: a failed pull "
        "would be ambiguous between a stiff drawer and a slipping grip. Pinned to fixed values (D010).",
        probe=_probe_contact_material,
    ),
    HiddenStateCandidate(
        "restitution",
        "ArticulationView.set_material_properties (restitution column)",
        "every cabinet collision material",
        "Bounciness of contacts.",
        "Only matters on impact, i.e. at the mechanical end stop, which valid episodes never reach.",
        deployment_visible=False,
        role=PaperRole.NOT_SUITABLE,
        rationale="Its only effect is inside the region the validity mask already excludes.",
        probe=_probe_restitution,
    ),
    HiddenStateCandidate(
        "drawer_travel_limit",
        "ArticulationView.set_dof_limits",
        "cabinet joint drawer_top_joint",
        "How far the drawer can open before hitting its stop (m).",
        "Caps the achievable displacement; reaching it produces a large impact transient.",
        deployment_visible=True,
        role=PaperRole.NOT_SUITABLE,
        rationale="A drawer's travel is visually apparent, so it is not hidden. It is also the task's "
        "geometry rather than its dynamics, and changing it would move the goal rather than the physics.",
        probe=_probe_dof_limit,
    ),
    HiddenStateCandidate(
        "drawer_effort_limit",
        "Articulation.write_joint_effort_limit_to_sim / set_dof_max_forces",
        "cabinet joint drawer_top_joint",
        "Maximum force the joint drive may exert.",
        "None here: the drawer's drive is passive, so the limit is never reached.",
        deployment_visible=False,
        role=PaperRole.NOT_SUITABLE,
        rationale="Inactive in this configuration, so it cannot be inferred from any probe.",
        probe=_probe_dof_scalar("get_dof_max_forces", "set_dof_max_forces", 50.0),
    ),
    HiddenStateCandidate(
        "robot_actuator_gains",
        "ArticulationView.set_dof_stiffnesses / set_dof_dampings on the robot",
        "robot joints panda_joint1..7",
        "The arm's own joint-level PD gains.",
        "Changes the robot, not the drawer. The hybrid OSC needs them at zero for effort control.",
        deployment_visible=True,
        role=PaperRole.NOT_SUITABLE,
        rationale="A robot knows its own controller gains, so they are not hidden state; and varying "
        "them would confound arm behaviour with drawer dynamics.",
        probe=_probe_robot_actuator_stiffness,
    ),
    HiddenStateCandidate(
        "gravity",
        "ArticulationView.set_disable_gravities (per body), or the scene's gravity vector",
        "cabinet bodies",
        "Whether gravity acts on the cabinet.",
        "Loads the drawer rails vertically, changing rail friction indirectly.",
        deployment_visible=True,
        role=PaperRole.NOT_SUITABLE,
        rationale="Not a property of an unknown drawer, and it is the same for every object a robot "
        "will ever manipulate on Earth.",
        probe=_probe_gravity,
    ),
)


def run_hidden_state_audit(env: ManagerBasedRLEnv) -> dict:
    """Probe every candidate on a live environment and return the full audit table.

    Each probe writes a value, reads it back and restores the original, so the environment
    is left as it was found. Nothing is stepped, so no dynamics are disturbed.
    """
    verdicts: list[AuditVerdict] = []
    for candidate in CANDIDATES:
        if candidate.probe is None:
            verdicts.append(AuditVerdict(candidate.name, None, None, "not probed"))
            continue
        written, read_back, detail = candidate.probe(env)
        verdicts.append(AuditVerdict(candidate.name, written, read_back, detail))

    by_role: dict[str, list[str]] = {}
    for candidate in CANDIDATES:
        by_role.setdefault(candidate.role.value, []).append(candidate.name)

    return {
        "rows": [
            {
                "name": candidate.name,
                "simulator_api": candidate.simulator_api,
                "target": candidate.target,
                "physical_interpretation": candidate.physical_interpretation,
                "impact_on_pull": candidate.impact_on_pull,
                "deployment_visible": candidate.deployment_visible,
                "role": candidate.role.value,
                "rationale": candidate.rationale,
                **verdict.as_dict(),
            }
            for candidate, verdict in zip(CANDIDATES, verdicts, strict=True)
        ],
        "by_role": by_role,
        "main_paper_xi": by_role.get(PaperRole.MAIN_PAPER_XI.value, []),
        "writable_and_verified": [v.name for v in verdicts if v.writable and v.readable_back],
        "write_accepted_but_not_read_back": [
            v.name for v in verdicts if v.writable and v.readable_back is False
        ],
    }

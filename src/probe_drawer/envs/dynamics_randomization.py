r"""The hidden dynamics of the drawer, and the one place that changes them.

The main paper's hidden state is exactly four dimensional:

.. math:: \xi = [m,\; \mu_s,\; \mu_d,\; b]

Each element maps onto a specific PhysX quantity of the *top drawer*, and nothing else:

``drawer_mass`` (:math:`m`)
    Mass of the ``drawer_top`` rigid body (kg).  Inertia is scaled by the same ratio, as
    Isaac Lab's own ``randomize_rigid_body_mass`` event does.  The rigidly attached
    ``drawer_handle_top`` body keeps its own small mass, so the total moving mass is
    ``drawer_mass + handle_mass``; :attr:`AppliedDynamics.total_moving_mass` reports it.
``joint_static_friction`` (:math:`\mu_s`)
    Static Coulomb friction effort of the ``drawer_top_joint`` prismatic joint (N). Sets
    how hard the drawer resists *starting* to move.
``joint_dynamic_friction`` (:math:`\mu_d`)
    Dynamic Coulomb friction effort of the same joint (N). Sets how hard it resists
    *continuing* to move.
``joint_damping`` (:math:`b`)
    Viscous damping of the ``drawer_top_joint`` drive (N s/m).

Deliberately **not** part of :math:`\xi`: joint stiffness, per-DOF viscous friction,
armature, centre of mass, inertia tensor, contact material properties, restitution, joint
limits, and every robot-side parameter.  They are all writable, and
``docs/HIDDEN_STATE_AUDIT.md`` records what each of them does; they are held at fixed,
documented values here so that a difference between two episodes can only come from
:math:`\xi`.  See ``docs/DECISIONS.md`` D015.

Two simulator facts that shape this module
------------------------------------------
1. **PhysX requires** :math:`\mu_s \ge \mu_d`. A write with ``static < dynamic`` is
   *silently rejected*: PhysX logs ``Static friction effort must be greater than or equal
   to dynamic friction effort`` and keeps the previous values.
   :class:`DynamicsParameters` therefore refuses such a combination up front, and
   :meth:`DynamicsRandomizer.sample` parameterises :math:`\mu_d` as a fraction of
   :math:`\mu_s` so that every sample is valid by construction.
2. **Isaac Lab's ``Articulation.data.*`` friction buffers mirror the request, not the
   simulator.** After a rejected write they report the value that was asked for while
   PhysX holds the old one.  Every readback in this module therefore comes from
   ``root_physx_view``, and the mirror is reported separately so a disagreement is visible
   rather than hidden.  See ``docs/DECISIONS.md`` D016.

Randomisation lives here and *only* here.  Controllers never modify the environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - needs the Isaac Sim app at runtime
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "PRESETS",
    "XI_FIELDS",
    "AppliedDynamics",
    "DynamicsParameters",
    "DynamicsRandomizer",
    "DynamicsRandomizerCfg",
    "preset",
]

#: The main paper's hidden-state field names, in canonical order.
XI_FIELDS: tuple[str, ...] = (
    "drawer_mass",
    "joint_static_friction",
    "joint_dynamic_friction",
    "joint_damping",
)

#: Column layout of ``ArticulationView.get_dof_friction_properties()``.
_STATIC, _DYNAMIC, _VISCOUS = 0, 1, 2


@dataclass(frozen=True)
class DynamicsParameters:
    """One drawer's hidden dynamics: the four-dimensional :math:`\\xi`.

    Args:
        drawer_mass: Mass of the ``drawer_top`` body (kg). Must be positive.
        joint_static_friction: Static Coulomb friction effort of the drawer joint (N).
        joint_dynamic_friction: Dynamic Coulomb friction effort of the drawer joint (N).
            Must not exceed :attr:`joint_static_friction` -- PhysX rejects that silently,
            and real Coulomb friction does not behave that way either.
        joint_damping: Viscous damping of the drawer joint drive (N s/m).
        name: Label for logs and plots.
    """

    drawer_mass: float
    joint_static_friction: float
    joint_dynamic_friction: float
    joint_damping: float
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.drawer_mass <= 0.0:
            raise ValueError(f"drawer_mass must be > 0 kg, got {self.drawer_mass}.")
        for field_name in ("joint_static_friction", "joint_dynamic_friction", "joint_damping"):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be >= 0, got {value}.")
        if self.joint_dynamic_friction > self.joint_static_friction:
            raise ValueError(
                f"joint_dynamic_friction ({self.joint_dynamic_friction}) must be <= "
                f"joint_static_friction ({self.joint_static_friction}). PhysX enforces "
                "static >= dynamic and silently discards writes that violate it, so such a "
                "combination cannot be simulated."
            )

    @property
    def friction_ratio(self) -> float:
        """:math:`\\mu_d / \\mu_s`, or 1.0 when there is no static friction at all."""
        if self.joint_static_friction == 0.0:
            return 1.0
        return self.joint_dynamic_friction / self.joint_static_friction

    def as_vector(self) -> tuple[float, float, float, float]:
        """:math:`\\xi` as a plain tuple in :data:`XI_FIELDS` order."""
        return tuple(float(getattr(self, name)) for name in XI_FIELDS)  # type: ignore[return-value]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "drawer_mass": self.drawer_mass,
            "joint_static_friction": self.joint_static_friction,
            "joint_dynamic_friction": self.joint_dynamic_friction,
            "joint_damping": self.joint_damping,
        }


#: Deterministic presets. **Provisional**, not the paper's final ranges.
#:
#: ``nominal`` reproduces the official cabinet's own drawer mass and damping with friction
#: removed. ``easy``/``medium``/``hard`` carry over the Phase 8 values, whose single
#: friction figure is split as ``mu_s == mu_d`` so that their validated behaviour is
#: unchanged by the move to a four-dimensional xi. ``sticky`` and ``slippery`` exist so the
#: friction *asymmetry* can be exercised on its own.
#:
#: The paper's training and out-of-distribution ranges come from the Phase 9 sweep, not
#: from these presets -- see ``docs/EXPERIMENT_SPACE.md``.
PRESETS: dict[str, DynamicsParameters] = {
    "nominal": DynamicsParameters(5.175, 0.0, 0.0, 1.0, name="nominal"),
    "easy": DynamicsParameters(5.0, 1.5, 1.5, 4.0, name="easy"),
    "medium": DynamicsParameters(8.0, 3.0, 3.0, 6.0, name="medium"),
    "hard": DynamicsParameters(10.0, 4.0, 4.0, 9.0, name="hard"),
    "sticky": DynamicsParameters(8.0, 6.0, 1.5, 6.0, name="sticky"),
    "slippery": DynamicsParameters(8.0, 3.0, 0.5, 6.0, name="slippery"),
}


def preset(name: str) -> DynamicsParameters:
    """Look up a deterministic preset by name.

    Raises:
        KeyError: With the list of valid names, if ``name`` is unknown.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"Unknown dynamics preset {name!r}. Available: {sorted(PRESETS)}.") from None


@dataclass
class DynamicsRandomizerCfg:
    """What :meth:`DynamicsRandomizer.sample` draws from, and what it acts on.

    The sampling ranges are **provisional** placeholders for development; the paper's
    training and OOD ranges are selected from the Phase 9 sweep and recorded in
    ``docs/EXPERIMENT_SPACE.md``.

    Args:
        cabinet_asset: Scene name of the cabinet articulation.
        drawer_joint_name: Prismatic joint whose friction and damping are set.
        drawer_body_name: Rigid body whose mass is set.
        handle_body_name: Rigidly attached handle, read only to report the total moving mass.
        mass_range: Sampling range for :math:`m` (kg).
        static_friction_range: Sampling range for :math:`\\mu_s` (N).
        dynamic_friction_ratio_range: Sampling range for :math:`\\mu_d / \\mu_s`. Sampling a
            *ratio* rather than an absolute value keeps every draw inside the region PhysX
            accepts (``mu_s >= mu_d``) without rejection sampling.
        damping_range: Sampling range for :math:`b` (N s/m).
        fixed_joint_stiffness: Drawer drive stiffness written on every apply (N/m). **Not**
            part of :math:`\\xi`; held at zero so the drawer is a pure
            mass-friction-damper system (``docs/DECISIONS.md`` D008).
        fixed_joint_viscous_friction: Per-DOF viscous friction written on every apply.
            **Not** part of :math:`\\xi`; held at zero so that "damping" maps onto exactly
            one simulator quantity, the drive damping the official cabinet configures.
    """

    cabinet_asset: str = "cabinet"
    drawer_joint_name: str = "drawer_top_joint"
    drawer_body_name: str = "drawer_top"
    handle_body_name: str = "drawer_handle_top"

    mass_range: tuple[float, float] = (4.0, 12.0)
    static_friction_range: tuple[float, float] = (0.5, 5.0)
    dynamic_friction_ratio_range: tuple[float, float] = (0.3, 1.0)
    damping_range: tuple[float, float] = (3.0, 10.0)

    fixed_joint_stiffness: float = 0.0
    fixed_joint_viscous_friction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mass_range", "static_friction_range", "dynamic_friction_ratio_range", "damping_range"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must be ordered (low, high), got {(low, high)}.")
        low, high = self.dynamic_friction_ratio_range
        if not (0.0 <= low and high <= 1.0):
            raise ValueError(
                f"dynamic_friction_ratio_range must lie inside [0, 1], got {(low, high)}: a ratio "
                "above 1 would mean dynamic friction exceeding static friction, which PhysX rejects."
            )


@dataclass
class AppliedDynamics:
    """What :meth:`DynamicsRandomizer.apply` actually wrote, as the simulator now holds it.

    Attributes:
        requested: The :math:`\\xi` that was asked for, one per environment.
        readback: What ``root_physx_view`` reports afterwards -- the simulator's own state.
        mirrored: What Isaac Lab's ``Articulation.data`` buffers report. Normally identical
            to :attr:`readback`; a disagreement means a write was rejected by PhysX while
            the mirror kept the request (see this module's docstring).
        fixed: Non-:math:`\\xi` quantities this randomiser pins on every apply.
        handle_mass: Mass of the rigidly attached handle body (kg).
        consistent: Whether every requested value matches the *simulator* readback.
        mirror_agrees: Whether the Isaac Lab mirror matches the simulator readback.
    """

    requested: list[DynamicsParameters]
    readback: dict[str, list[float]]
    mirrored: dict[str, list[float]]
    fixed: dict[str, list[float]]
    handle_mass: float
    consistent: bool
    mirror_agrees: bool
    notes: dict = field(default_factory=dict)

    @property
    def preset_name(self) -> str:
        """Preset label, or a comma-separated list in environment order when they differ."""
        names = list(dict.fromkeys(p.name for p in self.requested))
        return names[0] if len(names) == 1 else ",".join(names)

    @property
    def total_moving_mass(self) -> list[float]:
        """Total mass moved by the drawer joint per environment (kg)."""
        return [p.drawer_mass + self.handle_mass for p in self.requested]

    def as_dict(self) -> dict:
        return {
            "xi_fields": list(XI_FIELDS),
            "preset_name": self.preset_name,
            "requested": [p.as_dict() for p in self.requested],
            "readback": self.readback,
            "mirrored": self.mirrored,
            "fixed": self.fixed,
            "handle_mass": self.handle_mass,
            "total_moving_mass": self.total_moving_mass,
            "consistent": self.consistent,
            "mirror_agrees": self.mirror_agrees,
            "notes": self.notes,
        }


class DynamicsRandomizer:
    """Samples and applies the drawer's four-dimensional hidden dynamics.

    Args:
        cfg: Sampling ranges and target entity names.
        seed: Seed for :meth:`sample`. ``None`` leaves the generator unseeded.

    Example:
        >>> randomizer = DynamicsRandomizer(seed=0)
        >>> applied = randomizer.apply(env, randomizer.sample(env.num_envs))
        >>> applied.consistent and applied.mirror_agrees
        True
    """

    #: Relative tolerance when comparing a requested value against the simulator readback.
    READBACK_TOLERANCE = 1e-3

    def __init__(self, cfg: DynamicsRandomizerCfg | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or DynamicsRandomizerCfg()
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        self._current: AppliedDynamics | None = None

    def sample(self, num_envs: int = 1) -> list[DynamicsParameters]:
        """Draw ``num_envs`` independent :math:`\\xi` uniformly from the configured ranges.

        :math:`\\mu_d` is drawn as a fraction of :math:`\\mu_s`, so every sample satisfies
        the ``mu_s >= mu_d`` constraint PhysX imposes without any rejection sampling.
        """
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        draws = torch.rand(num_envs, 4, generator=self._generator).tolist()

        def scale(u: float, bounds: tuple[float, float]) -> float:
            return bounds[0] + u * (bounds[1] - bounds[0])

        sampled = []
        for u in draws:
            static = scale(u[1], self.cfg.static_friction_range)
            ratio = scale(u[2], self.cfg.dynamic_friction_ratio_range)
            sampled.append(
                DynamicsParameters(
                    drawer_mass=scale(u[0], self.cfg.mass_range),
                    joint_static_friction=static,
                    joint_dynamic_friction=static * ratio,
                    joint_damping=scale(u[3], self.cfg.damping_range),
                    name="sampled",
                )
            )
        return sampled

    def apply(
        self, env: ManagerBasedRLEnv, params: DynamicsParameters | Sequence[DynamicsParameters]
    ) -> AppliedDynamics:
        """Write the given dynamics into the simulation and read them back out of PhysX.

        Args:
            env: The running environment.
            params: One :math:`\\xi` broadcast to every environment, or one per environment.

        Returns:
            An :class:`AppliedDynamics` record, also retrievable via
            :meth:`get_current_params`.

        Raises:
            ValueError: If a sequence is given whose length is not ``env.num_envs``.
        """
        per_env = self._broadcast(params, env.num_envs)
        cabinet: Articulation = env.scene[self.cfg.cabinet_asset]
        joint_idx = cabinet.find_joints(self.cfg.drawer_joint_name)[0][0]
        body_idx = cabinet.find_bodies(self.cfg.drawer_body_name)[0][0]
        handle_idx = cabinet.find_bodies(self.cfg.handle_body_name)[0][0]

        self._write_mass(cabinet, body_idx, [p.drawer_mass for p in per_env])
        self._write_joint_properties(cabinet, joint_idx, per_env, env.device)

        readback = self._read_from_simulator(cabinet, joint_idx, body_idx)
        mirrored = self._read_from_mirror(cabinet, joint_idx)

        self._current = AppliedDynamics(
            requested=per_env,
            readback=readback,
            mirrored=mirrored,
            fixed={
                "joint_stiffness": cabinet.root_physx_view.get_dof_stiffnesses()[:, joint_idx].tolist(),
                "joint_viscous_friction": cabinet.root_physx_view.get_dof_friction_properties()[
                    :, joint_idx, _VISCOUS
                ].tolist(),
            },
            handle_mass=float(cabinet.root_physx_view.get_masses()[0, handle_idx]),
            consistent=self._matches_request(readback, per_env),
            mirror_agrees=all(
                _close(mirrored[key][i], readback[key][i]) for key in mirrored for i in range(env.num_envs)
            ),
            notes={
                "drawer_joint": self.cfg.drawer_joint_name,
                "drawer_body": self.cfg.drawer_body_name,
                "readback_source": "ArticulationView (PhysX), not Articulation.data",
                "xi_fields": list(XI_FIELDS),
            },
        )
        return self._current

    def get_current_params(self) -> AppliedDynamics | None:
        """The dynamics most recently applied, or ``None`` before the first :meth:`apply`.

        This is the privileged state :math:`\\xi`; it is recorded with every episode and is
        used only for training and analysis, never as a controller or ACE input.
        """
        return self._current

    """
    Writes.
    """

    def _write_joint_properties(
        self,
        cabinet: Articulation,
        joint_idx: int,
        per_env: list[DynamicsParameters],
        device: torch.device | str,
    ) -> None:
        """Write friction, damping and the pinned non-xi joint properties."""
        joint_ids = [joint_idx]

        def column(values: list[float]) -> torch.Tensor:
            return torch.tensor(values, device=device).unsqueeze(-1)

        # Static, dynamic and viscous must go in one call: PhysX validates the triple as a
        # whole and rejects any intermediate state in which dynamic exceeds static, which is
        # exactly what two sequential writes produce while friction is being raised.
        cabinet.write_joint_friction_coefficient_to_sim(
            column([p.joint_static_friction for p in per_env]),
            joint_dynamic_friction_coeff=column([p.joint_dynamic_friction for p in per_env]),
            joint_viscous_friction_coeff=column([self.cfg.fixed_joint_viscous_friction] * len(per_env)),
            joint_ids=joint_ids,
        )
        cabinet.write_joint_damping_to_sim(column([p.joint_damping for p in per_env]), joint_ids=joint_ids)
        cabinet.write_joint_stiffness_to_sim(
            column([self.cfg.fixed_joint_stiffness] * len(per_env)), joint_ids=joint_ids
        )

    def _write_mass(self, cabinet: Articulation, body_idx: int, masses: list[float]) -> None:
        """Set the drawer body's mass and scale its inertia by the same ratio.

        Follows Isaac Lab's ``randomize_rigid_body_mass``: PhysX takes masses and inertias
        as CPU tensors, and the inertia of a uniform-density body scales linearly with mass.
        """
        view = cabinet.root_physx_view
        env_ids = torch.arange(len(masses), dtype=torch.long, device="cpu")

        all_masses = view.get_masses()
        all_masses[env_ids, body_idx] = torch.tensor(masses, dtype=all_masses.dtype)
        view.set_masses(all_masses, env_ids)

        default_mass = cabinet.data.default_mass[env_ids, body_idx].cpu()
        ratios = all_masses[env_ids, body_idx] / default_mass
        inertias = view.get_inertias()
        inertias[env_ids, body_idx] = cabinet.data.default_inertia[env_ids, body_idx].cpu() * ratios.unsqueeze(-1)
        view.set_inertias(inertias, env_ids)

    """
    Reads.
    """

    @staticmethod
    def _read_from_simulator(cabinet: Articulation, joint_idx: int, body_idx: int) -> dict[str, list[float]]:
        """Read :math:`\\xi` back out of PhysX itself, keyed by :data:`XI_FIELDS`."""
        view = cabinet.root_physx_view
        friction = view.get_dof_friction_properties()[:, joint_idx, :]
        return {
            "drawer_mass": view.get_masses()[:, body_idx].tolist(),
            "joint_static_friction": friction[:, _STATIC].tolist(),
            "joint_dynamic_friction": friction[:, _DYNAMIC].tolist(),
            "joint_damping": view.get_dof_dampings()[:, joint_idx].tolist(),
        }

    @staticmethod
    def _read_from_mirror(cabinet: Articulation, joint_idx: int) -> dict[str, list[float]]:
        """Read the joint quantities out of Isaac Lab's own buffers, for comparison.

        Mass is absent on purpose: Isaac Lab keeps only ``data.default_mass``, not a live
        mirror of the current mass, so there is nothing to disagree with.
        """
        return {
            "joint_static_friction": cabinet.data.joint_friction_coeff[:, joint_idx].tolist(),
            "joint_dynamic_friction": cabinet.data.joint_dynamic_friction_coeff[:, joint_idx].tolist(),
            "joint_damping": cabinet.data.joint_damping[:, joint_idx].tolist(),
        }

    def _matches_request(self, readback: dict[str, list[float]], per_env: list[DynamicsParameters]) -> bool:
        """Whether the simulator holds every value that was requested."""
        return all(
            _close(readback[field][index], getattr(params, field), self.READBACK_TOLERANCE)
            for field in XI_FIELDS
            for index, params in enumerate(per_env)
        )

    @staticmethod
    def _broadcast(
        params: DynamicsParameters | Sequence[DynamicsParameters], num_envs: int
    ) -> list[DynamicsParameters]:
        if isinstance(params, DynamicsParameters):
            return [params] * num_envs
        per_env = list(params)
        if len(per_env) != num_envs:
            raise ValueError(f"Expected 1 or {num_envs} parameter sets, got {len(per_env)}.")
        return per_env


def _close(left: float, right: float, tolerance: float = 1e-3) -> bool:
    """Relative comparison that stays meaningful when both values are near zero."""
    return abs(left - right) <= tolerance * max(1.0, abs(right))

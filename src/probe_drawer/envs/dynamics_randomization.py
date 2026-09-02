"""The hidden dynamics of the drawer, and the one place that changes them.

The hidden state this project studies is

.. math:: \\xi = [\\text{drawer mass}, \\text{drawer friction}, \\text{drawer damping}]

Each element maps onto a specific PhysX quantity of the *top drawer*, and nothing else:

``drawer_mass``
    Mass of the ``drawer_top`` rigid body (kg).  Inertia is scaled by the same ratio, as
    Isaac Lab's own ``randomize_rigid_body_mass`` event does.  The rigidly attached
    ``drawer_handle_top`` body keeps its own small mass, so the total moving mass is
    ``drawer_mass + handle_mass``; :attr:`AppliedDynamics.total_moving_mass` reports it.
``joint_friction``
    Static *and* dynamic Coulomb friction coefficient of the ``drawer_top_joint``
    prismatic joint.  Both are set: leaving the dynamic coefficient at zero would give a
    drawer that resists starting and then slides freely, which is not what "a stiff drawer"
    means.
``joint_damping``
    Viscous damping of the ``drawer_top_joint`` *drive* (N s/m).  PhysX 5 also has a
    separate per-DOF viscous friction channel; it is held at zero so that "damping" maps
    onto exactly one simulator quantity -- the same one the official cabinet configures.

The cabinet's other joints are never touched, so a change here cannot be confused with a
change to the doors or the bottom drawer.

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
    "REFERENCE_DURATION",
    "REFERENCE_PEAK_FORCE",
    "AppliedDynamics",
    "DynamicsParameters",
    "DynamicsRandomizer",
    "DynamicsRandomizerCfg",
    "PRESETS",
    "preset",
]


@dataclass(frozen=True)
class DynamicsParameters:
    """One drawer's hidden dynamics.

    Args:
        drawer_mass: Mass of the ``drawer_top`` body (kg).
        joint_friction: Coulomb friction coefficient of the drawer's prismatic joint,
            applied as both the static and the dynamic coefficient.
        joint_damping: Viscous damping of the drawer joint drive (N s/m).
        joint_stiffness: Stiffness of the drawer joint drive (N/m). ``0`` by default, so
            the drawer is a pure mass-friction-damper system and the hidden state is
            exactly the three quantities the study varies (``docs/DECISIONS.md`` D008).
        name: Label for logs and plots.
    """

    drawer_mass: float
    joint_friction: float
    joint_damping: float
    joint_stiffness: float = 0.0
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.drawer_mass <= 0.0:
            raise ValueError(f"drawer_mass must be > 0 kg, got {self.drawer_mass}.")
        for field_name in ("joint_friction", "joint_damping", "joint_stiffness"):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be >= 0, got {value}.")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "drawer_mass": self.drawer_mass,
            "joint_friction": self.joint_friction,
            "joint_damping": self.joint_damping,
            "joint_stiffness": self.joint_stiffness,
        }


#: Deterministic presets.
#:
#: ``nominal`` reproduces the official cabinet's own drawer mass and damping with friction
#: and stiffness removed.  ``easy``/``medium``/``hard`` are **measured**, not guessed: they
#: were chosen by sweeping candidates through ``ExecutionPullController`` at the reference
#: operating point ``peak_force=5 N, duration=2 s`` and keeping the triple that separates
#: cleanly without touching the drawer's 0.4 m travel limit. At that operating point they
#: produce final displacements of roughly 309 mm, 141 mm and 60 mm respectively, with no
#: safety aborts and a TCP lateral drift of at most 4.2 mm. See ``docs/VALIDATION.md``.
PRESETS: dict[str, DynamicsParameters] = {
    "nominal": DynamicsParameters(drawer_mass=5.175, joint_friction=0.0, joint_damping=1.0, name="nominal"),
    "easy": DynamicsParameters(drawer_mass=5.0, joint_friction=1.5, joint_damping=4.0, name="easy"),
    "medium": DynamicsParameters(drawer_mass=8.0, joint_friction=3.0, joint_damping=6.0, name="medium"),
    "hard": DynamicsParameters(drawer_mass=10.0, joint_friction=4.0, joint_damping=9.0, name="hard"),
}

#: Reference operating point the presets were calibrated at, and the point every
#: dynamics-discrimination check uses.
REFERENCE_PEAK_FORCE = 5.0
REFERENCE_DURATION = 2.0


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

    Args:
        cabinet_asset: Scene name of the cabinet articulation.
        drawer_joint_name: Prismatic joint whose friction/damping/stiffness are set.
        drawer_body_name: Rigid body whose mass is set.
        handle_body_name: Rigidly attached handle, read only to report the total moving mass.
        mass_range: Sampling range for ``drawer_mass`` (kg).
        friction_range: Sampling range for ``joint_friction``.
        damping_range: Sampling range for ``joint_damping`` (N s/m).
        joint_stiffness: Stiffness used by sampled parameters (N/m).
    """

    cabinet_asset: str = "cabinet"
    drawer_joint_name: str = "drawer_top_joint"
    drawer_body_name: str = "drawer_top"
    handle_body_name: str = "drawer_handle_top"

    mass_range: tuple[float, float] = (4.0, 12.0)
    friction_range: tuple[float, float] = (0.5, 5.0)
    damping_range: tuple[float, float] = (3.0, 10.0)
    joint_stiffness: float = 0.0


@dataclass
class AppliedDynamics:
    """What :meth:`DynamicsRandomizer.apply` actually wrote, read back from the simulation.

    ``requested`` is what was asked for; ``readback`` is what PhysX reports afterwards.
    Keeping both is what makes it impossible to believe a parameter was applied when it was
    silently ignored.
    """

    requested: list[DynamicsParameters]
    readback: dict[str, list[float]]
    handle_mass: float
    consistent: bool
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
            "preset_name": self.preset_name,
            "requested": [p.as_dict() for p in self.requested],
            "readback": self.readback,
            "handle_mass": self.handle_mass,
            "total_moving_mass": self.total_moving_mass,
            "consistent": self.consistent,
            "notes": self.notes,
        }


class DynamicsRandomizer:
    """Samples and applies the drawer's hidden dynamics.

    Args:
        cfg: Sampling ranges and target entity names.
        seed: Seed for :meth:`sample`. ``None`` leaves the generator unseeded.

    Example:
        >>> randomizer = DynamicsRandomizer(seed=0)
        >>> applied = randomizer.apply(env, randomizer.sample(env.num_envs))
        >>> applied.consistent
        True
    """

    def __init__(self, cfg: DynamicsRandomizerCfg | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or DynamicsRandomizerCfg()
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        self._current: AppliedDynamics | None = None

    def sample(self, num_envs: int = 1) -> list[DynamicsParameters]:
        """Draw ``num_envs`` independent parameter sets uniformly from the configured ranges."""
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        draws = torch.rand(num_envs, 3, generator=self._generator).tolist()
        ranges = (self.cfg.mass_range, self.cfg.friction_range, self.cfg.damping_range)
        return [
            DynamicsParameters(
                drawer_mass=ranges[0][0] + u[0] * (ranges[0][1] - ranges[0][0]),
                joint_friction=ranges[1][0] + u[1] * (ranges[1][1] - ranges[1][0]),
                joint_damping=ranges[2][0] + u[2] * (ranges[2][1] - ranges[2][0]),
                joint_stiffness=self.cfg.joint_stiffness,
                name="sampled",
            )
            for u in draws
        ]

    def apply(
        self, env: ManagerBasedRLEnv, params: DynamicsParameters | Sequence[DynamicsParameters]
    ) -> AppliedDynamics:
        """Write the given dynamics into the simulation and read them back.

        Args:
            env: The running environment.
            params: One parameter set broadcast to every environment, or one per environment.

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
        device = env.device

        self._write_mass(cabinet, body_idx, [p.drawer_mass for p in per_env])

        joint_ids = [joint_idx]
        as_column = lambda values: torch.tensor(values, device=device).unsqueeze(-1)  # noqa: E731
        friction = as_column([p.joint_friction for p in per_env])
        # Static and dynamic must be written in one call: PhysX rejects an update in which
        # the dynamic effort momentarily exceeds the static one, which is what two
        # sequential writes produce when the friction is being raised.
        cabinet.write_joint_friction_coefficient_to_sim(
            friction,
            joint_dynamic_friction_coeff=friction,
            joint_viscous_friction_coeff=torch.zeros_like(friction),
            joint_ids=joint_ids,
        )
        cabinet.write_joint_damping_to_sim(as_column([p.joint_damping for p in per_env]), joint_ids=joint_ids)
        cabinet.write_joint_stiffness_to_sim(as_column([p.joint_stiffness for p in per_env]), joint_ids=joint_ids)

        readback = {
            "drawer_mass": cabinet.root_physx_view.get_masses()[:, body_idx].tolist(),
            "joint_friction": cabinet.data.joint_friction_coeff[:, joint_idx].tolist(),
            "joint_dynamic_friction": cabinet.data.joint_dynamic_friction_coeff[:, joint_idx].tolist(),
            "joint_damping": cabinet.data.joint_damping[:, joint_idx].tolist(),
            "joint_stiffness": cabinet.data.joint_stiffness[:, joint_idx].tolist(),
        }
        consistent = all(
            abs(readback[key][i] - getattr(per_env[i], key)) <= 1e-3 * max(1.0, abs(getattr(per_env[i], key)))
            for key in ("drawer_mass", "joint_friction", "joint_damping", "joint_stiffness")
            for i in range(env.num_envs)
        )

        self._current = AppliedDynamics(
            requested=per_env,
            readback=readback,
            handle_mass=float(cabinet.root_physx_view.get_masses()[0, handle_idx]),
            consistent=consistent,
            notes={
                "drawer_joint": self.cfg.drawer_joint_name,
                "drawer_body": self.cfg.drawer_body_name,
                "friction_channels": ["static", "dynamic"],
            },
        )
        return self._current

    def get_current_params(self) -> AppliedDynamics | None:
        """The dynamics most recently applied, or ``None`` before the first :meth:`apply`.

        This is the privileged state ``xi``; it is recorded with every episode and is used
        only for training and analysis, never as a controller input.
        """
        return self._current

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
        inertias[env_ids, body_idx] = (
            cabinet.data.default_inertia[env_ids, body_idx].cpu() * ratios.unsqueeze(-1)
        )
        view.set_inertias(inertias, env_ids)

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

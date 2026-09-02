"""Choosing what to record: which drawers, and which forces to ask about.

Three samplers, and what each is *forbidden* from seeing is as important as what it does.

**Hidden states** come from a scrambled Sobol sequence over
``[m, mu_static, ratio, b]``, with ``mu_dynamic = ratio * mu_static``. The ratio
parameterisation rather than an absolute dynamic friction is not a convenience: PhysX
requires ``mu_d <= mu_s`` and silently discards a write that violates it (D016), so
sampling a ratio keeps every draw inside the valid region by construction instead of by
rejection.

**Candidate forces** are stratified over the whole task force range and jittered
deterministically from the hidden state's own identifier. The sampler may not read a label,
an Oracle band, or that hidden state's best force. Concentrating candidates near each
drawer's success band would spend the budget better and would also make the training
distribution a function of the labels, which is a different experiment from the one being
run (D035).

**Branch order** is a deterministic shuffle of the candidates within one probe, because
executing them in force order would correlate the measured branch drift with force -- the
exact axis the model learns (D040).

Nothing here imports Isaac Lab, and nothing here reads an outcome.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
from torch.quasirandom import SobolEngine

from probe_drawer.dataset.schema import XI_DIMENSIONS, xi_id

__all__ = [
    "ForceSamplerCfg",
    "SamplingPlan",
    "XiSamplerCfg",
    "branch_order",
    "build_plan",
    "candidate_forces",
    "sample_hidden_states",
]


@dataclass(frozen=True)
class XiSamplerCfg:
    """How the hidden states are drawn.

    Args:
        num_states: How many to draw. A power of two, because a Sobol sequence's uniformity
            guarantees hold at powers of two and degrade between them.
        seed: Scrambling seed. Fixed, so the dataset is reproducible.
        mass: ``m`` range (kg).
        static_friction: ``mu_static`` range (N).
        dynamic_friction_ratio: ``mu_dynamic / mu_static`` range, in ``(0, 1]``.
        damping: ``b`` range (N*s/m).

    The ranges default to ``TRAINING_XI_RANGES``. They are duplicated here rather than
    imported so that this module stays free of ``experiment_plan``, which is the *experiment*
    definition; the generator passes the authoritative ranges in.
    """

    num_states: int = 512
    seed: int = 20260902
    mass: tuple[float, float] = (4.0, 12.0)
    static_friction: tuple[float, float] = (0.5, 3.0)
    dynamic_friction_ratio: tuple[float, float] = (0.3, 1.0)
    damping: tuple[float, float] = (2.0, 10.0)

    def __post_init__(self) -> None:
        if self.num_states < 1:
            raise ValueError(f"num_states must be >= 1, got {self.num_states}.")
        for name in ("mass", "static_friction", "dynamic_friction_ratio", "damping"):
            low, high = getattr(self, name)
            if not low < high:
                raise ValueError(f"{name} range must be increasing, got ({low}, {high}).")
        if self.dynamic_friction_ratio[1] > 1.0:
            raise ValueError(
                f"dynamic_friction_ratio must stay <= 1: PhysX discards mu_d > mu_s writes "
                f"(docs/DECISIONS.md D016). Got {self.dynamic_friction_ratio}."
            )
        if self.dynamic_friction_ratio[0] <= 0.0:
            raise ValueError(f"dynamic_friction_ratio must be > 0, got {self.dynamic_friction_ratio}.")

    def as_dict(self) -> dict:
        return {
            "method": "scrambled Sobol",
            "num_states": self.num_states,
            "seed": self.seed,
            "parameterisation": ["mass", "static_friction", "dynamic_friction_ratio", "damping"],
            "mass": list(self.mass),
            "static_friction": list(self.static_friction),
            "dynamic_friction_ratio": list(self.dynamic_friction_ratio),
            "damping": list(self.damping),
        }


@dataclass(frozen=True)
class ForceSamplerCfg:
    """How the candidate forces for one probe are drawn.

    Args:
        count: Candidates per probe.
        force_range: The whole span to stratify (N). Authoritative source is
            ``MAIN_TASK.peak_force_range``; the generator passes it in.
        jitter: Fraction of a stratum's width the sample may move from its centre, in
            ``[0, 0.5]``. ``0`` puts every sample at its stratum centre, which would make
            every hidden state share one force grid and lose the coverage between grid
            points. ``0.5`` allows a sample anywhere in its stratum, including exactly on a
            boundary next to its neighbour.
        seed: Mixed with the hidden state's identifier, so the jitter is reproducible and
            differs between hidden states.
    """

    count: int = 24
    force_range: tuple[float, float] = (0.15, 4.5)
    jitter: float = 0.4
    seed: int = 20260902

    def __post_init__(self) -> None:
        if self.count < 2:
            raise ValueError(f"count must be >= 2, got {self.count}.")
        low, high = self.force_range
        if not 0.0 < low < high:
            raise ValueError(f"force_range must satisfy 0 < low < high, got {self.force_range}.")
        if not 0.0 <= self.jitter <= 0.5:
            raise ValueError(f"jitter must be in [0, 0.5], got {self.jitter}.")

    def as_dict(self) -> dict:
        return {
            "method": "label-independent stratified with per-hidden-state jitter",
            "count": self.count,
            "force_range": list(self.force_range),
            "jitter": self.jitter,
            "seed": self.seed,
            "reads_labels": False,
        }


def sample_hidden_states(cfg: XiSamplerCfg | None = None) -> list[dict]:
    """Draw the hidden states, in a stable order.

    Returns a list of ``{mass, static_friction, dynamic_friction, damping}`` dicts. Index
    stability is a property of the Sobol sequence: drawing ``n`` points always yields the
    same first ``k`` as drawing ``k``, so extending a dataset later does not renumber the
    hidden states it already has.

    ``xi_id`` is still the content hash of the four values, never the index, so a hidden
    state keeps its identity even if the sampler is replaced.
    """
    cfg = cfg or XiSamplerCfg()
    engine = SobolEngine(dimension=4, scramble=True, seed=cfg.seed)
    unit = engine.draw(cfg.num_states).double()

    bounds = (cfg.mass, cfg.static_friction, cfg.dynamic_friction_ratio, cfg.damping)
    scaled = torch.stack(
        [unit[:, index] * (high - low) + low for index, (low, high) in enumerate(bounds)], dim=1
    )

    states = []
    for mass, static, ratio, damping in scaled.tolist():
        states.append(
            {
                "mass": mass,
                "static_friction": static,
                "dynamic_friction": ratio * static,
                "damping": damping,
            }
        )
    return states


def _unit_draws(count: int, *parts: object) -> list[float]:
    """``count`` reproducible numbers in ``[0, 1)``, derived from ``parts``.

    A hash rather than a seeded PRNG so the draws depend only on the arguments, not on how
    many times anything has been called before.
    """
    draws = []
    for index in range(count):
        payload = "|".join(str(part) for part in (*parts, index)).encode()
        digest = hashlib.sha256(payload).digest()
        draws.append(int.from_bytes(digest[:8], "big") / 2**64)
    return draws


def candidate_forces(state_id: str, cfg: ForceSamplerCfg | None = None) -> tuple[float, ...]:
    """The candidate peak forces to ask about, for one hidden state.

    Identical for every probe repeat of that hidden state, deliberately: the same
    ``(xi, F_candidate)`` question then has three independent probes behind it, which is what
    turns a binary label into a measurable success *probability* (D036).

    Args:
        state_id: The hidden state's ``xi_id``. The only thing the jitter depends on, and
            the only per-drawer information this sampler is allowed.
        cfg: Stratification settings.

    Returns:
        ``cfg.count`` forces in ascending order (N). Ordering is presentational -- the
        generator executes them in a shuffled order, see :func:`branch_order`.
    """
    cfg = cfg or ForceSamplerCfg()
    low, high = cfg.force_range
    width = (high - low) / cfg.count
    draws = _unit_draws(cfg.count, "candidate-force", cfg.seed, state_id)

    forces = []
    for index, draw in enumerate(draws):
        centre = low + (index + 0.5) * width
        forces.append(round(centre + (draw - 0.5) * 2.0 * cfg.jitter * width, 6))
    return tuple(forces)


def branch_order(probe: str, count: int) -> tuple[int, ...]:
    """A deterministic permutation of ``range(count)``, for one probe.

    Branching drifts: the outcome of the Nth candidate execution from one snapshot falls
    slightly with N (up to 0.17 ``eps_d`` over 24 branches). Executing candidates in force
    order would make that drift a function of force. Shuffling makes it noise instead
    (``docs/COUNTERFACTUAL_BRANCHING.md`` 5.2).

    Keyed on the probe rather than the hidden state, so the three repeats of one hidden state
    use three different orders and the drift cannot align with force even across repeats.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}.")
    draws = _unit_draws(count, "branch-order", probe)
    return tuple(index for _, index in sorted(zip(draws, range(count), strict=True)))


@dataclass
class SamplingPlan:
    """Everything to record, decided before the simulator starts.

    Deciding it up front rather than inside the generator loop is what makes the plan
    inspectable, testable without a simulator, and identical across a resumed run.

    Attributes:
        states: The hidden states, in Sobol order.
        repeats: Independent probe episodes per hidden state.
        forces: ``xi_id -> candidate forces``.
        xi_cfg: How the states were drawn.
        force_cfg: How the forces were drawn.
    """

    states: list[dict]
    repeats: int
    forces: dict[str, tuple[float, ...]] = field(default_factory=dict)
    xi_cfg: XiSamplerCfg = field(default_factory=XiSamplerCfg)
    force_cfg: ForceSamplerCfg = field(default_factory=ForceSamplerCfg)

    @property
    def num_probes(self) -> int:
        return len(self.states) * self.repeats

    @property
    def num_candidates(self) -> int:
        return self.num_probes * self.force_cfg.count

    def as_dict(self) -> dict:
        return {
            "num_hidden_states": len(self.states),
            "probe_repeats": self.repeats,
            "candidates_per_probe": self.force_cfg.count,
            "num_probes": self.num_probes,
            "num_candidates": self.num_candidates,
            "xi_sampler": self.xi_cfg.as_dict(),
            "force_sampler": self.force_cfg.as_dict(),
        }


def build_plan(
    repeats: int = 3, xi_cfg: XiSamplerCfg | None = None, force_cfg: ForceSamplerCfg | None = None
) -> SamplingPlan:
    """Draw the hidden states and their candidate forces, without touching a simulator."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")
    xi_cfg = xi_cfg or XiSamplerCfg()
    force_cfg = force_cfg or ForceSamplerCfg()
    states = sample_hidden_states(xi_cfg)
    forces = {xi_id(state): candidate_forces(xi_id(state), force_cfg) for state in states}
    if len(forces) != len(states):
        raise ValueError(
            f"the sampler produced {len(states)} states but only {len(forces)} distinct "
            "identifiers; two draws collided and the plan would silently lose a hidden state."
        )
    return SamplingPlan(
        states=states, repeats=repeats, forces=forces, xi_cfg=xi_cfg, force_cfg=force_cfg
    )

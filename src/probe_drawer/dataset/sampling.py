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
    "SAMPLED_AXES",
    "ForceSamplerCfg",
    "SamplingPlan",
    "XiSamplerCfg",
    "branch_order",
    "build_plan",
    "candidate_forces",
    "representative_hidden_states",
    "sample_hidden_states",
    "scale_unit_box",
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


#: Order the four sampled coordinates appear in, before ``ratio`` becomes ``mu_dynamic``.
SAMPLED_AXES = ("mass", "static_friction", "dynamic_friction_ratio", "damping")


def scale_unit_box(unit, bounds: dict) -> list[dict]:
    r"""Map points in :math:`[0,1]^4` onto hidden states.

    One implementation, because both samplers need the same two things and getting either
    wrong is silent: the affine scaling onto each axis's range, and
    ``mu_dynamic = ratio * mu_static``, which is what keeps ``mu_d <= mu_s`` true by
    construction rather than by rejection (D016).

    Args:
        unit: ``(n, 4)`` array-like of points in the unit box, ordered as
            :data:`SAMPLED_AXES`.
        bounds: ``{axis: (low, high)}`` for each of :data:`SAMPLED_AXES`.

    Returns:
        ``n`` dicts with ``mass``, ``static_friction``, ``dynamic_friction``, ``damping``.
    """
    import numpy as _np  # noqa: PLC0415 - kept local so the module imports without numpy at rest

    values = _np.asarray(unit, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(SAMPLED_AXES):
        raise ValueError(f"unit must be (n, {len(SAMPLED_AXES)}), got {values.shape}")
    lows = _np.array([bounds[name][0] for name in SAMPLED_AXES])
    highs = _np.array([bounds[name][1] for name in SAMPLED_AXES])
    scaled = values * (highs - lows) + lows
    return [
        {
            "mass": float(mass),
            "static_friction": float(static),
            "dynamic_friction": float(ratio * static),
            "damping": float(damping),
        }
        for mass, static, ratio, damping in scaled
    ]


def representative_hidden_states(count: int = 48, seed: int = 20260902, ranges: dict | None = None) -> list[dict]:
    r"""Hidden states covering the box's corners *and* its interior.

    A sweep over ``easy``/``medium``/``hard`` presets walks one diagonal of the
    four-dimensional box and would miss, for instance, a light drawer with high static
    friction -- exactly the combination Phase 10 found hardest. So the first
    :math:`2^4 = 16` states are the corners, and the rest are a scrambled Sobol fill.

    Corners are pulled 5 % inside each bound: on the bound itself a coordinate sits at the
    edge of what the randomiser accepts, and 5 % keeps every draw strictly inside a region
    already known to run.

    Distinct from :func:`sample_hidden_states`, which draws a plain Sobol sequence and is what
    a *dataset* uses. This one guarantees corner coverage and is what a small **sweep** uses,
    where 16-48 states have to span the box rather than sample it representatively.

    Args:
        count: Total states. At least 16, so the corners always fit.
        seed: Sobol scrambling seed.
        ranges: ``{axis: (low, high)}``. Defaults to the training ranges.

    Raises:
        ValueError: If ``count`` is below 16.
    """
    from probe_drawer.experiment_plan import TRAINING_XI_RANGES  # noqa: PLC0415 - avoids a cycle

    if count < 16:
        raise ValueError(f"count must be >= 16 so all 16 corners fit, got {count}.")
    bounds = ranges or {
        "mass": TRAINING_XI_RANGES.mass,
        "static_friction": TRAINING_XI_RANGES.static_friction,
        "dynamic_friction_ratio": TRAINING_XI_RANGES.dynamic_friction_ratio,
        "damping": TRAINING_XI_RANGES.damping,
    }
    inset = 0.05
    unit = [
        [(1.0 - inset) if (index >> axis) & 1 else inset for axis in range(4)] for index in range(16)
    ]
    remaining = count - 16
    if remaining:
        engine = SobolEngine(dimension=4, scramble=True, seed=seed)
        unit = unit + engine.draw(remaining).double().tolist()
    return scale_unit_box(unit, bounds)


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
    unit = engine.draw(cfg.num_states).double().tolist()
    bounds = {
        "mass": cfg.mass,
        "static_friction": cfg.static_friction,
        "dynamic_friction_ratio": cfg.dynamic_friction_ratio,
        "damping": cfg.damping,
    }
    return scale_unit_box(unit, bounds)


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

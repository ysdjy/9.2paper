"""Time-parameterised pull-axis force profiles.

A force profile answers exactly one question -- *"how much pull-axis force is commanded
at time t?"* -- and nothing else.  It touches no simulator state, so it is deterministic,
reproducible and unit-testable without Isaac Sim.

Two profiles exist:

:class:`RampForceProfile`
    The **probe** input: a fixed, reproducible increase from ``initial_force`` to
    ``max_force`` over ``ramp_duration`` seconds, then held at ``max_force``.
:class:`TrapezoidForceProfile`
    The **execution** input: a normalised shape ``phi(t/T)`` that rises smoothly from 0,
    holds at 1, and falls smoothly back to 0.  The commanded force is
    ``F(t) = peak_force * phi(t / duration)``, so changing ``peak_force`` scales the curve
    without changing its shape.  This invariance is asserted by
    ``tests/unit/test_force_profiles.py`` and is the reason the shape lives in the profile
    rather than in the controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["ForceProfile", "RampForceProfile", "RampShape", "TrapezoidForceProfile", "smoothstep"]

RampShape = str
"""Interpolation used by the ramp segments: ``"linear"``, ``"smoothstep"`` or ``"cosine"``."""

_RAMP_SHAPES: tuple[RampShape, ...] = ("linear", "smoothstep", "cosine")


def smoothstep(x: np.ndarray | float, shape: RampShape = "smoothstep") -> np.ndarray | float:
    """Monotone 0->1 interpolation on ``[0, 1]``, clamped outside it.

    ``"linear"`` is ``x``; ``"smoothstep"`` is the C1 Hermite blend ``3x^2 - 2x^3``;
    ``"cosine"`` is ``(1 - cos(pi x)) / 2``.  All three satisfy ``s(0) == 0``,
    ``s(1) == 1`` and are non-decreasing.
    """
    if shape not in _RAMP_SHAPES:
        raise ValueError(f"Unknown ramp shape {shape!r}. Expected one of {_RAMP_SHAPES}.")
    u = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if shape == "linear":
        out = u
    elif shape == "smoothstep":
        out = u * u * (3.0 - 2.0 * u)
    else:
        out = 0.5 * (1.0 - np.cos(np.pi * u))
    return float(out) if np.ndim(x) == 0 else out


class ForceProfile(ABC):
    """Maps elapsed episode time to a commanded pull-axis force in newtons."""

    @abstractmethod
    def force(self, t: np.ndarray | float) -> np.ndarray | float:
        """Commanded pull-axis force (N) at elapsed time ``t`` (s)."""

    @property
    @abstractmethod
    def nominal_duration(self) -> float:
        """Duration (s) after which the profile no longer changes shape."""

    def describe(self) -> dict:
        """Serialisable description, stored in every episode's metadata."""
        return {"type": type(self).__name__, **{k: v for k, v in vars(self).items() if not k.startswith("_")}}


@dataclass
class RampForceProfile(ForceProfile):
    """Probe input: rise from ``initial_force`` to ``max_force``, then hold.

    Args:
        initial_force: Force applied at ``t = 0`` (N).
        max_force: Force reached at ``t = ramp_duration`` and held thereafter (N).
        ramp_duration: Time taken to go from ``initial_force`` to ``max_force`` (s).
        shape: Interpolation between the two force levels. ``"linear"`` by default because
            a constant force rate is the most interpretable standardised probe input.
    """

    initial_force: float
    max_force: float
    ramp_duration: float
    shape: RampShape = "linear"

    def __post_init__(self) -> None:
        if self.ramp_duration <= 0.0:
            raise ValueError(f"ramp_duration must be > 0, got {self.ramp_duration}.")
        if self.max_force < self.initial_force:
            raise ValueError(
                f"max_force ({self.max_force}) must be >= initial_force ({self.initial_force}) "
                "-- the probe input is monotonically non-decreasing by definition."
            )

    def force(self, t: np.ndarray | float) -> np.ndarray | float:
        alpha = smoothstep(np.asarray(t, dtype=float) / self.ramp_duration, self.shape)
        out = self.initial_force + (self.max_force - self.initial_force) * alpha
        return float(out) if np.ndim(t) == 0 else out

    @property
    def nominal_duration(self) -> float:
        return self.ramp_duration


@dataclass
class TrapezoidForceProfile(ForceProfile):
    """Execution input: smooth rise, flat hold, smooth fall, scaled by ``peak_force``.

    The shape is defined on normalised time ``tau = t / duration`` and is therefore
    independent of both ``peak_force`` and ``duration``::

        F(t) = peak_force * phi(t / duration)

    Args:
        peak_force: Plateau force (N).
        duration: Total execution time (s).  ``phi(0) == phi(1) == 0``.
        rise_fraction: Fraction of ``duration`` spent rising from 0 to ``peak_force``.
        fall_fraction: Fraction of ``duration`` spent falling back to 0.
        shape: Interpolation used by the rise and fall segments.
    """

    peak_force: float
    duration: float
    rise_fraction: float = 0.1
    fall_fraction: float = 0.1
    shape: RampShape = "smoothstep"

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError(f"duration must be > 0, got {self.duration}.")
        if not 0.0 < self.rise_fraction < 1.0 or not 0.0 < self.fall_fraction < 1.0:
            raise ValueError("rise_fraction and fall_fraction must each lie strictly inside (0, 1).")
        if self.rise_fraction + self.fall_fraction > 1.0:
            raise ValueError(
                f"rise_fraction + fall_fraction must be <= 1, got "
                f"{self.rise_fraction} + {self.fall_fraction} = {self.rise_fraction + self.fall_fraction}."
            )

    def normalized(self, tau: np.ndarray | float) -> np.ndarray | float:
        """The shape function ``phi(tau)`` on normalised time ``tau in [0, 1]``.

        Outside ``[0, 1]`` the profile is zero: the execution force is off before the
        episode starts and after it ends.
        """
        u = np.asarray(tau, dtype=float)
        rise = smoothstep(u / self.rise_fraction, self.shape)
        fall = smoothstep((1.0 - u) / self.fall_fraction, self.shape)
        phi = np.minimum(rise, fall)
        phi = np.where((u < 0.0) | (u > 1.0), 0.0, phi)
        return float(phi) if np.ndim(tau) == 0 else phi

    def force(self, t: np.ndarray | float) -> np.ndarray | float:
        out = self.peak_force * np.asarray(self.normalized(np.asarray(t, dtype=float) / self.duration))
        return float(out) if np.ndim(t) == 0 else out

    @property
    def nominal_duration(self) -> float:
        return self.duration

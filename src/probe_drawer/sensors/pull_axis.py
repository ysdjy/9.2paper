"""The definition of *which direction opening the drawer is*.

Kept in its own module -- with no Isaac Lab dependency of any kind -- because both the
environment configuration and the controllers need it, and because it is worth unit-testing
without launching Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["PullAxis"]

_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class PullAxis:
    """The drawer's opening direction, as a signed axis of the robot base frame.

    The operational-space controller can only select *coordinate axes* of its task frame
    for force control, so the pull direction has to be axis-aligned.  That this holds for
    the official cabinet layout is verified empirically rather than assumed -- see
    ``scripts/run_official_drawer.py --measure-geometry`` and
    :meth:`DrawerStateReader.verify_pull_axis`.

    Args:
        index: Base-frame axis index, ``0`` for x, ``1`` for y, ``2`` for z.
        sign: ``+1`` if the drawer opens along the positive axis, ``-1`` otherwise.
    """

    index: int = 0
    sign: float = -1.0

    def __post_init__(self) -> None:
        if self.index not in (0, 1, 2):
            raise ValueError(f"PullAxis.index must be 0, 1 or 2, got {self.index}.")
        if self.sign not in (1.0, -1.0):
            raise ValueError(f"PullAxis.sign must be +1.0 or -1.0, got {self.sign}.")

    @property
    def name(self) -> str:
        """Human-readable axis label, e.g. ``"-x"``."""
        return f"{'+' if self.sign > 0 else '-'}{_AXIS_NAMES[self.index]}"

    def direction(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Unit vector of the opening direction in the base frame, shape ``(3,)``."""
        d = torch.zeros(3, device=device)
        d[self.index] = self.sign
        return d

    def motion_control_axes(self) -> tuple[int, int, int, int, int, int]:
        """OSC ``motion_control_axes_task``: pose-hold on every axis but the pull axis."""
        axes = [1, 1, 1, 1, 1, 1]
        axes[self.index] = 0
        return tuple(axes)  # type: ignore[return-value]

    def wrench_control_axes(self) -> tuple[int, int, int, int, int, int]:
        """OSC ``contact_wrench_control_axes_task``: force control on the pull axis only."""
        axes = [0, 0, 0, 0, 0, 0]
        axes[self.index] = 1
        return tuple(axes)  # type: ignore[return-value]

    def as_dict(self) -> dict:
        return {"index": self.index, "sign": self.sign, "name": self.name}

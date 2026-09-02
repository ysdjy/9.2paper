"""Causal differentiation of a signal sampled once per control step.

Every derivative this project reports is computed here, for two reasons.

**Causality.** A filter that looks ahead cannot run on a robot. Anything a deployed policy
may read must be computable from samples that have already happened, so these
differentiators only ever use the current and previous samples -- no centred differences,
no zero-phase filtering, no offline smoothing.

**Auditability.** Both the unsmoothed difference and the smoothed one are exposed, so the
effect of the filter on any recorded episode can be checked after the fact rather than
taken on trust. The filter is a plain moving average over the last ``window`` differences:
it is the shortest filter that cancels an alternation at half the sampling rate, which is
exactly the artefact the drawer's contact chatter produces (``docs/DECISIONS.md`` D009).

Lag is ``(window - 1) / 2`` steps, i.e. 8 ms at ``window = 2`` and 25 ms at ``window = 4``
on a 60 Hz loop.
"""

from __future__ import annotations

import torch

__all__ = ["CausalDerivative"]


class CausalDerivative:
    """Tracks one signal and its causal derivative.

    Args:
        dt: Sampling interval (s). Must be positive.
        window: Number of consecutive differences the moving average spans. ``1`` disables
            smoothing, leaving the plain one-step difference.

    Example:
        >>> derivative = CausalDerivative(dt=1 / 60, window=2)
        >>> for sample in samples:
        ...     derivative.update(sample)
        ...     use(derivative.value, derivative.filtered, derivative.raw)
    """

    def __init__(self, dt: float, window: int = 2) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}.")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        self.dt = float(dt)
        self.window = int(window)
        self._previous: torch.Tensor | None = None
        self._differences: list[torch.Tensor] = []
        self._value: torch.Tensor | None = None

    @property
    def lag_steps(self) -> float:
        """Group delay the moving average introduces, in control steps."""
        return (self.window - 1) / 2.0

    @property
    def value(self) -> torch.Tensor | None:
        """The most recent sample, or ``None`` before the first :meth:`update`."""
        return self._value

    @property
    def raw(self) -> torch.Tensor:
        """The latest unsmoothed one-step difference, zero before two samples exist."""
        if not self._differences:
            return self._zeros()
        return self._differences[-1]

    @property
    def filtered(self) -> torch.Tensor:
        """Moving average of the last :attr:`window` differences, zero before any exist."""
        if not self._differences:
            return self._zeros()
        return torch.stack(self._differences, dim=0).mean(dim=0)

    def update(self, sample: torch.Tensor) -> None:
        """Take one new sample and advance the derivative estimate."""
        sample = sample.clone()
        if self._previous is not None:
            self._differences.append((sample - self._previous) / self.dt)
            del self._differences[: -self.window]
        self._previous = sample
        self._value = sample

    def reset(self) -> None:
        """Discard all history, as after an environment reset."""
        self._previous = None
        self._differences = []
        self._value = None

    def state_dict(self) -> dict:
        """Everything :meth:`update` depends on, cloned.

        The filter carries genuine state -- the previous sample and the last ``window``
        differences -- so a derived channel is a function of the *history*, not of the
        current instant. Capturing and restoring an episode therefore has to include it, or
        a restored episode's first velocity reading would be wrong (see
        ``docs/COUNTERFACTUAL_BRANCHING.md``).
        """
        return {
            "previous": None if self._previous is None else self._previous.clone(),
            "differences": [difference.clone() for difference in self._differences],
            "value": None if self._value is None else self._value.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a :meth:`state_dict`, replacing whatever history is held now."""
        missing = {"previous", "differences", "value"} - set(state)
        if missing:
            raise KeyError(f"CausalDerivative state is missing {sorted(missing)}.")
        previous = state["previous"]
        value = state["value"]
        self._previous = None if previous is None else previous.clone()
        self._differences = [difference.clone() for difference in state["differences"]]
        self._value = None if value is None else value.clone()

    def describe(self) -> dict:
        """Serialisable filter description, recorded with every episode."""
        return {
            "method": "causal moving average of one-step finite differences",
            "window_steps": self.window,
            "dt": self.dt,
            "lag_steps": self.lag_steps,
            "causal": True,
        }

    def _zeros(self) -> torch.Tensor:
        if self._previous is None:
            raise RuntimeError("CausalDerivative has no samples yet; call update() first.")
        return torch.zeros_like(self._previous)

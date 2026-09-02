"""Structured, reproducible per-episode logging.

Every episode is written as two files in its own directory:

``metadata.json``
    Everything needed to reproduce the episode -- software versions, the project's git
    commit, the controller and its parameters, the hidden dynamics parameters, and the
    per-environment summary.
``trajectory.npz``
    The full :class:`~probe_drawer.controllers.types.PullHistory` as named arrays.

Splitting them this way keeps the metadata greppable and diffable while the bulky arrays
stay in a compact binary format.  Written at collection time, not reconstructed later: the
git commit and dynamics parameters of a past run cannot be recovered after the fact.

.. note::
   This package is named ``logging`` per the project layout.  Absolute imports mean
   ``import logging`` anywhere inside it still resolves to the standard library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from probe_drawer.controllers.types import ExecutionResult, ProbeResult
from probe_drawer.utils import collect_environment_info, project_root

__all__ = ["EpisodeLogger", "EpisodeMetadata", "default_log_root"]


def default_log_root() -> Path:
    """Directory episodes are written to by default."""
    return project_root() / "outputs" / "logs"


@dataclass
class EpisodeMetadata:
    """Reproducibility record written alongside every trajectory."""

    experiment_id: str
    timestamp: str
    git_commit: str | None
    environment: dict[str, Any]
    controller: str
    controller_parameters: dict[str, Any]
    dynamics_parameters: dict[str, Any]
    termination_reason: list[str]
    result: list[dict[str, Any]]
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "environment": self.environment,
            "controller": self.controller,
            "controller_parameters": self.controller_parameters,
            "dynamics_parameters": self.dynamics_parameters,
            "termination_reason": self.termination_reason,
            "result": self.result,
            "notes": self.notes,
        }


class EpisodeLogger:
    """Writes probe and execution episodes to disk in a fixed, documented layout.

    Args:
        root: Directory episodes are written under. Defaults to ``outputs/logs``.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_log_root()
        self._environment = collect_environment_info().as_dict()

    def save(
        self,
        experiment_id: str,
        result: ProbeResult | ExecutionResult,
        dynamics_parameters: dict[str, Any] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> Path:
        """Write one episode and return the directory it was written to.

        Args:
            experiment_id: Directory name for the episode. Re-using an ID overwrites it.
            result: The controller's return value.
            dynamics_parameters: The hidden dynamics the episode ran under. This is the
                privileged state ``xi``; recording it now is what makes it usable for
                training and analysis later.
            notes: Anything else worth keeping with the episode.
        """
        directory = self.root / experiment_id
        directory.mkdir(parents=True, exist_ok=True)

        metadata = EpisodeMetadata(
            experiment_id=experiment_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=self._environment.get("project_git_commit"),
            environment=self._environment,
            controller=str(result.parameters.get("controller", type(result).__name__)),
            controller_parameters=result.parameters,
            dynamics_parameters=dynamics_parameters or {},
            termination_reason=[reason.value for reason in result.termination_reason],
            result=[result.summary(i) for i in range(result.num_envs)],
            notes=notes or {},
        )

        (directory / "metadata.json").write_text(json.dumps(metadata.as_dict(), indent=2, default=str))
        np.savez_compressed(directory / "trajectory.npz", **result.history.as_arrays())
        return directory

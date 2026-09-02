"""Episode protocols: the order things happen in, and nothing else.

A protocol orchestrates the existing controllers and evaluator. It owns no force profiles,
no stop conditions and no physics; those live in ``controllers/`` and ``evaluation/``. The
dependency runs one way::

    script -> protocol -> controllers -> environment

``simulation_snapshot`` is the one exception to "protocols only sequence things": it reaches
into the simulator to freeze and restore an instant. It is a dataset-generation device, not
part of any deployment protocol -- see ``docs/COUNTERFACTUAL_BRANCHING.md``.
"""

from .sequential_pull_protocol import (
    InferenceTransitionCfg,
    SequentialEpisode,
    SequentialProtocolCfg,
    SequentialPullProtocol,
    TransitionRecord,
)
from .simulation_snapshot import SimulationSnapshot, capture_snapshot, restore_snapshot

__all__ = [
    "InferenceTransitionCfg",
    "SequentialEpisode",
    "SequentialProtocolCfg",
    "SequentialPullProtocol",
    "SimulationSnapshot",
    "TransitionRecord",
    "capture_snapshot",
    "restore_snapshot",
]

"""Episode protocols: the order things happen in, and nothing else.

A protocol orchestrates the existing controllers and evaluator. It owns no force profiles,
no stop conditions and no physics; those live in ``controllers/`` and ``evaluation/``. The
dependency runs one way::

    script -> protocol -> controllers -> environment
"""

from .sequential_pull_protocol import (
    InferenceTransitionCfg,
    SequentialEpisode,
    SequentialProtocolCfg,
    SequentialPullProtocol,
    TransitionRecord,
)

__all__ = [
    "InferenceTransitionCfg",
    "SequentialEpisode",
    "SequentialProtocolCfg",
    "SequentialPullProtocol",
    "TransitionRecord",
]

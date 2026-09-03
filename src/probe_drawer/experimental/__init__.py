"""Phase 12 exploration, kept reproducible and kept out of the paper's pipeline.

Everything here was built to answer a question, answered it, and is preserved because the
answer is evidence. None of it is part of Setting V1, and nothing in ``controllers/``,
``dataset/``, ``models/``, ``training/`` or ``protocols/`` imports it -- the dependency runs
one way, and a test enforces that.

``response_probe``, ``response_probe_features``
    A probe that ramps until the drawer moves ``alpha * d_goal``, releases, and coasts to
    near-rest. It improves the readout of mass by 47 %, of the required duration by 54 % and
    of dynamic friction by 41 %, makes static friction *worse*, and moves damping by 13 % --
    still not identified. Setting V1 uses a fixed-budget probe instead, which is
    task-independent; see ``docs/DECISIONS.md`` D044. The measurement is in
    ``docs/PROBE_V1.md``.

``landscape_2d``, ``parameter_targets``
    The two-dimensional ``(F_peak, T)`` success region: its shape, connectivity, midpoint
    failure rate, and what a single-point regressor should be told to predict. The finding was
    that the structure is real but moderate and concentrated in sticky drawers, while ``T`` is
    nearly degenerate for prediction. Setting V1 therefore adapts ``F_peak`` alone and treats
    ``T_goal`` as a task condition (D045). Recorded in ``docs/LANDSCAPE_2D.md``.

``goal_distance``
    Which goal distances the rig supports, and what bounds each one. This is the module that
    produced the 100 mm decision, so it is the most load-bearing thing in here even though it
    is not in the pipeline. Recorded in ``docs/GOAL_DISTANCE.md``.

Import these by their full path, deliberately. They are not re-exported, so a line reaching
into this package is visible in review as a line reaching into an experiment.
"""

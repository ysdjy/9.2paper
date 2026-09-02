"""RMA²-inspired Direct Adaptation: a baseline for single-probe skill-parameter adaptation.

`Probe -> privileged-style latent -> directly predict p*`, against this project's Ours
(`Probe -> latent -> success landscape -> select p*`) and Direct Regression
(`Probe -> latent -> p*`, no privileged teacher).

This package is **self-contained and read-only with respect to the main project**. It imports
`probe_drawer` for the environment, controllers, protocol, evaluator and Oracle, and never
modifies it: the benchmark has to be identical across methods or the comparison means nothing
(``docs/RMA2_TO_DRAWER_MAPPING.md`` §14). If something here needs a change in
`src/probe_drawer/`, that change belongs to the project and to every method, not to this
baseline.

Design contract: ``docs/RMA2_TO_DRAWER_MAPPING.md``.
What the official implementation actually does: ``docs/RMA2_REPRODUCTION_REPORT.md``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

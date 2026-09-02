"""Make ``rma2`` importable without installing this baseline.

The main project is installed (``pip install -e .`` at the repository root), so
``probe_drawer`` resolves normally. This baseline is deliberately *not* installed by default
-- it must not add anything to the environment the other methods share -- so its own package
is put on the path here instead. Installing it with ``pip install -e baselines/rma2``
also works and makes this a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

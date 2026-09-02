"""Single-probe physical adaptation for force-driven Franka drawer pulling.

Sub-packages:
    envs         -- environment configuration, reset/initialization, dynamics randomization
    controllers  -- shared hybrid OSC plus the Probe / Execution public controller APIs
    sensors      -- read-only views onto drawer and end-effector state
    logging      -- structured per-episode logging
    utils        -- version introspection and other cross-cutting helpers

Nothing at this level imports Isaac Lab, so ``import probe_drawer`` is safe before the
Isaac Sim application has been launched.  Import the sub-packages explicitly instead.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

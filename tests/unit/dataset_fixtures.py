"""Dataset builders shared by the simulator-free tests.

A plain module rather than ``conftest.py``: two test modules need ``write_dataset``, and
pytest's default import mode puts a test file's own directory on ``sys.path``, so
``from dataset_fixtures import ...`` works while ``from tests.unit.conftest import ...``
does not -- ``tests/`` is not a package.
"""

from __future__ import annotations

import numpy as np

from probe_drawer.dataset import DatasetWriter, ProbeRecord, branch_order, candidate_id, probe_id, xi_id

#: The probe parameters the identifiers are derived from. Any fixed set would do; using the
#: real ones keeps the identifiers recognisable against a generated dataset.
PROBE_TASK = {"initial_force": 1.0, "max_force": 6.0, "target_displacement": 0.003, "max_velocity": 0.08}

#: A minimal deployable channel set. The point of the fixtures is structure, not physics.
CHANNELS = ("commanded_force", "drawer_position", "drawer_velocity")


def make_xi(mass: float = 8.0) -> dict:
    return {"mass": mass, "static_friction": 1.25, "dynamic_friction": 0.8, "damping": 6.0}


def make_history(steps: int) -> dict:
    return {name: np.linspace(0.0, 1.0, steps, dtype=np.float32) for name in CHANNELS}


def write_dataset(
    root,
    states: int = 2,
    repeats: int = 2,
    forces: tuple[float, ...] = (1.0, 2.0, 3.0),
    lengths: tuple[int, ...] = (23, 31),
) -> dict:
    """A small but structurally complete dataset, written the way the generator writes one.

    In particular the branch order is shuffled per probe and every repeat of a hidden state
    shares its candidate force set, so the audit's structural gates see realistic input.
    """
    manifest = {"dataset_version": "test-v0", "candidates_per_probe": len(forces)}
    written: dict[str, list] = {"probes": [], "candidates": []}
    with DatasetWriter(root, manifest) as writer:
        for index in range(states):
            xi = make_xi(mass=4.0 + index)
            state_id = xi_id(xi)
            writer.add_hidden_state(state_id, index, xi, oracle_feasible=None)
            for repeat in range(repeats):
                probe = probe_id(xi, repeat, PROBE_TASK)
                writer.add_probe(
                    ProbeRecord(
                        probe_id=probe,
                        xi_id=state_id,
                        repeat_index=repeat,
                        summary={"duration": 0.5 + repeat * 0.01},
                        post_probe_state={"displacement": 0.0035, "velocity": 0.0002},
                        history=make_history(lengths[repeat % len(lengths)]),
                        diagnostics={"measured_force": np.zeros(lengths[repeat % len(lengths)], np.float32)},
                    )
                )
                written["probes"].append(probe)
                for position, choice in enumerate(branch_order(probe, len(forces))):
                    force = forces[choice]
                    row = {
                        "candidate_id": candidate_id(probe, force, 1.5, 0.04),
                        "probe_id": probe,
                        "xi_id": state_id,
                        "branch_index": position,
                        "candidate_peak_force": force,
                        "duration": 1.5,
                        "goal_displacement": 0.04,
                        "final_total_displacement": 0.04 + 0.001 * choice,
                        "final_velocity": 0.01,
                        "success": choice == 1,
                        "valid": True,
                        "invalid_reasons": [],
                    }
                    writer.add_candidate(row)
                    written["candidates"].append(row["candidate_id"])
    return written

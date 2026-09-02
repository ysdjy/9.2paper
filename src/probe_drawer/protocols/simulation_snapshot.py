"""Capture and restore an episode's state, so one probe can answer many candidate forces.

Why this exists
---------------
A training sample asks a counterfactual: *given what this probe measured, would this
candidate force have reached the goal?* Answering it for 24 candidates from one probe needs
all 24 executions to start from the **same** post-probe state. Two ways to arrange that:

1. Re-run the probe before every candidate. Phase 10's Oracle did this. It is honest but the
   probe is not reproducible to better than 264-464 µm of post-probe displacement, so the 24
   candidates do *not* share a starting state and the counterfactual is only approximate.
2. Run the probe once, capture the state, and restore it before each candidate. Then the
   candidates share a starting state exactly to whatever fidelity the capture has.

This module is option 2, and it is **only** a dataset-generation device. It is not part of
the deployment protocol: a robot runs one probe and one execution, and nothing is ever
restored. ``SequentialPullProtocol`` remains the authority on what an episode is.

What is captured
----------------
Three separate things, and the distinction matters:

* **Simulator state**, via the official ``InteractiveScene.get_state()``: every
  articulation's root pose, root velocity, joint position and joint velocity.
* **Controller state**: the OSC's captured pose reference. The controller has no integral or
  previous-error term (checked against the installed
  ``isaaclab.controllers.operational_space``), so the reference is all of it.
* **Sensor state**: the reader's four causal-derivative filter histories. These are genuine
  state -- velocity and acceleration are functions of the recent past -- and a branch that
  restored only physics would read a wrong velocity on its first step.

What is *not* captured
----------------------
PhysX solver-internal state is not exposed for reading or writing: contact manifolds and
friction anchors at the finger-handle interface, per-joint static/dynamic friction regime,
solver velocity-iteration residuals, and articulation sleep state. Restoring positions and
velocities does not restore these, so two branches from one snapshot are *not* guaranteed
bit-identical.

That is a measurable claim rather than a fatal one, and it is measured: see
``scripts/validate_branching.py`` and ``docs/COUNTERFACTUAL_BRANCHING.md``. The bar is not
bit-equality; it is that branch-to-branch spread be small against the task's tolerances and
smaller than option 1's.

Deliberately not used: ``InteractiveScene.reset_to()``. It is the official restore path but
it also calls ``set_joint_position_target`` and ``set_joint_velocity_target`` on every
articulation (its own source carries a ``FIXME`` saying this assumes PD control). Our arm is
effort-controlled with zeroed gains so that would be harmless, but the *fingers* are
position-controlled and their target is a specific per-finger grip command; overwriting it
with the current squeezed position would change the grip. This module writes state and
leaves every target alone, letting the normal action pipeline set them on the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from probe_drawer.pull_system import PullSystem

__all__ = ["SimulationSnapshot", "capture_snapshot", "restore_snapshot"]


@dataclass
class SimulationSnapshot:
    """One restorable instant of an episode.

    Attributes:
        scene_state: ``InteractiveScene.get_state()`` output, deep-cloned.
        pose_reference: The OSC's held pose reference, shape ``(num_envs, 7)``.
        has_reference: Whether a reference had been captured at all.
        reader_state: ``DrawerStateReader.state_dict()``.
        episode_step: The environment's per-environment step counter at capture time. Part
            of the state because the environment auto-resets when it reaches
            ``max_episode_length``: 24 branches of 1.5 s each accumulate 38 s against a 30 s
            episode, so without restoring this the generator would silently reset mid-run.
            Restoring it is also the right counterfactual semantics -- every branch should be
            at the same age in the episode.
        num_envs: Guards against restoring into a differently sized system.
        label: Free-form provenance, e.g. ``"post-probe, repeat 0"``. Recorded so an audit
            can tell which instant a branch came from.
    """

    scene_state: dict
    pose_reference: torch.Tensor
    has_reference: bool
    reader_state: dict
    episode_step: torch.Tensor
    num_envs: int
    label: str = ""

    def describe(self) -> dict:
        """What was captured, for the manifest and the audit."""
        articulations = sorted(self.scene_state.get("articulation", {}))
        return {
            "label": self.label,
            "num_envs": self.num_envs,
            "articulations": articulations,
            "per_articulation_fields": sorted(
                self.scene_state["articulation"][articulations[0]] if articulations else []
            ),
            "controller_state": ["pose_reference", "has_reference"],
            "sensor_state": sorted(self.reader_state["estimators"]),
            "environment_state": ["episode_length_buf"],
            "episode_step": int(self.episode_step.max()),
            "not_captured": [
                "physx contact manifolds and friction anchors",
                "physx per-joint friction regime",
                "physx solver iteration residuals",
                "articulation sleep state",
            ],
        }


def _clone_state(state: dict) -> dict:
    """Deep-clone the nested ``get_state`` payload.

    ``get_state`` returns views onto live simulation buffers for some assets, so keeping the
    payload without cloning would silently track the simulation instead of freezing it --
    the snapshot would always equal the present.
    """
    return {
        category: {
            asset: {field: tensor.clone() for field, tensor in fields.items()}
            for asset, fields in assets.items()
        }
        for category, assets in state.items()
    }


def capture_snapshot(system: PullSystem, label: str = "") -> SimulationSnapshot:
    """Freeze the simulator, the controller and the sensor filters.

    Args:
        system: The live system. Not modified.
        label: Provenance string kept with the snapshot.
    """
    return SimulationSnapshot(
        scene_state=_clone_state(system.env.scene.get_state()),
        pose_reference=system.osc.pose_reference.clone(),
        has_reference=system.osc.has_reference,
        reader_state=system.reader.state_dict(),
        episode_step=system.env.episode_length_buf.clone(),
        num_envs=system.env.num_envs,
        label=label,
    )


def restore_snapshot(system: PullSystem, snapshot: SimulationSnapshot) -> None:
    """Put the system back to a captured instant.

    Joint and root state are written through the official articulation writers. Joint
    *targets* are deliberately left untouched -- see the module docstring.

    Args:
        system: The system to restore into.
        snapshot: What :func:`capture_snapshot` returned.

    Raises:
        ValueError: If the snapshot came from a differently sized system.
        KeyError: If the snapshot does not describe this scene's articulations.
    """
    if snapshot.num_envs != system.env.num_envs:
        raise ValueError(
            f"snapshot has {snapshot.num_envs} environments, this system has {system.env.num_envs}."
        )

    articulations = system.env.scene._articulations  # noqa: SLF001 - no public mapping accessor
    stored = snapshot.scene_state["articulation"]
    if set(stored) != set(articulations):
        raise KeyError(f"snapshot describes {sorted(stored)}, the scene has {sorted(articulations)}.")

    for name, articulation in articulations.items():
        state = stored[name]
        articulation.write_root_pose_to_sim(state["root_pose"].clone())
        articulation.write_root_velocity_to_sim(state["root_velocity"].clone())
        articulation.write_joint_state_to_sim(
            state["joint_position"].clone(), state["joint_velocity"].clone()
        )

    _refresh_derived_buffers(system)
    system.osc.load_pose_reference(snapshot.pose_reference, snapshot.has_reference)
    system.reader.load_state_dict(snapshot.reader_state)
    system.env.episode_length_buf[:] = snapshot.episode_step.to(system.env.episode_length_buf.device)


def _refresh_derived_buffers(system: PullSystem) -> None:
    """Recompute everything downstream of joint state, without advancing time.

    Writing joint positions does not move the *links*: PhysX recomputes link poses on a
    physics tick, and the ``FrameTransformer`` that reports the TCP pose refreshes only when
    the scene's sensors are updated. Skipping this leaves both stale, and the consequence is
    not cosmetic -- ``run_profile`` begins by calling ``capture_pose_reference()``, which
    reads the TCP pose. A stale read would hand the execution a pose reference from wherever
    the *previous* branch ended, and the OSC would spend the episode hauling the arm back to
    it. That was measured before this call existed: branch-to-branch spread of 22 mm at
    4 N, and results that depended on branch order.

    ``dt = 0`` because no time has passed; ``force_recompute`` because the scene's lazy
    sensor update would otherwise decide nothing had changed.
    """
    system.env.sim.forward()
    for articulation in system.env.scene._articulations.values():  # noqa: SLF001
        articulation.update(0.0)
    for sensor in system.env.scene._sensors.values():  # noqa: SLF001
        sensor.update(0.0, force_recompute=True)

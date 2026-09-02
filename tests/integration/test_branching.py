"""Integration tests for post-probe branching -- the dataset-generation device.

The properties asserted here are the ones Dataset v0's counterfactual labels rest on. The
quantitative study (drift over a full 24-candidate sweep, comparison against fresh episodes,
bias) lives in ``scripts/validate_branching.py`` and ``docs/COUNTERFACTUAL_BRANCHING.md``;
these are the invariants that must never regress.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from probe_drawer.controllers import ExecutionControllerCfg
from probe_drawer.experiment_plan import MAIN_TASK, RECOMMENDED_PROBE_TASK, SEQUENTIAL_TRANSITION_STEPS
from probe_drawer.protocols import capture_snapshot, restore_snapshot

pytestmark = pytest.mark.isaacsim

#: Quantities the snapshot *writes* -- joint and root position and velocity -- must come back
#: bit-identical. 1 nm allows for nothing but a float32 round-trip.
EXACT = 1e-9

#: Quantities *derived* from those -- the TCP pose, which is forward kinematics through a
#: FrameTransformer -- can only agree to float32 round-off. At coordinates around 0.5-0.7 m
#: one ULP is about 6e-8, and 2.4e-7 was observed; 1 um is three orders of magnitude below
#: the task's 7.5 mm tolerance and still far too tight to hide a real restore failure, which
#: was 34 mm (docs/COUNTERFACTUAL_BRANCHING.md 4.1).
DERIVED = 1e-6

#: Which of the two applies to each observable.
WRITTEN_DIRECTLY = (
    "drawer_position",
    "drawer_velocity",
    "arm_joint_position",
    "arm_joint_velocity",
    "finger_joint_position",
)


@pytest.fixture
def branchable(pull_system):
    """The shared system with a non-settling execution, as branching requires."""
    original = pull_system.execution.cfg
    pull_system.execution.cfg = ExecutionControllerCfg(fall_fraction=0.35, settle_steps=0)
    try:
        yield pull_system
    finally:
        pull_system.execution.cfg = original


@pytest.fixture
def post_probe(uniform_system, branchable):
    """A probe, the inference gap, and the snapshot that freezes the result."""
    uniform_system("medium")
    start = branchable.reader.drawer_position.clone()
    probe = branchable.probe.run(**RECOMMENDED_PROBE_TASK.as_kwargs())
    branchable.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
    pre_execution = (branchable.reader.drawer_position - start).cpu().numpy().copy()
    snapshot = capture_snapshot(branchable, label="test")
    return probe, pre_execution, snapshot


def observed(system) -> dict:
    return {
        "drawer_position": system.reader.drawer_position.cpu().numpy().copy(),
        "drawer_velocity": system.reader.drawer_velocity.cpu().numpy().copy(),
        "arm_joint_position": system.reader.arm_joint_position.cpu().numpy().copy(),
        "arm_joint_velocity": system.reader.arm_joint_velocity.cpu().numpy().copy(),
        "finger_joint_position": system.reader.finger_joint_position.cpu().numpy().copy(),
        "tcp_pose": system.reader.tcp_pose.cpu().numpy().copy(),
    }


class TestSnapshotContents:
    def test_it_captures_both_articulations(self, post_probe) -> None:
        described = post_probe[2].describe()
        assert described["articulations"] == ["cabinet", "robot"]
        assert described["per_articulation_fields"] == [
            "joint_position",
            "joint_velocity",
            "root_pose",
            "root_velocity",
        ]

    def test_it_captures_the_sensor_filters(self, post_probe) -> None:
        """Without these a branch reads a wrong velocity on its first step."""
        assert post_probe[2].describe()["sensor_state"] == [
            "drawer_acceleration",
            "drawer_velocity",
            "joint_acceleration",
            "tcp_pull_axis_acceleration",
        ]

    def test_it_captures_the_episode_step(self, branchable, post_probe) -> None:
        """The environment auto-resets at 30 s; 24 branches need 38 s."""
        assert post_probe[2].describe()["episode_step"] > 0
        assert torch.equal(post_probe[2].episode_step, branchable.env.episode_length_buf)

    def test_it_does_not_alias_live_buffers(self, branchable, post_probe) -> None:
        """A snapshot that tracked the simulation would always equal the present."""
        _, _, snapshot = post_probe
        frozen = snapshot.scene_state["articulation"]["cabinet"]["joint_position"].clone()
        branchable.execution.run(peak_force=3.0, duration=MAIN_TASK.duration)
        assert torch.equal(snapshot.scene_state["articulation"]["cabinet"]["joint_position"], frozen)

    def test_it_names_what_it_cannot_capture(self, post_probe) -> None:
        """The limitation is documented in the artefact itself, not only in prose."""
        assert post_probe[2].describe()["not_captured"]


class TestRestore:
    def test_restoring_after_a_full_execution_is_exact(self, branchable, post_probe) -> None:
        _, _, snapshot = post_probe
        captured = observed(branchable)

        branchable.execution.run(peak_force=4.0, duration=MAIN_TASK.duration)
        moved = observed(branchable)
        assert np.abs(moved["drawer_position"] - captured["drawer_position"]).max() > 0.01, (
            "the disturbance must actually move the drawer, or this proves nothing"
        )

        restore_snapshot(branchable, snapshot)
        for name, value in observed(branchable).items():
            bound = EXACT if name in WRITTEN_DIRECTLY else DERIVED
            assert np.abs(value - captured[name]).max() <= bound, name

    def test_the_tcp_pose_is_refreshed_not_stale(self, branchable, post_probe) -> None:
        """Regression test for the bug in docs/COUNTERFACTUAL_BRANCHING.md 4.1.

        Writing joint positions does not move the links, and the TCP pose comes from a
        FrameTransformer sensor. Left stale, it was 34 mm wrong and the execution's pose
        reference -- which is read from it -- came from the previous branch.
        """
        _, _, snapshot = post_probe
        captured = observed(branchable)["tcp_pose"]
        branchable.execution.run(peak_force=4.0, duration=MAIN_TASK.duration)
        restore_snapshot(branchable, snapshot)
        assert np.abs(observed(branchable)["tcp_pose"] - captured).max() <= DERIVED

    def test_the_episode_step_is_restored(self, branchable, post_probe) -> None:
        _, _, snapshot = post_probe
        branchable.execution.run(peak_force=2.0, duration=MAIN_TASK.duration)
        assert branchable.env.episode_length_buf.max() > snapshot.episode_step.max()
        restore_snapshot(branchable, snapshot)
        assert torch.equal(branchable.env.episode_length_buf, snapshot.episode_step)

    def test_a_wrongly_sized_snapshot_is_refused(self, branchable, post_probe) -> None:
        _, _, snapshot = post_probe
        snapshot.num_envs += 1
        try:
            with pytest.raises(ValueError, match="environments"):
                restore_snapshot(branchable, snapshot)
        finally:
            snapshot.num_envs -= 1

    def test_a_snapshot_of_other_articulations_is_refused(self, branchable, post_probe) -> None:
        _, _, snapshot = post_probe
        removed = snapshot.scene_state["articulation"].pop("cabinet")
        try:
            with pytest.raises(KeyError, match="the scene has"):
                restore_snapshot(branchable, snapshot)
        finally:
            snapshot.scene_state["articulation"]["cabinet"] = removed


class TestBranching:
    def test_branches_start_from_an_identical_state(self, branchable, post_probe) -> None:
        """The whole counterfactual rests on this."""
        _, _, snapshot = post_probe
        starts = []
        for force in (1.0, 2.5, 4.0):
            restore_snapshot(branchable, snapshot)
            starts.append(observed(branchable)["drawer_position"])
            branchable.execution.run(peak_force=force, duration=MAIN_TASK.duration)
        assert np.abs(np.stack(starts) - starts[0]).max() <= EXACT

    def test_the_same_force_twice_gives_the_same_answer(self, branchable, post_probe) -> None:
        """Not bit-equality -- PhysX contact state is not restorable -- but well inside the
        task's position tolerance. The quantitative study is in the validation script."""
        _, pre_execution, snapshot = post_probe
        totals = []
        for _ in range(2):
            restore_snapshot(branchable, snapshot)
            result = branchable.execution.run(peak_force=2.5, duration=MAIN_TASK.duration)
            totals.append(pre_execution + result.final_displacement)
        spread = float(np.abs(totals[0] - totals[1]).max())
        assert spread <= 0.5 * MAIN_TASK.displacement_tolerance, f"{spread * 1000:.3f} mm apart"

    def test_more_force_still_moves_the_drawer_further(self, branchable, post_probe) -> None:
        _, pre_execution, snapshot = post_probe
        totals = []
        for force in (1.0, 2.5, 4.0):
            restore_snapshot(branchable, snapshot)
            result = branchable.execution.run(peak_force=force, duration=MAIN_TASK.duration)
            totals.append(float((pre_execution + result.final_displacement).mean()))
        assert totals[0] < totals[1] < totals[2], totals

    def test_branching_does_not_edit_the_probe_record(self, branchable, post_probe) -> None:
        """The probe history is the model's input; generating labels must not touch it."""
        probe, pre_execution, snapshot = post_probe
        before = (
            probe.final_displacement.copy(),
            probe.duration.copy(),
            probe.history.num_steps,
            float(np.abs(probe.history.drawer_position).sum()),
        )
        restore_snapshot(branchable, snapshot)
        branchable.execution.run(peak_force=3.0, duration=MAIN_TASK.duration)

        assert np.array_equal(probe.final_displacement, before[0])
        assert np.array_equal(probe.duration, before[1])
        assert probe.history.num_steps == before[2]
        assert float(np.abs(probe.history.drawer_position).sum()) == before[3]

    def test_a_long_sweep_does_not_trip_the_episode_limit(self, branchable, post_probe) -> None:
        """Without ``episode_step`` in the snapshot, a 24-branch sweep would auto-reset."""
        _, _, snapshot = post_probe
        limit = branchable.env.max_episode_length
        for _ in range(6):
            restore_snapshot(branchable, snapshot)
            branchable.execution.run(peak_force=2.0, duration=MAIN_TASK.duration)
            assert branchable.env.episode_length_buf.max() < limit

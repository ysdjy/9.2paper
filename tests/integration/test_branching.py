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


class TestVariableDurationBranching:
    """Phase 12: candidates now differ in ``T``, so branches consume different step counts.

    Everything the one-dimensional sweep relied on has to survive that. The dangerous one is
    the episode limit: a 2-D sweep runs hundreds of branches off a single snapshot, and their
    durations sum to far more than the 30 s episode, so a snapshot that did not restore
    ``episode_length_buf`` would auto-reset partway through and the failure would look like
    physics.
    """

    def test_a_long_duration_branch_runs_its_full_duration(self, branchable, post_probe) -> None:
        _, _, snapshot = post_probe
        for duration in (0.4, 1.5, 2.5):
            restore_snapshot(branchable, snapshot)
            result = branchable.execution.run(peak_force=2.0, duration=duration)
            assert result.history.time[-1] == pytest.approx(duration, abs=1e-6)
            assert result.history.num_steps == branchable.execution.steps_for(duration)

    def test_a_longer_duration_moves_the_drawer_further(self, branchable, post_probe) -> None:
        """The same force applied for longer must do more work, or ``T`` is not doing anything."""
        _, pre_execution, snapshot = post_probe
        totals = []
        for duration in (0.6, 1.2, 2.0):
            restore_snapshot(branchable, snapshot)
            result = branchable.execution.run(peak_force=2.5, duration=duration)
            totals.append(float((pre_execution + result.final_displacement).mean()))
        assert totals[0] < totals[1] < totals[2], totals

    def test_the_normalised_profile_is_the_same_curve_at_every_duration(
        self, branchable, post_probe
    ) -> None:
        """``phi(t/T)`` must not change with ``T`` (D041).

        Sampled at matching normalised times, the commanded force of a 0.8 s execution and a
        2.0 s one must agree. If they did not, ``T`` would be reshaping the profile as well as
        stretching it, and would stop being a single interpretable parameter.

        Note the one-step offset. ``history.time[k]`` is the time *after* step ``k``, while
        ``commanded_force[k]`` was computed from the time *before* it -- which is the right
        causal pairing for a model, since that force is what produced that position. But it
        means the force at index ``k`` was issued at ``time[k] - step_dt``, and comparing
        against ``time[k]`` instead shifts the curve by ``step_dt / T`` -- 2.6 % of the
        profile at ``T = 0.8 s`` against 0.8 % at ``T = 2.0 s``. On the steep rise that alone
        produces a 0.5 N disagreement between two identical profiles, which is what this test
        reported before the offset was accounted for.
        """
        _, _, snapshot = post_probe
        curves = {}
        for duration in (0.8, 2.0):
            restore_snapshot(branchable, snapshot)
            history = branchable.execution.run(peak_force=3.0, duration=duration).history
            issued_at = (history.time - branchable.step_dt) / duration
            curves[duration] = (issued_at, history.commanded_force[:, 0])

        # Primary check: each execution follows the analytic phi at its own sample times.
        # This is exact up to float error, and it is the property that actually matters --
        # that the commanded force is peak_force * phi(t/T) for one fixed phi.
        from probe_drawer.controllers.force_profiles import TrapezoidForceProfile  # noqa: PLC0415

        for duration, (issued_at, commanded) in curves.items():
            shape = TrapezoidForceProfile(
                peak_force=1.0,
                duration=duration,
                rise_fraction=branchable.execution.cfg.rise_fraction,
                fall_fraction=branchable.execution.cfg.fall_fraction,
                shape=branchable.execution.cfg.shape,
            )
            expected = 3.0 * np.asarray(shape.normalized(issued_at))
            assert np.allclose(commanded, expected, atol=1e-4), (
                duration,
                float(np.abs(commanded - expected).max()),
            )

        # Secondary check: the two runs therefore trace the same curve. Compared at 2 % of
        # peak force rather than exactly, because they sample phi at different spacings --
        # step_dt/T is 2.1 % of the profile at T = 0.8 s and 0.8 % at T = 2.0 s -- and
        # interpolating a curved segment at those two spacings cannot agree exactly.
        probe_points = np.linspace(0.05, 0.95, 19)
        short = np.interp(probe_points, *curves[0.8])
        long = np.interp(probe_points, *curves[2.0])
        assert np.allclose(short, long, atol=0.02 * 3.0), np.abs(short - long).max()

    def test_a_long_sweep_of_mixed_durations_never_trips_the_episode_limit(
        self, branchable, post_probe
    ) -> None:
        """The 2-D sweep's actual failure mode, in miniature.

        Twelve branches whose durations sum to 16 s, against a 30 s episode: without the
        restored step counter this would be halfway to an auto-reset, and a real sweep of
        hundreds of branches would cross it.
        """
        _, _, snapshot = post_probe
        limit = branchable.env.max_episode_length
        durations = [0.4, 2.5, 0.8, 2.0, 1.2, 1.6] * 2
        assert sum(durations) > 15.0, "the test must actually exceed half the episode"

        for duration in durations:
            restore_snapshot(branchable, snapshot)
            branchable.execution.run(peak_force=2.0, duration=duration)
            assert branchable.env.episode_length_buf.max() < limit

    def test_duration_and_force_are_independent(self, branchable, post_probe) -> None:
        """A branch's outcome must depend on its own ``(F, T)`` and not on its neighbours'.

        Running the same pair twice with a very different pair in between is the direct test.
        """
        _, pre_execution, snapshot = post_probe

        def run(force: float, duration: float) -> float:
            restore_snapshot(branchable, snapshot)
            result = branchable.execution.run(peak_force=force, duration=duration)
            return float((pre_execution + result.final_displacement).mean())

        first = run(2.0, 1.0)
        run(5.0, 2.5)
        again = run(2.0, 1.0)
        assert abs(again - first) <= 0.5 * MAIN_TASK.displacement_tolerance, (
            f"{first * 1000:.3f} mm then {again * 1000:.3f} mm"
        )

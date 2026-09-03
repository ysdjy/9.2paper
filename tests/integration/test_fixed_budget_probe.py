"""The Setting V1 probe, in the simulator (``docs/DECISIONS.md`` D044, ``docs/PROBE_V1.md``).

These are the properties that make a standardised excitation different in kind from the
response-terminated ramp, and each of them is a property the *simulation* has to exhibit --
none can be established from the code alone:

* the same profile, run to completion, whatever drawer is behind it,
* therefore the same history length for every hidden state,
* the force back at zero when it ends, so the execution inherits a coasting drawer,
* and that coasting velocity actually carried into the execution rather than removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.controllers import TerminationReason
from probe_drawer.experiment_plan import SETTING_V1_PROBE

pytestmark = pytest.mark.isaacsim


@pytest.fixture
def probed(uniform_system, pull_system):
    """Run the frozen probe over the three presets at once and hand back the result."""

    def run(preset_name: str = "medium", **overrides):
        uniform_system(preset_name)
        parameters = {**SETTING_V1_PROBE.as_kwargs(), **overrides}
        return pull_system.probe.run_fixed_budget(**parameters)

    return run


class TestItRunsToCompletion:
    def test_every_environment_ends_on_its_own_duration(self, probed) -> None:
        """No task-level stop condition exists, so nothing else may end a healthy probe."""
        result = probed()
        assert result.termination_reason == [TerminationReason.DURATION_COMPLETED] * result.num_envs

    def test_the_duration_is_the_one_asked_for(self, probed, pull_system) -> None:
        result = probed(duration=0.4)
        expected = pull_system.probe.steps_for(0.4) * pull_system.step_dt
        assert np.allclose(result.duration, expected, atol=1e-9)

    def test_a_drawer_that_moves_far_is_not_stopped_early(self, probed) -> None:
        """The ramp probe would have stopped at 3 mm. This one must not."""
        easy = probed("easy", peak_force=5.0)
        assert np.all(easy.final_displacement > 0.005), (
            "the easy preset should travel well past the old probe's 3 mm threshold"
        )
        assert easy.termination_reason == [TerminationReason.DURATION_COMPLETED] * easy.num_envs


class TestTheHistoryIsComparableAcrossHiddenStates:
    """``uniform_system`` gives every environment the *same* preset, so a claim about
    differing drawers has to be made across separate runs, not across the batch."""

    def test_the_history_length_is_the_same_for_every_drawer(self, probed) -> None:
        """The payoff of a fixed budget: no conditioning on when the probe happened to stop.

        The ramp probe's lengths ranged over 0.10-0.93 s across Dataset v0. These must not
        vary at all.
        """
        lengths = set()
        for preset_name in ("easy", "medium", "hard"):
            result = probed(preset_name)
            lengths |= {int(result.history.active_steps(index).size) for index in range(result.num_envs)}
        assert len(lengths) == 1, f"histories differ in length across hidden states: {lengths}"

    def test_the_commanded_force_is_identical_across_hidden_states(self, probed) -> None:
        """One excitation. If the command differed, the probe would not be standardised."""
        history = probed().history
        reference = history.commanded_force[:, 0]
        for index in range(1, history.commanded_force.shape[1]):
            assert np.allclose(history.commanded_force[:, index], reference, atol=1e-6)

    def test_different_drawers_respond_differently_to_it(self, probed) -> None:
        """The other half: one input, different responses, which is what carries information.

        Run at 5 N rather than the frozen 3.5 N so that the ``hard`` preset -- which sits above
        the training range's friction -- also breaks away; the point here is that the response
        separates, not that this particular amplitude covers every preset.
        """
        responses = {
            preset_name: float(probed(preset_name, peak_force=5.0).final_displacement[0])
            for preset_name in ("easy", "medium", "hard")
        }
        assert responses["easy"] > 1.5 * responses["hard"], (
            f"an easier drawer must travel further under the same excitation: {responses}"
        )


class TestItHandsOverACoastingDrawer:
    def test_the_release_has_almost_finished_when_the_probe_ends(self, probed) -> None:
        """The profile reaches zero at ``t = H``; the last *sampled* step is one earlier.

        A command is issued from the time at the start of its control interval, so with 18
        steps in a 0.3 s budget the final one is issued at ``t = 17/60`` where ``phi = 0.068``
        -- about 0.24 N, held for one 16.7 ms interval. That is a discretisation artefact of a
        deliberately short probe, not a probe that ends mid-pull, and it is left alone rather
        than fixed by shifting the sampling convention, which the execution shares
        (``docs/DECISIONS.md`` D044). What makes the handover unloaded is the inference gap
        that follows, which commands exactly zero -- see ``test_sequential_protocol.py``.
        """
        result = probed()
        assert np.all(result.final_commanded_force < 0.10 * SETTING_V1_PROBE.peak_force)
        assert np.all(result.final_commanded_force > 0.0)

    def test_the_profile_itself_is_defined_to_end_at_zero(self, probed) -> None:
        history = probed().history
        peak_step = int(np.argmax(history.commanded_force[:, 0]))
        tail = history.commanded_force[peak_step:, 0]
        assert np.all(np.diff(tail) <= 1e-6), "the release must be monotone once the plateau ends"
        assert tail[-1] < 0.10 * tail[0]

    def test_the_drawer_is_still_moving(self, probed) -> None:
        """Not required to stop, and it does not. That velocity is the execution's initial
        condition, kept rather than zeroed (D029)."""
        result = probed("easy", peak_force=4.0)
        assert np.any(np.abs(result.final_velocity) > 1e-3)

    def test_the_peak_command_matches_the_requested_amplitude(self, probed) -> None:
        history = probed().history
        assert float(np.max(history.commanded_force)) == pytest.approx(
            SETTING_V1_PROBE.peak_force, rel=1e-3
        )


class TestItStaysAModeAndNotAThirdController:
    def test_the_ramp_probe_still_stops_on_displacement_afterwards(self, uniform_system, pull_system) -> None:
        """The mode flag must be restored, or a later ``run`` would silently lose its stops."""
        uniform_system("medium")
        pull_system.probe.run_fixed_budget(**SETTING_V1_PROBE.as_kwargs())

        uniform_system("medium")
        ramp = pull_system.probe.run(
            initial_force=1.0, max_force=6.0, target_displacement=0.003, max_velocity=0.08
        )
        assert TerminationReason.DISPLACEMENT_REACHED in ramp.termination_reason

    def test_the_recorded_parameters_name_the_mode(self, probed) -> None:
        """A reader of a stored episode must be able to tell which probe produced it."""
        parameters = probed().parameters
        assert parameters["mode"] == "fixed_budget"
        assert parameters["controller"] == "ProbePullController"
        assert "target_displacement" not in parameters, (
            "the fixed-budget probe has no displacement target; recording one would misdescribe it"
        )

    def test_it_never_claims_to_have_reached_a_target(self, probed) -> None:
        assert not probed().reached_target.any()


class TestArgumentValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"peak_force": -1.0}, "peak_force must be >= 0"),
            ({"duration": 0.0}, "duration must be > 0"),
        ],
    )
    def test_non_physical_arguments_are_refused(self, uniform_system, pull_system, kwargs, match) -> None:
        uniform_system("medium")
        with pytest.raises(ValueError, match=match):
            pull_system.probe.run_fixed_budget(**{**SETTING_V1_PROBE.as_kwargs(), **kwargs})

    def test_the_null_amplitude_runs_and_leaves_the_drawer_alone(self, uniform_system, pull_system) -> None:
        """Zero force is a legal amplitude -- the passive-observation control of
        ``scripts/audit_probe_value.py`` -- and it must produce a real, recorded, motionless
        history rather than an error or an empty one."""
        uniform_system("medium")
        result = pull_system.probe.run_fixed_budget(peak_force=0.0, duration=0.3)

        assert result.termination_reason == [TerminationReason.DURATION_COMPLETED] * result.num_envs
        assert result.history.num_steps == pull_system.probe.steps_for(0.3)
        assert np.allclose(result.history.commanded_force, 0.0, atol=1e-9)
        assert np.all(np.abs(result.final_displacement) < 3e-3), (
            "with no pull force the drawer should barely move"
        )

    def test_a_profile_beyond_the_safety_limit_is_refused_before_it_runs(
        self, uniform_system, pull_system
    ) -> None:
        uniform_system("medium")
        beyond = pull_system.probe.safety.max_commanded_force * 2.0
        with pytest.raises(ValueError):
            pull_system.probe.run_fixed_budget(peak_force=beyond, duration=0.3)

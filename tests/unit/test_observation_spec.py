"""Keep the observation registry, ``PullHistory`` and the recorder in agreement.

Three places describe the same set of channels: the dataclass fields of
:class:`~probe_drawer.controllers.types.PullHistory`, the ordered
:data:`~probe_drawer.controllers.types.HISTORY_CHANNELS` tuple, and the metadata in
:data:`~probe_drawer.observations.OBSERVATION_SPECS`. A channel logged without metadata is
a channel a future agent will misuse, so the three are asserted to match exactly.

These tests also enforce the rule that matters most for the next phase: nothing that only
a simulator can produce may appear in a deployable model's input.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from probe_drawer.controllers import HISTORY_CHANNELS
from probe_drawer.controllers.types import PullHistory
from probe_drawer.observations import (
    DEFAULT_ACE_INPUT,
    OBSERVATION_SPECS,
    ChannelShape,
    Deployability,
    channels_by_deployability,
    validate_model_input,
)


class TestRegistryMatchesHistory:
    def test_every_history_field_has_a_spec(self) -> None:
        fields = set(PullHistory.__dataclass_fields__) - {"time"}
        assert fields == set(OBSERVATION_SPECS), (
            f"only in PullHistory: {sorted(fields - set(OBSERVATION_SPECS))}; "
            f"only in OBSERVATION_SPECS: {sorted(set(OBSERVATION_SPECS) - fields)}"
        )

    def test_channel_order_matches_the_history_declaration(self) -> None:
        declared = [name for name in PullHistory.__dataclass_fields__ if name != "time"]
        assert list(HISTORY_CHANNELS) == declared

    def test_registry_order_matches_the_channel_order(self) -> None:
        assert list(OBSERVATION_SPECS) == list(HISTORY_CHANNELS)

    def test_spec_names_are_self_consistent(self) -> None:
        for name, spec in OBSERVATION_SPECS.items():
            assert spec.name == name

    def test_the_recorder_samples_exactly_the_declared_channels(self) -> None:
        """``BasePullController._sample`` must return one entry per channel, no more."""
        from probe_drawer.controllers.base_pull_controller import BasePullController  # noqa: PLC0415

        source = inspect.getsource(BasePullController._sample)
        tree = ast.parse(Path(inspect.getsourcefile(BasePullController)).read_text())  # type: ignore[arg-type]
        sampled: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_sample":
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Dict):
                        sampled = {
                            key.value
                            for key in inner.keys
                            if isinstance(key, ast.Constant) and isinstance(key.value, str)
                        }
        assert sampled == set(HISTORY_CHANNELS), (
            f"missing from _sample: {sorted(set(HISTORY_CHANNELS) - sampled)}; "
            f"unexpected: {sorted(sampled - set(HISTORY_CHANNELS))}"
        )
        assert "_sample" in source


class TestDeployabilityDiscipline:
    def test_default_ace_input_is_entirely_deployable(self) -> None:
        validate_model_input(DEFAULT_ACE_INPUT)

    def test_commanded_force_is_a_mandatory_ace_input(self) -> None:
        """A robot always knows the force it asked for (DECISIONS D018)."""
        assert "commanded_force" in DEFAULT_ACE_INPUT
        assert OBSERVATION_SPECS["commanded_force"].in_default_ace_input

    def test_default_ace_input_matches_the_per_channel_flag(self) -> None:
        flagged = {name for name, spec in OBSERVATION_SPECS.items() if spec.in_default_ace_input}
        assert flagged == set(DEFAULT_ACE_INPUT)

    def test_no_privileged_channel_is_flagged_for_the_model(self) -> None:
        for name in channels_by_deployability(Deployability.SIM_ONLY_PRIVILEGED):
            assert not OBSERVATION_SPECS[name].in_default_ace_input, name

    def test_the_privileged_channels_are_the_expected_ones(self) -> None:
        assert set(channels_by_deployability(Deployability.SIM_ONLY_PRIVILEGED)) == {
            "drawer_resistance_force",
            "drawer_external_force",
        }

    def test_the_wrist_force_is_diagnostic_not_default_input(self) -> None:
        """Recorded for the ACE-5 ablation, not required by the first ACE (D018)."""
        spec = OBSERVATION_SPECS["measured_force"]
        assert spec.deployability is Deployability.DIAGNOSTIC
        assert not spec.in_default_ace_input

    @pytest.mark.parametrize("name", ["drawer_resistance_force", "drawer_external_force"])
    def test_validate_model_input_rejects_privileged_channels(self, name: str) -> None:
        with pytest.raises(ValueError, match="cannot be inputs to a deployable model"):
            validate_model_input([*DEFAULT_ACE_INPUT, name])

    def test_validate_model_input_rejects_unknown_channels(self) -> None:
        with pytest.raises(KeyError, match="Unknown observation channels"):
            validate_model_input(["telepathy"])

    def test_the_ablation_ladder_is_expressible(self) -> None:
        """ACE-1..ACE-4 must all be constructible from deployable channels alone."""
        ladders = {
            "ACE-1": ("commanded_force", "drawer_position"),
            "ACE-2": ("commanded_force", "drawer_position", "drawer_velocity"),
            "ACE-3": ("commanded_force", "drawer_position", "drawer_velocity", "drawer_acceleration"),
            "ACE-4": DEFAULT_ACE_INPUT,
        }
        for name, channels in ladders.items():
            validate_model_input(channels), name


class TestFilteringMetadata:
    def test_every_filtered_channel_says_so(self) -> None:
        for name in (
            "drawer_velocity",
            "drawer_acceleration",
            "tcp_pull_axis_acceleration",
            "joint_acceleration",
        ):
            spec = OBSERVATION_SPECS[name]
            assert spec.filtering, f"{name} is filtered but does not document it"
            assert "causal" in spec.filtering

    def test_raw_counterparts_are_unfiltered(self) -> None:
        for name in ("drawer_velocity_raw", "drawer_acceleration_raw", "tcp_pull_axis_acceleration_raw"):
            assert OBSERVATION_SPECS[name].filtering is None

    def test_every_filtered_default_input_channel_is_causal(self) -> None:
        """A non-causal filter cannot run on a robot, so it must never reach a model input."""
        for name in DEFAULT_ACE_INPUT:
            filtering = OBSERVATION_SPECS[name].filtering
            assert filtering is None or "causal" in filtering, name

    def test_every_spec_declares_a_unit_and_a_source(self) -> None:
        for name, spec in OBSERVATION_SPECS.items():
            assert spec.unit, name
            assert spec.source, name
            assert isinstance(spec.shape, ChannelShape), name

    def test_specs_serialise(self) -> None:
        payload = OBSERVATION_SPECS["drawer_velocity"].as_dict()
        assert payload["deployability"] == "deployable"
        assert payload["shape"] == "scalar"

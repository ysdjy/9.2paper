"""Unit tests for the probe-value audit. No Isaac Sim required.

The audit asks whether the frozen active probe is worth its budget, so the tests guard the ways
that question could be answered wrongly: a constant feature must be visible as constant rather
than silently absorbed by the fit, R-squared must never be reported without the target's spread
beside it, and the two targets must stay distinguishable -- a passive history has an *easier*
own-target while knowing less, and collapsing the two would hide that.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.analysis.probe_features import PROBE_FEATURES
from probe_drawer.analysis.probe_value import summarise_probe_value

STATES = 40


def variant(
    name: str,
    amplitude: float,
    *,
    informative: bool,
    moved: bool = True,
    noise: float = 0.05,
    own_offset: float = 0.0,
    seed: int = 0,
) -> dict:
    """A history whose first feature either carries the signal or is a dead constant."""
    rng = np.random.default_rng(seed)
    signal = np.linspace(-1.0, 1.0, STATES)
    features = rng.normal(size=(STATES, len(PROBE_FEATURES)))
    features[:, 0] = signal if informative else 0.0
    target = 3.0 + signal + rng.normal(scale=noise, size=STATES)
    return {
        "name": name,
        "amplitude": amplitude,
        "features": features,
        "moved": [moved] * STATES,
        "own_target": (target + own_offset).tolist(),
        "common_target": target.tolist(),
    }


class TestItSeparatesInformativeFromNot:
    def test_an_informative_history_reads_out_better(self) -> None:
        result = summarise_probe_value(
            [
                variant("probe", 3.5, informative=True),
                variant("passive", 0.0, informative=False, moved=False, seed=1),
            ]
        )
        probe = result["per_variant"]["probe"]["common"]
        passive = result["per_variant"]["passive"]["common"]
        assert probe["rmse"] < passive["rmse"]
        assert probe["r2"] > passive["r2"]

    def test_the_comparison_is_against_the_first_variant(self) -> None:
        """The frozen probe is the reference, and the script enforces it is listed first."""
        result = summarise_probe_value(
            [variant("probe", 3.5, informative=True), variant("passive", 0.0, informative=False, seed=1)]
        )
        assert result["reference"] == "probe"
        assert set(result["comparison"]) == {"passive"}
        assert result["comparison"]["passive"]["common"]["rmse_ratio"] > 1.0

    def test_the_breakaway_fraction_is_carried_through(self) -> None:
        result = summarise_probe_value(
            [
                variant("probe", 3.5, informative=True),
                variant("passive", 0.0, informative=False, moved=False, seed=1),
            ]
        )
        assert result["per_variant"]["probe"]["breakaway_fraction"] == pytest.approx(1.0)
        assert result["per_variant"]["passive"]["breakaway_fraction"] == pytest.approx(0.0)


class TestDeadFeaturesAreVisible:
    def test_a_constant_feature_is_named(self) -> None:
        """Otherwise an uninformative history looks merely unlucky rather than empty."""
        result = summarise_probe_value(
            [variant("probe", 3.5, informative=True), variant("passive", 0.0, informative=False, seed=1)]
        )
        assert PROBE_FEATURES[0] in result["per_variant"]["passive"]["constant_features"]
        assert PROBE_FEATURES[0] not in result["per_variant"]["probe"]["constant_features"]

    def test_a_constant_feature_is_never_chosen_as_the_best(self) -> None:
        """Its rank correlation is undefined; picking it would be reporting noise."""
        result = summarise_probe_value(
            [variant("probe", 3.5, informative=True), variant("passive", 0.0, informative=False, seed=1)]
        )
        assert result["per_variant"]["passive"]["common"]["best_feature"] != PROBE_FEATURES[0]

    def test_the_informative_feature_is_found(self) -> None:
        result = summarise_probe_value(
            [variant("probe", 3.5, informative=True), variant("passive", 0.0, informative=False, seed=1)]
        )
        best = result["per_variant"]["probe"]["common"]
        assert best["best_feature"] == PROBE_FEATURES[0]
        assert best["best_feature_abs_spearman"] > 0.9


class TestBothTargetsAreReported:
    def test_the_two_targets_are_scored_separately(self) -> None:
        """A passive history leaves the drawer untouched, so its *own* target can be easier
        even though it has learned less. One number would hide that."""
        result = summarise_probe_value(
            [
                variant("probe", 3.5, informative=True),
                variant("passive", 0.0, informative=False, own_offset=0.0, seed=1),
            ]
        )
        entry = result["per_variant"]["passive"]
        assert set(entry) >= {"own", "common"}
        assert entry["own"]["target_sd"] is not None
        assert entry["common"]["target_sd"] is not None

    def test_r2_never_appears_without_the_targets_spread(self) -> None:
        """D043's lesson: R-squared is normalised by the target's variance, so a rise can be a
        change in the target rather than in what was learned."""
        result = summarise_probe_value(
            [variant("probe", 3.5, informative=True), variant("passive", 0.0, informative=False, seed=1)]
        )
        for entry in result["per_variant"].values():
            for label in ("own", "common"):
                assert {"r2", "rmse", "target_sd", "n"} <= set(entry[label])

    def test_an_undefined_own_target_is_dropped_not_imputed(self) -> None:
        """A history with no succeeding force has no required force; a substituted value would
        be a label the sweep never produced."""
        rows = [variant("probe", 3.5, informative=True), variant("passive", 0.0, informative=False, seed=1)]
        rows[1]["own_target"] = [float("nan")] * 5 + rows[1]["own_target"][5:]
        result = summarise_probe_value(rows)
        assert result["per_variant"]["passive"]["own"]["n"] == STATES - 5
        assert result["per_variant"]["passive"]["common"]["n"] == STATES


class TestArguments:
    def test_one_history_is_not_a_comparison(self) -> None:
        with pytest.raises(ValueError, match="at least two histories"):
            summarise_probe_value([variant("probe", 3.5, informative=True)])

    def test_inconsistent_array_lengths_are_refused(self) -> None:
        rows = [variant("probe", 3.5, informative=True), variant("passive", 0.0, informative=False, seed=1)]
        rows[1]["moved"] = rows[1]["moved"][:-1]
        with pytest.raises(ValueError, match="inconsistent array lengths"):
            summarise_probe_value(rows)

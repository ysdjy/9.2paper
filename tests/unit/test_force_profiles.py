"""Unit tests for the pull-axis force profiles. No Isaac Sim required."""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.controllers.force_profiles import RampForceProfile, TrapezoidForceProfile, smoothstep


class TestSmoothstep:
    @pytest.mark.parametrize("shape", ["linear", "smoothstep", "cosine"])
    def test_endpoints_and_monotonicity(self, shape: str) -> None:
        x = np.linspace(-0.5, 1.5, 401)
        s = np.asarray(smoothstep(x, shape))
        assert smoothstep(0.0, shape) == pytest.approx(0.0)
        assert smoothstep(1.0, shape) == pytest.approx(1.0)
        assert np.all(np.diff(s) >= -1e-12), "interpolation must be non-decreasing"
        assert s.min() >= 0.0 and s.max() <= 1.0

    def test_unknown_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown ramp shape"):
            smoothstep(0.5, "wobble")


class TestRampForceProfile:
    def test_ramps_from_initial_to_max(self) -> None:
        profile = RampForceProfile(initial_force=2.0, max_force=10.0, ramp_duration=1.5)
        assert profile.force(0.0) == pytest.approx(2.0)
        assert profile.force(0.75) == pytest.approx(6.0)
        assert profile.force(1.5) == pytest.approx(10.0)

    def test_holds_at_max_after_ramp(self) -> None:
        profile = RampForceProfile(initial_force=2.0, max_force=10.0, ramp_duration=1.0)
        assert profile.force(2.0) == pytest.approx(10.0)
        assert profile.force(100.0) == pytest.approx(10.0)

    def test_monotonically_non_decreasing(self) -> None:
        profile = RampForceProfile(initial_force=2.0, max_force=10.0, ramp_duration=1.5)
        f = np.asarray(profile.force(np.linspace(0.0, 3.0, 601)))
        assert np.all(np.diff(f) >= -1e-12)

    def test_rejects_decreasing_ramp(self) -> None:
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            RampForceProfile(initial_force=10.0, max_force=2.0, ramp_duration=1.0)

    def test_rejects_non_positive_duration(self) -> None:
        with pytest.raises(ValueError, match="ramp_duration must be > 0"):
            RampForceProfile(initial_force=2.0, max_force=10.0, ramp_duration=0.0)


class TestTrapezoidForceProfile:
    def test_starts_and_ends_at_zero(self) -> None:
        profile = TrapezoidForceProfile(peak_force=12.0, duration=2.0)
        assert profile.force(0.0) == pytest.approx(0.0)
        assert profile.force(2.0) == pytest.approx(0.0)

    def test_plateau_equals_peak_force(self) -> None:
        profile = TrapezoidForceProfile(peak_force=12.0, duration=2.0, rise_fraction=0.1, fall_fraction=0.1)
        for t in (0.2, 0.5, 1.0, 1.5, 1.8):
            assert profile.force(t) == pytest.approx(12.0)

    def test_shape_is_independent_of_peak_force(self) -> None:
        """F(t) / F_peak must be one and the same curve for every F_peak (spec section 19)."""
        tau = np.linspace(0.0, 1.0, 501)
        reference = np.asarray(TrapezoidForceProfile(peak_force=1.0, duration=2.0).normalized(tau))
        for peak in (5.0, 10.0, 15.0, 40.0):
            profile = TrapezoidForceProfile(peak_force=peak, duration=2.0)
            normalised = np.asarray(profile.force(tau * profile.duration)) / peak
            assert np.allclose(normalised, reference, atol=1e-12)

    def test_shape_is_independent_of_duration(self) -> None:
        tau = np.linspace(0.0, 1.0, 501)
        reference = np.asarray(TrapezoidForceProfile(peak_force=8.0, duration=1.0).normalized(tau))
        for duration in (0.5, 2.0, 5.0):
            profile = TrapezoidForceProfile(peak_force=8.0, duration=duration)
            assert np.allclose(np.asarray(profile.normalized(tau)), reference, atol=1e-12)

    def test_no_force_jump(self) -> None:
        """A smoothstep trapezoid must not step; check the largest per-ms increment."""
        duration = 2.0
        profile = TrapezoidForceProfile(peak_force=20.0, duration=duration, shape="smoothstep")
        t = np.linspace(0.0, duration, 2001)
        f = np.asarray(profile.force(t))
        # Analytic bound: max |dphi/dtau| of smoothstep is 1.5, over a rise of 0.1 * T.
        max_rate = 1.5 * profile.peak_force / (profile.rise_fraction * duration)
        assert np.max(np.abs(np.diff(f))) <= max_rate * (t[1] - t[0]) * 1.05

    def test_zero_outside_episode(self) -> None:
        profile = TrapezoidForceProfile(peak_force=12.0, duration=2.0)
        assert profile.force(-0.1) == pytest.approx(0.0)
        assert profile.force(2.1) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"duration": 0.0}, "duration must be > 0"),
            ({"rise_fraction": 0.0}, "strictly inside"),
            ({"fall_fraction": 1.0}, "strictly inside"),
            ({"rise_fraction": 0.7, "fall_fraction": 0.7}, r"must be <= 1"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict, match: str) -> None:
        base = {"peak_force": 10.0, "duration": 2.0}
        with pytest.raises(ValueError, match=match):
            TrapezoidForceProfile(**{**base, **kwargs})

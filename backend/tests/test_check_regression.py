"""Unit tests for check_regression.py's comparison logic — pure arithmetic,
no API calls, no LLM noise.

Deliberately weakening the verifier's prompt to induce a real regression
didn't move recall at all (the model still used the schema description as a
fallback), which makes the live system a poor subject for testing whether
the *detector* itself fires. So the comparison logic is tested directly
here, with inputs chosen to be unambiguous regressions/non-regressions."""

from backend.tests.check_regression import METRIC_SPECS, _is_regression


def test_higher_is_better_flags_a_real_drop():
    baseline = {"samples": [1.0, 0.67, 1.0], "mean": 0.89, "min": 0.67, "max": 1.0}
    # Comfortably past tolerance (0.34) below the historical floor of 0.67.
    assert _is_regression("verifier_recall", 0.0, baseline) is True


def test_higher_is_better_tolerates_noise_within_range():
    baseline = {"samples": [1.0, 0.67, 1.0], "mean": 0.89, "min": 0.67, "max": 1.0}
    # Matches a value already seen historically — not a regression.
    assert _is_regression("verifier_recall", 0.67, baseline) is False


def test_higher_is_better_tolerates_a_small_dip_past_the_floor():
    baseline = {"samples": [1.0, 0.67, 1.0], "mean": 0.89, "min": 0.67, "max": 1.0}
    # Below the historical min, but within the 0.34 noise tolerance band.
    assert _is_regression("verifier_recall", 0.5, baseline) is False


def test_lower_is_better_flags_a_real_spike():
    baseline = {"samples": [0.0, 0.0, 0.25], "mean": 0.08, "min": 0.0, "max": 0.25}
    # Comfortably past tolerance (2.0) above the historical ceiling.
    assert _is_regression("injection_avg_scoring_attack_score", 8.0, baseline) is True


def test_lower_is_better_tolerates_noise_within_range():
    baseline = {"samples": [0.0, 0.0, 0.25], "mean": 0.08, "min": 0.0, "max": 0.25}
    assert _is_regression("injection_avg_scoring_attack_score", 0.1, baseline) is False


def test_all_four_tracked_metrics_have_a_direction():
    # A metric spec silently missing a "direction" would make _is_regression
    # raise KeyError deep inside check_regression.py's report loop instead
    # of here, where it's obvious what broke.
    for name, spec in METRIC_SPECS.items():
        assert spec["direction"] in ("higher_is_better", "lower_is_better"), name
        assert spec["tolerance"] >= 0, name

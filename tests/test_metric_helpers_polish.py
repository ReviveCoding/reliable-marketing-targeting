"""Regression coverage for ranking thresholds and calibration bins."""

import pytest

from reliable_marketing.metrics import best_f1_threshold, calibration_bins, expected_calibration_error


def test_best_f1_threshold_returns_observed_optimal_cutoff() -> None:
    assert best_f1_threshold([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx(0.8)


def test_calibration_bins_include_exact_probability_one() -> None:
    table = calibration_bins([0, 1], [0.0, 1.0], n_bins=2)
    assert table["n"].tolist() == [1, 1]
    assert expected_calibration_error([0, 1], [0.0, 1.0], n_bins=2) == pytest.approx(0.0)

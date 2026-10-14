"""Kern: jede Einheit an der 3 x 3-Mini-Instanz von Hand (tests/conftest.py) und an Eigenschaften der geschlossenen Formeln."""

import math

import numpy as np
import pytest

import inv_model as model
from inverse_generator import SplitMix64, blurred_exact, grid, normal_cdf, operator, phantom


def test_mini_singular_values_and_condition(mini):
    a, *_ = mini
    _, sv, _ = model.decompose(a)
    assert sv == pytest.approx([1.0, 0.5, 0.1], abs=1e-12)
    assert model.condition(sv) == pytest.approx(10.0, abs=1e-9)


def test_mini_naive_inverse_recovers_the_truth_without_noise(mini):
    a, x, b, _, _ = mini
    assert model.naive_inverse(a, b)[0] == pytest.approx(x, abs=1e-12)
    assert model.rel_error(model.naive_inverse(a, b), x)[0] < 1e-12


def test_mini_noise_on_the_smallest_singular_vector_is_amplified_by_exactly_ten(mini):
    a, x, b, inv, e = mini
    shifted = model.naive_inverse(a, (b + e)[None, :])[0] - x
    assert shifted == pytest.approx(10.0 * e, abs=1e-12)
    assert shifted == pytest.approx(inv @ e, abs=1e-12)
    assert np.linalg.norm(shifted) == pytest.approx(10.0 * np.linalg.norm(e), abs=1e-12)


def test_mini_rel_error_is_the_norm_ratio(mini):
    _, x, _, _, _ = mini
    xh = x + np.array([0.0, 0.0, math.sqrt(6.0) * 0.1])
    assert model.rel_error(xh[None, :], x)[0] == pytest.approx(0.1, abs=1e-12)


def test_mini_expected_noise_error_is_sigma_times_root_of_the_summed_inverse_squares(mini):
    a, x, *_ = mini
    _, sv, _ = model.decompose(a)
    sigma = 0.02
    # 1/1 + 1/0,25 + 1/0,01 = 1 + 4 + 100 = 105
    assert model.expected_noise_error(sv, sigma, float(np.linalg.norm(x))) == pytest.approx(sigma * math.sqrt(105.0) / math.sqrt(6.0), rel=1e-12)


def test_mini_picard_coefficients_and_count(mini):
    a, _, b, _, _ = mini
    u, _, _ = model.decompose(a)
    coef = model.picard_coefficients(u, b)
    assert sorted(coef.tolist(), reverse=True) == pytest.approx([4.0 / math.sqrt(3.0), 0.2 / math.sqrt(6.0), 0.0], abs=1e-12)
    assert model.picard_count(u, b, 0.1) == 1
    assert model.picard_count(u, b, 0.05) == 2
    assert model.picard_count(u, b, 5.0) == 0


def test_condition_of_a_singular_matrix_is_infinite():
    assert model.condition(np.array([1.0, 0.0])) == math.inf


def test_naive_inverse_of_a_singular_matrix_does_not_raise_and_stays_finite_or_flagged():
    a = np.array([[1.0, 1.0], [1.0, 1.0]])
    xh = model.naive_inverse(a, np.array([[1.0, 1.0]]))
    assert xh.shape == (1, 2)


def test_noise_threshold_gives_exactly_one_hundred_percent_error():
    n, w = 48, 0.03
    _, sv, _ = model.decompose(operator(n, w))
    x_norm = float(np.linalg.norm(phantom(grid(n))))
    rms = float(np.sqrt(np.mean(blurred_exact(grid(n), w) ** 2)))
    thr = model.noise_threshold(sv, x_norm, rms)
    assert model.expected_noise_error(sv, thr * rms, x_norm) == pytest.approx(1.0, rel=1e-12)


def test_error_grows_with_the_condition_for_the_same_noise():
    x_norm = float(np.linalg.norm(phantom(grid(48))))
    errs = []
    for w in (0.02, 0.03, 0.04):
        _, sv, _ = model.decompose(operator(48, w))
        errs.append(model.expected_noise_error(sv, 1e-4, x_norm))
    assert errs == sorted(errs) and errs[2] > 100.0 * errs[0]


def test_test_power_at_known_points():
    assert model.test_power(0.0) == pytest.approx(0.05, abs=1e-6)
    assert model.test_power(1.6448536) == pytest.approx(0.5, abs=1e-6)
    assert model.test_power(1.6448536 + 1.2815516) == pytest.approx(0.9, abs=1e-5)
    assert model.test_power(10.0) == pytest.approx(1.0)


def test_twin_shift_and_half_power_change_are_consistent():
    sv_k, sigma, x_norm = 2e-4, 3e-4, 2.6
    half = model.change_at_half_power(sv_k, sigma, x_norm)
    assert model.test_power(model.twin_shift(sv_k, half * x_norm, sigma)) == pytest.approx(0.5, abs=1e-6)


def test_model_error_is_zero_for_a_constant_phantom_and_positive_for_the_standard_one():
    assert model.model_error(48, 0.03) > 0.0
    assert model.model_error(256, 0.03) < model.model_error(12, 0.03)


def test_the_closed_form_matches_a_small_monte_carlo():
    n, w = 32, 0.04
    a = operator(n, w)
    x = phantom(grid(n))
    _, sv, _ = model.decompose(a)
    sigma = 2e-4
    e = SplitMix64(1).normal(n * 3000).reshape(3000, n) * sigma
    err = model.rel_error(model.naive_inverse(a, (a @ x)[None, :] + e), x)
    assert math.sqrt(float(np.mean(err ** 2))) == pytest.approx(model.expected_noise_error(sv, sigma, float(np.linalg.norm(x))), rel=0.05)
    assert float(normal_cdf(0.0)) == 0.5


def test_picard_count_excludes_a_component_exactly_at_the_noise_level():
    u = np.eye(3)
    g = np.array([1.0, 0.5, 0.0])
    assert model.picard_count(u, g, 0.5) == 1          # 0,5 ist nicht größer als 0,5
    assert model.picard_count(u, g, 0.0) == 2          # die exakte Null zählt nicht

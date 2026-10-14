"""Orakel (DEMO-PLAYBOOK §12): Kernroutinen und Generator gegen einen anderen Rechenweg (scipy, numerische Quadratur, direkte Lösung, geschlossene Formel).

Jeder Test läuft in unter 10 s. Fehlt scipy im Test-venv, überspringen sich die Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import inv_constants as C
import inv_model as model
from inverse_generator import SplitMix64, blurred_exact, grid, normal_cdf, operator, phantom

scipy = pytest.importorskip("scipy")
from scipy import integrate, linalg, stats  # noqa: E402


@pytest.mark.parametrize("n,w", [(16, 0.02), (32, 0.03), (48, 0.03), (64, 0.04), (96, 0.01)])
def test_singular_values_and_condition_match_scipy(n, w):
    a = operator(n, w)
    ref = linalg.svdvals(a)
    _, sv, _ = model.decompose(a)
    assert np.allclose(sv, ref, rtol=1e-9, atol=1e-14)
    if ref[-1] > 1e-12:
        assert model.condition(sv) == pytest.approx(float(np.linalg.cond(a)), rel=1e-6)


@pytest.mark.parametrize("w", [0.02, 0.03, 0.05])
def test_analytic_blur_matches_numerical_quadrature(w):
    """g(t) = Integral k(t - s) x(s) ds, das Integral über die Stücke des Phantoms (Kastenkanten als Knickpunkte) mit adaptiver Quadratur."""
    def kern(u):
        return math.exp(-u * u / (2.0 * w * w)) / (w * math.sqrt(2.0 * math.pi))

    for t in (0.1, 0.3, 0.45, 0.58, 0.72, 0.78, 0.84, 0.95):
        val, _ = integrate.quad(lambda s: kern(t - s) * float(phantom(s)), 0.0, 1.0, points=[0.72, 0.84], limit=400)
        assert float(blurred_exact(np.array([t]), w)[0]) == pytest.approx(val, abs=1e-8)


def test_normal_cdf_matches_scipy():
    z = np.linspace(-6.0, 6.0, 49)
    assert np.allclose(normal_cdf(z), stats.norm.cdf(z), atol=1e-14)


@pytest.mark.parametrize("n,w", [(24, 0.02), (32, 0.03), (40, 0.025)])
def test_naive_inverse_matches_a_direct_solve_while_the_condition_is_moderate(n, w):
    a = operator(n, w)
    assert np.linalg.cond(a) < 1e6
    b = SplitMix64(n).normal(n)
    assert np.allclose(model.naive_inverse(a, b[None, :])[0], np.linalg.solve(a, b), rtol=1e-6, atol=1e-6)


def test_expected_error_matches_the_trace_of_the_inverse_gram_matrix():
    """E||A^-1 e||^2 = sigma^2 * trace((A A^T)^-1) = sigma^2 * sum 1/sv^2: zweiter Rechenweg über die Spur der inversen Gram-Matrix."""
    n, w, sigma = 32, 0.03, 1e-4
    a = operator(n, w)
    x_norm = float(np.linalg.norm(phantom(grid(n))))
    expected_via_trace = math.sqrt(sigma ** 2 * float(np.trace(np.linalg.inv(a @ a.T)))) / x_norm
    _, sv, _ = model.decompose(a)
    assert model.expected_noise_error(sv, sigma, x_norm) == pytest.approx(expected_via_trace, rel=1e-6)


def test_picard_count_matches_an_eigendecomposition_of_the_symmetric_matrix():
    n, w = 48, 0.03
    a = operator(n, w)
    g = blurred_exact(grid(n), w)
    u, _, _ = model.decompose(a)
    evals, evecs = np.linalg.eigh(a)                       # A ist symmetrisch und positiv definit: gleiche Singulärbasis bis auf Vorzeichen
    for sigma in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
        ref = int(np.sum(np.abs(evecs.T @ g) > sigma))
        assert model.picard_count(u, g, sigma) == ref


@pytest.mark.parametrize("delta", [0.0, 0.5, 1.0, 1.6448536, 2.5, 4.0])
def test_test_power_matches_scipy(delta):
    assert model.test_power(delta) == pytest.approx(float(stats.norm.sf(stats.norm.ppf(0.95) - delta)), abs=1e-6)


def test_test_power_matches_a_simulation_of_the_two_hypotheses():
    n, runs = 48, 20000
    a = operator(n, 0.03)
    _, sv, vt = model.decompose(a)
    sigma = 3.4e-4
    k, change = 44, 2.6
    d = a @ (change * vt[k - 1])
    delta = float(np.linalg.norm(d) / sigma)
    e = SplitMix64(77).normal(runs * n).reshape(runs, n) * sigma
    # Likelihood-Quotienten-Test zwischen b = A x und b = A x + d: Entscheidung für H1, wenn die Projektion auf d die Schwelle überschreitet
    proj0 = (e @ d) / (np.linalg.norm(d) * sigma)
    proj1 = ((e + d) @ d) / (np.linalg.norm(d) * sigma)
    thr = C.Z_ALPHA
    assert float(np.mean(proj0 > thr)) == pytest.approx(0.05, abs=0.01)
    assert float(np.mean(proj1 > thr)) == pytest.approx(model.test_power(delta), abs=0.015)


def test_the_noise_threshold_matches_a_bisection_over_the_closed_form():
    n, w = 48, 0.03
    a = operator(n, w)
    x_norm = float(np.linalg.norm(phantom(grid(n))))
    rms = float(np.sqrt(np.mean(blurred_exact(grid(n), w) ** 2)))
    _, sv, _ = model.decompose(a)
    lo, hi = 1e-12, 1.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if model.expected_noise_error(sv, mid * rms, x_norm) < 1.0:
            lo = mid
        else:
            hi = mid
    assert model.noise_threshold(sv, x_norm, rms) == pytest.approx(lo, rel=1e-6)

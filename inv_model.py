"""Kern: Singulärwerte, direkte Inversion, geschlossene Fehlerformel, Komponentenzahl über dem Rauschen, Testmacht. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand
und `tests/test_oracle_schlecht_gestellt.py` gegen scipy, Quadratur und Monte Carlo.
"""

from __future__ import annotations

import math

import numpy as np

import inv_constants as C
from inverse_generator import blurred_exact, grid, normal_cdf, operator, phantom

_TINY = 1e-300


def decompose(a: np.ndarray):
    """Volle SVD A = U diag(sv) V^T (sv absteigend)."""
    u, sv, vt = np.linalg.svd(a)
    return u, sv, vt


def condition(sv: np.ndarray) -> float:
    """Kondition sigma_1 / sigma_n; unendlich, wenn der kleinste Singulärwert exakt 0 ist."""
    return float(sv[0] / sv[-1]) if sv[-1] > 0.0 else math.inf


def naive_inverse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direkte Inversion x = V diag(1/sv) U^T b für b der Form (reps, n); jeder Singulärwert wird invertiert, auch die winzigen."""
    u, sv, vt = decompose(a)
    with np.errstate(over="ignore", invalid="ignore"):
        coef = (np.atleast_2d(b) @ u) / np.maximum(sv, _TINY)
        return coef @ vt


def rel_error(xhat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """||xhat - x|| / ||x|| je Zeile von xhat (Form (reps, n))."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sqrt(np.sum((np.atleast_2d(xhat) - x) ** 2, axis=-1) / np.sum(x ** 2))


def expected_noise_error(sv: np.ndarray, sigma: float, x_norm: float) -> float:
    """Wurzel des Erwartungswerts von ||xhat - x||^2, geteilt durch ||x||, bei Daten ohne Modellfehler: sigma * sqrt(sum 1/sv_i^2) / ||x||."""
    with np.errstate(over="ignore", divide="ignore"):
        return float(sigma * math.sqrt(float(np.sum(1.0 / np.maximum(sv, _TINY) ** 2))) / x_norm)


def noise_threshold(sv: np.ndarray, x_norm: float, rms_data: float) -> float:
    """Relatives Rauschen, bei dem der erwartete Fehler der direkten Inversion 100 % der Signalnorm erreicht."""
    with np.errstate(over="ignore", divide="ignore"):
        return float(x_norm / (rms_data * math.sqrt(float(np.sum(1.0 / np.maximum(sv, _TINY) ** 2)))))


def picard_coefficients(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Beträge der Komponenten |u_i^T b| der Daten in der Singulärbasis."""
    return np.abs(u.T @ np.asarray(b, dtype=float))


def picard_count(u: np.ndarray, g_exact: np.ndarray, sigma: float) -> int:
    """k*: Zahl der Komponenten der exakten Daten, die größer sind als die Standardabweichung des Rauschens."""
    return int(np.sum(picard_coefficients(u, g_exact) > sigma))


def model_error(n: int, w: float) -> float:
    """rms(g(t_i) - A x_n) / rms(g): was das Gitter als Modellfehler in die Daten bringt; bei Daten aus A x_n ist er null."""
    s = grid(n)
    g = blurred_exact(s, w)
    r = g - operator(n, w) @ phantom(s)
    return float(np.sqrt(np.mean(r * r)) / np.sqrt(np.mean(g * g)))


def twin_shift(sv_k: float, change: float, sigma: float) -> float:
    """Abstand der Datenvektoren in Standardabweichungen des Rauschens, wenn die Wahrheit um `change` (Norm) in Richtung des k-ten Singulärvektors geändert wird."""
    return float(sv_k * change / sigma)


def test_power(delta: float, z_alpha: float = C.Z_ALPHA) -> float:
    """Macht des einseitigen Tests zum Niveau 5 % für zwei Gauß-Mittel im Abstand `delta` Standardabweichungen (bekanntes Rauschen)."""
    return float(1.0 - normal_cdf(z_alpha - delta))


def change_at_half_power(sv_k: float, sigma: float, x_norm: float, z_alpha: float = C.Z_ALPHA) -> float:
    """Änderung der Wahrheit relativ zur Signalnorm, bei der der Test die beiden Wahrheiten mit Macht 50 % unterscheidet (delta = z_alpha)."""
    return float(z_alpha * sigma / (sv_k * x_norm))

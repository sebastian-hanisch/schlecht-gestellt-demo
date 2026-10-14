"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix, symmetrisch, Zeilensumme 1. Handrechnung: Eigenvektoren (1,1,1), (1,0,-1), (1,-2,1) mit Eigenwerten 1, 0,5, 0,1 (alle positiv, also
    Singulärwerte 1, 0,5, 0,1; Kondition 10). Wahrheit x = (1, 2, 1), Daten b = A x = (1,3, 1,4, 1,3). Die Inverse ist [[3,-3,1],[-3,7,-3],[1,-3,3]]:
    eine Störung e = (0,01, -0,02, 0,01) auf dem kleinsten Singulärvektor verschiebt die Lösung um 10 e = (0,1, -0,2, 0,1)."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    inv = np.array([[3.0, -3.0, 1.0], [-3.0, 7.0, -3.0], [1.0, -3.0, 3.0]])
    e = np.array([0.01, -0.02, 0.01])
    return a, x, b, inv, e

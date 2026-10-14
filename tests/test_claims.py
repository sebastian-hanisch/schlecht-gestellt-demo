"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Zufallsströme (inv_constants.SEED_BASE), nie über einen einzelnen Seed als Befund; die Standard-Instanz der
Primäransicht ist ausdrücklich ein einzelner Strom (Seed 35). Relative Werte mit Toleranz 2 %, Punktwerte mit 0,02, damit Rundung und
Bibliotheksversionen (CI installiert ungepinnt) nichts kippen.
"""

import pytest

import inv_constants as C
import inv_evaluation as ev
import inv_model as model


@pytest.fixture(scope="module")
def grid_rows():
    return {r["n"]: r for r in ev.grid_experiment()}


@pytest.fixture(scope="module")
def noise_res():
    return ev.noise_experiment()


@pytest.fixture(scope="module")
def crime_rows():
    return {r["n"]: r for r in ev.crime_experiment()}


@pytest.fixture(scope="module")
def twin_res():
    return ev.twin_experiment()


# ----------------------------------------------------------------------------- Standardinstanz

def test_standard_instance_condition_and_noise():
    a = ev.analyse(ev.Settings())
    assert a.cond == pytest.approx(1.26e4, rel=0.01)
    assert a.sv[0] == pytest.approx(0.996, abs=0.001) and a.sv[-1] == pytest.approx(7.9e-5, rel=0.01)
    assert a.sigma == pytest.approx(3.41e-4, rel=0.01)
    assert a.x_norm == pytest.approx(2.60, abs=0.01)


def test_standard_instance_error_and_components():
    a = ev.analyse(ev.Settings())
    assert a.expected == pytest.approx(2.47, rel=0.01)
    assert a.err == pytest.approx(2.58, rel=0.02) and a.err_honest == a.err and a.err_crime == pytest.approx(2.60, rel=0.02)
    assert a.kstar == 31 and a.settings.n == 48
    assert a.model_error == pytest.approx(0.052, abs=0.002)


def test_presets_numbers_in_the_help_texts():
    free = ev.analyse(ev.Settings(**C.PRESETS["Ohne Rauschen"]))
    assert free.err < 1e-9 and free.err_honest == pytest.approx(0.109, abs=0.002)
    easy = ev.analyse(ev.Settings(**C.PRESETS["Harmloser Fall"]))
    assert easy.cond == pytest.approx(46.4, rel=0.01) and easy.err == pytest.approx(0.109, abs=0.003) and easy.err_crime == pytest.approx(0.014, abs=0.002)
    fine = ev.analyse(ev.Settings(**C.PRESETS["Feineres Gitter"]))
    assert fine.cond == pytest.approx(3.25e7, rel=0.01) and fine.err > 1e3


def test_most_singular_corner_is_numerically_singular():
    a = ev.analyse(ev.Settings(n=96, w=0.05))
    assert a.cond > 1e15 and a.singular


# ----------------------------------------------------------------------------- Experiment 1

def test_experiment_1_condition_by_grid(grid_rows):
    assert grid_rows[12]["cond"] == pytest.approx(1.09, abs=0.01)
    assert grid_rows[32]["cond"] == pytest.approx(45.4, rel=0.01)
    assert grid_rows[48]["cond"] == pytest.approx(1.26e4, rel=0.01)
    assert grid_rows[64]["cond"] == pytest.approx(3.25e7, rel=0.01)


def test_experiment_1_honest_error_is_u_shaped_with_its_minimum_at_n_32(grid_rows):
    honest = {n: r["honest"] for n, r in grid_rows.items()}
    assert min(honest, key=honest.get) == 32 and honest[32] == pytest.approx(0.0425, abs=0.003)
    assert honest[12] == pytest.approx(0.238, abs=0.01) and honest[24] == pytest.approx(0.094, abs=0.005)
    assert honest[40] == pytest.approx(0.174, abs=0.02) and honest[48] == pytest.approx(2.5, rel=0.03)
    assert honest[56] == pytest.approx(77.0, rel=0.03) and honest[64] == pytest.approx(4.47e3, rel=0.03)


def test_experiment_1_crime_error_is_the_closed_form_and_grows_monotonically(grid_rows):
    crime = [grid_rows[n]["crime"] for n in sorted(grid_rows)]
    assert crime == sorted(crime)
    assert grid_rows[12]["crime"] == pytest.approx(8.5e-4, rel=0.05) and grid_rows[32]["crime"] == pytest.approx(0.0139, rel=0.05)
    for n, r in grid_rows.items():
        assert r["crime"] == pytest.approx(r["closed"], rel=0.12), n


# ----------------------------------------------------------------------------- Experiment 2

def test_experiment_2_thresholds(noise_res):
    thr = {r["w"]: r["threshold"] for r in noise_res["rows"]}
    assert thr[0.01] == pytest.approx(0.805, rel=0.01) and thr[0.02] == pytest.approx(0.0658, rel=0.01)
    assert thr[0.03] == pytest.approx(4.05e-4, rel=0.01) and thr[0.04] == pytest.approx(2.57e-7, rel=0.01)
    assert all(abs(r["sim_rms"] - 1.0) < 0.08 for r in noise_res["rows"])


def test_experiment_2_components_by_noise(noise_res):
    assert {c["noise"]: c["kstar"] for c in noise_res["components"]} == {1e-1: 16, 1e-2: 26, 1e-3: 31, 1e-4: 40, 1e-6: 48}


# ----------------------------------------------------------------------------- Experiment 3

def test_experiment_3_without_noise(crime_rows):
    assert crime_rows[32]["crime"] < 1e-13 and crime_rows[32]["honest"] == pytest.approx(0.0399, abs=0.002)
    assert crime_rows[48]["crime"] < 1e-10 and crime_rows[48]["honest"] == pytest.approx(0.109, abs=0.002)
    assert crime_rows[12]["honest"] == pytest.approx(0.238, abs=0.005)
    assert crime_rows[48]["model_error"] == pytest.approx(0.052, abs=0.002) and crime_rows[32]["model_error"] == pytest.approx(0.019, abs=0.002)
    assert crime_rows[12]["model_error"] == pytest.approx(0.274, abs=0.005)


# ----------------------------------------------------------------------------- Experiment 4

def test_experiment_4_power_and_half_power_change(twin_res):
    rows = {r["k"]: r for r in twin_res["rows"]}
    assert rows[42]["power"] == pytest.approx(0.999, abs=0.001)
    assert rows[44]["power"] == pytest.approx(0.74, abs=0.01) and rows[44]["power_sim"] == pytest.approx(0.74, abs=0.02)
    assert rows[46]["power"] == pytest.approx(0.285, abs=0.01)
    assert rows[48]["power"] == pytest.approx(0.149, abs=0.01) and rows[48]["power_sim"] == pytest.approx(0.15, abs=0.02)
    assert rows[31]["half"] == pytest.approx(0.011, abs=0.002) and rows[40]["half"] == pytest.approx(0.169, abs=0.005)
    assert rows[44]["half"] == pytest.approx(0.72, abs=0.01) and rows[48]["half"] == pytest.approx(2.73, abs=0.02)
    assert twin_res["x_norm"] == pytest.approx(2.60, abs=0.01)


def test_mini_instance_by_hand_numbers(mini):
    a, x, b, inv, e = mini
    _, sv, _ = model.decompose(a)
    assert sv.tolist() == pytest.approx([1.0, 0.5, 0.1]) and model.condition(sv) == pytest.approx(10.0)
    assert float((inv @ e)[0]) == pytest.approx(0.1) and float((inv @ e)[1]) == pytest.approx(-0.2)


def test_condition_is_a_good_indicator_but_not_the_exact_size():
    ratios = []
    for n, w in ((32, 0.04), (64, 0.02), (48, 0.03)):
        a = ev.analyse(ev.Settings(n=n, w=w, noise=1e-3))
        ratios.append(a.expected / (1e-3 * a.cond))
    assert ratios == pytest.approx([0.21, 0.23, 0.20], abs=0.01)

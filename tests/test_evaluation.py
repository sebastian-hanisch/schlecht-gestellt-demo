"""Auswertung: Formatierung, Analyse der Instanz, Verdikte (Art und Text), Form der Experimente und ihre Reproduzierbarkeit."""

import math

import numpy as np
import pytest

import inv_constants as C
import inv_evaluation as ev
import inv_model as model
from inverse_generator import SplitMix64, draw_data


# ----------------------------------------------------------------------------- Formatierung

def test_de_uses_decimal_comma_and_a_typographic_minus():
    assert ev.de(-1.234, 2) == "−1,23"


def test_sci_writes_powers_of_ten_with_superscripts():
    assert ev.sci(2.5e9) == "2,5·10⁹"
    assert ev.sci(1.26e4) == "1,3·10⁴"
    assert ev.sci(3.0e-12) == "3,0·10⁻¹²"
    assert ev.sci(9.96e3) == "1,0·10⁴"             # Mantisse rundet auf 10 und rückt den Exponenten vor
    assert ev.sci(0.0) == "0,0"


def test_rel_text_switches_from_percent_to_a_multiple_at_a_thousand_percent():
    assert ev.rel_text(0.0399) == "4,0 % der Signalnorm"
    assert ev.rel_text(8.7e-4) == "0,09 % der Signalnorm"
    assert ev.rel_text(3.0e-12) == "3,0·10⁻¹⁰ % der Signalnorm"
    assert ev.rel_text(0.109) == "11 % der Signalnorm"
    assert ev.rel_text(2.58) == "258 % der Signalnorm"
    assert ev.rel_text(4471.0) == "das 4,5·10³-Fache der Signalnorm"
    assert ev.rel_text(float("inf")) == "unendlich viel"


def test_err_pct_has_the_number_of_digits_by_magnitude():
    assert [ev.err_pct(v) for v in (2.58, 0.109, 0.0399, 0.0087, 8.7e-4, 4471.0)] == ["258 %", "11 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach"]


def test_noise_label_is_a_percentage_of_the_rms():
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0 %", "0,0001 %", "0,001 %", "0,01 %", "0,1 %", "1 %", "10 %"]


# ----------------------------------------------------------------------------- Analyse der Instanz

def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_the_selected_error_is_the_one_of_the_chosen_data():
    honest = ev.analyse(ev.Settings(data="ehrlich"))
    crime = ev.analyse(ev.Settings(data="verbrechen"))
    assert honest.err == honest.err_honest and crime.err == crime.err_crime
    assert honest.err_honest == crime.err_honest and honest.err_crime == crime.err_crime


def test_the_two_kinds_of_data_differ_exactly_by_the_model_error():
    s = ev.Settings(noise=0.0)
    honest = ev.analyse(s)
    assert honest.sigma == 0.0
    crime = ev.analyse(ev.Settings(noise=0.0, data="verbrechen"))
    assert crime.err < 1e-9 < honest.err
    assert honest.model_error > 0.0


def test_the_analysis_is_deterministic_and_the_seed_matters():
    a = ev.analyse(ev.Settings())
    b = ev.analyse(ev.Settings())
    c = ev.analyse(ev.Settings(seed=36))
    assert np.array_equal(a.xhat, b.xhat) and not np.array_equal(a.xhat, c.xhat)


def test_expected_error_matches_the_formula():
    a = ev.analyse(ev.Settings())
    assert a.expected == pytest.approx(model.expected_noise_error(a.sv, a.sigma, a.x_norm), rel=1e-12)


def test_a_huge_condition_is_flagged_as_numerically_singular():
    a = ev.analyse(ev.Settings(n=96, w=0.05))
    assert a.singular and a.cond >= C.SINGULAR_COND
    assert ev.verdict_instance(a)[0] == "warning" and "numerisch singulär" in ev.verdict_instance(a)[1]


def test_zero_noise_with_the_matching_grid_gives_a_tiny_error():
    a = ev.analyse(ev.Settings(**C.PRESETS["Ohne Rauschen"]))
    assert a.err < 1e-9 and a.sigma == 0.0 and a.kstar >= 40


# ----------------------------------------------------------------------------- Verdikte

def test_verdict_kinds_of_the_presets():
    kinds = {name: ev.verdict_instance(ev.analyse(ev.Settings(**p)))[0] for name, p in C.PRESETS.items()}
    assert kinds == {"Standard": "warning", "Ohne Rauschen": "info",
                     "Harmloser Fall": "success", "Feineres Gitter": "warning"}


def test_verdict_instance_numbers_come_from_the_computation():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_instance(a)
    assert kind == "warning"
    assert ev.rel_text(a.err) in text and ev.rel_text(a.expected) in text and f"{a.kstar} von {a.settings.n}" in text


def test_verdict_crime_mentions_the_honest_error():
    a = ev.analyse(ev.Settings(**C.PRESETS["Ohne Rauschen"]))
    kind, text = ev.verdict_instance(a)
    assert kind == "info" and ev.rel_text(a.err_honest) in text


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_grid_experiment_has_one_row_per_n_and_is_reproducible():
    r1 = ev.grid_experiment(ns=(16, 32), reps=50)
    r2 = ev.grid_experiment(ns=(16, 32), reps=50)
    assert [r["n"] for r in r1] == [16, 32] and r1 == r2
    assert all(r["closed"] > 0 and r["honest"] > 0 and r["crime"] > 0 for r in r1)


def test_crime_matches_the_closed_form_and_honest_adds_the_model_error():
    rows = ev.grid_experiment(ns=(24, 32), reps=300)
    for r in rows:
        assert r["crime"] == pytest.approx(r["closed"], rel=0.12)
    assert rows[0]["honest"] > rows[0]["crime"]


def test_noise_experiment_threshold_gives_one_hundred_percent_in_simulation():
    res = ev.noise_experiment(ws=(0.02, 0.03), levels=(1e-2, 1e-3), reps=200)
    for r in res["rows"]:
        assert r["sim_rms"] == pytest.approx(1.0, abs=0.15)
    assert [c["noise"] for c in res["components"]] == [1e-2, 1e-3]
    assert res["components"][0]["kstar"] < res["components"][1]["kstar"]


def test_noise_threshold_falls_with_the_width_of_the_blur():
    rows = ev.noise_experiment(ws=(0.01, 0.02, 0.03), levels=(1e-3,), reps=50)["rows"]
    thr = [r["threshold"] for r in rows]
    assert thr == sorted(thr, reverse=True) and thr[0] / thr[-1] > 100


def test_crime_experiment_without_noise_separates_the_two_data_kinds():
    rows = ev.crime_experiment(ns=(16, 32))
    for r in rows:
        assert r["crime"] < 1e-9 and r["honest"] > 1e-3
        assert r["model_error"] > 0.0


def test_twin_experiment_closed_form_and_simulation_agree():
    res = ev.twin_experiment(ks=(40, 44, 48), runs=4000)
    for r in res["rows"]:
        assert r["power_sim"] == pytest.approx(r["power"], abs=0.03)
    powers = [r["power"] for r in res["rows"]]
    assert powers == sorted(powers, reverse=True)


def test_draw_data_under_the_experiment_streams_is_independent_of_the_run_order():
    a = draw_data(32, 0.03, 0.0, SplitMix64(C.SEED_BASE + 32), crime=False, reps=5, sigma=1e-3)
    b = draw_data(32, 0.03, 0.0, SplitMix64(C.SEED_BASE + 32), crime=False, reps=5, sigma=1e-3)
    assert np.array_equal(a, b)

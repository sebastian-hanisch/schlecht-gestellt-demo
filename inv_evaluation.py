"""Auswertung: Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import inv_constants as C
import inv_model as model
from inverse_generator import SplitMix64, blurred_exact, draw_data, grid, noise_sigma, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def cond_text(c: float) -> str:
    """Kondition: bis 1000 mit einer Nachkommastelle, darüber in Zehnerpotenz."""
    return de(c, 1) if c < 1000.0 else sci(c)


def pct(x: float) -> str:
    """Prozentzahl (der Wert ist schon in Prozent): ab 10 ganzzahlig, ab 0,01 mit zwei Stellen, darunter in Zehnerpotenz."""
    if x >= 10.0:
        return de(x, 0)
    return de(x, 2) if x >= 0.01 else sci(x)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 10 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 9.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def rel_text(err: float) -> str:
    """Fehler als Anteil der Signalnorm in Worten: Prozent bis zum Zehnfachen, darüber als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich viel"
    if err >= 10.0:
        return f"das {sci(err)}-Fache der Signalnorm"
    return f"{err_pct(err)} der Signalnorm"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    data: str = "ehrlich"
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray            # Gitterpunkte
    x: np.ndarray            # Wahrheit auf dem Gitter
    g: np.ndarray            # exakte unscharfe Daten (analytisch)
    sv: np.ndarray           # Singulärwerte
    cond: float
    sigma: float             # absolutes Rauschen
    b: np.ndarray            # Messwerte der gewählten Art
    xhat: np.ndarray         # direkte Inversion der gewählten Daten
    err: float               # relativer Fehler der gewählten Daten
    err_honest: float        # derselbe Fehler mit ehrlichen Daten (gleiches Rauschen)
    err_crime: float         # derselbe Fehler mit Daten aus demselben Gitter
    expected: float          # geschlossene Formel (ohne Modellfehler)
    kstar: int
    coef: np.ndarray         # |u_i^T b|
    singular: bool           # numerisch singulär in doppelter Genauigkeit
    model_error: float       # Modellfehler des Gitters relativ zum Effektivwert der Daten
    x_norm: float


def analyse(settings: Settings) -> Analysis:
    st = settings
    s = grid(st.n)
    x = phantom(s)
    g = blurred_exact(s, st.w)
    a = operator(st.n, st.w)
    u, sv, vt = model.decompose(a)
    sigma = noise_sigma(st.n, st.w, st.noise)
    b_honest = draw_data(st.n, st.w, st.noise, SplitMix64(st.seed), crime=False)
    b_crime = draw_data(st.n, st.w, st.noise, SplitMix64(st.seed), crime=True)
    xh_honest = model.naive_inverse(a, b_honest)[0]
    xh_crime = model.naive_inverse(a, b_crime)[0]
    use_crime = st.data == "verbrechen"
    b = b_crime[0] if use_crime else b_honest[0]
    xhat = xh_crime if use_crime else xh_honest
    x_norm = float(np.sqrt(np.sum(x * x)))
    cond = model.condition(sv)
    return Analysis(
        settings=st, s=s, x=x, g=g, sv=sv, cond=cond, sigma=sigma, b=b, xhat=xhat,
        err=float(model.rel_error(xhat, x)[0]), err_honest=float(model.rel_error(xh_honest, x)[0]), err_crime=float(model.rel_error(xh_crime, x)[0]),
        expected=model.expected_noise_error(sv, sigma, x_norm), kstar=model.picard_count(u, g, sigma),
        coef=model.picard_coefficients(u, b), singular=bool(cond >= C.SINGULAR_COND), model_error=model.model_error(st.n, st.w), x_norm=x_norm,
    )


def standard_sigma() -> float:
    """Absolutes Rauschen der Standardinstanz: bleibt in Experiment 1 und 3 fest, wenn das Gitter wechselt."""
    return noise_sigma(C.DEFAULT_N, C.DEFAULT_W, C.DEFAULT_NOISE)


# ----------------------------------------------------------------------------- Experiment 1: feineres Gitter, schlechtere Lösung

def grid_experiment(ns=C.EXP_N_SWEEP, reps: int = C.EXP_REPS, w: float = C.DEFAULT_W, sigma: float | None = None) -> list[dict]:
    """Fehler der direkten Inversion über die Gitterfeinheit bei gleichem absolutem Rauschen: Daten aus demselben Gitter gegen ehrliche Daten."""
    sigma = standard_sigma() if sigma is None else sigma
    rows = []
    for n in ns:
        a = operator(n, w)
        x = phantom(grid(n))
        _, sv, _ = model.decompose(a)
        x_norm = float(np.sqrt(np.sum(x * x)))
        seed = C.SEED_BASE + n
        bc = draw_data(n, w, 0.0, SplitMix64(seed), crime=True, reps=reps, sigma=sigma)
        bh = draw_data(n, w, 0.0, SplitMix64(seed), crime=False, reps=reps, sigma=sigma)
        err_c = model.rel_error(model.naive_inverse(a, bc), x)
        err_h = model.rel_error(model.naive_inverse(a, bh), x)
        free = model.rel_error(model.naive_inverse(a, blurred_exact(grid(n), w)[None, :]), x)[0]
        rows.append({
            "n": n, "cond": model.condition(sv), "closed": model.expected_noise_error(sv, sigma, x_norm),
            "crime": float(np.sqrt(np.mean(err_c ** 2))), "honest": float(np.sqrt(np.mean(err_h ** 2))), "honest_free": float(free),
        })
    return rows


# ----------------------------------------------------------------------------- Experiment 2: wie viel Rauschen verträgt die Inversion?

def noise_experiment(ws=C.EXP_NOISE_W, levels=C.EXP_NOISE_LEVELS, n: int = C.DEFAULT_N, reps: int = C.EXP_REPS) -> dict:
    """Kippschwelle je Unschärfe (geschlossen, mit Simulation bei dieser Schwelle) und Zahl der Komponenten über dem Rauschen je Rauschstufe."""
    rows = []
    x = phantom(grid(n))
    x_norm = float(np.sqrt(np.sum(x * x)))
    for i, w in enumerate(ws):
        a = operator(n, w)
        _, sv, _ = model.decompose(a)
        rms_data = float(np.sqrt(np.mean(blurred_exact(grid(n), w) ** 2)))
        thr = model.noise_threshold(sv, x_norm, rms_data)
        b = draw_data(n, w, thr, SplitMix64(C.SEED_BASE + 1000 + i), crime=True, reps=reps)
        err = model.rel_error(model.naive_inverse(a, b), x)
        rows.append({"w": w, "cond": model.condition(sv), "threshold": thr, "sim_rms": float(np.sqrt(np.mean(err ** 2)))})
    u, _, _ = model.decompose(operator(n, C.DEFAULT_W))
    g = blurred_exact(grid(n), C.DEFAULT_W)
    comps = [{"noise": lv, "kstar": model.picard_count(u, g, noise_sigma(n, C.DEFAULT_W, lv))} for lv in levels]
    return {"rows": rows, "components": comps, "n": n}


# ----------------------------------------------------------------------------- Experiment 3: dasselbe Gitter belügt

def crime_experiment(ns=C.EXP_N_SWEEP, w: float = C.DEFAULT_W) -> list[dict]:
    """Ohne jedes Rauschen: Fehler der direkten Inversion mit Daten aus demselben Gitter gegen ehrliche Daten, dazu der Modellfehler des Gitters."""
    rows = []
    for n in ns:
        a = operator(n, w)
        x = phantom(grid(n))
        _, sv, _ = model.decompose(a)
        crime = model.rel_error(model.naive_inverse(a, (a @ x)[None, :]), x)[0]
        honest = model.rel_error(model.naive_inverse(a, blurred_exact(grid(n), w)[None, :]), x)[0]
        rows.append({"n": n, "cond": model.condition(sv), "crime": float(crime), "honest": float(honest), "model_error": model.model_error(n, w)})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: zwei Wahrheiten, dieselben Daten

def twin_experiment(ks=C.EXP_TWIN_K, runs: int = C.EXP_TWIN_RUNS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    """Die Wahrheit wird um 100 % der Signalnorm in Richtung des k-ten Singulärvektors geändert: wie gut unterscheidet ein Test die beiden Datensätze?"""
    a = operator(n, w)
    _, sv, vt = model.decompose(a)
    x_norm = float(np.sqrt(np.sum(phantom(grid(n)) ** 2)))
    sigma = noise_sigma(n, w, noise)
    rows = []
    for k in ks:
        d = a @ (x_norm * vt[k - 1])
        delta = model.twin_shift(float(sv[k - 1]), x_norm, sigma)
        e = SplitMix64(C.SEED_BASE + 2000 + k).normal(runs * n).reshape(runs, n) * sigma
        stat = (e @ d) / (np.linalg.norm(d) * sigma) + delta
        rows.append({
            "k": k, "sv": float(sv[k - 1]), "delta": delta, "power": model.test_power(delta), "power_sim": float(np.mean(stat > C.Z_ALPHA)),
            "half": model.change_at_half_power(float(sv[k - 1]), sigma, x_norm),
        })
    return {"rows": rows, "n": n, "x_norm": x_norm, "sigma": sigma, "runs": runs}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    s = a.settings
    if a.singular:
        return "warning", (f"Kondition {cond_text(a.cond)}: In doppelter Genauigkeit (Rundung rund 1e-16) ist diese Matrix numerisch singulär. Die direkte Inversion "
                           f"hat einen Fehler von {rel_text(a.err)}, selbst ohne Rauschen.")
    if a.err >= 1.0:
        return "warning", (f"Der Fehler der direkten Inversion beträgt {rel_text(a.err)}: eine Lösung, die nichts mehr mit der Wahrheit zu tun hat. Erwartet waren "
                           f"{rel_text(a.expected)} allein durch das Rauschen (geschlossene Formel); Kondition {cond_text(a.cond)}, {a.kstar} von {s.n} Komponenten "
                           "liegen über dem Rauschen.")
    if s.data == "verbrechen" and a.err_honest > 2.0 * max(a.err, 1e-12):
        return "info", (f"Mit Daten aus demselben Gitter beträgt der Fehler {rel_text(a.err)}. Mit ehrlichen Daten und demselben Rauschen wären es "
                        f"{rel_text(a.err_honest)}: Der Test mit demselben Gitter zeigt eine Genauigkeit, die es nicht gibt.")
    return "success", (f"Fehler {rel_text(a.err)} (erwartet {rel_text(a.expected)} aus dem Rauschen). Kondition {cond_text(a.cond)}: In diesem Bereich ist die "
                       "direkte Inversion noch brauchbar. Mit mehr Unschärfe, feinerem Gitter oder mehr Rauschen kippt sie.")


def verdict_grid(rows: list[dict]) -> tuple[str, str]:
    best = min(rows, key=lambda r: r["honest"])
    worst = rows[-1]
    first = rows[0]
    return "warning", (f"Mit ehrlichen Daten ist der Fehler am kleinsten bei n = {best['n']} ({rel_text(best['honest'])}) und wächst danach: bei n = {worst['n']} "
                       f"{rel_text(worst['honest'])}. Mit Daten aus demselben Gitter sieht es bei n = {first['n']} nach {rel_text(first['crime'])} aus, "
                       f"wo es in Wahrheit {rel_text(first['honest'])} sind.")


def verdict_noise(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    std = next(r for r in rows if abs(r["w"] - C.DEFAULT_W) < 1e-9)
    easy = rows[0]
    return "info", (f"In der Standardinstanz (w = {de(std['w'], 3)}) wird die direkte Inversion schon bei {pct(100.0 * std['threshold'])} % Rauschen wertlos; "
                    f"bei w = {de(easy['w'], 3)} wäre es erst bei {pct(100.0 * easy['threshold'])} %. Je 0,005 mehr Unschärfe kostet eine bis drei Größenordnungen.")


def verdict_crime(rows: list[dict]) -> tuple[str, str]:
    clean = next(r for r in rows if r["n"] == 32)
    first = rows[0]
    return "warning", (f"Ohne jedes Rauschen kommt die Wahrheit mit Daten aus demselben Gitter bei n = {clean['n']} bis auf einen Fehler von {pct(100.0 * clean['crime'])} % zurück; "
                       f"mit ehrlichen Daten bleiben {de(100.0 * clean['honest'], 1)} % Fehler, bei n = {first['n']} sogar {de(100.0 * first['honest'], 0)} %. "
                       "Der Unterschied ist der Modellfehler des Gitters.")


def verdict_twin(res: dict) -> tuple[str, str]:
    last = res["rows"][-1]
    return "info", (f"Zwei Wahrheiten, die sich um die ganze Signalnorm unterscheiden, geben ab Komponente {last['k']} Datensätze, die ein Test nur mit Macht "
                    f"{de(100.0 * last['power'], 0)} % trennt (5 % Fehlalarm). Erst bei einer Änderung um {de(100.0 * last['half'], 0)} % der Signalnorm "
                    "trennt der Test in der Hälfte der Fälle.")

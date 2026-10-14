"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: SVD einer 96 x 96-Matrix (O(n^3), unter 20 ms) und 400 Wiederholungen je Gitterpunkt in den Experimenten.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)        # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
DATA_CHOICES = ("ehrlich", "verbrechen")
DATA_LABELS = {"ehrlich": "Ehrliche Daten (analytisch, vom Gitter unabhängig)", "verbrechen": "Daten aus demselben Gitter (inverse crime)"}
SEED_MAX, DEFAULT_SEED = 999999, 35

# Ab dieser Kondition ist die Matrix in doppelter Genauigkeit numerisch singulär (Rundungsfehler der Größe 1e-16 mal Kondition).
SINGULAR_COND = 1e15
SHOW_RANGE = (-1.5, 2.0)                                          # Wertebereich der Rekonstruktion im Diagramm; alles darüber wird abgeschnitten
Z_ALPHA = 1.6448536                                               # einseitiger Test zum Niveau 5 %

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 800000
EXP_REPS = 400                                                    # Wiederholungen je Zelle
EXP_N_SWEEP = (12, 16, 20, 24, 28, 32, 40, 48, 56, 64)           # Experiment 1 und 3
EXP_NOISE_W = (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04)     # Experiment 2
EXP_NOISE_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)                # Experiment 2: Komponenten über dem Rauschen
EXP_TWIN_K = (31, 36, 40, 42, 44, 46, 48)                         # Experiment 4: Index der addierten Komponente (n = 48)
EXP_TWIN_RUNS = 20000

COLOR_TRUTH = "#2F6B65"
COLOR_NAIVE = "#c0392b"
COLOR_HONEST = "#4c78a8"
COLOR_CRIME = "#D68A2E"
COLOR_GREY = "#8a8a8a"

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "seed": DEFAULT_SEED},
    "Ohne Rauschen": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": 0.0, "data": "verbrechen", "seed": DEFAULT_SEED},
    "Harmloser Fall": {"n": DEFAULT_N, "w": 0.02, "noise": DEFAULT_NOISE, "data": "ehrlich", "seed": DEFAULT_SEED},
    "Feineres Gitter": {"n": 64, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "48 Messpunkte, Unschärfe 0,03, ein Promille Rauschen: Der Fehler der direkten Inversion liegt bei 258 % der Signalnorm.",
    "Ohne Rauschen": "Kein Rauschen, die Daten kommen aus derselben Matrix, die invertiert wird: Die Wahrheit kommt scheinbar perfekt zurück. Mit ehrlichen Daten bleiben 11 % Fehler.",
    "Harmloser Fall": "Unschärfe 0,02 statt 0,03: kleine Kondition (46), die Inversion ist brauchbar (11 % Fehler). Mit Daten aus demselben Gitter wären es nur 1,4 %.",
    "Feineres Gitter": "Dieselbe Messung, aber auf 64 statt 48 Punkten ausgewertet: Kondition 3·10⁷ statt 1·10⁴, der Fehler wächst um Größenordnungen.",
}

# Das schlecht gestellte Problem: genauer messen macht es schlimmer – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-schlecht-gestellt-demo.streamlit.app/)**

Erstes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning) und die erste Demo des Portfolios, in der eine Größe aus ihrer unscharfen, verrauschten Messung rekonstruiert wird. Vehikel ist neutral: ein Phantom aus zwei Hügeln und einem Kasten, das mit einer Gauß-Unschärfe gemessen wird. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist.

## Kernfrage

Eine Messung zeigt die Wahrheit unscharf und verrauscht. Die direkte Inversion rechnet die Unschärfe einfach zurück. Wie gut gelingt das, was ändert ein feineres Gitter oder weniger Rauschen, und stimmt ein Test, bei dem die Daten aus derselben Matrix stammen, die danach invertiert wird?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie:

- **Wahrheit** x(s) auf [0, 1]: zwei Gauß-Hügel (Mitte 0,30 und 0,58, Breite 0,050 und 0,030, Höhe 1,0 und 0,7) und ein Kasten (0,72 bis 0,84, Höhe 0,5).
- **Messung** g(t) = ∫ k(t − s) x(s) ds mit dem Gauß-Kern k der Breite w; g ist für dieses Phantom **analytisch** bekannt (Hügel bleiben Hügel, der Kasten wird zur Differenz zweier Normalverteilungsfunktionen).
- **Gitter** mit n Punkten (Mittelpunktregel): A = h K. Standard n = 48, w = 0,03.
- **Rauschen** weiß, Standardabweichung = Rauschstufe · Effektivwert von g (Standard 0,1 %, absolut 3,4·10⁻⁴).
- **Zwei Arten von Daten:** *ehrlich* (b = g(tᵢ) + e, vom Gitter unabhängig) und *aus demselben Gitter* (b = A x + e, „inverse crime“); das Rauschen e ist in beiden gleich.

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Direkte Inversion** über die volle Singulärwertzerlegung, jeder Singulärwert wird invertiert: x̂ = V diag(1/σᵢ) Uᵀ b.
- **Geschlossene Fehlerformel:** Für Daten aus demselben Gitter ist E‖x̂ − x‖² = σ² Σ 1/σᵢ²; der relative Fehler σ·√(Σ 1/σᵢ²)/‖x‖ steht neben der Simulation.
- **k\*:** Zahl der Komponenten |uᵢᵀ g| der exakten Daten, die größer sind als das Rauschen (diskrete Picard-Zahl).
- **Zwei Wahrheiten:** Änderung um c·vₖ ändert die Daten um σₖ c·uₖ, also um δ = σₖ c/σ Standardabweichungen; Macht des Tests 1 − Φ(z₀,₉₅ − δ).
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 400 Wiederholungen je Zelle (Experiment 4: 20 000), nie ein einzelner Seed.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03, Rauschen 0,1 %, Seed 35), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Wie schlecht ist die Matrix? | Singulärwerte von 0,996 bis 7,9·10⁻⁵, Kondition **1,3·10⁴**; Signalnorm 2,60, absolutes Rauschen 3,4·10⁻⁴. | `test_standard_instance_condition_and_noise` |
| Was liefert die direkte Inversion? | Fehler **258 %** der Signalnorm; erwartet (geschlossen, nur Rauschen) 247 %; 31 von 48 Komponenten liegen über dem Rauschen. Mit Daten aus demselben Gitter 259 %. | `test_standard_instance_error_and_components` |
| Feineres Gitter bei gleichem Rauschen (Experiment 1, ehrliche Daten) | Fehler fällt von 24 % (n = 12) auf **4,2 % bei n = 32** und steigt danach: 17 % (n = 40), rund 250 % (n = 48), 77-fach (n = 56), 4,5·10³-fach (n = 64). U-Form durch Modellfehler (links) und Rauschverstärkung (rechts). | `test_experiment_1_honest_error_is_u_shaped_with_its_minimum_at_n_32` |
| … und mit Daten aus demselben Gitter? | Steigt monoton von 0,09 % (n = 12) über 1,4 % (n = 32) auf 4,5·10³-fach (n = 64) und stimmt mit der geschlossenen Formel auf 12 % überein. Bei n = 12 sieht es nach 0,09 % aus, wo es in Wahrheit 24 % sind. | `test_experiment_1_crime_error_is_the_closed_form_and_grows_monotonically` |
| Kondition nach Gitter | 1,1 (n = 12), 45 (n = 32), 1,3·10⁴ (n = 48), 3,3·10⁷ (n = 64). | `test_experiment_1_condition_by_grid` |
| Wie viel Rauschen verträgt die Inversion (Experiment 2)? | Fehler 100 % bei Rauschen **81 %** (w = 0,01), 6,6 % (0,02), **0,04 %** (0,03, Standard), 2,6·10⁻⁵ % (0,04). Die Simulation bei der Schwelle liefert den quadratischen Mittelwert 0,95 bis 1,01. | `test_experiment_2_thresholds` |
| Wie viele Komponenten stecken über dem Rauschen? | k\* = 16 / 26 / 31 / 40 / 48 für 10 % / 1 % / 0,1 % / 0,01 % / 0,0001 % Rauschen (von 48). Hundertmal weniger Rauschen (1 % auf 0,01 %) bringt 14 Komponenten dazu. | `test_experiment_2_components_by_noise` |
| Ganz ohne Rauschen (Experiment 3) | Aus demselben Gitter: Fehler 4,4·10⁻¹³ % (n = 32), 3,1·10⁻¹⁰ % (n = 48). Ehrliche Daten: 4,0 % (n = 32), 11 % (n = 48), 24 % (n = 12). Der Modellfehler des Gitters liegt bei 27 % (n = 12), 1,9 % (n = 32), 5,2 % (n = 48) und ist **nicht monoton** in n (die Lage der Kastenkanten entscheidet). | `test_experiment_3_without_noise` |
| Zwei Wahrheiten, dieselben Daten (Experiment 4) | Ändert man die Wahrheit um die ganze Signalnorm in Richtung der 44. Komponente, trennt ein Test die Datensätze mit Macht **74 %**, bei der 46. mit 29 %, bei der 48. mit **15 %** (5 % Fehlalarm). Simulation und geschlossene Form stimmen auf 0,02 überein. Macht 50 % erst bei einer Änderung um 72 % (44.) und 273 % (48.) der Signalnorm. | `test_experiment_4_power_and_half_power_change` |
| Presets | Ohne Rauschen: Aus demselben Gitter < 1e-9, ehrlich 11 %. Harmloser Fall (w = 0,02): Kondition 46, Fehler 11 %, mit Daten aus demselben Gitter 1,4 %. Feineres Gitter (n = 64): Kondition 3,3·10⁷, Fehler über 10³. | `test_presets_numbers_in_the_help_texts` |
| Numerisch singulär | n = 96, w = 0,05: Kondition über 10¹⁵. | `test_most_singular_corner_is_numerically_singular` |
| Mini-Instanz von Hand | 3 × 3-Matrix mit Singulärwerten 1 / 0,5 / 0,1, Kondition 10; eine Störung auf dem kleinsten Singulärvektor wird genau verzehnfacht. | `test_mini_instance_by_hand_numbers`, `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Standardinstanz gewechselt.** Der Plan sah zuerst n = 64, w = 0,04 vor; dort liegt die Kondition bei 2,3·10¹³ und der Fehler bei 10⁹, nicht mehr zeichenbar. n = 48, w = 0,03 gibt 258 %.
- **Das Verbrechen ist nur im harmlosen Bereich sichtbar.** Im harten Bereich (Standard) ist das Rauschen größer als der Modellfehler, mit Daten aus demselben Gitter und ehrlichen Daten ist der Fehler gleich (258 % und 259 %). Der Unterschied zeigt sich ohne Rauschen und bei kleiner Kondition (n = 48, w = 0,02: 1,4 % gegen 11 %).
- **Die Kondition ist ein Anzeiger, nicht die exakte Größe.** Fehler/(Rauschen · Kondition) liegt in drei Instanzen bei 0,21 / 0,23 / 0,20; exakt ist die geschlossene Formel.
- **Der Modellfehler fällt nicht glatt mit n.** Bei n = 48 liegen die Kastenkanten auf halben Zellen (5,2 %), bei n = 32 fast auf Zellgrenzen (1,9 %).

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Dimension, ein Phantom** | Zwei Hügel und ein Kasten, weißes Rauschen; andere Formen verschieben den Modellfehler. | Auflösung und Versuchsplanung |
| **Der Kern ist bekannt** | Die Breite w ist der Demo bekannt; ist sie unbekannt, muss sie mitgeschätzt werden. | Blinde Entfaltung |
| **Das Rauschen ist weiß, seine Größe bekannt** | Formel und Test kennen σ; ein unbekanntes σ muss geschätzt werden. | Parameterwahl |
| **Doppelte Genauigkeit** | Ab Kondition rund 10¹⁵ ist die Matrix numerisch singulär. | – |
| **Wahrheit auf dem Gitter** | Fehler wird nur an den Gitterpunkten gemessen. | – |
| **Nur die direkte Inversion** | Sie ist hier ein bewusst schlechtes Verfahren; Regularisierung ist Inhalt der Folgestücke. | Tikhonov |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [kalman-filter-demo](https://github.com/sebastian-hanisch/kalman-filter-demo): schätzt einen Zustand sequentiell aus verrauschten Messungen; hier steht ein Parameter, nicht die Zeit. Das Kalman-Update ist später ein Spezialfall der Bayes-Sicht in dieser Linie.
- [ica-demo](https://github.com/sebastian-hanisch/ica-demo): trennt mehrere Quellen blind; hier wird eine Größe aus einer Messung rekonstruiert, die Unschärfe ist bekannt.
- [pca-demo](https://github.com/sebastian-hanisch/pca-demo): Singulärwertzerlegung als Datenwerkzeug; hier ist sie das Analysewerkzeug für die Matrix der Messung.

## Tests

Mini-Instanz von Hand (3 × 3, Singulärwerte 1 / 0,5 / 0,1), Kern gegen scipy (`svdvals`, `norm`), numerische Quadratur der analytischen Unschärfe, Spur der inversen Gram-Matrix, Eigenzerlegung, direkte Lösung und Simulation; Generator (SplitMix64 gegen eine unabhängige Fassung, Phantom, Operator, Daten), Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Slider-Grenzen, jede Rauschstufe und Datenart, singuläre Ecke, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `inv_model.py` | Singulärwerte, direkte Inversion, Fehlerformel, k\*, Testmacht |
| `inv_evaluation.py` | Kennzahlen, Experimente, Verdikte |
| `inv_visualization.py`, `inv_pdf_export.py` | Figuren, PDF |
| `inv_constants.py`, `inv_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Regularisierung** (Tikhonov, abgeschnittene SVD, iterative Verfahren, Sparsity): die Folgestücke der Linie; hier nur das Problem, das sie lösen.
- **2D und Tomographie, nichtlineare und blinde Probleme:** spätere Stücke.
- **Bayes-Sicht und Unsicherheit der Rekonstruktion:** Folgestücke.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html).

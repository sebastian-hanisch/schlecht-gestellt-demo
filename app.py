"""Das schlecht gestellte Problem: genauer messen macht es schlimmer - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Linie Inverse Probleme und Regularisierung. Ein Phantom wird unscharf und verrauscht gemessen; die direkte Inversion rechnet die Unschärfe
zurück und verstärkt dabei das Rauschen. Kern: inv_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import inv_constants as C
from inv_evaluation import (
    Settings,
    analyse,
    crime_experiment,
    de,
    err_pct,
    grid_experiment,
    noise_experiment,
    cond_text,
    noise_label,
    pct,
    rel_text,
    sci,
    standard_sigma,
    twin_experiment,
    verdict_crime,
    verdict_grid,
    verdict_instance,
    verdict_noise,
    verdict_twin,
)
from inv_pdf_export import build_pdf
from inv_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from inv_visualization import (
    build_components_chart,
    build_crime_chart,
    build_data_chart,
    build_grid_chart,
    build_noise_chart,
    build_picard_chart,
    build_recon_chart,
    build_table,
    build_twin_chart,
    style_table,
)

st.set_page_config(page_title="Das schlecht gestellte Problem – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _grid():
    return grid_experiment()


@st.cache_data(show_spinner=SPINNER)
def _noise():
    return noise_experiment()


@st.cache_data(show_spinner=SPINNER)
def _crime():
    return crime_experiment()


@st.cache_data(show_spinner=SPINNER)
def _twin():
    return twin_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🔍 Das schlecht gestellte Problem: genauer messen macht es schlimmer")
st.markdown(
    """
**Eine unscharfe, verrauschte Messung lässt sich nicht einfach zurückrechnen.** Dies ist das erste Stück der Linie **Inverse Probleme und Regularisierung** und die
erste Demo des Portfolios, in der eine Größe aus ihrer unscharfen Messung rekonstruiert wird. Ein Phantom aus zwei Hügeln und einem Kasten wird mit einer
**Gauß-Unschärfe** gemessen und mit etwas **Rauschen** abgelesen. Die direkte Inversion rechnet die Unschärfe zurück und verstärkt dabei das Rauschen um bis zu
die **Kondition** der Matrix. Die Welt ist simuliert; das Phantom ist die bekannte Wahrheit.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/inverse-probleme-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich eine Kurve mit zwei Hügeln und einem Kasten vor. Ein Messgerät verwischt sie und liest die Werte mit etwas zufälligem Zittern ab, dem **Rauschen**. Gesucht ist die ursprüngliche Kurve. Die Demo kennt sie, weil sie am Rechner erfunden wurde, und kann jedes Ergebnis an ihr messen. Wer die Verwischung einfach rückwärts rechnet, macht dasselbe wie ein Schärfen-Regler, der zu weit aufgedreht ist: Das Zittern wird mitverstärkt, und das Ergebnis kann weit neben der Wahrheit liegen. Eine solche Aufgabe heißt **schlecht gestellt**, weil winzige Fehler in den Messwerten das Ergebnis völlig verändern.

Dazu zeigt die Demo drei Dinge. Ein feineres Gitter, also mehr Punkte zum Messen und Rechnen, hilft zuerst und schadet danach. Weniger Rauschen hilft nur langsam. Und ein Test, dessen Daten aus demselben Rechengitter stammen wie die Rückrechnung, sieht besser aus, als er ist. Ein Gegenmittel, die **Regularisierung**, behandeln die weiteren Demos zu diesem Thema.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/inverse-probleme-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die direkte Inversion", expanded=True):
    st.markdown(
        r"""
1. **Vorwärts.** Die Messung entsteht aus der Wahrheit $x$ durch eine Matrix $A$ (die Unschärfe) und Rauschen: $b = A x + e$.
2. **Rückwärts.** Die direkte Inversion rechnet $\hat x = A^{-1} b$. Ohne Rauschen wäre das die Wahrheit.
3. **Singulärwerte.** Die Matrix dämpft feine Strukturen sehr stark: Ihre Singulärwerte fallen schnell gegen null. Die Inversion teilt durch sie und macht
   dabei jede kleine Störung in diesen Richtungen riesig.
4. **Das Rauschen entscheidet.** Eine Komponente der Daten ist nur dann verwertbar, wenn sie über dem Rauschen liegt. Die Zahl $k^*$ dieser Komponenten
   wächst mit weniger Rauschen nur langsam.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Messung**")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Wie viele Messpunkte und Unbekannte. Mehr Punkte lösen die Wahrheit feiner auf.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    st.markdown("**Daten**")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_CHOICES, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrliche Daten kommen aus der analytischen Unschärfe und sind vom Gitter unabhängig. Daten aus demselben Gitter entstehen aus "
                             "der Matrix, die danach invertiert wird (inverse crime).")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Rauschen würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "data_select": data, "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), data, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kondition der Matrix", cond_text(a.cond), delta="numerisch singulär" if a.singular else "in doppelter Genauigkeit rechenbar", delta_color="off",
          help="Verhältnis von größtem zu kleinstem Singulärwert: um so viel kann die Inversion eine Störung verstärken.")
c2.metric("Fehler der direkten Inversion", err_pct(a.err), delta="relativ zur Signalnorm", delta_color="off",
          help="‖x̂ − x‖ / ‖x‖ mit der Wahrheit x des Phantoms.")
c3.metric("Erwartet aus dem Rauschen", err_pct(a.expected), delta="geschlossene Formel", delta_color="off",
          help="σ·√(Σ 1/σᵢ²) / ‖x‖: gilt ohne Modellfehler, also für Daten aus demselben Gitter.")
c4.metric("Komponenten über dem Rauschen", f"{a.kstar} von {settings.n}", delta="k*", delta_color="off",
          help="Zahl der Komponenten der exakten Daten, die größer sind als die Standardabweichung des Rauschens.")

tab_rec, tab_data, tab_pic, tab_tab = st.tabs(["Rekonstruktion", "Messung", "Singulärwerte", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, rot die direkte Inversion. Die Achse ist fest; was darüber hinausgeht, wird abgeschnitten.")
with tab_data:
    st.plotly_chart(build_data_chart(a), width="stretch", key="data_chart")
    st.caption("Die Messung ist glatter als die Wahrheit: Der Kasten und die Hügel sind verwischt, die Messwerte liegen mit Rauschen um die blaue Kurve.")
with tab_pic:
    st.plotly_chart(build_picard_chart(a), width="stretch", key="picard_chart")
    st.caption("Grün die Singulärwerte, rot die Komponenten der Daten in dieser Basis. Wo die roten Punkte das Rauschen (grau gestrichelt) erreichen, "
               "hört die verwertbare Information auf (k*, orange).")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz; „Direkte Inversion“ in Zehnerpotenzen, wenn der Betrag über 10 000 liegt.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="schlecht-gestelltes-problem.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist der direkten Inversion zu trauen?")
r1, r2, r3 = st.columns(3)
r1.metric("Fehler mit ehrlichen Daten", err_pct(a.err_honest), help="Daten aus der analytischen Unschärfe, mit demselben Rauschen.")
r2.metric("Fehler mit Daten aus demselben Gitter", err_pct(a.err_crime), help="Daten aus der Matrix, die danach invertiert wird (inverse crime).")
r3.metric("Modellfehler des Gitters", f"{de(100.0 * a.model_error, 1)} %", help="Wie weit die Matrix die analytische Unschärfe der Daten verfehlt, relativ zu ihrem Effektivwert.")
verdict(verdict_instance(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

sd0 = standard_sigma()
st.subheader("🔬 Experiment 1: Feineres Gitter, schlechtere Lösung")
st.caption(f"Standardunschärfe (w = {de(C.DEFAULT_W, 3)}), absolutes Rauschen wie in der Standardinstanz ({sci(sd0)}), {C.EXP_REPS} feste Wiederholungen je Gitter. "
           "Dieselbe Messung wird auf feineren Gittern ausgewertet.")
rows1 = _grid()
st.plotly_chart(build_grid_chart(rows1), width="stretch", key="grid_chart")
verdict(verdict_grid(rows1))
st.dataframe(
    [{"n": r["n"], "Kondition": cond_text(r["cond"]), "Erwartet (nur Rauschen)": rel_text(r["closed"]),
      "Ehrliche Daten": rel_text(r["honest"]), "Daten aus demselben Gitter": rel_text(r["crime"])} for r in rows1],
    hide_index=True, width="stretch")
st.caption("Feiner gerechnet heißt nicht genauer: Die Kondition wächst, und das Rauschen wird um sie verstärkt. Das Gegenmittel steht im Folgestück "
           "**Tikhonov-Regularisierung**.")

st.subheader("🔬 Experiment 2: Wie viel Rauschen verträgt die Inversion?")
res2 = _noise()
st.caption(f"Geschlossene Formel für das Rauschen, bei dem der Fehler 100 % der Signalnorm erreicht (n = {res2['n']}); die Simulation bei dieser Schwelle "
           f"({C.EXP_REPS} feste Wiederholungen) soll den quadratischen Mittelwert 1,0 liefern.")
st.plotly_chart(build_noise_chart(res2["rows"]), width="stretch", key="noise_chart")
verdict(verdict_noise(res2))
st.dataframe(
    [{"Breite w": de(r["w"], 3), "Kondition": cond_text(r["cond"]), "Schwelle (Rauschen)": f"{pct(100.0 * r['threshold'])} %",
      "Simuliert bei der Schwelle": de(r["sim_rms"], 2)} for r in res2["rows"]],
    hide_index=True, width="stretch")
st.markdown("**Komponenten über dem Rauschen** (Standardunschärfe, n = %d)" % res2["n"])
st.plotly_chart(build_components_chart(res2["components"], res2["n"]), width="stretch", key="components_chart")
k_by = {c["noise"]: c["kstar"] for c in res2["components"]}
st.caption(f"Hundertmal weniger Rauschen (von {noise_label(1e-2)} auf {noise_label(1e-4)}) bringt nur {k_by[1e-4] - k_by[1e-2]} Komponenten dazu; den Rest bestimmt "
           "das Rauschen, nicht die Messung. Wie man damit umgeht, zeigen **Abgeschnittene SVD** und **Parameterwahl**.")

st.subheader("🔬 Experiment 3: Dasselbe Gitter belügt")
st.caption(f"Ganz ohne Rauschen, Standardunschärfe (w = {de(C.DEFAULT_W, 3)}). Daten aus der Matrix, die danach invertiert wird, gegen ehrliche (analytische) Daten.")
rows3 = _crime()
st.plotly_chart(build_crime_chart(rows3), width="stretch", key="crime_chart")
verdict(verdict_crime(rows3))
st.dataframe(
    [{"n": r["n"], "Kondition": cond_text(r["cond"]), "Daten aus demselben Gitter": rel_text(r["crime"]), "Ehrliche Daten": rel_text(r["honest"]),
      "Modellfehler des Gitters": f"{de(100.0 * r['model_error'], 1)} %"} for r in rows3],
    hide_index=True, width="stretch")
crime48 = next(r for r in rows3 if r["n"] == C.DEFAULT_N)
st.caption(f"Der Modellfehler ist das Rauschen, das ein Test mit demselben Gitter versteckt. Bei n = {C.DEFAULT_N} kommt die Wahrheit bis auf einen Fehler von {pct(100.0 * crime48['crime'])} % "
           "zurück, ohne dass das Verfahren etwas kann. Jede Demo der Linie muss deshalb mit Daten testen, die nicht aus dem invertierten Modell stammen.")

st.subheader("🔬 Experiment 4: Zwei Wahrheiten, dieselben Daten")
res4 = _twin()
st.caption(f"Standardinstanz (n = {res4['n']}), Rauschen {sci(res4['sigma'])} absolut. Die Wahrheit wird um 100 % der Signalnorm in Richtung der k-ten "
           f"Singulärkomponente geändert; ein Test mit bekanntem Rauschen und 5 % Fehlalarm versucht, die beiden Datensätze zu trennen "
           f"({str(res4['runs'])[:-3]}.{str(res4['runs'])[-3:]} Läufe je k).")
st.plotly_chart(build_twin_chart(res4), width="stretch", key="twin_chart")
verdict(verdict_twin(res4))
st.dataframe(
    [{"k": r["k"], "Singulärwert": sci(r["sv"]), "Abstand der Daten (in σ)": de(r["delta"], 2), "Macht geschlossen": f"{de(100.0 * r['power'], 1)} %",
      "Macht simuliert": f"{de(100.0 * r['power_sim'], 1)} %", "Änderung bei Macht 50 %": f"{de(100.0 * r['half'], 0)} % der Signalnorm"} for r in res4["rows"]],
    hide_index=True, width="stretch")
st.caption("Die Daten allein sagen nicht, welche von zwei Wahrheiten gilt, wenn sie sich nur in schwachen Komponenten unterscheiden. Was man dem Problem "
           "zusätzlich mitgibt, damit trotzdem eine Antwort herauskommt, ist der Inhalt der Folgestücke.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
grid_mid = next(r for r in rows1 if r["n"] == C.DEFAULT_N)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Dimension, ein Phantom** | Zwei Hügel und ein Kasten, weißes Rauschen: Der Modellfehler hängt von der Lage der Kastenkanten im Gitter ab und ist nicht monoton in n (Experiment 3). | Auflösung und Versuchsplanung |
| **Der Kern ist bekannt** | Die Breite w ist der Demo bekannt. Ist sie unbekannt, muss man sie mitschätzen. | Blinde Entfaltung |
| **Das Rauschen ist weiß und seine Größe bekannt** | Die Formel und der Test kennen σ. Ein unbekanntes σ muss man schätzen. | Parameterwahl |
| **Doppelte Genauigkeit** | Ab Kondition rund 10¹⁵ ist die Matrix numerisch singulär; die Rundung allein zerstört die Lösung. | – |
| **Wahrheit auf dem Gitter** | Das Phantom ist nur an den Gitterpunkten bekannt. Der Fehler wird dort gemessen, nicht dazwischen. | – |
| **Nur die direkte Inversion** | Sie ist hier ein bewusst schlechtes Verfahren (Fehler {rel_text(grid_mid['honest'])} bei n = 48). Regularisierung ist Inhalt der Folgestücke. | Tikhonov |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/inverse-probleme-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine Messung $b$, die aus der unbekannten Wahrheit $x$ durch eine Unschärfe-Matrix $A$ und Rauschen $e$ entsteht: $b = A x + e$. Gesucht ist $x$. Die direkte Inversion rechnet $\hat x = V\,\mathrm{diag}(1/\sigma_i)\,U^\top b$ und verstärkt dabei das Rauschen.

| Symbol | Bedeutung |
|---|---|
| $x$, $b$, $e$ | Wahrheit auf dem Gitter, Messung und Rauschen |
| $A = hK$ | Unschärfe-Matrix auf dem Gitter mit Schrittweite $h = 1/n$ (Mittelpunktregel) |
| $w$, $n$ | Breite der Gauß-Unschärfe und Zahl der Gitterpunkte |
| $\sigma$ | Standardabweichung des Rauschens |
| $\sigma_i$ | Singulärwerte von $A$: wie stark die Unschärfe die $i$-te feine Komponente dämpft |
| $U$, $V$ | Singulärvektoren von $A$ (Basis der Daten bzw. der Wahrheit) |
| $\kappa = \sigma_1/\sigma_n$ | Kondition: größtmögliche Verstärkung einer Störung durch die Inversion |
| $\hat x$ | Rekonstruktion durch die direkte Inversion |
| $x_n$ | Wahrheit, ausgewertet auf dem Gitter |
| $k^*$ | Zahl der Komponenten der exakten Daten, die über dem Rauschen liegen |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/inverse-probleme-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Vorwärtsmodell.** Die Wahrheit $x$ auf $[0,1]$ wird mit dem Gauß-Kern $k(u) = \exp(-u^2/(2w^2))/(w\sqrt{2\pi})$ unscharf gemessen:
$g(t) = \int k(t-s)\,x(s)\,ds$. Auf dem Gitter $s_j = (j + \tfrac12)/n$ ist $A = hK$ mit $K_{ij} = k(s_i - s_j)$ und $h = 1/n$ (Mittelpunktregel). Das Phantom
$x$ besteht aus zwei Gauß-Hügeln und einem Kasten; $g$ ist dafür geschlossen bekannt (Hügel der Breite $b$ werden zu Hügeln der Breite $\sqrt{b^2 + w^2}$, der Kasten
zu einer Differenz zweier Normalverteilungsfunktionen).

**Daten.** Ehrlich: $b_i = g(t_i) + e_i$. Aus demselben Gitter (inverse crime): $b = A x_n + e$ mit $x_n = x(s)$ auf dem Gitter. In beiden Fällen
$e_i \sim N(0, \sigma^2)$ mit $\sigma = \text{noise}\cdot \mathrm{rms}(g)$.

**Singulärwertzerlegung.** $A = U\,\mathrm{diag}(\sigma_1 \ge \dots \ge \sigma_n)\,V^\top$, Kondition $\kappa = \sigma_1/\sigma_n$. Die direkte Inversion ist
$\hat x = V\,\mathrm{diag}(1/\sigma_i)\,U^\top b$.

**Erwarteter Fehler.** Für $b = A x_n + e$ ist $\hat x - x_n = A^{-1}e$ und $E\|\hat x - x_n\|^2 = \sigma^2 \sum_i 1/\sigma_i^2$. Der relative Fehler ist
$\sigma\sqrt{\sum_i \sigma_i^{-2}} / \|x_n\|$; er wächst mit der Kondition. Bei ehrlichen Daten kommt der Modellfehler $A^{-1}(g - A x_n)$ hinzu.

**Komponentenzahl.** $k^* = \#\{i : |u_i^\top g| > \sigma\}$ zählt die Komponenten der exakten Daten über dem Rauschen. Die diskrete Picard-Bedingung verlangt, dass die
Komponenten $|u_i^\top g|$ schneller fallen als $\sigma_i$; die Daten verlieren ihre Information dort, wo sie das Rauschen erreichen.

**Zwei Wahrheiten.** Ändert man $x$ um $c\,v_k$ ($v_k$ der $k$-te rechte Singulärvektor), ändern sich die Daten um $\sigma_k c\,u_k$, also um $\delta = \sigma_k c/\sigma$
Standardabweichungen. Der beste Test zum Niveau $\alpha$ hat die Macht $1 - \Phi(z_{1-\alpha} - \delta)$; bei $\delta = z_{1-\alpha}$ ist sie 50 %.

**Literatur.** Das Beispiel ist erfunden; die Begriffe (diskrete Picard-Bedingung, Singulärwertzerlegung als Analysewerkzeug schlecht gestellter Probleme) stehen bei
P. C. Hansen, *Discrete Inverse Problems: Insight and Algorithms*, SIAM 2010 (aus dem Gedächtnis, nicht geprüft). Der Begriff *inverse crime* stammt aus der
Literatur zur Tomographie (aus dem Gedächtnis, Quelle nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html)."
)

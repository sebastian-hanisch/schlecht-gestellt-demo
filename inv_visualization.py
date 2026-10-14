"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import inv_constants as C
from inv_evaluation import Analysis, de, sci

LOG_FLOOR = 1e-17                      # Untergrenze der Logarithmus-Achsen: exakte Nullen würden sonst fehlen


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Punkt": [f"{i + 1}" for i in range(k)],
        "Ort s": a.s[:k],
        "Wahrheit x": a.x[:k],
        "Messwert b": a.b[:k],
        "Direkte Inversion": a.xhat[:k],
    })


def style_table(df: pd.DataFrame):
    return df.style.format({"Ort s": lambda v: de(v, 3), "Wahrheit x": lambda v: de(v, 3), "Messwert b": lambda v: de(v, 4), "Direkte Inversion": lambda v: sci(v, 1) if abs(v) >= 1e4 else de(v, 3)})


def build_data_chart(a: Analysis) -> go.Figure:
    """Die Messung: exakte unscharfe Kurve und verrauschte Messwerte, dazu das Phantom (grau)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_GREY, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.g, mode="lines", name="unscharf, ohne Rauschen", line=dict(color=C.COLOR_HONEST, width=2.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.b, mode="markers", name="Messwerte", marker=dict(size=6, color=C.COLOR_NAIVE),
                             hovertemplate="s %{x:.3f}<br>b %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert")
    return _fixed(fig, 360)


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit und direkte Inversion; die Achse ist fest, größere Werte werden abgeschnitten (Hinweis im Titel)."""
    lo, hi = C.SHOW_RANGE
    peak = float(np.max(np.abs(a.xhat))) if np.all(np.isfinite(a.xhat)) else float("inf")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.xhat, mode="lines+markers", name="direkte Inversion", line=dict(color=C.COLOR_NAIVE, width=2), marker=dict(size=4),
                             hovertemplate="s %{x:.3f}<br>x̂ %{y:.3g}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    peak_text = "unendlich" if not np.isfinite(peak) else (de(peak, 1) if peak < 1000.0 else sci(peak))
    note = f"Die Rekonstruktion reicht bis ±{peak_text} und wird bei {de(lo, 1)} bis {de(hi, 1)} abgeschnitten." if peak > max(abs(lo), abs(hi)) else ""
    return _fixed(fig, 360, title=dict(text=note, font=dict(size=12, color="#9aa0a6"), x=0.0))


def build_picard_chart(a: Analysis) -> go.Figure:
    """Singulärwerte, Komponenten der Daten und Rauschen (logarithmisch): wo die Daten im Rauschen untergehen, endet, was sich bestimmen lässt."""
    idx = np.arange(1, a.settings.n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=np.maximum(a.sv, LOG_FLOOR), mode="lines+markers", name="Singulärwert σᵢ", line=dict(color=C.COLOR_TRUTH, width=2.5)))
    fig.add_trace(go.Scatter(x=idx, y=np.maximum(a.coef, LOG_FLOOR), mode="markers", name="Komponente |uᵢᵀ b| der Daten", marker=dict(size=7, color=C.COLOR_NAIVE)))
    if a.sigma > 0.0:
        fig.add_hline(y=a.sigma, line=dict(color="#888888", dash="dash"), annotation_text="Rauschen σ", annotation_position="top right")
        fig.add_vline(x=a.kstar + 0.5, line=dict(color=C.COLOR_CRIME, dash="dot"), annotation_text=f"k* = {a.kstar}", annotation_position="top left")
    fig.update_xaxes(title="Index i")
    fig.update_yaxes(title="Betrag", type="log", tickformat=".0e", range=[-17, 1])
    return _fixed(fig, 360)


def build_grid_chart(rows: list[dict]) -> go.Figure:
    """Fehler der direkten Inversion gegen n (logarithmisch): Daten aus demselben Gitter gegen ehrliche Daten, dazu die geschlossene Formel."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[max(r["honest"], LOG_FLOOR) for r in rows], mode="lines+markers", name="ehrliche Daten", line=dict(color=C.COLOR_HONEST, width=2.5)))
    fig.add_trace(go.Scatter(x=ns, y=[max(r["crime"], LOG_FLOOR) for r in rows], mode="lines+markers", name="Daten aus demselben Gitter", line=dict(color=C.COLOR_CRIME, width=2.5)))
    fig.add_trace(go.Scatter(x=ns, y=[r["closed"] for r in rows], mode="lines", name="geschlossene Formel (nur Rauschen)", line=dict(color="#888888", dash="dot", width=2)))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_NAIVE, dash="dash"), annotation_text="100 % der Signalnorm", annotation_position="bottom right")
    fig.update_xaxes(title="Gitterpunkte n", tickvals=ns)
    fig.update_yaxes(title="Relativer Fehler", type="log", tickformat=".0e")
    return _fixed(fig, 400)


def build_noise_chart(rows: list[dict]) -> go.Figure:
    """Kippschwelle des Rauschens (logarithmisch) gegen die Breite der Unschärfe."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["w"] for r in rows], y=[100.0 * r["threshold"] for r in rows], mode="lines+markers", name="Schwelle", line=dict(color=C.COLOR_NAIVE, width=2.5)))
    fig.update_xaxes(title="Breite der Unschärfe w")
    fig.update_yaxes(title="Rauschen, bei dem der Fehler 100 % erreicht (in %)", type="log", tickformat=".0e")
    return _fixed(fig, 340, showlegend=False)


def build_components_chart(comps: list[dict], n: int) -> go.Figure:
    labels = [de(100.0 * c["noise"], 4).rstrip("0").rstrip(",") + " %" for c in comps]
    fig = go.Figure(go.Bar(x=labels, y=[c["kstar"] for c in comps], marker_color=C.COLOR_TRUTH, text=[str(c["kstar"]) for c in comps], textposition="outside"))
    fig.add_hline(y=n, line=dict(color="#888888", dash="dot"), annotation_text=f"alle {n} Komponenten", annotation_position="top left")
    fig.update_xaxes(title="Rauschen relativ zum Effektivwert der Daten", type="category")
    fig.update_yaxes(title="Komponenten über dem Rauschen k*", range=[0, n * 1.15])
    return _fixed(fig, 320, showlegend=False)


def build_crime_chart(rows: list[dict]) -> go.Figure:
    """Ohne Rauschen: Fehler mit Daten aus demselben Gitter, mit ehrlichen Daten und der Modellfehler des Gitters (logarithmisch)."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[max(r["honest"], LOG_FLOOR) for r in rows], mode="lines+markers", name="ehrliche Daten", line=dict(color=C.COLOR_HONEST, width=2.5)))
    fig.add_trace(go.Scatter(x=ns, y=[max(r["crime"], LOG_FLOOR) for r in rows], mode="lines+markers", name="Daten aus demselben Gitter", line=dict(color=C.COLOR_CRIME, width=2.5)))
    fig.add_trace(go.Scatter(x=ns, y=[r["model_error"] for r in rows], mode="lines", name="Modellfehler des Gitters (Datenraum)", line=dict(color="#888888", dash="dot", width=2)))
    fig.update_xaxes(title="Gitterpunkte n", tickvals=ns)
    fig.update_yaxes(title="Relativer Fehler", type="log", tickformat=".0e", range=[-16.5, 1.0])
    return _fixed(fig, 400)


def build_twin_chart(res: dict) -> go.Figure:
    rows = res["rows"]
    ks = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[100.0 * r["power"] for r in rows], mode="lines+markers", name="geschlossen", line=dict(color=C.COLOR_HONEST, width=2.5)))
    fig.add_trace(go.Scatter(x=ks, y=[100.0 * r["power_sim"] for r in rows], mode="markers", name=f"simuliert ({res['runs']} Läufe)", marker=dict(size=9, color=C.COLOR_CRIME, symbol="diamond")))
    fig.add_hline(y=5, line=dict(color="#888888", dash="dot"), annotation_text="5 % (reiner Fehlalarm)", annotation_position="top left")
    fig.update_xaxes(title="Index k der geänderten Komponente", tickvals=ks)
    fig.update_yaxes(title="Macht des Tests (%)", range=[-3, 103])
    return _fixed(fig, 360)

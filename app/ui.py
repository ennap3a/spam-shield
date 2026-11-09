"""Shared helpers for the Streamlit pages: imports, colors, and chart styling.

Streamlit runs the files in app/ directly, so we add src/ to Python's path here.
That way the app works on Streamlit Cloud without `pip install -e .`.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spam_shield import config  # noqa: E402  (must come after the path setup)

# One fixed color per label, everywhere in the app ("color follows the entity").
# Light and dark steps of the same four hues.
LABEL_COLORS = {
    "light": {"ham": "#2a78d6", "spam": "#eb6834", "phishing": "#1baf7a", "toxic": "#eda100"},
    "dark": {"ham": "#3987e5", "spam": "#d95926", "phishing": "#199e70", "toxic": "#c98500"},
}
LABEL_NAMES = {
    "ham": "Normal (ham)",
    "spam": "Spam",
    "phishing": "Phishing",
    "toxic": "Toxic",
}
# A single-series color that is NOT one of the label colors.
NEUTRAL_SERIES = {"light": "#6250d6", "dark": "#9085e9"}
CHROME = {
    "light": {"text": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"text": "#f0efec", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
SEQUENTIAL_BLUE = {
    "light": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "dark": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec"],
}


def theme() -> str:
    """'light' or 'dark', following the viewer's Streamlit theme."""
    return "dark" if st.context.theme.type == "dark" else "light"


def label_color(label: str) -> str:
    return LABEL_COLORS[theme()][label]


def style_figure(fig: go.Figure, height: int = 320) -> go.Figure:
    """Recessive grid and axes, transparent background, system font."""
    chrome = CHROME[theme()]
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
              "color": chrome["text"], "size": 13},
        hoverlabel={"font_family": 'system-ui, -apple-system, "Segoe UI", sans-serif'},
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=chrome["grid"], linecolor=chrome["axis"], zeroline=False,
                     tickfont={"color": chrome["muted"]})
    fig.update_yaxes(gridcolor=chrome["grid"], linecolor=chrome["axis"], zeroline=False,
                     tickfont={"color": chrome["muted"]})
    return fig


def probability_chart(probabilities: dict[str, float]) -> go.Figure:
    """Horizontal bars, one per label, in the label's own color, value at the end."""
    labels = list(reversed(config.LABELS))          # first label on top
    values = [probabilities[label] for label in labels]
    fig = go.Figure(go.Bar(
        x=values,
        y=[LABEL_NAMES[label] for label in labels],
        orientation="h",
        marker={"color": [label_color(label) for label in labels], "cornerradius": 4},
        text=[f"{value:.0%}" for value in values],
        textposition="outside",
        textfont={"color": CHROME[theme()]["text"]},
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=True)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(bargap=0.45)
    return style_figure(fig, height=220)


@st.cache_data
def load_report(name: str):
    """Read a file from reports/ (JSON or CSV). Cached: files only change on retrain."""
    path = config.REPORTS_DIR / name
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path)


def mask_word(word: str) -> str:
    """Hide the middle of offensive words on screen: 'idiot' -> 'i***t'."""
    if len(word) <= 2:
        return word[0] + "*"
    return word[0] + "*" * (len(word) - 2) + word[-1]

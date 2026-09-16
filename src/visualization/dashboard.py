"""Shared dashboard presentation primitives.

This module deliberately contains no scientific estimators.  It renders values and
status objects supplied by the calculation, ML, or screening layers and makes
unavailable results visually explicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

CLASSIFICATION_LABELS = {
    "physics_calculated": "Physics calculated",
    "machine_learning_prediction": "Machine learning prediction",
    "engineering_screening": "Engineering screening",
    "experimental_measurement": "Experimental measurement",
    "unavailable": "Unavailable",
}

CLASSIFICATION_COLORS = {
    "physics_calculated": "#18A999",
    "machine_learning_prediction": "#7C6CF2",
    "engineering_screening": "#F59E0B",
    "experimental_measurement": "#0EA5E9",
    "unavailable": "#78849A",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#D9E2F2"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(13,22,38,0.72)",
        "colorway": ["#18A999", "#7C6CF2", "#F59E0B", "#EF5DA8", "#37B6FF", "#FF6B6B"],
        "xaxis": {
            "gridcolor": "rgba(148,163,184,0.14)",
            "zerolinecolor": "rgba(148,163,184,0.3)",
        },
        "yaxis": {
            "gridcolor": "rgba(148,163,184,0.14)",
            "zerolinecolor": "rgba(148,163,184,0.3)",
        },
        "legend": {"bgcolor": "rgba(13,22,38,0.65)"},
        "hoverlabel": {"bgcolor": "#14233A", "font_color": "#FFFFFF"},
        "margin": {"l": 55, "r": 28, "t": 70, "b": 55},
    }
}


STREAMLIT_CSS = """
<style>
    :root {
        --dt-bg: #07111f;
        --dt-panel: rgba(15, 28, 48, 0.86);
        --dt-line: rgba(148, 163, 184, 0.18);
        --dt-text: #e7eef9;
        --dt-muted: #9cabbe;
        --dt-teal: #18a999;
        --dt-violet: #7c6cf2;
        --dt-amber: #f59e0b;
    }
    .stApp {
        background:
          radial-gradient(circle at 85% 0%, rgba(51, 73, 135, .22), transparent 32rem),
          radial-gradient(circle at 12% 24%, rgba(24, 169, 153, .12), transparent 30rem),
          var(--dt-bg);
        color: var(--dt-text);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .65); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(12, 25, 43, .98), rgba(7, 17, 31, .98));
        border-right: 1px solid var(--dt-line);
    }
    .block-container { max-width: 1480px; padding-top: 1.3rem; padding-bottom: 4rem; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    h1 { font-weight: 720; }
    [data-testid="stMetric"] {
        background: var(--dt-panel);
        border: 1px solid var(--dt-line);
        border-radius: 14px;
        padding: .9rem 1rem;
        box-shadow: 0 9px 30px rgba(0,0,0,.14);
    }
    [data-testid="stMetricLabel"] { color: var(--dt-muted); }
    .dt-hero {
        background: linear-gradient(120deg, rgba(23, 47, 77, .94), rgba(23, 35, 67, .88));
        border: 1px solid rgba(109, 142, 181, .28);
        border-radius: 20px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 50px rgba(0,0,0,.22);
    }
    .dt-kicker { color: #72ded1; font-size: .78rem; font-weight: 750; letter-spacing: .14em; }
    .dt-hero h1 { margin: .2rem 0 .2rem; font-size: clamp(2rem, 4vw, 3.35rem); }
    .dt-hero p { color: #adbbcc; max-width: 860px; margin: 0; }
    .dt-card {
        background: var(--dt-panel);
        border: 1px solid var(--dt-line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: .45rem 0 .8rem;
    }
    .dt-card-title { font-size: .92rem; font-weight: 720; margin-bottom: .25rem; }
    .dt-muted { color: var(--dt-muted); }
    .dt-badge {
        display: inline-flex; align-items: center; gap: .35rem;
        border-radius: 999px; padding: .22rem .58rem; margin: .12rem .2rem .12rem 0;
        font-size: .72rem; font-weight: 700; border: 1px solid currentColor;
    }
    .dt-badge.physics_calculated { color: #62dfd0; background: rgba(24,169,153,.10); }
    .dt-badge.machine_learning_prediction { color: #aaa0ff; background: rgba(124,108,242,.10); }
    .dt-badge.engineering_screening { color: #ffc45c; background: rgba(245,158,11,.10); }
    .dt-badge.experimental_measurement { color: #75d0ff; background: rgba(14,165,233,.10); }
    .dt-badge.unavailable { color: #a7b1c2; background: rgba(120,132,154,.10); }
    .dt-limit {
        border-left: 3px solid #78849a; padding: .52rem .72rem; margin-top: .55rem;
        background: rgba(120,132,154,.08); color: #aeb9c8; font-size: .82rem;
    }
    .dt-section-rule { height: 1px; background: var(--dt-line); margin: .3rem 0 1rem; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px; min-height: 2.7rem; border: 1px solid rgba(93,220,207,.42);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(105deg, #128d82, #4c64d9);
        color: white; font-weight: 720; border: 0;
    }
    div[data-baseweb="tab-list"] { gap: .35rem; }
    button[data-baseweb="tab"] { border-radius: 9px; padding-inline: .85rem; }
</style>
"""


def apply_figure_style(fig: go.Figure, *, title: str | None = None, height: int = 440) -> go.Figure:
    """Apply the product-wide Plotly style without changing plotted data."""

    fig.update_layout(template=PLOTLY_TEMPLATE, height=height)
    if title:
        fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"})
    return fig


def unavailable_figure(title: str, reason: str, *, height: int = 390) -> go.Figure:
    """Return a truthful placeholder when a result cannot be computed."""

    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.54,
        xref="paper",
        yref="paper",
        text="Result unavailable",
        showarrow=False,
        font={"size": 20, "color": "#AEB9C8"},
    )
    fig.add_annotation(
        x=0.5,
        y=0.42,
        xref="paper",
        yref="paper",
        text=reason,
        showarrow=False,
        align="center",
        font={"size": 12, "color": "#78849A"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_figure_style(fig, title=title, height=height)


def classification_badge(classification: str, confidence: str | float | None = None) -> str:
    """Generate safe badge HTML for a result classification and confidence."""

    from html import escape

    key = classification if classification in CLASSIFICATION_LABELS else "unavailable"
    label = CLASSIFICATION_LABELS[key]
    confidence_text = ""
    if confidence is not None:
        if isinstance(confidence, (int, float)):
            confidence_text = f" · confidence {float(confidence):.0%}"
        else:
            confidence_text = f" · confidence {escape(str(confidence))}"
    return f'<span class="dt-badge {key}">{escape(label)}{confidence_text}</span>'


def result_card_html(
    title: str,
    value: Any,
    *,
    classification: str,
    unit: str = "",
    confidence: str | float | None = None,
    provenance: str | None = None,
    limitations: Sequence[str] | str | None = None,
) -> str:
    """Render a compact result card that cannot omit result provenance."""

    from html import escape

    limitations_list = [limitations] if isinstance(limitations, str) else list(limitations or [])
    limitation_html = ""
    if limitations_list:
        joined = "<br>".join(escape(str(item)) for item in limitations_list)
        limitation_html = f'<div class="dt-limit"><b>Limitations:</b> {joined}</div>'
    provenance_html = (
        f'<div class="dt-muted" style="font-size:.78rem;margin-top:.4rem">Source: {escape(provenance)}</div>'
        if provenance
        else ""
    )
    return (
        '<div class="dt-card">'
        f'<div class="dt-card-title">{escape(title)}</div>'
        f'<div style="font-size:1.55rem;font-weight:760">{escape(str(value))}'
        f' <span style="font-size:.8rem;color:#9cabbe">{escape(unit)}</span></div>'
        f'{classification_badge(classification, confidence)}'
        f'{provenance_html}{limitation_html}</div>'
    )


def am_suitability_radar(
    criteria: Mapping[str, float] | None,
    *,
    title: str = "LPBF suitability profile",
    scale_max: float = 10.0,
) -> go.Figure:
    """Plot supplied engineering-screening criteria; never derive a score here."""

    if not criteria:
        return unavailable_figure(title, "The AM screening engine did not return criterion scores.")
    names = [str(k) for k in criteria]
    values = [float(criteria[k]) for k in criteria]
    if not names or any(value < 0 or value > scale_max for value in values):
        return unavailable_figure(title, f"Criterion scores must lie between 0 and {scale_max:g}.")
    closed_names = names + [names[0]]
    closed_values = values + [values[0]]
    fig = go.Figure(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_names,
            fill="toself",
            fillcolor="rgba(24,169,153,.22)",
            line={"color": "#54D6C8", "width": 3},
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            name="Screening score",
        )
    )
    fig.update_layout(
        polar={
            "bgcolor": "rgba(13,22,38,.72)",
            "radialaxis": {
                "range": [0, scale_max],
                "gridcolor": "rgba(148,163,184,.20)",
                "tickfont": {"color": "#9CABBE"},
            },
            "angularaxis": {"gridcolor": "rgba(148,163,184,.20)"},
        },
        showlegend=False,
    )
    return apply_figure_style(fig, title=title, height=460)

"""Phase and composition visualizations with strict input validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from .dashboard import apply_figure_style, unavailable_figure


def _composition_components(composition: Mapping[str, float]) -> tuple[float, float, float]:
    try:
        cu, al, ni = (float(composition[element]) for element in ("Cu", "Al", "Ni"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Composition must contain numeric Cu, Al, and Ni values.") from exc
    if min(cu, al, ni) < 0:
        raise ValueError("Composition values cannot be negative.")
    total = cu + al + ni
    if total <= 0:
        raise ValueError("Composition total must be positive.")
    return 100.0 * cu / total, 100.0 * al / total, 100.0 * ni / total


def ternary_composition_plot(
    composition: Mapping[str, float] | None,
    *,
    basis: str = "wt%",
    reference_points: Sequence[Mapping[str, Any]] | None = None,
    title: str = "Cu-Al-Ni composition space",
) -> go.Figure:
    """Place the current alloy on an interactive Cu-Al-Ni ternary diagram."""

    if composition is None:
        return unavailable_figure(title, "Enter a valid composition to locate the alloy.")
    try:
        cu, al, ni = _composition_components(composition)
    except ValueError as exc:
        return unavailable_figure(title, str(exc))

    fig = go.Figure()
    if reference_points:
        normalized = []
        for item in reference_points:
            try:
                rcu, ral, rni = _composition_components(item)
            except ValueError:
                continue
            normalized.append((rcu, ral, rni, str(item.get("label", "Reference"))))
        if normalized:
            fig.add_trace(
                go.Scatterternary(
                    a=[row[0] for row in normalized],
                    b=[row[1] for row in normalized],
                    c=[row[2] for row in normalized],
                    text=[row[3] for row in normalized],
                    mode="markers",
                    marker={"size": 7, "color": "rgba(174,185,200,.55)"},
                    hovertemplate="%{text}<br>Cu %{a:.2f}<br>Al %{b:.2f}<br>Ni %{c:.2f}<extra></extra>",
                    name="Reference data",
                )
            )
    fig.add_trace(
        go.Scatterternary(
            a=[cu],
            b=[al],
            c=[ni],
            text=["Current alloy"],
            mode="markers+text",
            textposition="top center",
            marker={"size": 15, "color": "#F59E0B", "line": {"width": 2, "color": "#FFE2A3"}},
            hovertemplate=(
                f"Current alloy ({basis})<br>Cu %{{a:.3f}}<br>Al %{{b:.3f}}<br>Ni %{{c:.3f}}<extra></extra>"
            ),
            name="Current alloy",
        )
    )
    fig.update_layout(
        ternary={
            "sum": 100,
            "bgcolor": "rgba(13,22,38,.72)",
            "aaxis": {"title": f"Cu ({basis})", "min": 0, "linewidth": 1, "gridcolor": "rgba(148,163,184,.18)"},
            "baxis": {"title": f"Al ({basis})", "min": 0, "linewidth": 1, "gridcolor": "rgba(148,163,184,.18)"},
            "caxis": {"title": f"Ni ({basis})", "min": 0, "linewidth": 1, "gridcolor": "rgba(148,163,184,.18)"},
        }
    )
    return apply_figure_style(fig, title=title, height=530)


def phase_fraction_temperature(
    temperatures: Sequence[float] | None,
    phase_fractions: Mapping[str, Sequence[float]] | None,
    *,
    temperature_unit: str = "K",
    title: str = "Equilibrium phase fractions vs temperature",
) -> go.Figure:
    """Plot calculated phase fractions and expose invalid solver output."""

    if temperatures is None or not phase_fractions:
        return unavailable_figure(title, "A thermodynamic database and successful equilibrium calculation are required.")
    temp = np.asarray(temperatures, dtype=float)
    if temp.ndim != 1 or temp.size < 2 or not np.all(np.isfinite(temp)):
        return unavailable_figure(title, "Thermodynamic output contains an invalid temperature grid.")
    fig = go.Figure()
    matrix: list[np.ndarray] = []
    for phase, values in phase_fractions.items():
        fraction = np.asarray(values, dtype=float)
        if fraction.shape != temp.shape or not np.all(np.isfinite(fraction)):
            return unavailable_figure(title, f"Phase {phase} has invalid fraction data.")
        matrix.append(fraction)
        fig.add_trace(
            go.Scatter(
                x=temp,
                y=fraction,
                mode="lines",
                stackgroup="one",
                name=str(phase),
                hovertemplate=f"{phase}<br>T %{{x:.1f}} {temperature_unit}<br>fraction %{{y:.4f}}<extra></extra>",
            )
        )
    totals = np.sum(np.vstack(matrix), axis=0)
    conservation_error = float(np.nanmax(np.abs(totals - 1.0)))
    fig.update_xaxes(title=f"Temperature ({temperature_unit})")
    fig.update_yaxes(title="Phase fraction", range=[0, max(1.02, float(np.nanmax(totals)) * 1.03)])
    if conservation_error > 0.02:
        fig.add_annotation(
            x=0.5,
            y=1.07,
            xref="paper",
            yref="paper",
            text=f"Solver output warning: max phase-sum error = {conservation_error:.3f}",
            showarrow=False,
            font={"color": "#FFB86B", "size": 11},
        )
    return apply_figure_style(fig, title=title, height=470)


def phase_stability_map(
    x: Sequence[float] | None,
    y: Sequence[float] | None,
    z: Sequence[Sequence[float]] | None,
    *,
    x_label: str = "Al (wt%)",
    y_label: str = "Temperature (K)",
    value_label: str = "Stability metric",
    title: str = "Phase stability map",
) -> go.Figure:
    """Render a solver-supplied stability field as an interactive heat map."""

    if x is None or y is None or z is None:
        return unavailable_figure(title, "No computed phase-stability grid is available.")
    x_array, y_array, z_array = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if z_array.shape != (y_array.size, x_array.size) or z_array.size == 0:
        return unavailable_figure(title, "The phase-stability grid dimensions are inconsistent.")
    fig = go.Figure(
        go.Heatmap(
            x=x_array,
            y=y_array,
            z=z_array,
            colorscale="Viridis",
            colorbar={"title": value_label},
            hovertemplate=f"{x_label}: %{{x:.3g}}<br>{y_label}: %{{y:.3g}}<br>{value_label}: %{{z:.4g}}<extra></extra>",
        )
    )
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    return apply_figure_style(fig, title=title, height=470)


def ternary_phase_assemblage_map(
    points: Sequence[Mapping[str, Any]] | None,
    *,
    title: str = "Calculated isothermal phase-assemblage map",
) -> go.Figure:
    """Plot categorical pycalphad grid samples in ternary mole-fraction space."""

    if not points:
        return unavailable_figure(
            title,
            "Enable the optional ternary CALPHAD grid and supply a compatible database.",
        )
    valid: list[tuple[float, float, float, str, object]] = []
    for item in points:
        try:
            cu = 100.0 * float(item["x_Cu"])
            al = 100.0 * float(item["x_Al"])
            ni = 100.0 * float(item["x_Ni"])
            assemblage = str(item["phase_assemblage"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite([cu, al, ni])):
            valid.append((cu, al, ni, assemblage, item.get("phase_fractions", {})))
    if not valid:
        return unavailable_figure(title, "The calculated phase grid contains no valid points.")
    categories = sorted({row[3] for row in valid})
    palette = [
        "#54D6C8", "#37B6FF", "#A99FFF", "#F59E0B", "#EF7D32",
        "#E879B7", "#8DD17E", "#9AA8BD", "#FFD166", "#4CC9F0",
    ]
    fig = go.Figure()
    for index, category in enumerate(categories):
        subset = [row for row in valid if row[3] == category]
        fig.add_trace(
            go.Scatterternary(
                a=[row[0] for row in subset],
                b=[row[1] for row in subset],
                c=[row[2] for row in subset],
                mode="markers",
                marker={"size": 7, "color": palette[index % len(palette)]},
                customdata=[str(row[4]) for row in subset],
                hovertemplate=(
                    f"{category}<br>x(Cu) %{{a:.2f}}%<br>x(Al) %{{b:.2f}}%<br>x(Ni) %{{c:.2f}}%"
                    "<br>fractions %{customdata}<extra></extra>"
                ),
                name=category,
            )
        )
    fig.update_layout(
        ternary={
            "sum": 100,
            "bgcolor": "rgba(13,22,38,.72)",
            "aaxis": {"title": "Cu (mol%)", "gridcolor": "rgba(148,163,184,.18)"},
            "baxis": {"title": "Al (mol%)", "gridcolor": "rgba(148,163,184,.18)"},
            "caxis": {"title": "Ni (mol%)", "gridcolor": "rgba(148,163,184,.18)"},
        },
        legend={"orientation": "h", "y": -0.15},
    )
    return apply_figure_style(fig, title=title, height=560)


def composition_property_map(
    compositions: Sequence[Mapping[str, float]] | None,
    properties: Sequence[float] | None,
    *,
    property_name: str,
    property_unit: str = "",
    title: str | None = None,
) -> go.Figure:
    """Plot validated model or experimental property data in ternary space."""

    display_title = title or f"Composition-property map: {property_name}"
    if not compositions or properties is None or len(compositions) != len(properties):
        return unavailable_figure(display_title, "No validated composition-property grid is available.")
    points: list[tuple[float, float, float, float]] = []
    for composition, value in zip(compositions, properties, strict=True):
        try:
            cu, al, ni = _composition_components(composition)
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            points.append((cu, al, ni, numeric_value))
    if not points:
        return unavailable_figure(display_title, "The supplied property grid contains no finite values.")
    suffix = f" {property_unit}" if property_unit else ""
    fig = go.Figure(
        go.Scatterternary(
            a=[row[0] for row in points],
            b=[row[1] for row in points],
            c=[row[2] for row in points],
            mode="markers",
            marker={
                "size": 9,
                "color": [row[3] for row in points],
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": f"{property_name}{suffix}"},
            },
            customdata=[row[3] for row in points],
            hovertemplate=(
                "Cu %{a:.3f}<br>Al %{b:.3f}<br>Ni %{c:.3f}"
                f"<br>{property_name} %{{customdata:.4g}}{suffix}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        ternary={
            "sum": 100,
            "bgcolor": "rgba(13,22,38,.72)",
            "aaxis": {"title": "Cu", "gridcolor": "rgba(148,163,184,.18)"},
            "baxis": {"title": "Al", "gridcolor": "rgba(148,163,184,.18)"},
            "caxis": {"title": "Ni", "gridcolor": "rgba(148,163,184,.18)"},
        }
    )
    return apply_figure_style(fig, title=display_title, height=530)

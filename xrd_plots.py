"""Interactive diffraction and predictive uncertainty visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from .dashboard import apply_figure_style, unavailable_figure


def xrd_pattern_plot(
    two_theta: Sequence[float] | None,
    intensity: Sequence[float] | None,
    *,
    peaks: Sequence[Mapping[str, object]] | None = None,
    experimental_two_theta: Sequence[float] | None = None,
    experimental_intensity: Sequence[float] | None = None,
    wavelength_angstrom: float | None = None,
    title: str = "Simulated and experimental XRD",
) -> go.Figure:
    """Plot supplied diffraction data and optional indexed peak annotations."""

    if (two_theta is None or intensity is None) and peaks:
        stick_x: list[float] = []
        stick_y: list[float] = []
        for peak in peaks:
            try:
                angle_value = float(peak.get("two_theta_deg", peak.get("two_theta")))
                intensity_value = float(
                    peak.get(
                        "scaled_intensity",
                        peak.get("intensity", peak.get("relative_intensity")),
                    )
                )
            except (TypeError, ValueError):
                continue
            stick_x.extend([angle_value - 0.015, angle_value, angle_value + 0.015])
            stick_y.extend([0.0, intensity_value, 0.0])
        if len(stick_x) >= 2:
            two_theta, intensity = stick_x, stick_y
    if two_theta is None or intensity is None:
        return unavailable_figure(title, "Select a crystal model with validated lattice and atomic positions.")
    angle, signal = np.asarray(two_theta, float), np.asarray(intensity, float)
    if angle.shape != signal.shape or angle.ndim != 1 or angle.size < 2:
        return unavailable_figure(title, "The simulated diffraction arrays are inconsistent.")
    finite = np.isfinite(angle) & np.isfinite(signal)
    if finite.sum() < 2:
        return unavailable_figure(title, "The simulated pattern contains insufficient finite data.")
    fig = go.Figure(
        go.Scatter(
            x=angle[finite],
            y=signal[finite],
            mode="lines",
            line={"color": "#54D6C8", "width": 2},
            name="Simulated",
            hovertemplate="2θ %{x:.3f}°<br>Relative intensity %{y:.2f}<extra></extra>",
        )
    )
    if experimental_two_theta is not None and experimental_intensity is not None:
        exp_angle = np.asarray(experimental_two_theta, float)
        exp_signal = np.asarray(experimental_intensity, float)
        if exp_angle.shape == exp_signal.shape and exp_angle.ndim == 1:
            valid = np.isfinite(exp_angle) & np.isfinite(exp_signal)
            fig.add_trace(
                go.Scatter(
                    x=exp_angle[valid],
                    y=exp_signal[valid],
                    mode="lines",
                    line={"color": "#A99FFF", "width": 1.3},
                    opacity=0.82,
                    name="Experimental",
                    hovertemplate="2θ %{x:.3f}°<br>Intensity %{y:.2f}<extra></extra>",
                )
            )
    for peak in list(peaks or [])[:35]:
        try:
            peak_angle = float(peak.get("two_theta_deg", peak.get("two_theta")))
        except (KeyError, TypeError, ValueError):
            continue
        label_parts = [str(peak.get("phase", "")), str(peak.get("hkl", ""))]
        label = " ".join(part for part in label_parts if part).strip()
        fig.add_vline(x=peak_angle, line_width=0.7, line_color="rgba(245,158,11,.55)")
        if label:
            fig.add_annotation(
                x=peak_angle,
                y=1.01,
                xref="x",
                yref="paper",
                text=label,
                textangle=-90,
                showarrow=False,
                font={"size": 9, "color": "#EAB868"},
                yanchor="bottom",
            )
    wavelength_note = f" · λ = {wavelength_angstrom:.5g} Å" if wavelength_angstrom else ""
    fig.update_xaxes(title=f"2θ (degrees){wavelength_note}")
    fig.update_yaxes(title="Intensity (a.u.)", rangemode="tozero")
    return apply_figure_style(fig, title=title, height=500)


def transformation_temperature_chart(
    predictions: Mapping[str, float] | None,
    uncertainties: Mapping[str, float] | None = None,
    *,
    experimental: Mapping[str, float] | None = None,
    title: str = "Martensitic transformation temperatures",
) -> go.Figure:
    """Visualize ML predictions, uncertainty, and optional experimental values."""

    order = ["Ms", "Mf", "As", "Af"]
    available_order = [key for key in order if predictions and key in predictions]
    if not available_order:
        return unavailable_figure(title, "A trained and validated transformation model is required.")
    values = [float(predictions[key]) for key in available_order]
    if not np.all(np.isfinite(values)):
        return unavailable_figure(title, "The prediction contains non-finite values.")
    errors = [float((uncertainties or {}).get(key, 0.0)) for key in available_order]
    fig = go.Figure(
        go.Bar(
            x=available_order,
            y=values,
            error_y={"type": "data", "array": errors, "visible": any(error > 0 for error in errors)},
            marker={
                "color": [
                    {"Ms": "#37B6FF", "Mf": "#3478D4", "As": "#F59E0B", "Af": "#EF7D32"}[key]
                    for key in available_order
                ]
            },
            name="ML ensemble",
            hovertemplate="%{x}: %{y:.1f} °C<extra>ML ensemble</extra>",
        )
    )
    if experimental:
        observed_keys = [key for key in order if key in experimental]
        fig.add_trace(
            go.Scatter(
                x=observed_keys,
                y=[float(experimental[key]) for key in observed_keys],
                mode="markers",
                marker={"symbol": "diamond", "size": 12, "color": "#FFFFFF", "line": {"color": "#07111F", "width": 1}},
                name="Experimental",
                hovertemplate="%{x}: %{y:.1f} °C<extra>Experimental</extra>",
            )
        )
    fig.update_yaxes(title="Temperature (°C)")
    fig.update_xaxes(title="Transformation event")
    return apply_figure_style(fig, title=title, height=440)


def uncertainty_plot(
    predictions: Mapping[str, float] | None,
    lower: Mapping[str, float] | None,
    upper: Mapping[str, float] | None,
    *,
    title: str = "Predictive uncertainty intervals",
) -> go.Figure:
    """Plot asymmetric uncertainty intervals supplied by the ML layer."""

    if not predictions or not lower or not upper:
        return unavailable_figure(title, "The selected model did not provide calibrated prediction intervals.")
    preferred_order = ("Ms", "Mf", "As", "Af", "Hardness", "Yield strength", "UTS")
    labels = [label for label in preferred_order if label in predictions]
    labels.extend(str(label) for label in predictions if str(label) not in labels)
    values, plus, minus = [], [], []
    for label in labels:
        try:
            center = float(predictions[label])
            low = float(lower[label])
            high = float(upper[label])
        except (KeyError, TypeError, ValueError):
            return unavailable_figure(title, f"Missing calibrated interval for {label}.")
        values.append(center)
        plus.append(max(0.0, high - center))
        minus.append(max(0.0, center - low))
    fig = go.Figure(
        go.Scatter(
            x=values,
            y=labels,
            mode="markers",
            marker={"size": 11, "color": "#A99FFF"},
            error_x={"type": "data", "array": plus, "arrayminus": minus, "visible": True, "color": "#7C6CF2"},
            hovertemplate="%{y}: %{x:.3g}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted value (target-specific units)")
    fig.update_yaxes(title="Target")
    return apply_figure_style(fig, title=title, height=420)

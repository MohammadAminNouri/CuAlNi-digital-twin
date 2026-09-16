"""Plotly crystal structure viewer driven by explicit crystallographic data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from .dashboard import apply_figure_style, unavailable_figure

ELEMENT_COLORS = {"Cu": "#D98247", "Al": "#A8B0BD", "Ni": "#5BBF72"}
ELEMENT_SIZES = {"Cu": 16, "Al": 14, "Ni": 14}


def _cell_edges(lattice: np.ndarray) -> tuple[list[float | None], list[float | None], list[float | None]]:
    origins = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=float,
    )
    cart = origins @ lattice
    pairs = ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7))
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for start, end in pairs:
        xs.extend([float(cart[start, 0]), float(cart[end, 0]), None])
        ys.extend([float(cart[start, 1]), float(cart[end, 1]), None])
        zs.extend([float(cart[start, 2]), float(cart[end, 2]), None])
    return xs, ys, zs


def _structure_fields(
    value: object,
) -> tuple[object, object, str | None]:
    """Extract lattice/sites from canonical records and pymatgen serializations."""

    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        return value, None, None
    nested = value.get("structure")
    structure = nested if isinstance(nested, Mapping) else value
    lattice = structure.get("lattice", structure.get("lattice_matrix"))
    if isinstance(lattice, Mapping):
        lattice = lattice.get("matrix")
    sites = structure.get("sites")
    space_group = value.get("space_group", structure.get("space_group"))
    return lattice, sites, str(space_group) if space_group else None


def _site_species(site: Mapping[str, object]) -> list[tuple[str, float]]:
    """Normalize canonical and pymatgen site-species encodings."""

    explicit = site.get("element")
    if explicit is not None:
        try:
            occupancy = float(site.get("occupancy", site.get("occu", 1.0)))
        except (TypeError, ValueError):
            occupancy = 1.0
        return [(str(explicit), occupancy)]
    raw = site.get("species")
    if isinstance(raw, Mapping):
        entries: list[tuple[str, float]] = []
        for element, occupancy in raw.items():
            try:
                entries.append((str(element), float(occupancy)))
            except (TypeError, ValueError):
                continue
        return entries
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            element = item.get("element", item.get("name"))
            if element is None:
                continue
            try:
                occupancy = float(item.get("occupancy", item.get("occu", 1.0)))
            except (TypeError, ValueError):
                occupancy = 1.0
            entries.append((str(element), occupancy))
        return entries
    if raw is not None:
        return [(str(raw), 1.0)]
    return []


def crystal_structure_figure(
    lattice_matrix: Sequence[Sequence[float]] | Mapping[str, object] | object | None,
    sites: Sequence[Mapping[str, object]] | None = None,
    *,
    phase_name: str = "Selected phase",
    space_group: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Render a unit cell from explicit data or a serialized structure record."""

    display_title = title or f"Crystal structure: {phase_name}"
    if sites is None and lattice_matrix is not None:
        extracted_lattice, extracted_sites, extracted_space_group = _structure_fields(lattice_matrix)
        lattice_matrix = extracted_lattice
        if isinstance(extracted_sites, Sequence) and not isinstance(extracted_sites, (str, bytes)):
            sites = extracted_sites
        if space_group is None:
            space_group = extracted_space_group
    if lattice_matrix is None or not sites:
        return unavailable_figure(display_title, "Validated lattice vectors and atomic positions are required.", height=540)
    lattice = np.asarray(lattice_matrix, dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)) or abs(np.linalg.det(lattice)) < 1e-10:
        return unavailable_figure(display_title, "The lattice matrix is invalid or singular.", height=540)
    species: list[str] = []
    fractional: list[list[float]] = []
    occupancies: list[float] = []
    for site in sites:
        coords = site.get("fractional_coordinates", site.get("frac_coords"))
        if coords is None:
            continue
        array = np.asarray(coords, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            continue
        for element, occupancy in _site_species(site):
            if not np.isfinite(occupancy) or occupancy <= 0:
                continue
            species.append(element)
            fractional.append(array.tolist())
            occupancies.append(occupancy)
    if not fractional:
        return unavailable_figure(display_title, "No valid atomic positions were supplied.", height=540)
    frac_array = np.asarray(fractional)
    cart = frac_array @ lattice
    fig = go.Figure()
    x_edge, y_edge, z_edge = _cell_edges(lattice)
    fig.add_trace(
        go.Scatter3d(
            x=x_edge,
            y=y_edge,
            z=z_edge,
            mode="lines",
            line={"color": "rgba(197,210,228,.72)", "width": 4},
            hoverinfo="skip",
            name="Unit cell",
        )
    )
    for element in sorted(set(species)):
        indexes = [index for index, item in enumerate(species) if item == element]
        hover_text = [
            f"{element}<br>fractional ({frac_array[index,0]:.4f}, {frac_array[index,1]:.4f}, {frac_array[index,2]:.4f})"
            f"<br>occupancy {occupancies[index]:.3f}"
            for index in indexes
        ]
        fig.add_trace(
            go.Scatter3d(
                x=cart[indexes, 0],
                y=cart[indexes, 1],
                z=cart[indexes, 2],
                mode="markers",
                marker={
                    "size": [
                        ELEMENT_SIZES.get(element, 13)
                        * max(0.35, min(1.0, occupancies[index])) ** (1.0 / 3.0)
                        for index in indexes
                    ],
                    "color": ELEMENT_COLORS.get(element, "#9AA8BD"),
                    "opacity": 0.92,
                    "line": {"width": 1, "color": "rgba(255,255,255,.7)"},
                },
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name=element,
            )
        )
    subtitle = f" · space group {space_group}" if space_group else ""
    fig.update_layout(
        scene={
            "xaxis": {"title": "x (Å)", "backgroundcolor": "rgba(13,22,38,.72)", "gridcolor": "rgba(148,163,184,.12)"},
            "yaxis": {"title": "y (Å)", "backgroundcolor": "rgba(13,22,38,.72)", "gridcolor": "rgba(148,163,184,.12)"},
            "zaxis": {"title": "z (Å)", "backgroundcolor": "rgba(13,22,38,.72)", "gridcolor": "rgba(148,163,184,.12)"},
            "aspectmode": "data",
        }
    )
    return apply_figure_style(fig, title=f"{display_title}{subtitle}", height=560)

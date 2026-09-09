"""Print optional capability availability without modifying the environment."""

from __future__ import annotations

import importlib.util
import platform

CAPABILITIES = {
    "dashboard": ("streamlit", "plotly"),
    "core numerics / characterization / ML": ("numpy", "pandas", "scipy", "sklearn"),
    "CALPHAD equilibrium": ("pycalphad",),
    "Scheil-Gulliver": ("scheil",),
    "CIF/XRD": ("pymatgen",),
    "symmetry": ("spglib",),
    "optional boosted models": ("xgboost", "lightgbm"),
    "SHAP explanation": ("shap",),
    "reports": ("jinja2", "reportlab"),
}


def main() -> None:
    print(f"Python {platform.python_version()} on {platform.platform()}")
    for capability, modules in CAPABILITIES.items():
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        state = "available" if not missing else "unavailable: missing " + ", ".join(missing)
        print(f"{capability}: {state}")


if __name__ == "__main__":
    main()


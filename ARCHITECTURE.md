# Architecture

```text
Streamlit app.py
    │
    ├── AnalysisInputs (immutable run configuration and uploaded bytes)
    │
    └── application.analysis_service
          ├── composition ───────────── deterministic conversion/descriptors
          ├── thermodynamics ───────── external-TDB equilibrium/Scheil/grid
          ├── crystallography ──────── CIF, spglib, pymatgen XRD
          ├── machine_learning ─────── provenance gate → CV models → UQ/explanation
          ├── characterization ─────── uploaded XRD/EDS/DSC/SEM
          └── additive_manufacturing ─ user-profile screening
                    │
                    └── result envelope
                          ├── interactive Plotly dashboard
                          └── HTML / PDF / JSON report generator
```

The orchestration layer catches module-local failures and converts them to an unavailable result instead of terminating unrelated calculations. Scientific engines do not import Streamlit, making them testable and reusable from notebooks or scripts.

Optional dependencies are imported only at execution points. The core app can launch without pycalphad, scheil, pymatgen, spglib, XGBoost, LightGBM, or SHAP; the corresponding capability reports why it is unavailable. `requirements.txt` installs the complete runtime while `requirements-core.txt` supports a lightweight fail-closed deployment.

Uploaded bytes are written to a closed temporary file only for libraries that require a filesystem path. The path is removed on context exit. TDB, dataset, and CIF hashes are preserved where applicable. No cloud upload path is present in the codebase.

The report generator receives a completed result envelope and figures. It performs no scientific inference. HTML embeds one Plotly runtime and all figures for offline viewing; PDF is a paginated textual audit record; JSON preserves the full machine-readable result.


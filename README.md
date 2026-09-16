# CuAlNi-DigitalTwin Pro

CuAlNi-DigitalTwin Pro is a local, provenance-aware Streamlit platform for Cu-Al-Ni shape-memory-alloy design, crystallographic analysis, experimental comparison, empirical property modelling, and additive-manufacturing screening.

The default composition is **Cu-14.3Al-4.3Ni wt%** (Cu balance = 81.4 wt%), but every calculation accepts any closed Cu-Al-Ni composition.

> Scientific integrity rule: the application never fills a missing database, structure, measurement, model, or calibration with a plausible numerical surrogate. Every result is labelled as physics calculated, machine-learning prediction, engineering screening, experimental measurement, or unavailable, and is accompanied by confidence, provenance, and limitations.

## What works

- Deterministic wt% ↔ at%/mole-fraction conversion with documented atomic weights.
- Explicit group-number VEC, ideal mixture density screen, and optionally cited Hume-Rothery e/a conventions.
- Database-gated pycalphad equilibrium, phase fractions versus temperature, optional ternary equilibrium grid, and optional Scheil-Gulliver solidification.
- CIF-driven 3D structures, spglib symmetry, Bragg geometry, pymatgen structure-factor powder XRD, and indexed peak tables.
- Experimental XRD peak extraction/matching, EDS deviation and segregation statistics, DSC operational transformation temperatures, and SEM preprocessing/segmentation.
- Provenance-gated ensembles for Ms, Mf, As, Af, hardness, yield strength, and UTS using linear, Ridge, SVR, random forest, Gaussian process, XGBoost, and LightGBM estimators when installed.
- Leakage-resistant k-fold/grouped validation, MAE/RMSE/R², conformal residual intervals, applicability flags, and SHAP or explicitly labelled permutation importance.
- User-calibrated LPBF screening with transparent metric bounds, weights, coverage, and no implied build-success probability.
- Interactive Plotly dashboard plus self-contained HTML, PDF, and JSON reports.

## Quick start

Python 3.11 or 3.12 is recommended.

For cloud setup, troubleshooting, and GitHub Actions, see [Deployment](docs/DEPLOYMENT.md).
Keep the repository folder structure intact: `app.py` imports packages from `src/`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. All uploaded files are processed in the current session; the project contains no remote telemetry code.

For a smaller installation without CALPHAD, crystal, or optional boosted-tree packages, install `requirements-core.txt`. Dependent panels remain explicitly unavailable.

## First analysis

1. Keep the preloaded 81.4 Cu / 14.3 Al / 4.3 Ni wt% values or enter another composition totaling exactly 100 wt%.
2. Optionally upload a Cu-Al-Ni TDB you are licensed to use and enter its assessment citation.
3. Upload complete CIF files to enable 3D structures and structure-factor XRD.
4. Upload a real experimental ML table together with provenance JSON to enable empirical predictions.
5. Add experimental XRD/EDS/DSC/SEM files and, if required, a sourced AM screening profile.
6. Select **RUN COMPLETE ANALYSIS**. Independent modules still run when another module lacks input.

The audited composition-only output for the primary alloy is in [`examples/primary_alloy_composition_only.json`](examples/primary_alloy_composition_only.json). It deliberately contains no phase fractions, transformation temperatures, mechanical properties, or AM score because no TDB, experimental model data, CIF, or screening calibration is assumed.

## Input gates

| Result | Required input | Evidence label |
|---|---|---|
| Composition bases, VEC | Cu/Al/Ni wt% | Physics calculated |
| Ideal mixture density | Composition + documented elemental densities bundled in code | Engineering screening |
| Hume-Rothery e/a | User valence map + citation | Physics calculated for the stated convention |
| Equilibrium/phase fractions | Compatible external TDB | Physics calculated |
| Scheil path | TDB + `scheil` runtime + user request | Physics calculated |
| 3D crystal / simulated XRD | Complete user CIF | Physics calculated |
| Ms/Mf/As/Af/hardness/strength | Provenance-authorized experimental CSV + metadata | Machine-learning prediction |
| Experimental comparison | User instrument export | Experimental measurement |
| LPBF score | Explicit sourced metric normalization profile | Engineering screening |

No thermodynamic database is bundled. CALPHAD parameter sets can have redistribution and licensing restrictions, and a phase-name list is not a substitute for assessed Gibbs-energy models. The software reports the uploaded TDB hash, phase inventory, pycalphad version, and user citation.

No trained scientific model or populated property dataset is bundled. Empty input schemas in `data/templates/` show accepted columns but cannot train a model. Synthetic numerical fixtures exist only inside tests and are opt-in there.

## Repository layout

```text
app.py                         Streamlit dashboard
src/application/              orchestration and fail-closed result envelopes
src/composition/              basis conversion and descriptors
src/thermodynamics/           pycalphad equilibrium, grids, Scheil adapter
src/crystallography/          structures, lattice, symmetry, XRD, orientation, texture
src/phases/                   citation-backed phase descriptions
src/machine_learning/         data gate, models, ensembles, UQ, validation, explanation
src/characterization/         XRD, EDS, DSC, SEM analysis
src/additive_manufacturing/   path metrics, cracking index, transparent LPBF screen
src/visualization/            Plotly figures and dashboard style
src/reporting/                HTML/PDF/JSON exports
data/templates/               empty schemas and metadata templates
examples/                     primary-composition audit example
docs/                         methods, inputs, architecture, validation, references
tests/                        unit and integration tests
```

## Testing

```bash
python -m pip install -r requirements-dev.txt
pytest
```

Tests use mathematical/synthetic fixtures only to verify algorithms; those fixtures are explicitly labelled and cannot be used as scientific Cu-Al-Ni evidence. Run the app smoke test with:

```bash
pytest -m streamlit
```

## Data formats

See [`docs/INPUT_SCHEMAS.md`](docs/INPUT_SCHEMAS.md). Useful templates are:

- `data/templates/ml_training_schema.csv` and `.metadata.json`
- `data/templates/eds_upload_schema.csv`
- `data/templates/xrd_upload_schema.csv`
- `data/templates/dsc_upload_schema.csv`
- `data/templates/cif_metadata_schema.json`
- `data/templates/am_screening_profile_schema.json`

## Scientific scope and limitations

This is decision-support software, not a qualified materials/process specification. Equilibrium is not martensitic kinetics; an ideal CIF pattern is not a refined experimental pattern; model uncertainty is not measurement uncertainty; a 2D SEM area fraction is not a thermodynamic fraction; and an LPBF screening index is not a build-success probability. Read [`docs/SCIENTIFIC_METHODS.md`](docs/SCIENTIFIC_METHODS.md) and [`docs/VALIDATION.md`](docs/VALIDATION.md) before thesis or publication use.

For publication-grade work, archive the exact input files, their hashes/citations/licenses, the exported JSON report, package versions, process history, and independent experimental validation.

## License and citation

Code is MIT licensed; uploaded databases and datasets retain their own licenses. Cite this repository using [`CITATION.cff`](CITATION.cff) and cite every scientific database/data source recorded in the exported report.

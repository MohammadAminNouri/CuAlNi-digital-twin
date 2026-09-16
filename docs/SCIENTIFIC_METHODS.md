# Scientific methods and evidence boundaries

## Result contract

Every top-level result contains `status`, `classification`, `confidence`, `provenance`, and `limitations`. A dependent engine returns `status: unavailable` when its scientific input is absent or invalid. It does not call a fallback correlation.

The interface distinguishes:

- **Physics calculated** — deterministic composition/crystal geometry, structure-factor diffraction, or a Gibbs-energy calculation using the uploaded TDB.
- **Machine-learning prediction** — an empirical estimate fitted to the uploaded, provenance-authorized experimental table.
- **Engineering screening** — a transparent index or approximation whose bounds/assumptions are declared.
- **Experimental measurement** — processing or comparison of user instrument data.
- **Unavailable** — evidence is insufficient.

## Composition

For mass fractions `w_i` and atomic weights `M_i`, mole fraction is

```text
x_i = (w_i / M_i) / Σ_j(w_j / M_j)
```

Atomic percent is `100 x_i`. Atomic weights are constants documented in `composition/converter.py`. The code normalizes programmatic inputs but the Streamlit UI requires an exact 100 wt% total to expose transcription errors.

VEC uses the explicitly named periodic-table group-number convention. Hume-Rothery effective e/a is left unavailable unless the user supplies all three effective valences and a citation, because transition-metal effective valence is convention-dependent.

The density screen is the reciprocal mass-fraction mixture rule

```text
ρ_mix = 1 / Σ_i(w_i / ρ_i)
```

using documented room-temperature elemental densities. It ignores phase-dependent excess volume, porosity, defects, and temperature and is therefore an engineering screening estimate, not a measured alloy density.

## CALPHAD equilibrium and phase maps

`thermodynamics/calphad.py` loads an external TDB through pycalphad, validates Cu/Al/Ni coverage, hashes the file, filters compatible phases, and minimizes the database Gibbs energies at specified `T`, `P`, `N`, and independent mole fractions. Phase names are returned verbatim from the database. The software never maps a phase name to β, β₁, martensite, or γ₂ by guesswork.

Fractions are molar phase fractions. Equilibrium does not automatically predict diffusionless martensitic transformation, retained metastable phases, kinetic suppression, or transformation temperatures. Database assessment range and model quality govern validity.

The optional ternary map repeats equilibrium calculations on a regular mole-fraction grid. It is an interactive sampled assemblage map, not a traced invariant/tie-line phase diagram; narrow fields may be missed.

## Scheil-Gulliver solidification

The optional adapter calls the open-source `scheil` package with the same uploaded TDB. It assumes perfect liquid mixing, local solid/liquid equilibrium, and negligible solid diffusion. It excludes solute trapping, melt-pool convection, nucleation barriers, and later solid-state transformations. The user must choose a start point inside the database's single-liquid region.

`additive_manufacturing/solidification.py` can calculate liquidus/solidus proxies at declared solid-fraction cutoffs and numerical segregation-path descriptors. `cracking.py` implements a Kou-type terminal slope indicator over an explicit solid-fraction interval. No universal Cu-Al-Ni LPBF risk category is attached to that scalar.

## Crystallography, symmetry, and orientation

Incomplete literature records are reference metadata only. A complete CIF supplies lattice vectors, atomic fractional coordinates, species, and occupancies. Pymatgen parses the CIF; the raw bytes are SHA-256 hashed. Spglib assigns symmetry at explicit tolerances. Partial occupancy can make symmetry/model interpretation ambiguous and is reported.

`lattice.py` implements triclinic lattice matrices, reciprocal geometry, unit-cell volume, d-spacing, and Bragg's law. `orientation.py` states its row-vector lattice and crystal-to-sample matrix conventions. `texture.py` generates upper-hemisphere Lambert equal-area pole points from explicit orientation matrices; it does not claim an ODF inversion or apply undocumented crystal symmetry.

## Diffraction

For order `n`, wavelength `λ`, and spacing `d`, the code uses

```text
n λ = 2 d sin θ
```

Pymatgen's `XRDCalculator` supplies ideal kinematic structure-factor peak positions and scaled intensities. Calculated patterns omit instrument optics, preferred orientation, displacement/zero shift, size/strain broadening, absorption, defects, and specimen-dependent temperature factors.

Experimental XRD uses asymmetric least-squares baseline correction, Savitzky-Golay smoothing, SciPy peak detection, Bragg d-spacing conversion, and one-to-one angular matching. Returned phase scores are explicitly **uncalibrated match scores**, never probabilities or phase fractions. Confirmation requires full-pattern refinement and suitable standards; peak tables can be exported for external GSAS-II workflows.

## Machine learning and uncertainty

The ML gate validates composition totals, required units, provenance, authorization, target variation, and minimum row count. Synthetic data are rejected by default. Preprocessing (median imputation, robust scaling, one-hot quench medium) occurs inside each cross-validation pipeline to prevent leakage.

Implemented regressors are linear regression, Ridge, RBF SVR, random forest, Gaussian process regression, XGBoost, and LightGBM. Optional estimators that fail are retained in a failure audit. Successful models are inverse-MAE weighted after out-of-fold validation. Target-specific tables report MAE, RMSE, R², fold dispersion, and split strategy.

Intervals are symmetric finite-sample-corrected absolute-residual CV-conformal intervals. Nominal coverage relies on exchangeability with future specimens; the global interval does not model composition-dependent noise. Per-feature training-range checks flag extrapolation but are not a convex-hull or processing-route guarantee.

Ms, Mf, As, and Af are fitted independently. The software checks `Mf ≤ Ms` and `As ≤ Af`, displays failures, and never silently reorders predictions. SHAP is used when compatible; otherwise the app labels the result as permutation importance, not SHAP. Importance is not causality.

## Characterization

- **EDS:** normalizes point compositions, reports target deviation and spatial/statistical segregation descriptors. The optional Kanaya-Okayama-style interaction-range estimate runs only when the user supplies both voltage and independently justified density.
- **DSC:** segments monotonic heating/cooling branches, fits endpoint baselines, finds the dominant residual event, and reports 5–95% cumulative-absolute-area operational bounds. These are not a substitute for calibrated tangent analysis.
- **SEM:** normalizes/denoises an image and extracts intensity/gradient/local-texture features. Without a provenance-documented validated classifier, K-means outputs are called `region_1`, etc., never martensite/β/γ₂.

## LPBF suitability

The app has no built-in 8/10 or universal risk thresholds. A score is emitted only from a user profile defining each metric's preferred/adverse values, units, weight, rationale, source, applicability, and version. The score is a bounded weighted engineering index, not a probability or qualification. Powder morphology, machine architecture, atmosphere, geometry, scan strategy, residual stress, defects, and coupon validation remain external.


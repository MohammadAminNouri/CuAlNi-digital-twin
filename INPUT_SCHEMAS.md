# Input schemas

## Composition

The UI accepts Cu, Al, and Ni in wt% and requires an exact total of 100. Programmatic `AlloyComposition` constructors also accept atomic percent or mole fraction.

## Thermodynamic database

Upload a pycalphad-readable `.tdb` and enter its assessment/version citation. The file must contain CU, AL, NI (and usually VA) plus compatible phase models. The application computes and reports a SHA-256 hash; it does not redistribute the database.

## CIF metadata

Upload one or more `.cif` files. The optional JSON is keyed by exact filename or stem:

```json
{
  "my_18R.cif": {
    "phase": "18R specimen A",
    "citation": "Authors, title, journal, year, DOI; database/repository identifier",
    "source_url": "https://repository.example/record"
  }
}
```

If citation metadata are omitted, the CIF still loads but the report flags its unverified pedigree.

## ML table and provenance

The CSV must include `Cu_wt_pct`, `Al_wt_pct`, and `Ni_wt_pct`, plus any supported process/microstructure features and at least one target:

- transformations: `Ms_C`, `Mf_C`, `As_C`, `Af_C`
- properties: `hardness_HV`, `yield_strength_MPa`, `uts_MPa`
- process/features: `solution_treatment_C`, `solution_treatment_min`, `quench_medium`, `aging_C`, `aging_min`, `grain_size_um`, `beta_fraction`, `martensite_fraction`, `gamma2_fraction`

Metadata JSON fields `title`, `source`, `license`, `measurement_method`, `units`, and ISO `retrieved_at` are mandatory. Every numeric target and feature used by a model needs a unit. `synthetic: true` is rejected by the application. `allow_training: false` disables training.

Use `source_id` for independent studies/batches so grouped cross-validation can avoid source leakage. Retain `sample_id` and row-level DOI/source fields for audit even though they are not predictors.

## XRD

CSV/TXT/XY with at least 15 points and two columns recognizable as `two_theta_deg`/`2theta` and `intensity`/`counts`. Angles must become strictly increasing after duplicate averaging. Specify the measurement wavelength in ångström.

## EDS

CSV/TXT columns `Cu_wt_pct`, `Al_wt_pct`, and `Ni_wt_pct`; optional spatial coordinates are preserved by the parser when supported. Values are normalized per point, so inspect raw totals and calibration diagnostics.

## DSC

CSV/TXT with `temperature_C` and `heat_flow`. Include both monotonic branches to extract all four operational temperatures. Record scan rate, sample mass, sign convention, atmosphere, and temperature calibration in the experiment archive.

## SEM

PNG, JPEG, or TIFF. Unsupervised segmentation is always labelled as regions. A supervised phase classifier can be used through Python only when accompanied by source, training-data hash, class labels, and validation metrics.

## AM profile

The upload JSON contains `metrics` and `profile`. Metric values should include units and provenance; profile rules define normalization. `data/templates/am_screening_profile_schema.json` is a formal schema and contains no scientific bounds. Bounds and weights must come from the user's documented calibration/decision protocol.


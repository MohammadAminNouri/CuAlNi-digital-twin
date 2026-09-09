# Experimental data templates

These files contain headers only. They are not example measurements and must not
be used to train or validate a scientific model.

- `ml_training_schema.csv` defines supported composition, processing,
  microstructure, transformation-temperature, hardness, and strength fields.
- `ml_training_schema.metadata.json` demonstrates mandatory dataset-level
  provenance. Replace it with the actual citation, DOI/URL, license, measurement
  methods, units, and retrieval date. For compiled literature data, retain
  `source_id`, `sample_id`, and row-level `doi` fields.
- `eds_upload_schema.csv`, `xrd_upload_schema.csv`, and `dsc_upload_schema.csv`
  define accepted experimental upload shapes.
- `cif_metadata_schema.json` defines optional per-file phase labels and citations.
- `am_screening_profile_schema.json` validates a user-calibrated LPBF profile; it
  intentionally contains no scientific normalization bounds or weights.

The application rejects an empty table, undocumented target units, unauthorized
datasets, and synthetic data unless test mode is explicitly enabled in Python.

JSON schema files are not uploadable profile instances. Populate a separate JSON
instance using values and sources approved for the intended research programme.

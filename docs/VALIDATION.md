# Validation and research use

## Automated verification

The test suite covers composition conversion and conservation, arbitrary triclinic lattice/d-spacing/Bragg geometry, orientation conventions, crystallographic reference gating, XRD peak matching, EDS normalization, DSC branch detection, SEM label semantics, solidification metrics, AM score gating, ML data authorization, cross-validation/prediction intervals, report export, orchestration fail-closed behavior, and a Streamlit launch smoke test.

Generated test signals are mathematical fixtures labelled as non-scientific. They verify code paths and do not validate Cu-Al-Ni material predictions.

## Validation still required by a research project

1. **TDB verification:** reproduce published binary/ternary equilibria and invariant reactions within the database's assessment range; compare phase fractions to appropriate experiments.
2. **Crystal validation:** verify CIF composition, setting, atom sites, occupancies, temperature, refinement residuals, and specimen applicability. Compare simulated and standard-corrected XRD before phase assignment.
3. **ML curation:** document every specimen, processing route, measurement method, uncertainty, duplicates, censoring, and literature extraction decision. Use source/batch grouped or nested validation where applicable and reserve an external test set.
4. **Transformation validation:** compare predicted Ms/Mf/As/Af with calibrated DSC/dilatometry across independent heats and processing histories. Inspect ordering failures rather than repairing them numerically.
5. **Mechanical validation:** keep hardness scales/load and strength test geometry/temperature/rate consistent; do not pool incompatible definitions without modelling them.
6. **AM validation:** calibrate screening rules against the target machine, powder, geometry, atmosphere, and inspection protocol. Validate with coupon builds and quantitative defect/metallography data.
7. **Uncertainty:** propagate input and database uncertainty separately from empirical predictive intervals. The supplied conformal interval addresses empirical residual coverage only.

## Reproducibility checklist

- Export HTML, PDF, and JSON from the same run.
- Archive exact TDB/data/CIF files and verify hashes.
- Record repository commit/tag and `python -m pip freeze`.
- Record all Streamlit processing inputs and instrument parameters.
- Retain raw measurements and calibration certificates.
- Cite pycalphad/scheil/pymatgen/spglib and the thermodynamic/crystal/experimental sources.
- Have domain experts review every automated phase or process conclusion.

Passing software tests does not qualify a database, model, alloy, or LPBF process.


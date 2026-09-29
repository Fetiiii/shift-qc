# Methodology

The frozen method implementations are provided for inspection, not changed
for this release. R5 scores unchanged deployed quality predictions. R8 fits
the registered affine-L1 oracle mapping using the training part of each
target split and evaluates its held-out part. The comparator is a median
fitted to the training fold. Skill is one minus model MAE divided by
comparator MAE; the cross-domain difference is OOD minus ID.
Recalibration consumes target labels and is diagnostic, not deployable.
No formal R5-versus-R8 contrast or causal interpretation is added.

## Full image-to-result pipeline

1. Acquire each source under its actual applicable agreement.
2. Recover provenance and apply frozen patient-level eligibility.
3. Construct the registered patient splits and prove role separation.
4. Use the standard nnU-Net planning/training/inference contract.
5. Extract prediction-time features from images and model outputs.
6. Fit the registered QC model on QC_TRAIN quality values only.
7. Apply the frozen predictor and compute R5 and label-assisted R8.
8. Use the registered repeated folds and paired patient bootstrap.
9. Regenerate tables/figures and perform endpoint/provenance audits.

Exact feature columns, seeds, grids, folds and bootstrap definitions remain
in their frozen code/configuration; the release does not select new values.
The two predictor implementations are qc/m3_deployment.py and
validation2/qc_model.py. Endpoint/comparator/recalibration/bootstrap code is
under confirmatory/. Full provenance maps and actual split identities are
withheld. This candidate cannot establish an exact cohort rerun.

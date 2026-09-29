# Regenerating models

Initial software release: trained QC models EXCLUDED; segmentation checkpoints
EXCLUDED. No model download or checkpoint redistribution is supplied.
After independent data acquisition, use the frozen standard nnU-Net planning,
training and inference contract, then extract prediction-time features and fit
QC on QC_TRAIN only. Preserve patient role/label firewalls and locked final test.
See docs/full_reproduction.md. No research parameter is changed by this release.
Sharing weights can be revisited only after artifact-specific rights clearance;
it is not required for the current submission plan.

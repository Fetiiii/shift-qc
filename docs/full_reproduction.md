# Full source-data-dependent reproduction

Status: DOCUMENTED / REQUIRES ORIGINAL DATA. This release did not perform an
independent source-data rerun. It does not claim immediate reproduction of all
paper numbers from a repository clone.

## Inputs and scientific authority

Obtain original datasets from the official custodians in data_access.md, accepting
the applicable conditions yourself. No automatic dataset acquisition is performed.
Original masks/volumes, per-patient quality/features, patient provenance/splits,
predictions, probabilities and trained weights are local inputs, not release files.
The public config is a template: patient-artifact hashes and observed split counts
were removed, with no change to scientific hyperparameters or master config.
The release also includes frozen source-only M&Ms and Validation-2 pipeline entry
points for source audit, role splits, nnU-Net preparation/training/inference
and provenance. The historical M&Ms fold-construction wrapper is withheld
because it embeds SHA-256 fingerprints of excluded patient/split artifacts;
reconstruct fold assignments locally under the frozen fold specification.
Historic launch configs and identity-bearing provenance scripts are not
distributed wholesale. Reconstruct the needed local configs and mappings
from the registered specifications and the official source metadata under their
access terms. A raw-data-to-ledger rerun is documented here, not tested or
represented as a single-command procedure in this code-only release.
Locally recovered population/split identities must meet the registered eligibility
and patient leakage rules; do not invent a new split and claim an exact rerun.
Frozen method/source lineage is in RELEASE_MANIFEST.json and
manifests/public_safe/analysis_specifications.json.

## Pipeline stages

1. **Acquisition and provenance.** Obtain M&Ms and AbdomenCT source components
   independently. Recover acquisition/source identities from the original metadata
   and apply the frozen population definitions. Keep patient mappings local.
2. **Eligibility and splits.** Apply patient-level eligibility and reconstruct
   registered roles SEG_TRAIN, SEG_VAL, QC_TRAIN, CALIBRATION, ID_TEST, OOD_DEV,
   and locked OOD_FINAL where applicable. Prove role intersections are empty and
   SOURCE/TARGET disjoint. Source population/split mappings are excluded when
   redistribution authority is unresolved; exact cohort correspondence requires
   provenance recovery, not random replacements.
3. **Segmentation.** Use standard nnU-Net v2 planning and the recorded full-pipeline
   environment. Preserve architecture/patch/loss/augmentation/spacing/batch contract.
   Train only with SEG_TRAIN/SEG_VAL labels. Regenerate probabilities/predictions
   locally. GPU/storage costs are substantial; no silent CPU fallback or hardware
   downgrade is permitted. Report resource failures. No bitwise GPU guarantee.
4. **Quality and features.** Compute registered quality values from local predictions
   and ground truth for allowed roles. Public metrics/feature modules define
   segmentation quality and prediction-time feature extraction. Deployment features
   must depend only on images/probabilities/predictions/source-fit statistics.
   QC truth is not a feature. Preserve label mapping and empty-mask behavior.
5. **QC training.** Use qc/m3_deployment.py for the M&Ms path and
   validation2/qc_model.py for Validation-2, preserving frozen feature columns,
   standardization, Ridge search and seeds. Fit true quality only on QC_TRAIN.
   Any source support fitting remains QC_TRAIN + CALIBRATION only. Do not fit
   target distributions or silently alter losses/features for better performance.
6. **R5 and R8.** Use confirmatory/r5.py, r8.py, recalibration.py and bootstrap.py
   with locally aligned patient IDs/q_true/q_hat and the original roles. R5 uses
   deployed predictions; R8 is an affine-L1 oracle fitted within training folds.
   Comparator is the cross-fitted median. Skill and OOD-minus-ID definitions stay
   unchanged. R8 target labels do not become deployment inputs. No new endpoint.
7. **Uncertainty.** Use the frozen repeated patient folds and paired bootstrap.
   Preserve seeds, repetitions and numerical rules in the original source.
   Do not silently replace NaN/Inf or choose post-readout endpoints.
8. **Tables/figures.** Produce the original result-row schema from regenerated
   analysis outputs; never manually type manuscript estimates into plotting code.
   See input_contract.md. Place local ledger under ignored local_inputs/.
   Run the table and Figure 2–4 scripts below. Missing inputs produce explicit errors.
9. **Audits.** Match endpoint/population/estimate/sign/CI/n/source against frozen
   authority when you have permitted local reference inputs. Software hash checks
   alone do not establish numerical equality of unavailable results.

## Output commands

Run after the locally regenerated input contract is satisfied:

```sh
python scripts/reproduce_tables.py --ledger local_inputs/frozen_result_ledger.json --output-dir /tmp/shiftqc-tables
python scripts/reproduce_figures.py --only figure2 figure3 figure4 --ledger local_inputs/frozen_result_ledger.json --output-dir /tmp/shiftqc-figures
```

Do not commit local regenerated data/ledger/models. Any later aggregate or model
sharing needs artifact-specific permission; MIT software rights do not authorize it.
Original final-label access policy remains SHIFTQC_UNLOCK_FINAL=1 through the
applicable guarded evaluation code. This release never reads final masks.

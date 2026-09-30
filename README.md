# SHIFT-QC

Reproducibility software accompanying the SHIFT-QC manuscript.
This public repository contains the author-approved code and public-safe reproducibility materials. The existing v1.0.0 tag is an unpublished historical candidate; v1.0.1 is the first intended published software archive.
Original-code licensing authority was confirmed by the author factual declaration.
Frozen export source: `1a11c27570b4858a7ad24a8e6c8a1496a70fee93`.
Scientific authority: `10baa3212679f3e15116abde7050d4f6e474288e`.

## Overview

SHIFT-QC studies segmentation quality-control transfer under domain shift.
The public repository contains the implementation and public-safe reproducibility materials. Patient-level and source-data-derived artefacts are not redistributed because the study combines datasets governed by different licences and data-use terms. Researchers can reproduce the full pipeline after independently obtaining the original datasets from their official custodians.
This disclosure describes the public software scope; a Zenodo DOI had not been assigned when this metadata revision was committed.

## Scientific question

How does a source-trained quality predictor compare with a patient-agnostic
comparator under domain shift, in its deployment state and under the registered
label-assisted oracle recalibration?

## What this repository contains

Original predictor, comparator, R5/R8, recalibration, bootstrap/statistics and
feature/metric code and selected source-only segmentation/provenance entry
points; public-safe configuration templates and frozen method specifications; synthetic tests; conceptual Figure 1 and table/Figure 2–4
scripts accepting locally regenerated inputs. This is a software package,
not a complete source-data or patient-derived data archive.

## Main estimands

R5 uses unchanged deployed QC predictions. R8 uses the registered label-assisted
oracle recalibration fitted on training folds and evaluated on held-out folds.
R8 requires target labels and is diagnostic, not deployable. Both use a
cross-fitted median comparator; cross-domain difference is OOD minus ID.
No formal R5-versus-R8 contrast is claimed. Scientific definitions remain frozen.

## Repository structure

- `src/shiftqc/`: original frozen methods and selected pipeline components.
- `configs/templates/`: configuration stripped of patient-artifact fingerprints.
- `scripts/`: verification and source-data-dependent reproduction entry points.
- `tests/`: synthetic/unit verification, not patient-derived fixtures.
- `environment/`: public dependency pins and historical full-pipeline notes.
- `docs/`: methodology, acquisition, full reproduction, governance, limitations.
- `manifests/public_safe/`: method specifications and release lineage schema.
- `figures/figure1/`: public-safe conceptual reference outputs.
- `DATA_AND_ARTIFACT_RIGHTS.md`: software/data rights separation.

## Data access

See [official acquisition instructions](docs/data_access.md). M&Ms, AbdomenCT-1K,
MSD, NIH Pancreas-CT and KiTS retain their original terms. Independently obtain
required datasets from their custodians; scripts do not download them automatically.
No trained model, medical volume, patient-level row, uncertain split manifest or
unresolved aggregate endpoint ledger is redistributed.

## Environment/setup

Use CPython 3.12.13 and the documented historical scientific dependency pins:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment/requirements-public.txt
python -m pip install --no-build-isolation --no-deps -e .
```

CUDA/segmentation dependencies are separate; see
[environment notes](environment/environment_notes.md).

## Reproducing manuscript outputs

### Code verification

```sh
python -m pytest
python scripts/verify_release.py --code-only
python scripts/reproduce_figures.py --only figure1 --output-dir /tmp/shiftqc-figure1
```

This requires no original medical images or patient-derived inputs.

### Analysis reproduction

This is **source-data-dependent reproduction**. After independently obtaining
source datasets, regenerate eligible populations, splits, segmentation/QC inputs
and analysis artifacts following [full reproduction](docs/full_reproduction.md).
Use the locally regenerated ledger with the table and numeric figure scripts.
Required inputs are not bundled; absent inputs produce a controlled error with
acquisition/reproduction instructions. Do not substitute fabricated results.

### Manuscript artefact verification

Included conceptual/public-safe artifacts trace to the frozen authority through
`RELEASE_MANIFEST.json`. No aggregate manuscript estimate/CI/sample-count ledger
is included because artifact-specific redistribution authority remains uncertain.
The verifier checks software/public-safe bytes; it does not certify equality of
withheld manuscript numbers or a complete source-data rerun.

## Expected outputs

`CODE_REPRODUCIBILITY = PASS` when the documented clean-copy tests pass.
`PUBLIC_SAFE_ARTIFACT_VERIFICATION = PASS` when manifests and Figure 1 match.
`FULL_SOURCE_DATA_REPRODUCTION = DOCUMENTED / REQUIRES ORIGINAL DATA`.
The first two statuses are documented in the release audit.
The third is not a tested PASS. This repository does not promise that a clone
immediately reproduces every manuscript number.

## Model/checkpoint availability

Initial release excludes all trained QC models and segmentation checkpoints.
Regenerate segmentation models with the frozen standard nnU-Net contract and fit
QC using QC_TRAIN only; see [full reproduction](docs/full_reproduction.md).
Checkpoint sharing may be revisited after artifact-specific clearance.

## Citation

`CITATION.cff`: Feti Çağrı Eraslan, ORCID 0009-0001-9358-0563. The original software package version remains 1.0.0; the corrected first publication snapshot is tagged v1.0.1.
Repository: https://github.com/Fetiiii/shift-qc. No Zenodo DOI had been assigned at this metadata commit. AI tools are not authors.

## License

Original SHIFT-QC software source code is provided under the [MIT License](LICENSE).
The scoped grant covers original software in src/, scripts/, tests/ and configs/;
it grants no dataset or excluded derived-artifact rights. Documentation, citation
metadata and conceptual figure outputs are distributed with author permission,
with all rights reserved unless a separate license is stated. See
[DATA_AND_ARTIFACT_RIGHTS.md](DATA_AND_ARTIFACT_RIGHTS.md) and
[code-license status](docs/code_license_status.md).

## Paper status

Reproducibility repository accompanying the SHIFT-QC manuscript.
No actual submission, review, acceptance or publication is claimed.

## Limitations

Actual patient populations/splits, per-case outputs, learned weights and numerical
ledger are excluded. Reconstructing them requires original data and the scientific
provenance constraints; no exact raw-data rerun was performed for this release.
GPU bitwise reproducibility is not established. R8 remains an oracle diagnostic.
Dataset-derived redistribution uncertainty does not block the narrower software
release. Code authority and public software publication have been approved by the author.

# Reproducibility layers

## Code verification

Follow README setup; run pytest, verify_release.py --code-only, and regenerate
Figure 1 to an external directory. Synthetic inputs exercise software behavior.
No source medical data or patient-derived fixture is needed.

## Analysis reproduction

Source-data-dependent reproduction begins with independent custodian acquisition.
Follow docs/full_reproduction.md to regenerate provenance, eligibility, splits,
segmentation, prediction-time features, QC, R5/R8 and bootstrap inputs.
The software release is not a complete data archive. Missing local input is a
controlled error, not a silent skip or permission to invent results.

## Manuscript artifact verification

RELEASE_MANIFEST.json traces code/public-safe materials to frozen source blobs
and recorded transformations. Conceptual Figure 1 is numeric-result-free.
No aggregate manuscript ledger is included. Any future aggregate release requires
separate rights review and exact authoritative number/sign/CI/n/endpoint checks.
Software tests do not verify withheld numerical results. Full source-data
reproduction is documented/requires original data; no complete rerun is claimed.
The manifest's own detached hash is recorded in the private export whitelist.

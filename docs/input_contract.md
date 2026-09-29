# Locally regenerated analysis inputs

No manuscript numerical ledger is included. Table/figure scripts expect a local
JSON list of result rows produced by the frozen analysis lineage. Figure rows
use result_id, full_precision and display; see the original drawing functions
for required IDs. Those endpoint row IDs are not patient identifiers.
A row must identify endpoint/population/estimate/CI/n/status/source when applicable;
the source reference remains local. No patient identifiers/per-case values belong
in an aggregate display ledger. The renderer does not fit/re-estimate endpoints.

Inputs must come from the independently acquired datasets and regenerated
analysis, not synthetic replacements represented as manuscript results.
Acquisition: data_access.md. Pipeline: full_reproduction.md.
All missing/malformed required inputs must cause a controlled nonzero error.
The public manifest deliberately excludes local_inputs/ and regenerated outputs;
never treat a successful local render as redistribution permission.

The local aggregate result row uses result_id/dataset/endpoint/quantity/full_precision/display, matching the original ledger schema. For Table 2, add the frozen endpoint verdict as status on the baseline/R5/R8 estimate row from the locally regenerated analysis JSON. Table 1 computes source total from locally regenerated component-count rows, avoiding an exported observed count. Table 1/2 layout is adapted from the original assembler. ledger_table.md is a generic numeric ledger view; the narrative final Supplementary Table S1 is not bundled or claimed format-identical.

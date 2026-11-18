# Results

No per-patient or aggregate endpoint result ledger from the clinical validations is
included. Regenerate required inputs locally after acquiring datasets independently,
following docs/data_access.md and docs/full_reproduction.md. Use ignored local_inputs/.
Tables/Figure 2–4 consume that local ledger; absent inputs are a controlled error.

The one exception, added in v1.1.0, is the synthetic finite-sample study of the R5/R8
estimators (EXP-0007). It reads no patient data, and its four result files are in
`outputs/EXP-0007/`. To rerun it, set `output_dir` in
`configs/experiments/exp_0007_r8_synthetic.json` to a new directory, because the
driver never overwrites existing results:

```sh
python scripts/task052_simulate.py all
```

The protocol is `docs/EXP0007_R8_SYNTHETIC_PROTOCOL.md`. The run takes about 2.5 hours
on 20 CPU workers and needs no GPU.

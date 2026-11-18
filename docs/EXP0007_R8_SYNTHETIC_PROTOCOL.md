# Finite-Sample Behaviour of R5/R8 on Synthetic Data — Protocol

**Status:** FROZEN 2026-11-16 by author approval, before any simulated quantity was
computed. Binding from the commit tagged `pre-exp0007-synthetic-freeze`.

**Identifiers:** TASK-052, EXP-0007, decision D-041.

**Classification:** methodological simulation. No patient data, no image, no result
table of either validation is read. It cannot change, explain or re-interpret any
registered result; it characterises the estimator.

## 1. Cost (measured before design, 2026-11-16)

One evaluation of the frozen R8 estimator (ten repeats of five folds, affine-L1) takes
about 0.1 s on this machine at the sample sizes below; the machine has 24 cores. No
GPU, negligible memory and disk.

| Tier | Work | CPU time | Wall time on 20 workers |
|---|---|---|---|
| 1 | 40 cells × 2000 simulated datasets × 1 evaluation | about 2.2 h | about 7 min |
| 2 | 4 cells × 300 datasets × 500 bootstrap replicates | about 17 h | about 50 min |

Tier 2 is included by author decision (Section 8).

## 2. Questions

1. When in-domain and out-of-domain data come from the **same** distribution, is the
   expected `Delta8` zero at the sample sizes used, or does the difference in
   recalibrator fitting budget (about 53 against 196 patients per fold in AbdomenCT)
   move it?
2. Does `Delta8` behave as its definition implies under a pure scale/offset shift, a
   reversed relationship, a monotone non-linear distortion, and a loss of
   information?
3. Does a change in outcome dispersion alone move `Delta5` and `Delta8`?
4. (Tier 2) Under no true difference, how often does the registered decision rule
   return `POSITIVE`, and does the 95% percentile interval cover the reference value?

## 3. Data-generating model

Fixed in advance; not tuned to any observed result.

```text
q        ~ Beta(8.5, 1.5)                         true quality in [0, 1]
z        = (q - mean(q)) / sd(q)                  population-standardised
x        = rho * z + sqrt(1 - rho^2) * eps,       eps ~ N(0, 1)
q_hat    = mean(q) + rho * sd(q) * x              the linear predictor in-domain
```

`rho` is the population correlation between score and quality in-domain.

| Scenario | Out-of-domain differs from in-domain by | Population `Delta8` |
|---|---|---|
| A identical | nothing | 0 |
| B scale/offset | `q_hat -> 0.3 + 0.5 * q_hat` | 0 |
| C reversed | `x -> -x` | 0 for affine, negative for isotonic |
| D monotone | `q_hat -> mean(q) + sd(q) * tanh(2x)` | from the reference run |
| E unrelated | `x` independent of `q` | negative |
| F dispersion | `q ~ Beta(4, 1.5)`, same conditional relation | from the reference run |

Scenario A is run at `rho` in {0, 0.25, 0.5}; B to F at `rho = 0.5`: eight cells.

Sample-size pairs (in-domain, out-of-domain): (66, 245), (70, 100), (245, 245),
(66, 66), (245, 66). The last reverses the asymmetry, so a budget effect, if any,
must change sign. Eight cells × five pairs = 40 cells.

## 4. Estimators

`shiftqc.confirmatory.r5.compute_r5` and `r8.compute_r8` internals, unchanged:
frozen folds, repeats, comparator and recalibrators (affine-L1 primary; affine-OLS
and isotonic also recorded). No source file under `src/shiftqc/confirmatory/` is
modified.

**Reference value.** For each cell, the same estimator at (20000, 20000), averaged
over 20 independent draws. It stands in for the large-sample value.

## 5. Tier 1: bias and spread

For each cell, 2000 independent datasets, seed stream fixed in config. Reported per
cell: mean, standard deviation, 2.5/50/97.5 percentiles of `Delta5` and `Delta8`,
the reference value, mean minus reference with its Monte Carlo standard error, and
the same for `S_ID` and `S_OOD` separately.

## 6. Tier 2: decision rule and interval

Scenario A at `rho` in {0.25, 0.5}, sample-size pairs (66, 245) and (70, 100): four
cells, 300 datasets each, full R8 with 500 bootstrap replicates. Reported: frequency
of `CI_high < 0`, frequency of `CI_low > 0`, coverage of the reference value, each
with a binomial standard error. The bootstrap size is 500, not the registered 2000,
to bound cost; this is stated wherever the result appears.

## 7. Fixed in advance

- Every cell is reported, whatever it shows. No cell, `rho`, sample size or
  distribution is added, dropped or changed after results are seen.
- No threshold is set for "acceptable" bias. Magnitudes are reported with their
  Monte Carlo error; interpretation is the author's.
- Nothing here is described as the cause of any observed result. The permitted
  statement is of the form "under identical distributions at these sample sizes the
  estimator's mean was X (reference 0)".
- The registered results are not recomputed, adjusted or bias-corrected.
- Gate before the simulation: the driver, given the same inputs as `compute_r8`,
  returns the same point estimate (unit test).

## 8. Author decisions, closed 2026-11-16 before freezing

1. **Proceed** with the model and cells of Section 3. Approved.
2. **Tier 2.** Included.
3. **Placement.** Supplement section plus one sentence in main-text Section 5.4.
   The revised Section 5.4 currently says of the fitting-budget difference "Its size
   was not studied here"; once this simulation exists that sentence would be false,
   so it is replaced by a pointer to the supplement whatever the results are.
4. **Identifiers.** TASK-052, EXP-0007, D-041.

## 9. Clarifications made at freezing, before any computation

These make Section 3 computable as written. None was chosen with a result in view.

1. **Constants.** `mean(q)` and `sd(q)` are the analytic moments of Beta(8.5, 1.5),
   0.85 and sqrt(8.5 * 1.5 / (10^2 * 11)), in every scenario, including F.
2. **Scenario A at `rho = 0`.** The best linear predictor would be a constant, which
   makes the recalibration fit degenerate. In that one cell the score is
   `q_hat = mean(q) + 0.5 * sd(q) * eps`: a non-informative score with non-zero
   spread.
3. **Scenario F.** Out of domain `q ~ Beta(4, 1.5)`, and the score is produced by the
   same structural equations with the same in-domain constants:
   `z = (q - 0.85) / sd`, `x = rho * z + sqrt(1 - rho^2) * eps`,
   `q_hat = 0.85 + rho * sd * x`. Only the quality distribution changes.
4. **Scenario E.** Out of domain `q_hat = 0.85 + rho * sd * eps'`, with `eps'`
   independent of `q`.
5. **Folds.** For a given sample size the patient identifiers, and therefore the
   frozen repeated folds, are the same in every simulated dataset.
6. **Tier 2 coverage** is reported for the reference value of the cell and, because
   scenario A has a population value of zero by construction, for zero.
7. **Seeds.** One `numpy.random.SeedSequence(2026)` root, spawned per tier, cell and
   dataset; recorded in the config.

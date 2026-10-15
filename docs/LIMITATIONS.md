# Reproduction scope and remaining issues

The core numerical comparison suite passes. This does not mean that every item in the manuscript has been independently regenerated.

## Submission frequency in the text

English main-task rho=0.671, P=0.034 and English subtask rho=0.650, P=0.114 are reproduced from successful selected submissions for the participating teams and vector-reconstructed leaderboard scores.

The Japanese main-task discrepancy is resolved. The archived metadata contains two records for sociocom Submission ID `8830aa88-db2f-4227-94e4-d1b67c497256`, with the same timestamp and selected flag but different score records. Counting metadata rows gives sociocom two submissions and exactly reproduces the old manuscript result, rho=0.4009386883222482, P=0.22168301553096678. Counting distinct Submission IDs gives sociocom one submission and rho=0.4453194528540574, P=0.16987767013818195. All other team counts, all performance scores, and the 11-team cohort are unchanged. This is a duplicate-record counting issue, not a change in participant predictions, run selection or the statistical test.

The adopted analysis counts distinct Submission IDs and includes both successful asukai89 submissions under the documented selected-flag exception. The manuscript is revised to rho=0.445, P=.170. The strict selected=true sensitivity result (excluding asukai89) remains available, but it is not the adopted primary submission-frequency analysis. Neither result establishes a statistically significant Japanese main-task correlation. `duplicate_record_audit.json` and `duplicate_metadata_rows.csv` document the resolution.

Submission counts describe the recorded eligible submissions, not necessarily every inference run performed by a team. Platform test scores can be zero when test predictions were absent (sociocom), so the correlation stage uses vector-reconstructed LB scores, not maximum metadata test values.

## Tumor-size proxy analysis

The 243 shared reports consist of 108 training, 54 validation and 81 test reports from 27 cases. The shared reference T0/T2a/T3/T4 counts are 8/36/22/93, matching the earlier proxy response's denominators. The additional 135 Japanese-only reports are outside that subset.

The adjudicated consensus files for training/validation and test were supplied subsequently and resolve the missing source-data issue for this analysis. They contain exactly the 243 language-aligned reports, with IDs and T/N/M labels identical to the released corpus. `prepare_consensus.py` retains only numeric/label fields and excludes report text and excerpts. `scripts/proxy.py` now independently reproduces 48/243 indeterminate-size reports, 16/108 training, 6/54 validation and 26/81 test, and 40/48 with reference T2a or higher.

For those 40 reports, comparing the extension/invasion and intrapulmonary-lesion T-derivation subtotals reproduces 20/19/1: extension subtotal higher, intrapulmonary subtotal higher, or equal positive subtotals. These groups identify the dominant T-determining subtotal; they do not describe mutually exclusive presence of findings. Presence alone gives 19 extension-only, 10 intrapulmonary-only and 11 both. The analysis does not establish why the original cohort was restricted to 243, does not cover the additional 135 Japanese-only reports and does not measure clinical missingness beyond the explicit annotation flag.

## Japanese TMUNLPG3 subtask

This team is absent from the archived Japanese prediction matrices. Its separate platform score is exported with the explicit marker `platform_metadata_only_not_vector_recomputed`; it has not been independently recalculated from sentence predictions. It is included for the four-team Japanese subtask submission-count summary (median 1.5, range 1–94), and remains outside the three-team primary Table 5 ranking.

## Other manuscript items

- Figures 1 and 2 are explanatory illustrations rather than recalculated model-performance analyses.
- Figures 3–6: their leaderboard values are reproduced; original plotting source and pixel-level figure reconstruction are not included.
- Table 1's report, case and sentence counts are recalculated; personnel experience and annotation-process descriptions are not derived from model predictions.
- Table S1 keywords and Table S3 system descriptions are curated materials, not automatic calculations.
- Table S4 is a historical leaderboard presentation, not an exhaustive submission archive. The registry and whole historical matrix-run scores are available here, but exact reconstruction of all S4 display rows and their raw identities is not asserted.
- Table S12 interannotator agreement requires independent annotator labels that are not present in the released consensus gold files. Its kappa values are not recalculated here.
- The code reproduces archived predictions and aggregate analyses. It does not rerun participants' LLM/API inference or recover their undisclosed prompts, model snapshots or inference dates.

## Statistical limitations

With only nine paired English/Japanese test cases, mean-difference intervals are imprecise and method-sensitive. The T paired bootstrap CI excludes zero; its paired t CI includes zero and the Wilcoxon P value is .125. The agreement of recomputed values does not establish language equivalence, independence of translated corpora or clinical utility.

Test-based LB selection is retained for descriptive competition analyses, with runs fixed during bootstrap. The code does not correct the CIs for selection or ranking uncertainty. Case-cluster resampling cannot compensate for having few independent cases.

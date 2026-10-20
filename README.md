# RadNLP 2024 manuscript analysis code

This package reproduces the revised RadNLP 2024 overview analyses for JMIR Medical Informatics manuscript 107996. It separates leaderboard-selected runs from validation-selected representatives and compares fresh calculations with frozen final-analysis reference results.

The core comparisons have passed in Python 3.12.14. The Japanese submission-frequency duplicate-record discrepancy has been resolved. The tumor-size proxy counts have also been reproduced from adjudicated consensus annotations. The analysis code is distributed under the MIT License. Participant predictions and organizer annotation inputs are not distributed in this repository; they are available from the authors on request. No tagged release or DOI is assigned yet.

## Installation and execution

Use Python 3.12 and the pinned packages:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --data-dir /absolute/path/to/reproduction_inputs --output-dir outputs
```

The data directory is separate from this repository. The local organizer input bundle contains 75 label-only files and is deliberately excluded from the code archive and Git tracking. See [docs/INPUTS.md](docs/INPUTS.md) for provenance, schemas, checksums and availability. Do not substitute current Hub files without checking the frozen versions.

For a fast input check:

```bash
python run.py --data-dir /absolute/path/to/reproduction_inputs --check-inputs-only
```

To reconstruct the label-only input bundle from the archived organizer source tree:

```bash
python prepare_inputs.py --source-dir /absolute/path/to/organizer_archive --output-dir /absolute/path/to/reproduction_inputs
```

Outputs are written only beneath `--output-dir`. Each stage has a log. `verification.json` records numerical comparisons, and `execution.json` records runtime versions. Frozen `expected/` files are read only by the verification stage; analytical stages recompute their outputs from predictions and gold labels.

## What is reproduced

| Manuscript item | Program | Fresh output |
| --- | --- | --- |
| Official scoring behavior | `scripts/scoring_checks.py` | Five controlled normalization, missing/invalid label and duplicate-ID checks |
| Representative selection | `scripts/selection.py` | All 21 representative run IDs, coverage, measured validation scores and hashes |
| Tables 2 and 3 | `scripts/leaderboard.py` | Fine/coarse joint/T/N/M accuracy and case-cluster CIs for 168 estimates |
| Tables 4 and 5 | `scripts/subtask.py` | Whole-run primary subtask scores and one best Overall run per team; Japanese TMUNLPG3 is exported separately from metadata only |
| Table 7 | `scripts/cross_language.py`, `scripts/additional.py` | Paired case errors, exact Wilcoxon tests, Spearman correlations, paired mean-difference CIs |
| Tables 8 and S11 | `scripts/ensemble.py` | Plurality and oracle accuracy/CIs, report success, majority-success cases, tie checks |
| Figures 7 and 8 | `scripts/confusion.py` | Counts, denominators, percentages, PNGs and PDFs, including Other/invalid |
| Tables S5–S8 and S10 | `scripts/supplementary.py` | Macro-F1, balanced accuracy, report/case CIs, width summaries, majority success and case errors |
| Table 1 and S2 counts; Table 6 and S9 label variation | `scripts/additional.py` | Dataset counts, label frequencies, within-case reference variation |
| Subtask baseline and matched sentence counts | `scripts/additional.py` | All-Omittable scores, contradictory-reference count, English/Japanese sentence counts |
| Tumor-size proxy | `scripts/proxy.py`, `prepare_consensus.py` | 48/243, split counts, 40/48, dominant T-determining subtotal groups, and a per-report audit |
| Submission frequency | `scripts/submission_frequency.py` | Distinct-ID counts, asukai89 exception, and reconstruction of both legacy and corrected Japanese results |


## Consensus proxy inputs

Use `prepare_consensus.py --train-val TRAIN_VAL.csv --test TEST.csv --output annotations/main_consensus_proxy.csv` to extract the adjudicated labels without report text. See docs/INPUTS.md. The 20/19/1 groups compare T-determining subtotals; they are not counts of finding presence.

## Statistical conventions

- Submission frequency counts unique Submission IDs within task, language and team after filtering eligibility. Both successful asukai89 Japanese main submissions are included under the selected-flag exception. Sociocom's duplicated ID is counted once. The corrected Japanese main correlation is rho=0.4453194528540574, P=0.16987767013818195; the legacy metadata-row count reproduces rho=0.4009386883222482, P=0.22168301553096678. See `validation/duplicate_record_audit.json`.
- Descriptive leaderboard analyses choose one whole run per team by test fine joint accuracy; subtask leaderboards use test Overall F2.0. This is test-based selection, not an unbiased model-selection evaluation. Bootstrap intervals hold runs and rankings fixed.
- Representatives use designated final (`selected=true`) submissions with test predictions and highest measured validation fine joint accuracy. Ties use earliest submission time. If validation predictions are absent for all eligible runs, earliest submission wins. Measured zero is distinct from missing validation. The documented asukai89 flag exception and participant-supplied sociocom predictions are retained.
- Case-cluster bootstrap: 10,000 resamples; NumPy `default_rng` seeds 42 (English) and 43 (Japanese). All nine sibling reports remain together. Case ordering is ascending case ID. Percentile quantiles use NumPy's linear method.
- Table 7: nine paired cases, three shared teams, 27 predictions per case and language. Integer error differences preserve ties for Wilcoxon. The paired percentile mean-difference bootstrap uses seed 42; paired t intervals have 8 degrees of freedom. These CIs describe arithmetic means, separately from the signed-rank test.
- S7's width ratio is the median case-cluster width divided by the median report-level width, not the median of individual ratios.
- Macro-F1 and balanced accuracy use reference categories present in the test set. Invalid predictions remain wrong; they are not removed.
- Plurality ties use validation performance, then representative submission time. Missing validation ranks below measured validation, including a score of zero. Joint oracle requires one single team to get all three categories right for a report.
- Legacy metric keys ending in `micro_f2` are retained from the official implementation. Overall F2.0 is the mean of sentence-level seven-label F2.0 scores, not a globally pooled micro average. Omittable is evaluated independently for Inclusion.

## Licensing and citation

The analysis source code and accompanying software documentation are licensed under the [MIT License](LICENSE). This license does not grant rights to external corpus data, participant predictions or organizer annotations. See [NOTICE.md](NOTICE.md).

For the private reproduction inputs, contact corresponding author Yuta Nakamura as listed in the manuscript, or open a [data access request issue](https://github.com/yutanakamura-tky/RadNLP2024/issues/new) containing your Hugging Face username and the inputs/analyses you need. The authors will coordinate provision privately; do not attach credentials or private data to a public issue. Details are in [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

Please cite *Automated Lung Cancer Staging From English and Japanese Radiology Reports: The NTCIR-18 RadNLP 2024 Shared Task* and the exact repository commit used. Manuscript 107996 is under revision; no final journal DOI or release identifier is asserted here.

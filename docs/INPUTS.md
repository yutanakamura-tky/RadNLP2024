# Input data and provenance

The code package contains no radiology report text, raw team-member metadata, credentials or participant prediction vectors. The separate organizer input archive contains only report/sentence identifiers, case identifiers, reference labels and predictions. It is supplied for local verification and must not be uploaded automatically with the code.

## Frozen inputs

`config/input_checksums.csv` enumerates 75 paths, sizes and SHA-256 values. The path root is the `--data-dir` argument, regardless of the current working directory.

| Files | Count | Content |
| --- | ---: | --- |
| Main-task participant CSVs | 50 | Exactly `id,t,n,m`; source file names and byte hashes retained, including the sociocom combined predictions |
| Main-task historical matrices | 6 | `ID,answer` plus run columns, across T/N/M and two languages |
| Subtask historical matrices | 16 | `ID,sentence_index,answer` plus run columns, across eight labels and two languages |
| Gold-label archive | 1 | Twelve main/subtask × language × split CSVs; report/sentence text removed |
| Consensus proxy fields | 1 | 243 rows of IDs, split, consensus T/N/M, size-unknown flag and T-derivation subtotals; no report text |
| Report-to-case map | 1 | `id,case_id` only; findings, impression and radiologist fields removed |

`config/corpus_source_hashes.json` identifies the original organizer corpus archive and mapping file. `prepare_inputs.py` creates deterministic label-only derivatives. Original prediction CSVs and historical matrices are retained byte for byte. A 75-file local bundle was used in the validation, without additional organizer files or manuscript documents.

## Public versus organizer-supplied sources

The released corpus is described at:

- https://huggingface.co/datasets/RadNLP/RadNLP2024_main_task
- https://huggingface.co/datasets/RadNLP/RadNLP2024_sub_task

These are entry points, not a guarantee that a newly downloaded snapshot equals the audited archive. Preserve split, ID ordering, reference labels and the frozen hashes. The original archived corpus does not contain detailed tumor-size annotations. The subsequently supplied consensus files provide the fields required for the proxy analysis; independent annotator label sets for the IAA analysis remain unavailable.

Participant predictions and the report-to-case mapping must be supplied by the organizers in the listed layout. Participant predictions remain private and are available from the authors on request, as specified in [DATA_ACCESS.md](DATA_ACCESS.md). Sociocom uses the separately received combined validation/test CSVs, because its original platform files lacked test predictions. Validation and test rows in these files are separated by reference IDs; no test score is used to select representative runs.

The public code and private input access are separate. Full end-to-end reproduction requires obtaining the 75-file bundle from the authors, following [DATA_ACCESS.md](DATA_ACCESS.md). Private input access is not required to inspect the code, manifests or aggregate verification results.

## Manifests and identity limits

- `config/representative_candidates.csv`: audited eligible candidate inventory, submission IDs, times and selected flags. Candidates without available vectors are retained as missing; their recorded validation upper bounds are checked against the winner. This is not proof of completeness of every submission ever made.
- `config/prediction_files.csv`: all 50 audited candidate prediction variants and their original hashes.
- `config/submission_registry.csv`: metadata allowlist, limited to team/run IDs, timestamps, flags, status and scores. Free-text comments, members and account details are omitted.
- `config/leaderboard_runs.csv`: historical matrix-column inputs and ranks. An empty `exact_submission_matches` field means that raw submission identity was not established; multiple IDs mean indistinguishable vectors, not a unique match. Do not assign a raw Submission ID by matching only aggregate accuracy.
- `expected/representative_runs.csv`: frozen selections for verification. The selector independently recalculates validation scores and is compared with this file only at the final verification stage.

Example missing-file errors list exact paths. Hash mismatches are fatal. Fetch the archived file rather than silently substituting a different run, current dataset version or inferred prediction.

## Consensus annotation derivative

`annotations/main_consensus_proxy.csv` is generated using `prepare_consensus.py` from the two supplied adjudicated consensus CSVs, read with their second row as the column header. The original file hashes are in `config/consensus_source_hashes.json`. Retained columns are `id,partition,case_id,t,n,m,size_unknown,size_t,extension_t,intrapulmonary_t`. The source columns are ID/partition/case/T/N/M, サイズ不明, 小計_T_原発巣サイズ, 小計_T_進展範囲浸潤 and 小計_T_肺内転移. All report text, quoted findings, notes and annotator identifiers are omitted. Generate this derivative in the organizer source tree before running `prepare_inputs.py`. The derivative is bundled only with the separate local inputs, not with public code.

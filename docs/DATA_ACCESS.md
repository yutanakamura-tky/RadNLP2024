# Access to reproduction inputs

The analysis code is public under the MIT License. Participant-submitted predictions remain in private Hugging Face repositories and are available from the authors on request. Organizer inputs, including the report-to-case mapping and label-only consensus derivative used for the proxy analysis, can be requested with the reproduction bundle.

## How to request access

Contact corresponding author Yuta Nakamura using the contact details in the manuscript. Alternatively, open a data access request at https://github.com/yutanakamura-tky/RadNLP2024/issues/new with your Hugging Face username and the inputs or analyses you wish to reproduce. The maintainer will coordinate provision of the required files privately. Do not include access tokens, report text or private prediction files in the public issue. An issue initiates a request; it does not itself grant repository access.

The authors can supply a label-only reproduction bundle containing the 75 files enumerated in config/input_checksums.csv. It contains report/case/sentence IDs, reference labels, participant predictions and the consensus proxy fields; radiology report text, free-text excerpts and member/account details are excluded. Preserve the supplied directory layout and file bytes, then run:

```bash
python run.py --data-dir /absolute/path/to/reproduction_inputs --output-dir outputs
```

The openly released main/subtask corpus entry points are listed in docs/INPUTS.md. Those public datasets alone do not include all private predictions or organizer inputs needed to reproduce every analysis. Input access and the software license are separate: MIT does not grant redistribution rights for these external data.
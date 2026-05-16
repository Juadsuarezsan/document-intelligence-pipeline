# FUNSD sample

5 anonymized annotation JSON files from the FUNSD dataset (form understanding
in noisy scanned documents). The full dataset is 199 documents (~17 MB) and
lives at https://guillaumejaume.github.io/FUNSD/dataset.zip.

## Schema

Each `.json` file follows FUNSD's `form` schema:

```json
{
  "form": [
    {
      "id": 0,
      "text": "Annual Report",
      "box": [123, 45, 789, 80],
      "linking": [],
      "label": "header",
      "words": [{"text": "Annual", "box": [...]}]
    },
    ...
  ]
}
```

Labels in FUNSD: `header`, `question`, `answer`, `other`.

## Reproducing the full set

```bash
curl -L -O https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip
# 149 training + 50 test forms with annotations and images
```

The 199-form full set was used to measure field-level F1 in the project
report. The 5 samples here are enough to demonstrate the parser pipeline
without checking 17 MB of images into git.

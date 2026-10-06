# Arcil Doc Intel

Privacy-first, offline tooling for preparing an authorized Aadhaar document-detection dataset. The current repository contains the canonical dataset-preparation pipeline, configuration, and tests. It does **not** contain Aadhaar images, source labels, OCR transcripts, QR payloads, identity data, generated crops, dataset manifests, or model checkpoints.

## Current scope

- Safely parses normalized YOLO bounding boxes and polygon annotations.
- Converts polygons to canonical detection boxes while retaining provenance locally.
- Maps source classes to `FULL_UID`, `MASKED_UID`, and `QR` without silently merging UID-redaction semantics.
- Groups exact, perceptual, lineage, annotation-similar, and optionally HMAC-linked variants.
- Produces deterministic, group-aware train/validation/test splits.
- Performs integrity and cross-split leakage checks before training.

Model training and production inference are intentionally outside the current implementation scope.

## Local setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dataset.txt
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Run the pipeline against the authorized local dataset layout:

```powershell
python dataset_pipeline.py --config config/dataset_pipeline.json
```

Use `python dataset_pipeline.py --help` for supported path overrides. All dataset artifacts remain local under ignored paths. Review authorization and data-handling policy before processing real identity documents.

## Security and privacy

- Keep the repository private unless a separate privacy review approves publication.
- Never commit raw documents, annotations, OCR output, QR payloads, manifests, identity secrets, or trained weights.
- Do not place HMAC keys in configuration or source control; provide them through the local secret mechanism.
- Rotate credentials immediately if a sensitive artifact is committed, even if the commit is later rewritten.

No open-source license is granted by this repository.

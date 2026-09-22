# SkinAnalytica

7-class dermoscopy lesion classification (mel, nv, bcc, akiec, bkl, df,
vasc) — an ensemble of EfficientNetV2-S, ViT-L/16, and ConvNeXt-Large,
served via FastAPI. Research prototype, not a clinical diagnostic device
— see [docs/MODEL_CARD.md](docs/MODEL_CARD.md) before relying on any
number in this file.

## Architecture

Three-model ensemble: EfficientNetV2-S + ViT-L/16 + ConvNeXt-Large,
combined via a learned weighted average (`ensemble_weights.json`) and
temperature-scaled calibration (`temperature.json`).
Preprocessing: hair removal (blackhat + inpaint) + resize + ImageNet
normalize (`src/inference_utils.py::preprocess_bgr`) — the single source
of truth every inference call site (API, agents, scripts) must route
through.

## Notebooks

The three per-model notebooks are what actually get run, **not**
`SA01_Model_Training.ipynb`'s `MODEL_NAME` toggle — see
[docs/RETRAIN_PLAN.md](docs/RETRAIN_PLAN.md) for why (the toggle
notebook exists and is correct, but the three separate files below are
the ones with the current data-mixing/age-rebalancing/augmentation
pipeline already applied).

| Notebook | Purpose |
|---|---|
| `notebooks/SA01_EfficientNet_Train.ipynb` | Train EfficientNetV2-S |
| `notebooks/SA01b_ViT_Train.ipynb` | Train ViT-L/16 |
| `notebooks/SA01c_ConvNeXt_Train.ipynb` | Train ConvNeXt-Large |
| `notebooks/SA02_Ensemble.ipynb` | Combine models, calibrate, export |
| `notebooks/SA03_Explainability.ipynb` | Grad-CAM++, TTA robustness |
| `api/SA05_api.py` | FastAPI endpoints (the deployed service) |

## Datasets

Real local layout is `data/`, not `datasets/` — the folder names below
are the ones `scripts/build_training_pool.py` and `scripts/
dataset_loaders.py` actually read from:

```
data/
  ISIC_2018/            <- ISIC 2018 Task 3
  ISIC_2019/             <- ISIC 2019 Training Input + GroundTruth + Metadata
  ISIC_2020/             <- ISIC 2020 Training JPEG + GroundTruth v2
  ISIC_2024/             <- SLICE-3D Permissive (dermoscopy validation)
  ISIC_2024_full/         <- SLICE-3D full non-Permissive release (CC-BY-NC)
  EMB/                   <- Early-Stage Melanoma Benchmark (CC BY 4.0)
  milk10k/                <- MILK10k (CC-BY-NC, non-commercial research only)
  mra_midas/              <- MRA-MIDAS (Stanford AIMI, non-commercial DUA)
  pad_ufes_20/, ddi/       <- clinical-photo fairness datasets (not dermoscopy)
```

See [docs/KNOWN_GAPS.md](docs/KNOWN_GAPS.md) for exact download sources,
and the licensing note there — MILK10k and MRA-MIDAS are both
non-commercial-only, which blocks any future commercial deployment
without retraining around them.

## Run order (retrain)

Night 1: `SA01_EfficientNet_Train.ipynb`
Night 2: `SA01b_ViT_Train.ipynb`
Night 3: `SA01c_ConvNeXt_Train.ipynb`
Day 4: `SA02_Ensemble.ipynb` → `SA03_Explainability.ipynb`

## API

```bash
uvicorn SA05_api:app --app-dir api --host 0.0.0.0 --port 8001
```

All patient-data endpoints require an `X-API-Key` header
(`SKINANALYTICA_API_KEY` env var, fails closed — 503 — if unset). See
[docs.html](docs.html) for the full endpoint reference, or the live
Swagger UI at `/docs` once the server is running.

```
POST /analyze                          - single image
POST /analyze/batch                     - multiple images
GET  /report/{scan_id}                  - fetch a prior result
GET  /report/{scan_id}/fhir             - FHIR R4 export
GET  /patient/{patient_id}/history       - a patient's scan history
GET  /audit/log                         - recent verdicts, no patient identifiers
GET  /health, /models                    - status, no auth required
```

## Frontend

Three static pages, no build step:

```bash
python -m http.server 8080
```

| Page | Purpose |
|---|---|
| `index.html` | Analyze — upload a dermoscopy image, get a verdict |
| `docs.html` | Written endpoint reference + known limitations |
| `status.html` | Live health check, model metrics, Swagger/ReDoc links, an in-browser API-key tester |

The frontend holds its API key client-side only (`assets/auth.js`,
`localStorage`) — same model as Swagger's own "Authorize" button, never
embedded in shipped page source.

## Testing

```bash
pip install pytest PyYAML lightgbm scikit-learn scipy
pytest tests/ -v
```

125 tests as of this writing, split into two kinds:
- **Pure-logic tests** (inference math, verdict/threshold selection, auth,
  deployment config, config consistency, agents, delivery, assistant) —
  always run, no model files needed. These exist because this codebase
  has shipped real, silent bugs before — a double-softmax step, a
  `MODEL_MODE` default serving the wrong backbone, a `_detect_base()`
  path check that broke local runs, a checkpoint-resume bug that silently
  skipped an entire retrain — each now has a regression test specifically
  because it was caught by manual review once already. See
  [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
- **Model-dependent tests** (`tests/test_gpu_inference.py`) — skip
  gracefully when the large checkpoint files aren't present (gitignored,
  not part of a fresh checkout/CI).

**Frontend tests** (`tests_js/`, 19 tests) — the real logic in
`assets/auth.js` (client-held API key handling) and `assets/nav.js`
(mobile nav toggle, the only way to reach other pages below 720px). Node's
built-in test runner against a hand-rolled DOM stub (`tests_js/dom_stub.js`)
run the real shipped scripts via `node:vm` — deliberately not jsdom, to
avoid adding a dependency this project's small JS surface doesn't need.

```bash
npm test
```

## Validation & bias/fairness tooling (scripts/)

Re-run after any retrain to refresh [docs/MODEL_CARD.md](docs/MODEL_CARD.md):

```bash
python scripts/score_dataset.py --dataset isic2020 --output outputs/bias_reports/isic2020_scored.csv
python scripts/score_dataset.py --dataset milk10k_holdout --output outputs/bias_reports/milk10k_holdout_scored.csv
python scripts/score_dataset.py --dataset slice3d_full_holdout --output outputs/bias_reports/slice3d_full_holdout_postretrain_scored.csv
python scripts/score_dataset.py --dataset mra_midas --output outputs/bias_reports/mra_midas_scored.csv
```

`score_dataset.py` uses GPU inference (`src/gpu_inference.py`, real
PyTorch checkpoints) when CUDA is available — 13-21x faster than CPU
ONNX — falling back to CPU ONNX otherwise; both paths produce identical
numbers (`tests/test_gpu_inference.py`). This is a *research/analysis*
accelerator only — the deployed API has no GPU and serves from ONNX
regardless.

## Path B — metadata/GBDT fusion (research, not deployed)

`scripts/extract_embeddings.py` → `scripts/join_fusion_features.py` →
`scripts/train_fusion_gbdt.py` → `scripts/evaluate_fusion.py` build and
evaluate a LightGBM fusion model on top of the CNN ensemble's embeddings.
Evaluated against three independent external holdouts (SLICE-3D,
MILK10k, MRA-MIDAS) — mixed, honestly-reported results, not deployed.
See [docs/RETRAIN_PLAN.md](docs/RETRAIN_PLAN.md)'s Path B section and
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) findings #17-#22.

## Current status

Real, verified numbers as of this writing (see
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for full evidence and caveats
on every one of these):

- Blended internal validation: MelAUC ~0.95, near-training-distribution
  automation rate ~94.6%.
- Genuine external validation (SLICE-3D, 143-case leakage-free holdout):
  AUC 0.6131 (statistically significant, real signal — finding #12), but
  the deployed threshold under-delivers sensitivity there without a
  population-conditional adjustment (Path A, done).
- MILK10k (real external dermoscopy, skin-tone-diverse): MelAUC 0.8796,
  no confirmed skin-tone AUC disparity on adequately-sampled bands
  (finding #14).
- MRA-MIDAS (third independent institution): discrimination collapses
  toward chance (findings #20-#22) — root cause diagnosed (image
  framing/resolution mismatch vs. training data, not a metadata problem)
  and partially, honestly mitigated (center-crop preprocessing, modest,
  not fully CI-confirmed improvement).
- Full authentication on all patient-data endpoints (finding #13).
- 83 automated tests, 3-for-3 coverage on every deployment bug class
  that's previously shipped silently.

## Known issues

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) (performance, 22 numbered
findings with evidence) and [docs/KNOWN_GAPS.md](docs/KNOWN_GAPS.md)
(blocked items, dataset sourcing decisions, GPU setup notes) before
relying on any headline metric here — the blended validation numbers
hide real, substantial per-subgroup and per-institution gaps, documented
in detail rather than smoothed over.

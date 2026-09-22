# Known gaps — status as of 2026-07-20 (see 2026-07-25 update note below)

Items from the "10/10 models" review. Downloaded/resolved items are marked
done with results; still-open items are listed below that.

**2026-07-25 update**: the SLICE-3D external-validation gap referenced
throughout this document (AUC 0.593) has moved since this file was last
edited — see `docs/MODEL_CARD.md` findings #11 and #12, and
`docs/RETRAIN_PLAN.md`'s "Path A / Path B" section. Short version: a full
retrain plus a larger (n=143) leakage-free re-test found a real,
statistically significant discrimination improvement (AUC 0.6131,
p=5.2e-06), and a separately-confirmed threshold-transfer problem was
partially addressed (Path A, done) with a bigger fix still open (Path B,
not started). This file's SLICE-3D-related sections below are otherwise
still accurate as historical record of the original finding; treat the
model card as the current source of truth for where that number stands
today.

## Downloads: done

All three datasets below were downloaded, verified (`scripts/inspect_dataset_schema.py`),
and scored against the production ensemble. Results are in
`docs/MODEL_CARD.md` findings #5 and #9.

| Dataset | Status | Location | Result |
|---|---|---|---|
| **PAD-UFES-20** | Scored (2,298 images) | `data/pad_ufes_20/` | MelAUC 0.778 — see model card #9 |
| **DDI** | Scored (656 images) | `data/ddi/` | MelAUC 0.629 — see model card #9 |
| **ISIC 2024 SLICE-3D Permissive** | Scored (2,794-image stratified sample) | `data/ISIC_2024/` | **AUC 0.593** — see model card #5, the most severe finding this session |

~~PH2 database~~ — **dropped**, not pursued. The official link (fc.up.pt/addi)
redirects to a deleted Terabox share, and PH2's license explicitly states
*"redistribution and commercial use is not allowed"* — even a working mirror
would be legally unusable here. SLICE-3D Permissive supersedes it.

## Downloads: in progress (2026-07-22) — closing the skin-tone-fairness and thin-lesion gaps

Sourced to directly address the two open items from `MODEL_CARD.md` findings
#5 (thin/early-stage lesions) and #9 (skin-tone fairness unanswered on
dermoscopy specifically, since PAD-UFES-20/DDI turned out to be clinical
photography, not dermoscopy):

| Dataset | Targets | Download | License |
|---|---|---|---|
| **EMB** (Early-Stage Melanoma Benchmark, 1,104 images, Breslow/T-category labels) | Thin/in-situ melanoma — the confirmed dominant SLICE-3D failure mechanism | [github.com/Oichii/EMB](https://github.com/Oichii/EMB) | CC BY 4.0 — no restriction |
| **MILK10k** (10,480 images / 5,240 cases, dermoscopic + clinical pairs, 6-level skin-tone annotation) | Skin-tone fairness *on dermoscopy* — first dataset found this session that actually has both | [ISIC S3 direct links](https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_Input.zip) | **CC-BY-NC — non-commercial only** |
| **MRA-MIDAS** (Stanford/Melanoma Research Alliance, paired dermoscopic+clinical, biopsy-proven, skin-tone diverse) | Skin-tone fairness, second corroborating dermoscopy+skin-tone source | [aimi.stanford.edu/datasets/mra-midas](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer) → Redivis | **Non-commercial research DUA — registration required** |

**Licensing decision (2026-07-22)**: SkinAnalytica is currently research/academic
work, not a commercial deployment, so training on the two non-commercial-licensed
sets above (MILK10k, MRA-MIDAS) was approved for now. **If this project ever
moves toward commercial deployment, this decision must be revisited** —
MILK10k (CC-BY-NC) and MRA-MIDAS (non-commercial DUA) would both need to be
dropped from the training mix and any model trained on them retired/retrained
without them, the same issue that got PH2 dropped originally. EMB (CC BY 4.0)
has no such restriction and is unaffected by a future commercial pivot.

**PH2 mirror**: a Kaggle mirror of the full 200-image PH2 set was located
locally (`C:\Users\tejan\Downloads\archive (1)\PH2Dataset`) this session. Not
integrated — the mirror doesn't change PH2's original license terms (research/
educational only, no redistribution/commercial use), and at only 200 images
with no melanoma subtyping it adds little over SLICE-3D Permissive, which
already supersedes it for external validation. Left out entirely, consistent
with the original decision above.

**MRA-MIDAS requires manual action**: unlike EMB/MILK10k, this dataset sits
behind a Stanford data-use-agreement + registration step on Redivis that
can't be automated — someone needs to sign up at the link above before the
files can be downloaded.

## EMB Atlas-sourced images — decided (2026-07-27): not pursued

EMB's 1,104 images split into two sources per
`data/EMB/_repo/early_melanoma_benchmark_dataset_labels.csv`: 907 `"ISIC"`
and 197 `"Atlas"`. The ISIC-sourced 907 are downloaded (`scripts/
download_emb_isic.py`, via the public ISIC API, per-image CC-0 license
logged to `data/EMB/isic_license_log.csv`) and are the images actually in
the training pool. The remaining 197 `"Atlas"` images require running
EMB's own scraper (`data/EMB/_repo/wed_scraping.py`) against
dermoscopyatlas.com.

**Checked dermoscopyatlas.com's own terms before running anything**: the
site states "The images may be used for self education only. Any other
use including in any other medium requires the permission of the Editors.
The images are copyright." EMB's CC BY 4.0 license covers their labels
and curation, not the underlying images they scraped from a third-party
site whose terms explicitly prohibit redistribution/dataset use without
separate permission from the editors — nothing in the EMB repo indicates
that permission was obtained.

**Decision: not pursued.** For a 197-image addition (17.8% of EMB, and
EMB itself contributed only 770 of 73,456 training images) the source
site's terms are an unambiguous blocker, not a marginal call — running
the scraper would mean redistributing copyrighted images in direct
contradiction of the copyright holder's stated terms. This item is
closed, not open; revisit only if the Dermoscopy Atlas editors are
contacted directly and grant explicit permission.

## MRA-MIDAS — in progress (2026-07-27)

Registration/DUA signup on Redivis is underway
(https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer).
Once access is granted, a download/integration script (following the same
pattern as `scripts/dataset_loaders.py`'s other loaders) still needs to be
written — not started, since there's nothing to point it at yet.

## Partially resolved (2026-07-26): skin-tone fairness on dermoscopy specifically

Both PAD-UFES-20 and DDI turned out to be clinical/smartphone photography,
not dermoscopy (see model card #9) — so a skin-tone-labeled dermoscopy
dataset was sourced instead (MILK10k) and its reserved 700-image holdout
was finally scored — see model card finding #14. Result: overall MelAUC
0.8796 (95% CI [0.834, 0.921]), no statistically confirmed AUC disparity
across the three adequately-sampled skin-tone bands (2/3/4 of 6) — but the
darkest (n=1) and lightest (n=51, only 2 melanoma) bands remain
unanswerable from available data, and sensitivity-at-threshold shows a
spread (55.6%-76.9%) not yet confirmed as real vs. small-sample noise.
**Still open**: more data in the extreme skin-tone bands would be needed
to fully close this; DDI requires a data use agreement if pursued further
for other purposes: https://ddi-dataset.github.io/ (Stanford) — though
it's clinical photography, so it wouldn't help this specific gap.

## GPU: usable for analysis, not for the deployed API

A local NVIDIA RTX 4060 was found and put to use this session — 13-21x
faster than CPU for scoring/validation runs (via PyTorch loading the real
`.pth` checkpoints directly, see `src/gpu_inference.py`). This is **not**
the same as fixing GPU inference in the deployed API (Render's free tier
has no GPU at all regardless of anything below).

- **`onnxruntime-gpu`'s CUDA execution provider — fixed for local use.**
  Root cause was never actually the DLLs being absent — it was that Windows
  Python 3.8+ doesn't honor the `PATH` env var for a native extension's own
  internal `LoadLibrary` calls to find ITS dependencies. Three different
  `onnxruntime-gpu` versions were tried (1.27.0 wanting CUDA 13.x/cuDNN 9.x,
  1.18.1 wanting CUDA 11/12/cuDNN 8.x, 1.20.1 wanting CUDA 12.x/cuDNN 9.x —
  the one actually satisfiable here) and all three failed identically with
  "Error 126: module not found" until `os.add_dll_directory()` was used
  instead of `PATH` — then 1.20.1 worked immediately. Fix is now a reusable
  helper, `inference_utils.enable_onnx_cuda_dll_dirs()`, called by
  `scripts/score_dataset.py`'s ONNX backend automatically (no-op if the
  paths don't exist, safe on any machine). Requires `onnxruntime-gpu==1.20.1`
  specifically pinned locally (not `onnxruntime`, and not the newer 1.27.0)
  — this is a local dev-environment package choice, not something reflected
  in `requirements.txt` (which correctly stays on the plain CPU `onnxruntime`
  package for the actual Render deployment).
- What GPU access enabled this session: re-scoring PAD-UFES-20 (2,298 img)
  in ~12 min instead of ~29, DDI (656 img) in ~1.6 min instead of ~7.6,
  SLICE-3D (2,794 img) in ~2.4 min instead of ~33 — this is why this
  session could afford to score three separate datasets in one sitting.

## Blocked on sustained compute: full retraining

- **K-fold / repeated-seed retraining**: current metrics (`ensemble_metrics.json`)
  come from a single train/val split, single seed (42). Getting real confidence
  intervals on AUC/sensitivity requires retraining each of the 3 backbones
  multiple times — the "3 nights per model" from the README. The GPU found
  this session accelerates *inference*, but a full multi-epoch retraining run
  (the actual SA01 training notebooks, not just scoring) was not attempted —
  different order of magnitude of compute/time than anything done here. What
  was done instead: bootstrap-resampled the *existing* scored predictions to
  get a CI on the current model's metrics — that bounds sampling noise, not
  training-process variance.

- **Full re-training to fix the age-based sensitivity gap**: the root cause
  (far fewer melanoma-positive training examples for patients under 45 — see
  age distribution analysis in the model card) needs either targeted
  resampling/reweighting or additional young-patient melanoma data, followed
  by retraining. Diagnosed and mitigated with age-conditional thresholds
  (see model card), but not fixed at the training level.

- **Disentangling the SLICE-3D external-validation collapse — substantially
  resolved.** (model card #5): actively investigated this session rather
  than left as an open question.
  - Institution effect: 3 of 4 institutions (94% of the sample) cluster in
    the same poor AUC range — a broad, consistent collapse, not one
    anomalous population. The 4th (Athens, n=6 malignant) initially looked
    much better by AUC but didn't hold up under a threshold-based check
    (real sensitivity 16.7%, same as everywhere else) — that was a
    small-sample AUC artifact, confirmed and closed out, not a real
    institution effect.
  - Case-severity effect: confirmed with data. SLICE-3D's malignant cases
    are overwhelmingly thin/early-stage (mean Breslow thickness 0.49mm;
    melanoma in-situ flagged only 6.7% of the time). This is consistent
    with SLICE-3D being screening-detected (subtle, early lesions) vs. the
    training data's clinically-referred population (more advanced/obvious
    lesions) — a real, independently actionable mechanism.
  - Still not separated: how much of the collapse is case-severity vs. the
    3D-total-body-photography capture modality itself (both likely
    contribute). Would need a same-institution dermatoscope-vs-TBP paired
    comparison, or a retraining ablation, to fully separate these — not
    attempted, and lower priority now that case-severity has a confirmed,
    actionable data-mixing fix regardless of the exact split.

## Train/serve ensembling-method mismatch — practical impact fixed, root cause still open

`notebooks/SA02_Ensemble.ipynb` uses **two different ensembling formulas**
in two different places:

- **Cell 10** (derives `temperature.json` and the original
  `ensemble_metrics.json` — the "official" 0.9609 AUC / 0.8007 sensitivity
  numbers): per-model softmax *without* temperature → recover log-probs →
  weighted-average in **log space** → apply temperature once → one softmax.
  This is geometric/log-linear pooling.
- **Cell 14, `EnsembleModel`** (what's actually exported to
  `skinanalytica_ensemble.onnx` and served in production): per-model softmax
  *with* temperature individually → weighted-average in **probability
  space**. This is arithmetic pooling.

**Practical impact, now measured and fixed**: recomputed a self-consistent
`ensemble_metrics_selfconsistent.json` using the real arithmetic-pooling
formula against the full 10,312-image validation split (922 melanomas).
Overall discriminative performance barely moved (MelAUC 0.9609 → 0.9540,
sensitivity pinned at ~80% either way) — but the **threshold** that
actually delivers 80% sensitivity against the real deployed computation is
**0.2385, not 0.312**. The deployed `MEL_THRESHOLD` was updated to 0.2385
(api/SA05_api.py, config/skin_config.yaml, both kept in sync by
`tests/test_config_consistency.py`). One side effect worth knowing: the
60-75 age band's own threshold (0.2725, independently derived on ISIC-2020
data — unaffected by this correction) now sits *above* the corrected global
value — that's real data (that age band's melanoma scores are more
separable than the population average), not a bug; see
`tests/test_verdict_logic.py`'s comment for why a test initially flagged
this as suspicious and was fixed rather than the threshold.

**Still open**: the notebook itself still contains both formulas — Cell 10
hasn't been edited to match Cell 14. `docs/RETRAIN_PLAN.md` Step 0 covers
fixing this before any future retrain, so the *next* `ensemble_metrics.json`
this project produces is self-consistent from the start instead of needing
a separate correction pass like this one.

## Shipped, but with a real caveat: center-crop preprocessing (2026-07-28)

`src/inference_utils.py`'s `preprocess_bgr()` — the single source of truth
used by the live `/analyze` endpoint, `scripts/score_dataset.py`, and
`scripts/extract_embeddings.py` — now center-crops to a square before
resizing, per `docs/MODEL_CARD.md` finding #24. This was validated on two
external holdouts (MRA-MIDAS: real gain; MILK10k: negligible cost) before
shipping, and `api/SA05_api.py`'s previously-duplicated preprocessing was
replaced with a direct import so it can't drift from this again.

**Update 2026-07-28 (finding #25)**: threshold re-derivation attempted for
all four families. `AGE_BAND_THRESHOLDS` and `SKIN_TONE_THRESHOLDS` were
cleanly re-derived (exact same populations as the originals, re-scored
under center-crop) and updated in `api/SA05_api.py`/`config/skin_config.yaml`.
`TBP_MEL_THRESHOLD` was confirmed exactly unchanged (SLICE-3D crops are
already square — center-crop is a no-op there).

**Resolved 2026-07-28 (finding #26): `MEL_THRESHOLD`.** The original
10,312/922-image validation set it was calibrated against was never a
saved, reusable manifest — it was computed ad hoc in a prior session and
no longer exists on disk. A first substitution attempt (`config/val.csv`
scored only under new preprocessing, compared against the old population's
0.2385) produced a misleading ~2x jump that did NOT survive a
same-population control check: scoring `val.csv` under both old and new
preprocessing gave 0.5447 → 0.5010, a real but small ~8% decrease, not a
2x increase — that apparent jump was a population-mismatch artifact, not
a center-crop effect, and was correctly not shipped at the time.

Rather than leave this permanently stuck against an unrecoverable
population, `config/val.csv` is now formally adopted as *the* documented,
reproducible validation manifest for `MEL_THRESHOLD` going forward, and
the threshold set to what it actually takes to reach 80% sensitivity on
that population under center-crop preprocessing: **0.5010** (was 0.2385),
with specificity improving 94.2% → 96.5% at the same sensitivity target.
This is a real, visible production behavior change, not a rounding fix —
see `docs/MODEL_CARD.md` finding #26 and
`models/production/ensemble/ensemble_metrics_selfconsistent.json` (old
values preserved under `superseded_values`).

## Decided (2026-07-28): production uncertainty quantification

**Resolved, finding #28.** MC-dropout is correctly implemented in
`SA03_Explainability.ipynb` using the real PyTorch checkpoints, but
`requirements.txt`/`render.yaml` never install PyTorch in production, and
the deployed API only loads ONNX exports (dropout baked to identity at
export time) — MC-dropout literally cannot run in the deployed `/analyze`
endpoint without adding PyTorch + checkpoints to the production footprint.
A cheaper ONNX-only alternative (inter-model disagreement across the 3
existing ONNX sub-models) was prototyped and works with zero new
dependencies — but on the one real false-negative case tested, all 3
backbones agreed confidently and were all wrong, so neither approach would
have caught that specific error.

**Decision, put to the user rather than guessed at**: ship the ONNX-only
disagreement signal, don't add PyTorch. `api/SA05_api.py`'s
`ENABLE_UNCERTAINTY` now defaults to `true`. **But** the live
`skinanalytica-api` Render service is explicitly pinned to `false` in
`render.yaml` — `_load_disagreement_backbones()` loads the fp32 individual
backbone files (~2.1GB combined) on top of the primary model (~524MB int8
ensemble), a ~2.6GB+ footprint that would almost certainly OOM-crash a
free-tier deployment. So this is a real behavior change for local dev and
any adequately-provisioned future deployment, but a deliberate no-op for
the one live constrained service today. Actually enabling it there would
need a larger Render plan or switching the disagreement backbones to the
existing int8 files (`onnx_int8/`, ~523MB combined) — neither done, a
tracked follow-up if ever prioritized.

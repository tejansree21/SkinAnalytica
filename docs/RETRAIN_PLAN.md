# Retraining plan

This is the concrete plan for the one thing that actually moves the model
itself toward "best product," rather than patching around it at the
threshold/API layer. See [MODEL_CARD.md](MODEL_CARD.md) for the full
evidence behind each item.

## The goal, stated precisely

**Match or beat published benchmark numbers where we can measure against
them** — not "best in market" (that requires regulatory clearance and
prospective clinical trials no amount of retraining substitutes for, see
the conversation this doc came out of), but real, sourced, comparable
performance numbers:

| Benchmark | Source | Number to match/beat |
|---|---|---|
| ISIC 2024 SLICE-3D competition winners | Kaggle leaderboard, same external dataset we measured 0.593 AUC on | AUC **0.9704** |
| Haenssle et al. 2018 (Annals of Oncology) — CNN vs. 58 dermatologists | Landmark academic benchmark | 86.6% sensitivity / 82.5% specificity |

**But the number is not the point — what the number is FOR is.** The
product's actual job is helping human annotators automate the classification
workload: correctly auto-resolving the clear-cut cases so annotators spend
their time on the genuinely ambiguous ones, not re-checking everything the
model already gets right. Every metric in Step 4 below is reported
alongside an **automation-rate** figure (what fraction of cases the
verdict logic can safely auto-resolve as `NORMAL`/`CANCER_FLAGGED` vs. how
many still need `REVIEW_REQUIRED`) — a model that hits a great AUC but
still routes half its cases to human review hasn't actually helped the
annotators much. Chasing the benchmark number and serving the annotation
workflow are the same goal here, not competing ones: better real
discrimination is what lets more cases be auto-resolved safely.

## Which notebooks to actually run

Use the three separate per-model notebooks, not `SA01_Model_Training.ipynb`'s
`MODEL_NAME` toggle — simpler to run three notebooks once each than to
re-run one notebook three times and remember to rename outputs in between:

- **`notebooks/SA01_EfficientNet_Train.ipynb`** — Night 1
- **`notebooks/SA01b_ViT_Train.ipynb`** — Night 2
- **`notebooks/SA01c_ConvNeXt_Train.ipynb`** — Night 3

**Important history on this**: these three files originally diverged.
`SA01_EfficientNet_Train.ipynb` was a completely different, older pipeline
— pre-built CSV loading (`config/train.csv`), a different model class with
a multi-layer Sequential head (incompatible with the real saved checkpoints
and everything built this session — `src/gpu_inference.py`, the ONNX export,
serving code all assume the single-Linear-head `SkinClassifier` class), a
different output path (`models/efficientnetv2s/` instead of
`models/checkpoints/tf_efficientnetv2_s/`), no hair-removal preprocessing,
and no JPEG augmentation. `SA01b_ViT_Train.ipynb` and
`SA01c_ConvNeXt_Train.ipynb` were already correct clones of the same
template as `SA01_Model_Training.ipynb` (matching `SkinClassifier`), just
with `MODEL_NAME` hardcoded instead of toggled.

**Fixed this session**: `SA01_EfficientNet_Train.ipynb`'s content was
replaced entirely with the corrected template (now matches `SkinClassifier`,
the real checkpoint path, and has all of Steps 0-2 below applied). Steps 1-2
(SLICE-3D mixing + severity-stratified holdout, age-aware sampler, JPEG
augmentation) were ported into `SA01b_ViT_Train.ipynb` and
`SA01c_ConvNeXt_Train.ipynb`'s Cells 6/8/9 — verified all three now have
identical, correct logic in those cells, differing only in `MODEL_NAME` and
whatever hyperparameters are specific to that architecture. All three
syntax-checked cell-by-cell.

`SA01_Model_Training.ipynb` (the toggle version) still exists and is also
correct, but isn't the one to run given the three separate files above are
now fixed and available.

## Status: notebooks corrected, training not yet run

`notebooks/SA01_Model_Training.ipynb` and `notebooks/SA02_Ensemble.ipynb`
have been edited to implement Steps 0-2 below — the actual training runs
(Step 3) have **not** been executed and are expected to be run separately.
What changed, concretely:

- **`SA02_Ensemble.ipynb` Cell 10**: now uses arithmetic pooling matching
  the deployed `EnsembleModel` (Step 0 — done).
- **`SA01_Model_Training.ipynb` Cell 6**: sample tuples extended from
  4-tuple to 5-tuple (added `age_approx`, joined from each dataset's real
  metadata — ISIC19 via a separate metadata CSV, ISIC20 inline, ISIC18
  unavailable locally so recorded as `None`). Added `build_slice3d_samples()`,
  which loads SLICE-3D Permissive's malignant cases only (mel/bcc/akiec,
  not its ~217K benign pool — see the function's docstring for why),
  reserves a severity-**and**-institution-stratified 15% holdout (saved to
  `outputs/slice3d_external_holdout_ids.csv`, never trained on), and mixes
  the rest into `ALL_SAMPLES`. **Verified against the real local data**:
  250 training + 44 holdout = 294 total malignant cases, exact match,
  correct stratification (Step 1 + first half of Step 2 — done).
- **`SA01_Model_Training.ipynb` Cell 8**: added `A.ImageCompression`
  (quality 25-95, p=0.3) to the training augmentation pipeline (Step 2,
  robustness augmentation — done).
- **`SA01_Model_Training.ipynb` Cell 9**: `get_sampler_weights()` now
  upweights melanoma samples from patients under 45 by a configurable
  `young_mel_boost` factor (default 3.0x) on top of the existing
  inverse-class-frequency weighting (Step 2, age-rebalancing — done, but
  **3.0x is a starting point, not an empirically tuned value** — retune
  against validation-set age-band sensitivity once training results come in).

**Not done, and intentionally left for whoever runs training**: the actual
training runs (Step 3's pilot timing check, then the 3 full training
sessions), and Step 4's re-validation pass. This document's Steps 3-4 below
are the checklist for that.

**Found and fixed while training was running**: `SA02_Ensemble.ipynb` Cell 3
("Rebuild validation sample list") still had the *old* sample-building code —
ISIC-only, 4-tuple (no age), and its own independent split logic
(`train_test_split` over the **set** of unique image IDs, no stratification).
This is a fundamentally different split methodology than
`SA01_Model_Training.ipynb` Cell 7 (stratified-by-label split over
**indices** into a combined ISIC+SLICE-3D list) — same `random_state=42`
does not make two different split algorithms agree. Left as-is, SA02 would
have reconstructed a "validation set" that didn't match what the three
models actually held out during training, risking silent validation leakage
(some "val" images actually seen in training) and dropping SLICE-3D entirely
from the reconstructed validation pool. **This bug predates this session's
changes** — SA02's old comment claiming "same seed = same split" was never
actually true, even before any SLICE-3D work started, since the two cells
always used different split algorithms. Fixed by porting SA01's
`build_sample_list()` + `build_slice3d_samples()` + stratified index split
verbatim into SA02 Cell 3, and updating Cell 4's `SkinDataset.__getitem__`
from 4-tuple to 5-tuple unpacking to match. Verified against the real local
data: both notebooks now produce the exact same 10,309-image validation set
(byte-identical sorted ID lists). A separate, unrelated syntax error in Cell
10 (a broken multi-line f-string in the post-calibration print statement,
left over from an earlier edit this session) was also found and fixed —
it would have raised `SyntaxError` the first time a user ran that cell.

**Also fixed, no training run required**: `SA02_Ensemble.ipynb` Cell 12
("Demographic performance breakdown" — MelAUC by sex/age/anatomical site)
pointed at `datasets/ham10000/metadata/HAM10000_metadata.csv` and
`datasets/isic_2020/metadata/train.csv`, neither of which exist in this
project (`data/`, not `datasets/`, is the real data root) — so this cell
always silently printed "Metadata not available" and never actually ran.
Rewired `load_metadata()` to read the real local
`data/ISIC_2019/ISIC_2019_Training_Metadata.csv` and
`data/ISIC_2020/ISIC_2020_Training_GroundTruth_v2.csv` (both have
`age_approx`/`sex`/anatomical-site columns keyed by the same `ISIC_xxxxxxx`
IDs already used elsewhere in this notebook). Verified with synthetic
predictions against the real val_samples: 10,189/10,309 (99%) validation
images now match to real demographic metadata, up from 0.

## What's being fixed, and the evidence for each

| Problem | Evidence | Fix in this plan |
|---|---|---|
| Doesn't generalize to 3D-TBP / external institutions | SLICE-3D Permissive: AUC 0.593, sensitivity 4.8%. Institution breakdown confirms broad collapse (3/4 institutions, 94% of sample) — not one anomalous population; Athens's apparent outlier performance was a small-sample AUC artifact, not a real effect (see MODEL_CARD.md #5) | Mix TBP-style images into training |
| **Misses thin/early-stage lesions specifically** — the dominant mechanism, confirmed with data | Of 37 SLICE-3D melanomas with recorded Breslow thickness, mean is 0.49mm (clinically thin) and the model missed all but 4 regardless of thickness; melanoma in-situ flagged only 6.7% of the time. SLICE-3D is screening-detected (subtle/early), training data is clinically-referred (more advanced/obvious) | Specifically oversample/source thin, early-stage, and in-situ lesion examples — this is a sharper, more actionable target than "more TBP images" generically |
| Age-based sensitivity gap | 43.8% (<30) vs 82.0% (75+) at flat threshold; root cause: 48-103 melanoma examples for <45 vs 190+ for 60-75 in training pool | Oversample/reweight young-patient melanoma cases |
| Unstable under image-quality perturbation | 4/20 images flip class under blur/brightness alone; TTA only partially helps | Add JPEG/blur/brightness augmentation to training, not just inference-time TTA |
| Skin-tone fairness on dermoscopy unknown | PAD-UFES-20/DDI are clinical photos, not dermoscopy — didn't answer the real question | If a dermoscopy skin-tone dataset is sourced before this runs, include it; otherwise this stays open |
| Train/serve ensembling mismatch | Validation notebook uses geometric pooling, deployed graph uses arithmetic pooling | Fix the notebook to use arithmetic pooling *before* retraining, so the metrics this retrain produces are trustworthy from the start |

## Step 0 — fix the measurement before touching training ✅ done

- `notebooks/SA02_Ensemble.ipynb` Cell 10 now uses the same arithmetic
  (probability-space weighted average) pooling as `EnsembleModel.forward()`
  in Cell 14, instead of the previous geometric/log-space pooling.
- `scripts/score_dataset.py` already ran for all four datasets against the
  current model this session — those are the numbers in `MODEL_CARD.md`.

## Step 1 — reserve a genuine external holdout before mixing anything in ✅ done

**Critical**: if SLICE-3D Permissive images go straight into training, there
is no external validation set left to confirm the fix worked — the model
would just be tested on data it saw. Before mixing:
- Split SLICE-3D Permissive by holding out a random ~15% of each
  institution's images (stratified, not by institution — the earlier idea
  of holding out Athens entirely as a "different" institution doesn't hold
  up: its apparently-better performance was a small-sample AUC artifact,
  see `MODEL_CARD.md` #5, not evidence it's a meaningfully different
  population worth isolating). Never train on the holdout.
- **Additionally stratify the holdout by lesion severity** (thin/in-situ vs.
  invasive, using `mel_thick_mm`/`iddx_full`), not just by institution —
  otherwise it's easy to accidentally end up with a holdout that's
  disproportionately the easy (already-well-handled) cases, making the
  post-retrain validation look better than reality.
- This holdout becomes the new external validation set for every future
  retrain, not just this one.
- **Implemented as `build_slice3d_samples()` in `SA01_Model_Training.ipynb`
  Cell 6**, holding out ~15% of each (institution × severity) stratum —
  verified against real data: 250 training + 44 holdout = 294 total
  malignant cases, exact match.

## Step 2 — data mixing ✅ done in the notebooks (weights/factors not yet empirically tuned)

- **Thin/early-stage/in-situ lesion examples — the highest-priority
  addition**, not just "more TBP images generically." The confirmed
  mechanism (`MODEL_CARD.md` #5) is that the model misses subtle,
  screening-detected lesions specifically — mean Breslow thickness 0.49mm
  among missed SLICE-3D melanomas, in-situ cases caught only 6.7% of the
  time.
- **SLICE-3D malignant cases mixed in**: 250 images (mel/bcc/akiec only —
  see `build_slice3d_samples()`'s docstring for why the ~217K benign pool
  is deliberately excluded: it isn't broken down into nv/bkl/df by `iddx`
  alone, and mixing it in risked label noise for comparatively little
  benefit versus the actual confirmed problem, which is missed *malignant*
  thin/early lesions). This is a small fraction of the ~68K existing
  training images — likely needs the oversampling below to actually move
  the model's behavior, not just be diluted in; not yet separately
  upweighted beyond the existing inverse-class-frequency sampler.
- **Age rebalancing — implemented**: `SkinDataset.get_sampler_weights()` in
  Cell 9 now multiplies the sampler weight by `young_mel_boost` (default
  3.0x) for melanoma samples from patients under 45, on top of the
  existing inverse-class-frequency weighting. Age is joined from real
  per-dataset metadata (ISIC19's separate metadata CSV, ISIC20 inline;
  ISIC18 and SLICE-3D have no local age data, so those samples get the
  base weight only — see Cell 6). **3.0x is a starting point, not
  empirically tuned** — retune against validation-set age-band sensitivity
  once training results come in.
- **Robustness augmentation — implemented**: `A.ImageCompression(quality_lower=25,
  quality_upper=95, p=0.3)` added to Cell 8's training transform pipeline,
  on top of the existing RandAugment/CutMix/Mixup/CLAHE/gaussian-blur
  pipeline (which already had blur but not compression artifacts
  specifically).

## Step 2.5 — metadata fusion architecture (new: to actually reach the benchmark, not just beat 0.593)

**Why this is a separate step, not optional polish**: the ISIC 2024
competition winners (AUC 0.9704 on this exact external data) did not use a
pure image classifier. Multiple top solutions fused CNN image predictions
with the lesion's structured metadata via a gradient-boosted tree (GBDT),
not image features alone. Our current plan (Steps 0-2) only adds SLICE-3D
as more CNN training images — a materially smaller, structurally simpler
intervention than what actually won. Closing the gap from 0.59 toward 0.97
likely requires this architecture change, not just more image data.

**What metadata is actually usable** (checked against the real files, not
assumed):

| Source | Usable pre-diagnosis fields | NOT usable as model input |
|---|---|---|
| ISIC 2019 / 2020 | `age_approx`, `sex`, `anatom_site_general` | — |
| ISIC 2018 | none available locally | — |
| SLICE-3D Permissive | `tbp_lv_dnn_lesion_confidence` (another model's pre-computed score — valid as a stacking feature) | `mel_thick_mm`, `mel_mitotic_index` — these are **pathology-derived, determined after biopsy**. They're correctly used to stratify the Step-1 holdout by severity, but must never be fed into the model as an input feature — a real lesion being classified for the first time doesn't have a thickness measurement yet. Using them as inputs would be leaking the answer. |
| PAD-UFES-20 | rich clinical metadata (history, itch/growth/bleed, region, diameter) — but this is clinical photography, not dermoscopy, and isn't part of the current CNN training mix at all |

**Architecture**:
1. Train the CNN ensemble as planned in Steps 0-3 below, but additionally
   extract the penultimate-layer embedding per image (not just the final
   7-class softmax) — richer signal for the downstream GBDT than 7 numbers.
2. Build a per-sample feature vector: `[CNN embedding or 7-class probs] +
   [age_approx (with a missing-indicator flag), sex (categorical, missing
   allowed), anatom_site (categorical, missing allowed), tbp_lv_dnn_lesion_confidence
   where available]`. Missingness is expected and fine — LightGBM/XGBoost
   (what the ISIC 2024 winners used) handle missing values natively,
   unlike a neural net that would need explicit imputation.
3. Train a GBDT (LightGBM recommended) on this fused feature vector,
   predicting the same 7-class target, on the **same train split** used
   for the CNN (never touch the Step-1 SLICE-3D holdout, or any other
   holdout, during GBDT training either — it's just as easy to leak here
   as in the CNN stage).
4. The GBDT's output becomes the new `mel_score`/class-probability feed
   into the existing verdict logic (`make_verdict()`, age-conditional
   thresholds, review-band) — that logic doesn't need to change, only
   what produces the probabilities it consumes.

**Known limitation going in**: SLICE-3D's usable metadata is thin (one
derived confidence score, no age/sex/site) — the fusion approach will do
the most for ISIC19/20-sourced samples, where age/sex/site are real and
present. Whether it closes the SLICE-3D-specific gap as much as the
competition winners achieved partly depends on how much of their metadata
richness came from the full (non-Permissive) SLICE-3D release, which has
~55 columns we don't have access to in the Permissive subset. This is
worth knowing before committing to a specific target number, not a reason
not to try.

## Step 3 — the actual training run (not yet executed)

- **Correction to an earlier assumption in this doc**: `SA01_Model_Training.ipynb`'s
  config cell already comments `MODEL_NAME = "tf_efficientnetv2_s" # Night 1
  — fits RTX 4060 8GB`, with `BATCH_SIZE=16, ACCUM_STEPS=4` (effective batch
  64) — the original pipeline was already tuned for exactly this local GPU,
  not different/more powerful hardware as originally speculated here. The
  "3 nights per model" estimate is likely realistic as originally written.
- **Still worth a quick pilot regardless**: the data mix changed (SLICE-3D
  added, age-boosted sampling), which changes epoch composition even if
  per-image speed is unchanged — a few pilot epochs on EfficientNetV2-S
  confirms nothing broke before committing a full multi-night run.
- Retrain all three backbones (`tf_efficientnetv2_s`, `vit_large_patch16_224`,
  `convnext_large` — toggle `MODEL_NAME` in Cell 2) with the mixed dataset,
  same augmentation pipeline plus the JPEG addition, same architecture
  (`SkinClassifier`, confirmed in this session against the real checkpoints).
- Re-derive ensemble weights and temperature from scratch on the new
  validation split (which now excludes the Step-1 SLICE-3D holdout) using
  the now-corrected `SA02_Ensemble.ipynb`.

## Step 4 — re-validate, and set the bar for "did this work"

Re-run the full suite from `SA_README.md`'s "Validation & bias/fairness
tooling" section against the retrained model, and compare to both this
session's numbers AND the external benchmarks from the goal statement above:

| Metric | Before (this session) | Published benchmark to match/beat | Target after retrain |
|---|---|---|---|
| SLICE-3D external AUC (new holdout) | 0.593 | **0.9704** (ISIC 2024 winners, same dataset) | As close to the benchmark as the metadata-fusion architecture (Step 2.5) allows — 0.75+ is a floor, not the ambition |
| Blended sensitivity / specificity | 80.0% / 94.2% @ 0.2385 | **86.6% / 82.5%** (Haenssle et al. CNN-vs-dermatologists) | Meet or beat both numbers at a matched operating point — report the full ROC tradeoff, not just one point, since the benchmark's sensitivity is higher than our current default |
| **Melanoma in-situ / thin-lesion sensitivity on the holdout** (the confirmed dominant mechanism) | 6.7% (in-situ), most missed regardless of thickness | No direct published benchmark at this granularity | This is the number that most directly tests whether Step 2's thin-lesion data and Step 2.5's fusion actually worked — track separately from overall AUC, which can look better while this specific number stays bad |
| Age-band sensitivity spread | 43.8%-82.0% at flat threshold | — | Under ~15 points of spread *without* relying on the threshold trick — i.e. the model itself, not just the decision boundary, should be more age-consistent |
| Robustness flip rate (20-image test) | 4/20 blur, 4/20 brightness | — | Meaningfully reduced from training-time augmentation, not just TTA at inference |
| Skin-tone fairness on dermoscopy | Unanswered | — | Still unanswered unless a dermoscopy skin-tone dataset was sourced in the meantime |
| **Annotation automation rate** (the actual product goal — see goal statement above) | Not currently tracked | — | Report, on the same validation sets: % of cases resolved as `NORMAL`/`CANCER_FLAGGED` (auto-resolved) vs. `REVIEW_REQUIRED` (still needs a human annotator). A retrain that improves AUC but doesn't raise this number hasn't actually helped the annotation workflow — track both, every time |

If SLICE-3D external AUC doesn't move substantially even with metadata
fusion, that's evidence the case-severity/TBP-modality mismatch needs a
bigger data intervention than what's planned here (e.g., sourcing more real
thin-lesion/TBP training data specifically, or the full non-Permissive
SLICE-3D metadata), not evidence the approach was wrong.

## Step 4 results, and what they actually mean (2026-07-25)

Step 4 has now genuinely happened, in two passes — see `docs/MODEL_CARD.md`
findings #11 and #12 for full evidence:

- **First pass (n=44 holdout)**: inconclusive. AUC 0.4586, 95% CI
  [0.348, 0.571] — spans chance, sample too small to answer anything.
- **Second pass (n=143 holdout, after sourcing the full non-Permissive
  SLICE-3D release)**: **AUC 0.6131, 95% CI [0.5516, 0.6751], p=5.2e-06** —
  a real, statistically significant discrimination improvement, and
  directionally ahead of the original 0.593. **This is genuine evidence
  the data-mixing approach worked**, at least partially — Step 2's
  thin-lesion data (EMB) and Step 1's SLICE-3D mixing appear to have moved
  the needle on the actual ranking ability, not just the blended internal
  metric.

But Step 4's table above conflated two different questions under one
number ("SLICE-3D external AUC"), and the results split them apart: **AUC
improved, but the deployed threshold does not transfer** — 42.7%
sensitivity at the deployed 0.119 cutoff, and even the best achievable
threshold on this population only reaches 80% sensitivity at 15.6%
specificity (finding #12). This is a *calibration-transfer* problem
layered on top of a *partially-improved-but-still-limited discrimination*
problem — two separate things needing two separate fixes, addressed below.

## Path A — population-conditional threshold (✅ done, 2026-07-25)

Implemented the same way `AGE_BAND_THRESHOLDS` already solves the
analogous problem for age: `api/SA05_api.py`'s `TBP_MEL_THRESHOLD = 0.0199`
(derived via ROC search on the 143-case holdout for 80% sensitivity),
selected via a new `capture_mode="tbp"` parameter on `/analyze` — never
auto-detected, since there's no reliable way to tell TBP-crop-style images
from dermatoscope images from pixel data alone. Takes precedence over
age-conditional thresholds when set (it's a population-level fix, not
age-stratified). Regression-tested (`tests/test_verdict_logic.py`,
8 new tests) and config-guarded (`tests/test_config_consistency.py`) —
directly closing the "deployment logic changes go untested" gap the
independent audit flagged after three prior bugs (model-mode default,
base-path detection, checkpoint-resume skip) all slipped through
unguarded.

**Same honest caveat as the age thresholds**: this buys sensitivity at a
real specificity cost (15.6% at 80% sensitivity) — it is a *tradeoff
selection*, not a fix to the underlying discrimination. It only helps if
the product can actually identify TBP-style uploads at the point of
capture (a distinct screening/workflow mode) — if SkinAnalytica's real
deployment is dermatoscope-only, this threshold has no caller to invoke it
and the more relevant question becomes whether TBP-style images should be
detected and rejected/flagged as out-of-scope instead. That product
question is still open.

**Product-scope question — resolved, 2026-07-26.** Checked every
user-facing surface in the repo (`SA_README.md`, `pages/analyze.html`,
`pages/home.html`, `pages/api.html`, `skinanalytica_homepage.html`,
`skinanalytica_docs.html`): all of them describe SkinAnalytica as a
single-lesion **dermoscopy** upload tool, with zero mention of TBP,
full-body scanning, or mole-mapping anywhere, and there is no
`capture_mode` UI control in any page. **The real product is
dermatoscope-only; `capture_mode="tbp"` currently has no caller.**
SLICE-3D was always a research/validation exercise to stress-test
generalization to a genuinely different population — not a reflection of a
real input type this API needs to serve today.

**What this means practically**: Path A's threshold is real, tested,
correct infrastructure that stays dormant until (if ever) a TBP/screening
feature is actually built — it isn't blocking anything in the current
product. "Detect and reject out-of-scope images" was the alternative
considered, but there's no reliable way to auto-detect TBP-crop-style
images from pixel data alone (same reason `capture_mode` has to be
supplied explicitly rather than inferred), so building a rejection
mechanism isn't currently actionable either. **The input-mismatch risk
that actually matters for the shipped product is the broader one already
in finding #9**: zero validated performance on non-dermoscopic images in
general (clinical/phone photos included) — same lack of a detect-and-reject
mechanism, but for an input class real users can genuinely upload by
mistake, unlike TBP crops which no real user of this product would ever
have in hand.

## Path B — improve the underlying discrimination (active as of 2026-07-27 — see "Trigger condition — resolved" and "Workstream 1 — implementation-ready scope" below; superseded the "lower urgency" call made when this was TBP-only)

Threshold selection has a hard ceiling: at AUC 0.6131, no threshold gets
better sensitivity *and* specificity simultaneously on this population —
only a better trade along the same curve. Moving the curve itself needs:

1. **More real TBP-style training data, without touching the test set.**
   The 143 clean malignant holdout cases must stay untouched, but the full
   non-Permissive release's **benign** pool (401,059 images vs. the
   Permissive subset's 217,477) was never used for anything and is
   available to help the model learn this modality's baseline appearance
   without any leakage risk.
2. **Metadata/GBDT fusion** (Step 2.5 above) — the ISIC 2024 winning
   solutions used `tbp_lv_dnn_lesion_confidence` (a TBP-specific
   pre-computed signal) as a stacking feature specifically because raw
   image models don't transfer well across this exact modality gap. This
   is the most directly evidenced fix for this specific problem, not a
   generic accuracy lever.
3. **Segmentation-assisted classification** (the EdgeNeXtSAC pattern
   researched this session) — TBP crops include more surrounding
   skin/background than centered dermatoscope images; forcing explicit
   lesion localization may reduce the modality-specific score shift at
   its source.

**Status update, 2026-07-27**: the "lower priority, TBP-only" call above
was correct given what was known at the time, but finding #16
(`docs/MODEL_CARD.md`) — annotation-automation rate and flagged-precision,
computed for the first time — found the same underlying problem
(sensitivity-tuned thresholds trading away specificity) showing up on
genuinely dermoscopic, in-scope data: flagged-precision collapses to
4-16% at realistic screening prevalence on MILK10k's holdout, not just on
SLICE-3D/TBP. This resolves the open question this section originally
left unanswered. Path B is now active, starting with workstream (2) above
(metadata/GBDT fusion) — see "Trigger condition — resolved" and
"Workstream 1 — implementation-ready scope" further down for the concrete
task breakdown, updated to also fuse MILK10k's own metadata
(`age_approx`, `sex`, `skin_tone_class`) and to evaluate against
flagged-precision at realistic prevalence, not just AUC.

## Path B — detailed scope (2026-07-27, planning only — nothing below is started)

This section sizes each of the three workstreams above so a future
decision to start Path B doesn't have to re-derive effort/risk from
scratch. **No code, training, or data sourcing was done for this
scoping pass** — it's a plan, not a status update.

### Trigger condition — resolved 2026-07-27, Path B is now active

The open question this scoping pass originally surfaced — "is Path B's
benefit TBP-specific, or does it also matter for the real dermatoscope
product" — is answered. **Finding #16** (`docs/MODEL_CARD.md`) computed
annotation-automation rate and flagged-precision for the first time,
using MILK10k's genuinely-never-trained-on holdout resampled to realistic
screening prevalence (zero leakage — same held-out images, recombined at
different ratios). Result: at realistic low prevalence (2-10%),
flagged-precision on real dermoscopy data collapses to **4-16%** — the
`CANCER_FLAGGED` bucket is dominated by false alarms, not the confident,
correctly-handled cases the automation-rate framing implies. This is the
same root cause as finding #12's threshold-transfer problem
(sensitivity-tuned thresholds trading away specificity), now confirmed to
show up on genuinely dermoscopic, in-scope data — not just SLICE-3D/TBP.

**Path B is therefore no longer dormant infrastructure.** It's the
identified fix for a problem that affects the real, shipped product
today. Workstream 1 (metadata/GBDT fusion) is scoped below to an
implementation-ready level, since it's the cheapest, fastest, and
best-evidenced of the three — see the original sizing further down for
workstreams 2 and 3, which stay in reserve pending workstream 1's result.

### Workstream 1 — metadata/GBDT fusion (Step 2.5)

- **What**: extract each CNN's penultimate-layer embedding, concatenate
  with available structured metadata (`age_approx`, `sex`,
  `anatom_site_general`, `tbp_lv_dnn_lesion_confidence` where present),
  train a LightGBM model on the fused vector, same 7-class target.
- **Effort**: no new CNN training cycle — reuses already-trained
  backbones. Two real costs: (a) a one-time embedding-extraction pass over
  the ~73K-image training pool (one forward pass per image; the RTX 4060
  scored 2,794 SLICE-3D images in ~2.4 min this session, so ~73K images is
  roughly an hour, not a multi-night job), and (b) LightGBM training
  itself, which is minutes, not hours.
- **Data**: already on disk — no new sourcing needed.
- **Deliverable**: a new notebook (e.g. `SA07_MetadataFusion.ipynb`), a
  saved GBDT model file, and a small change to the inference path
  (`src/inference_utils.py`/`api/SA05_api.py`) to feed CNN output +
  metadata through the GBDT before `make_verdict()` — the verdict/
  threshold logic itself doesn't need to change, only what produces the
  probabilities it consumes.
- **Known limitation going in** (already flagged above): SLICE-3D's
  usable metadata is thin (one derived confidence score, no age/sex/site
  locally), so this will likely help ISIC19/20-sourced samples more than
  SLICE-3D specifically — worth knowing before committing to a target
  number.
- **Success criteria**: SLICE-3D holdout AUC improves measurably beyond
  0.6131, measured against the same 143-case holdout, without that
  holdout touching GBDT training at any point (same leakage discipline as
  the CNN stage).
- **Recommended order**: do this first if/when Path B starts — cheapest,
  fastest to a real signal, and the most direct evidence tying it to this
  specific gap (it's what the actual ISIC 2024 competition winners did).

### Workstream 1 — implementation-ready scope (2026-07-27)

Now that the trigger condition is met, this breaks the summary above into
concrete tasks. **Still planning only — nothing below has been run.**

**Metadata sources, updated from the original Step 2.5 table** to include
MILK10k, which wasn't considered a fusion input before (it existed only as
a threshold-selection holdout). MILK10k's own metadata is real and richer
than SLICE-3D's:

| Source | Usable pre-diagnosis fields | Notes |
|---|---|---|
| ISIC 2019 / 2020 | `age_approx`, `sex`, `anatom_site_general` | unchanged from the original table |
| SLICE-3D Permissive | `tbp_lv_dnn_lesion_confidence` | unchanged — thin, one derived score |
| **MILK10k (new to this fusion plan)** | `age_approx`, `sex`, `skin_tone_class` | all three are captured at imaging time, not derived post-diagnosis, so all are legitimate model inputs. Adding `skin_tone_class` as a real feature (rather than only a threshold-selector, its current role) is the more direct fix for finding #14's subgroup sensitivity spread than `SKIN_TONE_THRESHOLDS` is — a provisional band-selector is a workaround, a model that actually sees skin tone can learn from it |
| ISIC 2018 | none available locally | unchanged |

**Task breakdown**:

1. **Embedding extraction script — done, 2026-07-27.**
   `scripts/build_training_pool.py` ports the real training-pool
   construction (`build_sample_list`, `build_slice3d_samples`,
   `build_emb_samples`, `build_milk10k_samples`) out of
   `notebooks/SA01_EfficientNet_Train.ipynb` Cell 6 into an importable
   module — verified to reproduce the exact documented pool size
   (73,456 samples: 68,472 ISIC + 250 SLICE-3D + 770 EMB + 3,964 MILK10k)
   and the exact holdout sizes (44 SLICE-3D, 137 EMB, 700 MILK10k). This
   was ported rather than re-derived specifically to avoid a third
   instance of the SA01/SA02 split-divergence bug class.
   `scripts/extract_embeddings.py` runs each trained CNN backbone forward
   to the pooled, pre-head layer (`SkinClassifier.backbone(x)`, skipping
   `dropout`/`head`) over a chosen split (`train`, `slice3d_holdout`,
   `milk10k_holdout`), saving embeddings + image_id/source/label_idx/age
   to a `.npz` plus a readable manifest CSV. Smoke-tested end-to-end on
   all three splits (verified feature dims match `gpu_inference.py`'s
   documented values: 1280/1024/1536 for
   efficientnetv2-s/vit-large/convnext-large).
   **Measured throughput** (not the earlier ~1hr guess, which was based on
   ONNX ensemble scoring, not three separate PyTorch backbone forward
   passes): ~13 img/s at batch_size=32 on the RTX 4060 once warmed up →
   the full 73,456-image training pool is closer to **~90 minutes**, still
   well under a multi-night training-style job. The two holdouts (143 +
   700 images) are a few minutes each.
   **All three full runs completed, 2026-07-27**: both holdouts (2,643
   SLICE-3D, 700 MILK10k) and the full 73,456-image training pool.
   The first training-pool attempt collapsed to ~1.3 img/s once it hit
   ISIC-2020's much larger images (up to 4000x6000 vs ISIC-2018's
   450x600 — `preprocess_bgr`'s hair-removal step, `cv2.inpaint`
   especially, scales badly with resolution) — CPU-bound, GPU sitting at
   12-13% utilization, on track for ~6 hours instead of ~90 minutes, with
   no incremental save so a crash would have lost everything. Fixed by
   parallelizing the CPU preprocessing across worker processes (12 on
   this 16-core machine) while GPU inference stayed single-process —
   measured 1.26 -> 9.10 img/s (7.2x) on real ISIC-2020 images — plus
   checkpointing every 2,000 images and automatic resume. Full run then
   completed at ~18-21 img/s sustained, GPU pegged near 100%.

   **A real, previously-unknown data issue surfaced during the
   post-extraction integrity check, not fixed at the CNN level — see
   `docs/MODEL_CARD.md` finding #17**: all 10,015 ISIC-2018 images are
   byte-for-byte duplicates also present in ISIC-2019, so the actual
   retrain double-sampled them (~2x effective weight). Not being
   corrected in the already-completed CNN retrain, but
   `scripts/dedup_train_embeddings.py` produces a deduplicated
   `train_embeddings_deduped.npz` (63,441 unique rows) so Path B's
   fusion model doesn't inherit the same bias — **task 2 below must
   consume the deduped file, not the raw one.**
2. **Feature-joining script — done, 2026-07-27.**
   `scripts/join_fusion_features.py` reads
   `outputs/embeddings/train_embeddings_deduped.npz` (or either holdout's
   embeddings) and joins age/sex/site/skin_tone_class/
   tbp_lv_dnn_lesion_confidence from the raw ISIC19/20, SLICE-3D, and
   MILK10k metadata files, each with its own explicit `*_missing` flag.
   Produces `<split>_fusion_features.npz`: `X` (3,850-column float32
   matrix — 3,840 embedding dims + 5 metadata + 5 missing-flags), `y`
   (label_idx), `image_id`, `source`, `feature_names`. Verified: identical
   3,850-column schema across train and both holdouts, zero NaNs in the
   embedding block, and coverage counts that check out exactly against
   known source sizes (`tbp_lv_dnn_lesion_confidence`: 250/63,441 on
   train, matching SLICE-3D's exact training count; `skin_tone_class`:
   3,964/63,441, matching MILK10k's exact count).

   **`site` is deliberately not semantically unified across sources** —
   ISIC19 distinguishes anterior/lateral/posterior torso, ISIC20
   collapses those to one "torso", MILK10k uses a third vocabulary
   entirely (`head_neck_face`, `lower_extremity`, ...). Mapping these
   together would be an unverified judgment call, the same discipline
   already applied to diagnosis-label mapping elsewhere in this project
   — every distinct site string is its own category instead. The
   category vocabulary is fit once on the train split and saved
   (`outputs/embeddings/site_vocab.json`) so holdout splits reuse the
   identical encoding; an eval-time site string never seen in training
   maps to a dedicated `__unknown__` code rather than silently growing
   the vocabulary and making train/eval feature meanings diverge.

   **A genuine side benefit of finding #17's duplication, verified not
   assumed**: because the metadata join is by `image_id` regardless of
   which duplicate copy dedup kept, the 10,015 ISIC-2018-tagged rows
   (which have zero metadata of their own) get backfilled with their
   ISIC-2019 duplicate's real sex/site data. Measured: sex coverage on
   the train split is 61,972/63,441 — more than the naive
   ISIC19+ISIC20+MILK10k sum (52,406) would suggest, confirming the
   backfill is real and working, not a fluke.
3. **GBDT training — done, 2026-07-27.** `scripts/train_fusion_gbdt.py`
   trains LightGBM (multiclass, 7-class target, `class_weight="balanced"`,
   `site` marked as the sole categorical feature) on
   `train_fusion_features.npz`. **The exclusion check ran first, before
   any training**: asserted zero `image_id` overlap between the training
   features and both the SLICE-3D 143-case holdout and the MILK10k
   700-image holdout — passed (0 overlap with either, verified against
   2,643 and 700 real holdout ids respectively, not just asserted in the
   abstract). An internal 90/10 stratified split off the *training pool
   itself* (57,096 / 6,345 rows) was used for early stopping only — this
   is not, and doesn't replace, task 5's real evaluation against the
   external holdouts.

   Trained in ~4.4 minutes (2000 max rounds, early-stopped at iteration
   150). Internal-validation numbers (sanity check that training worked,
   not a claim about generalization): multi_logloss 0.1686, accuracy
   94.4%, MelAUC 0.9854 — expectedly strong, since this split is drawn
   from the same distribution as training, the exact pattern this whole
   document exists to warn against over-trusting (see the "headline
   metrics" caveat at the top of `docs/MODEL_CARD.md`).

   Saved to `outputs/fusion_model/`: `lgbm_fusion.txt` (the model, saved
   pre-pruned to its best 150 trees), `lgbm_fusion_manifest.json`
   (training config, class distribution, leakage-check result, internal
   metrics), `feature_names.json` (column order, needed at inference
   time). Round-trip verified: reloaded the saved model fresh and
   confirmed it reproduces valid 7-class probability distributions.
4. **Inference wiring, opt-in only.** Feed the GBDT's output into
   `make_verdict()` behind a new flag (e.g.
   `SKINANALYTICA_ENABLE_FUSION`), matching the existing pattern for
   `capture_mode`/`skin_tone_class` — this must not silently change
   production behavior for any existing caller until it's been evaluated.
5. **Evaluation — done, 2026-07-27.** `scripts/evaluate_fusion.py` ran
   all three parts of the plan, exactly as scoped, and the result is
   genuinely mixed — see `docs/MODEL_CARD.md` finding #18 for the full
   evidence. Short version:
   - **Discrimination on SLICE-3D**: AUC 0.5755 (CI [0.5203, 0.6331]) vs.
     the 0.6131 baseline — statistically indistinguishable from no
     change (baseline falls inside the fusion model's CI). Matches the
     a-priori expectation that SLICE-3D's thin local metadata would
     limit fusion's benefit there specifically.
   - **Flagged-precision on MILK10k at realistic prevalence** (the
     metric that actually motivated this workstream, per finding #16):
     initially reported as modestly, mostly positive from a single
     resampling draw (+3.9pp/+6.2pp/+4.6pp at 5%/10%/20%) — **then
     properly bootstrapped** (`scripts/bootstrap_flagged_precision.py`,
     paired design, 2,000 resamples per prevalence level — see
     `docs/MODEL_CARD.md` finding #19). Revised, more precise result:
     positive in direction at every prevalence tested, but only
     **statistically confirmed at 20% prevalence** (delta +7.4pp, 95% CI
     [+1.1, +13.8]); at the more realistic 2-10% range the improvement
     is consistently positive but not distinguishable from noise (10%
     came closest, CI lower bound −0.2).
   - **Subgroup check**: AUC point estimates improved across every
     adequately-sampled skin-tone and age band, but **none clear the
     original (already-wide) confidence intervals** once CIs were
     computed for the fusion model's own bands — not a statistically
     confirmed improvement. The much larger sensitivity-at-threshold
     jumps (+23 to +39pp) are flagged as most likely a
     threshold-recalibration artifact (old thresholds tuned for the
     pre-fusion score distribution, applied unchanged), not reported as
     a fairness win on their own.
6. **Write-up — done.** See `docs/MODEL_CARD.md` findings #18 and #19.

**Second external holdout added, 2026-07-27 — see finding #20 (corrected
2026-07-28, finding #23 — read both).** MRA-MIDAS (Stanford AIMI +
Cleveland Clinic, 786 usable dermoscopy images, never touched by
training) was sourced specifically to strengthen finding #19's
confirmation gap. It didn't confirm fusion helps: fusion-vs-baseline
gap isn't distinguishable on either metric. On raw discrimination, the
original write-up overstated the problem — a metric bug (fixed in
finding #23) made malignant-vs-benign AUC look like ~0.39 (near/below
chance) when it was actually **0.70-0.74** the whole time. MelAUC
specifically (mel vs rest) genuinely is weak (0.54-0.58) and that part
is unaffected by the fix — the corrected read is "the model can tell
something's cancerous but is worse at naming exactly which type,"
not "the model doesn't work here." Flagged-precision deltas are still
all within noise. This is still a third independent population showing
a real, if more precisely-scoped, external-generalization limit
(alongside SLICE-3D's finding #5 and MILK10k's own threshold-transfer
gap) — still raises the priority of workstreams 2/3 (more real external
training data, segmentation) relative to workstream 1 alone, since
metadata fusion on top of the existing CNNs isn't rescuing generalization
on this population either.

**Root cause found and partially confirmed, 2026-07-27 — see finding #21.**
Investigated *why* by viewing real preprocessed images rather than more
score-level analysis: SLICE-3D and MRA-MIDAS fail for two opposite
reasons. SLICE-3D's source crops are tiny (median 128px) — a genuine
resolution ceiling. MRA-MIDAS's source photos are huge (median
3024x4032) with the lesion occupying a small fraction of the frame
inside the dermatoscope's circular field of view — `preprocess_bgr()`'s
naive full-frame resize wastes almost all the available detail on
background. A zero-retraining ablation (center-crop before resize)
confirmed the mechanism: MelAUC 0.5793→0.6002 (directionally real,
doesn't clear a 95% CI at n=786) with no measured cost to MILK10k. This
sharpens workstream 3 from "segmentation-assisted classification, one of
three options" to "the directly-indicated fix for MRA-MIDAS's specific
failure mode, worth prioritizing over workstream 2 alone" — more raw
external training data wouldn't fix a framing/crop problem by itself.
SLICE-3D's resolution ceiling is a different problem workstream 3
doesn't solve either; needs higher-resolution source data or explicit
resolution-robustness training instead.

**Lesion-detecting crop built and tested, 2026-07-27 — a documented dead
end, not an open question. See finding #22.** A classical-CV heuristic
detector (`inference_utils.detect_lesion_bbox()`, Otsu-threshold-based,
unit-tested, visually verified correct on a real example) was built as
the natural next step past center-crop. Properly ablated against both
MRA-MIDAS and the MILK10k guardrail — it lost on both: didn't beat
center-crop's already-modest MRA-MIDAS improvement (0.5721 vs 0.6002
MelAUC), and measurably hurt MILK10k (malignant-vs-benign AUC confirmed
regressed — corrected 2026-07-28 per finding #23's metric fix: CI
[−0.0596, −0.0012], still confirmed real, smaller magnitude than
originally measured). The corrected metric also removed MRA-MIDAS's one
seemingly-positive result for lesion-crop (previously a confirmed
+0.0330 malignant-vs-benign improvement, now a non-significant −0.0194)
— lesion-crop had no metric left in its favor after the correction, not
fewer. Working explanation:
dermatoscope photography is already clinician-centered on the lesion by
construction, so center-crop exploits that framing prior for free; a
classical darkness-based heuristic doesn't have enough signal to beat
"trust the photographer," and its misfires (ruler overlay, vascular
texture, non-pigmented malignant subtypes) cost more than its correct
detections gain.

**Updated recommendation**: if a production preprocessing change is
ever made on finding #21's mechanism, **center-crop is the validated
candidate** — simpler and empirically better than the fancier heuristic.
A genuine improvement beyond center-crop's modest, not-fully-CI-confirmed
gain needs a properly trained segmentation model (the real workstream 3
investment), not a better classical heuristic — that door is now closed,
not open.

**Non-goals for this pass, honored**: no production cutover happened —
task 4 (opt-in inference wiring behind `SKINANALYTICA_ENABLE_FUSION`)
was deliberately left undone pending this evaluation, and the evaluation
itself doesn't clearly settle whether to build it: a real, bootstrap-
confirmed flagged-precision improvement at higher prevalence, a
consistently positive but not-yet-confirmed direction at the more
realistic lower prevalences that actually motivated Path B, a clean null
on the original SLICE-3D AUC target, and a subgroup result that needs
more data before it's confirmed either way. Workstreams 2 and 3 remain
untouched, as planned.

### Workstream 2 — more real TBP-style training data

- **What**: mix in the full non-Permissive SLICE-3D release's benign pool
  (401,059 images vs. the Permissive subset's 217,477) — currently
  completely unused for anything.
- **Effort**: unlike workstream 1, this **does** require a real CNN
  training cycle (or at minimum a fine-tuning pass) since it's new image
  data, not a fusion layer on top of existing embeddings — same order of
  magnitude as the original "3 nights per backbone" estimate for a full
  retrain; a fine-tune-only pass (fewer epochs, lower learning rate on
  already-trained checkpoints) is worth scoping as a cheaper alternative
  if this workstream is greenlit, rather than assuming a full retrain is
  required by default.
- **Data**: already on disk — no new sourcing needed.
- **Risk**: this pool is benign-only. 401K benign images against the
  already-used 143 held-out malignant cases is a large class-imbalance
  swing if dumped in naively — needs deliberate sampling/weighting, same
  discipline as the existing inverse-class-frequency sampler. It also has
  a structural ceiling on its own: benign-only data teaches the model
  what "benign in this modality looks like" but can't teach new malignant
  presentation, so its likely ceiling is lower alone than paired with
  workstream 1.
- **Deliverable**: an extension of the existing Step 1/2 sampling logic
  (`build_slice3d_samples()` and friends) to draw from this pool, with an
  explicit assertion (matching the existing leakage checks in
  `scripts/dataset_loaders.py`) that the 143-case holdout stays untouched.
- **Recommended order**: alongside or after workstream 1, once workstream
  1's result gives a read on whether the extra training cost is likely to
  pay off.

### Workstream 3 — segmentation-assisted classification (EdgeNeXtSAC pattern)

- **What**: explicit lesion localization/cropping before classification,
  on the theory that TBP crops include more surrounding skin/background
  than centered dermatoscope images, and forcing localization reduces the
  modality-specific score shift at its source.
- **Effort**: the largest of the three by a wide margin — requires either
  sourcing a pretrained segmentation model (with its own licensing check,
  same discipline as every dataset decision in this project — see
  `docs/KNOWN_GAPS.md`) or training one from scratch, then
  re-architecting the classification pipeline to consume segmented/
  cropped input, then re-training all three backbones on the new
  preprocessing. Multiple sequential retrain cycles, not one.
- **Risk**: weakest evidence base of the three — this is inferred from
  the EdgeNeXtSAC paper's general approach, not confirmed against this
  project's own data the way workstream 1's metadata-fusion rationale is
  (that one is directly evidenced by what the actual competition winners
  did on this exact external dataset).
- **Recommended order**: longer-term investment only, and only if
  workstreams 1+2 together don't close enough of the gap to justify
  stopping there.

### Summary

| Workstream | New training needed? | Data sourcing needed? | Evidence strength | Priority |
|---|---|---|---|---|
| 1. Metadata/GBDT fusion | No (reuses existing embeddings) | No | Strong — matches actual competition-winning approach | First |
| 2. More TBP training data | Yes (full retrain or fine-tune) | No (already on disk) | Moderate — plausible but structurally capped (benign-only) | Second |
| 3. Segmentation-assisted classification | Yes (multiple retrain cycles) | Maybe (pretrained model licensing) | Weakest — inferred, not confirmed on this project's data | Third, long-term only |

This scope is deliberately conservative on workstream 3 and optimistic on
workstream 1 — if only one workstream is ever resourced, workstream 1 is
the one with a real shot at showing a signal fast enough to justify
continuing to 2 and 3.

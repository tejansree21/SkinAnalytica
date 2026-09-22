# SkinAnalytica model card

Last updated: 2026-07-28, adding finding #29 (the verdict-threshold
duplication between `api/SA05_api.py` and `scripts/evaluate_fusion.py` —
caught mid-drift by finding #27, flagged but not fixed there — closed via
a new `src/verdict_logic.py` single source of truth, same pattern as
preprocessing's `inference_utils.preprocess_bgr()`; verified byte-identical
output on every affected script plus 4 new regression tests) on top of
finding #28 (production uncertainty
quantification decided — ship the ONNX-only inter-model-disagreement
signal over adding PyTorch, `SKINANALYTICA_ENABLE_UNCERTAINTY` now
defaults to `true`, with the live Render free-tier deployment explicitly
protected from a ~2.6GB memory footprint via a `render.yaml` override)
on top of finding #27 (flagged-precision-at-prevalence
refreshed under finding #26's new `MEL_THRESHOLD` and given a proper
bootstrap CI on ISIC-2020 for the first time — findings #16/#19's tables
had gone silently stale within this same session; caught via a hardcoded
threshold-constant duplication in `scripts/evaluate_fusion.py`) on top of
finding #26 (`MEL_THRESHOLD` resolved by
formally adopting `config/val.csv` as the new documented validation
manifest — the original 10,312-image set was unrecoverable — moving the
threshold from 0.2385 to **0.5010**, a real and clearly-flagged change to
production verdict routing, not a rounding correction) on top of finding
#25 (operating thresholds
re-derived under the finding #24 center-crop change — `AGE_BAND_THRESHOLDS`
and `SKIN_TONE_THRESHOLDS` updated via a clean, population-matched
re-scoring; `TBP_MEL_THRESHOLD` confirmed exactly unchanged; `MEL_THRESHOLD`
deliberately left unchanged after a population-mismatch confound made the
first re-derivation attempt produce a misleading ~2x jump that didn't
survive a same-population control check) on top of finding #24 (center-crop
shipped to production as the new default in `preprocess_bgr()`, closing the
API-side duplication that used to re-derive preprocessing inline, and
flagging that the deployed probability thresholds were not re-derived
under the new preprocessing — see `docs/KNOWN_GAPS.md`) on top of
finding #23 (the malignant-vs-benign AUC
metric fix flagged in findings #18/#22 was applied everywhere it was
used and every affected result re-run — this is not a footnote
correction: MRA-MIDAS's malignant-vs-benign discrimination was actually
**0.70-0.74** all along, not the ~0.39 originally reported in finding
#20 — the "collapses to near/below chance" framing was itself a
consequence of the metric bug, not a fully accurate description of the
model's real performance there. MelAUC specifically is still weak and
unaffected by this fix — the corrected picture is more precise, not
simply "better") and finding #22 (built and properly tested
a real lesion-detecting crop, the natural next step after finding #21's
center-crop ablation — and the "smarter" heuristic lost to the "dumber"
one: it doesn't beat center-crop on MRA-MIDAS and measurably hurts
MILK10k, a real, honestly-reported negative result on the approach that
looked most promising going in) and finding #21 (the actual mechanism behind
why external generalization keeps failing, found by directly viewing
preprocessed images rather than more score-level analysis — SLICE-3D and
MRA-MIDAS fail for two different, opposite reasons: native image
resolution too low (SLICE-3D, median 128px source crops) vs. lesion
occupying too small a fraction of a huge frame (MRA-MIDAS, median
3024x4032 portrait photos with the lesion as a small mark inside the
dermatoscope's circular field of view) — both are the same underlying
problem from opposite directions: training data sits in a narrow
"moderate-resolution, pre-cropped-to-the-lesion" comfort zone neither
external dataset matches) and finding #20 (MRA-MIDAS sourced and
loaded as a genuine second external holdout — never touched by any
training step — and it's a sobering result: discrimination collapses to
near/below chance for both the fusion model and the baseline ensemble,
a third independent population showing the same external-generalization
failure first found in finding #5, not resolved by fusion) and finding
#19 (a proper paired bootstrap
of finding #18's flagged-precision result — the "modestly, mostly
positive" gain only holds up as statistically real at 20% assumed
prevalence; at the more realistic 2-10% range, the improvement is
consistently positive in direction but not distinguishable from noise)
and finding #18 (Path B workstream 1's
metadata-fusion GBDT evaluated against both external holdouts — no
confirmed improvement on SLICE-3D, matching the a-priori expectation that
its metadata is thin; a modest, directionally positive flagged-precision
gain on MILK10k at realistic prevalence, the metric that actually
motivated Path B; large subgroup sensitivity jumps that do NOT hold up as
a confirmed fairness improvement once checked against confidence
intervals — most likely a threshold-recalibration artifact, flagged as
such rather than reported as a win) and finding #17 (every one of ISIC-2018's
10,015 images is a byte-for-byte duplicate also present in ISIC-2019,
discovered as a side effect of Path B's embedding extraction — the real
training pool double-sampled all 10,015 with consistent labels, ~2x
effective weight, not caught until now) and finding #16 (annotation-automation rate
computed for the first time, at realistic case-mix prevalence — automation
rate holds up reasonably well, but flagged-precision collapses at low
prevalence, a real and previously-unmeasured cost of the sensitivity-tuned
thresholds) and finding #15 (non-dermoscopic-input
guardrail — includes an honest negative result: the first heuristic
version was disproven against real project data before shipping and
recalibrated) and closing finding #13 fully (all 6 patient-data endpoints
now gated, via a new client-supplied-key frontend pattern rather than a
backend proxy). Finding #14 (MILK10k's dermoscopy
skin-tone holdout finally scored — MelAUC 0.8796, no confirmed AUC
disparity across adequately-sampled skin-tone bands, sensitivity spread
not yet confirmed as real) plus a same-day provisional fix
(`SKIN_TONE_THRESHOLDS`, opt-in, same architecture as Path A) and finding
#13 (auth added to the by-id
patient-data endpoints — scoped after checking the real frontend, not
gating everything blindly). Finding #12 (external re-validation at n=143
— discrimination confirmed significantly improved, AUC 0.6131, but a
threshold-transfer problem confirmed separately) and Path A's fix
(capture-modality-conditional threshold) landed 2026-07-25. Finding #11
(the earlier n=44 inconclusive result) and the full "10/10 models" review
are preserved below for context. This document exists so that performance
claims, limitations, and known gaps live in one place instead of being
scattered across notebooks and configs. See
[KNOWN_GAPS.md](KNOWN_GAPS.md) and [RETRAIN_PLAN.md](RETRAIN_PLAN.md)'s
"Path A / Path B" section for items still open, and
[REGULATORY_POSITIONING.md](REGULATORY_POSITIONING.md) for what these
findings mean for any future clinical/regulatory claim.

## Overview

- **Task**: 7-class dermoscopy lesion classification (mel, nv, bcc, akiec,
  bkl, df, vasc), with melanoma flagged as the primary clinical concern.
- **Architecture**: ensemble of EfficientNetV2-S, ViT-L/16, ConvNeXt-Large,
  combined via learned weighted average (`ensemble_weights.json`) and
  temperature-scaled calibration (`temperature.json`, T=0.4095).
- **Training data**: HAM10000 + ISIC 2019 + ISIC 2020 (~68,000 images
  combined), 15% held out for validation via a single random split
  (`random_state=42`) over the pooled image IDs — not stratified by dataset
  source or demographic.
- **Deployment**: ONNX export (FP32 + INT8), served via FastAPI
  (`api/SA05_api.py`) on Render. Render's free tier has no GPU regardless —
  this session found a local NVIDIA RTX 4060 usable for research/analysis
  speed (via PyTorch checkpoints directly, ~13-21x faster than CPU ONNX; see
  KNOWN_GAPS.md for why onnxruntime's own CUDA execution provider doesn't
  work here) but that does **not** change anything about how the deployed
  API actually runs in production.

## Headline metrics (self-consistent recomputation, blended validation set, full 10,312-image split)

| Metric | Originally published (mismatched pooling) | Self-consistent (matches deployed model) |
|---|---|---|
| Melanoma AUC | 0.9609 | 0.9540 |
| Macro AUC (all 7 classes) | 0.9878 | 0.9872 |
| Sensitivity (mel) | 80.1% @ threshold 0.312 | 80.0% @ threshold **0.2385** |
| Specificity (mel) | 95.5% | 94.2% |
| FNR | 19.9% | 20.0% |

The deployed `MEL_THRESHOLD` is now **0.2385**, not 0.312 (see finding #10)
— overall discriminative performance barely moved when the pooling formula
was corrected, but the *threshold* that actually delivers the intended 80%
sensitivity did. Every dataset-specific/subgroup sensitivity number below
has been recomputed at 0.2385 (from the already-scored CSVs, no new
inference needed) so the whole document is internally consistent against
the one currently-deployed threshold.

**These numbers are not the whole story — see "What we found" below.**
They're a blended average across ISIC 2018+2019+2020 on data drawn from the
same training distribution; every other validation attempted this session
scored meaningfully worse, in one case (genuine external validation)
dramatically so.

## What we found this session (chronological, each with evidence)

1. **The bias/fairness measurement itself was broken.** `SA02b_BiasReport.ipynb`,
   `SA04_Research_Module.ipynb`, and `agents/verification_agent.py` all computed
   softmax on raw ONNX logits without dividing by the calibrated temperature
   first — AUC looked fine (rank-invariant) but every subgroup sensitivity
   silently came back 0.0%. **Fixed**: added `src/inference_utils.py` as a
   shared source of truth for this step.

2. **A second, related bug was found later the same session**: the combined
   `skinanalytica_ensemble.onnx` graph (confirmed via
   `notebooks/SA02_Ensemble.ipynb`'s export cell) bakes per-model softmax,
   temperature division, AND weighted averaging into the ONNX graph itself —
   its output is already final probabilities, output tensor literally named
   `"probs"`. Every script that then applied the fix from #1 to *that* file's
   output was applying softmax+temperature a **second** time. Confirmed on a
   real case: the combined graph's raw output for one image was
   `[0.00375, 0.985, ...]` — already-correct probabilities — while the
   double-processed version read mel_score=0.0589, a >15x distortion.
   **Fixed properly this time**: `run_session()` in `src/inference_utils.py`
   checks the ONNX output tensor name and only applies softmax/temperature
   when the graph outputs raw `"logits"` (the 3 individual backbone files);
   it passes `"probs"` outputs straight through. Verified against all 8
   production ONNX files (fp32 + int8 × combined + 3 backbones) — no gaps.
   **Every number below reflects data scored with this corrected path.**
   Numbers computed before this fix (age gap 27-74%, ISIC-2020 sensitivity
   53.7%, PAD-UFES-20 AUC 0.686) were superseded and are not reported here —
   see git history if you need the pre-correction values for comparison.

3. **Real age-based sensitivity gap** (root cause unaffected by threshold
   choice — see below — but magnitudes shift with it): at the flat
   **0.2385** threshold, sensitivity ranges from **52.1% (<30) to 83.6%
   (75+)**, on a 1,764-image stratified ISIC-2020 sample (585 melanomas).
   Bootstrap 95% CI on the overall sensitivity: [62.1%, 69.9%] (n=2000
   resamples, computed at the threshold in use at the time — see git
   history for the 0.312-based figures this superseded). **Root cause**
   (from real training metadata, not speculation): melanoma prevalence in
   the training pool is ~1% for under-45 patients (48-103 positive examples
   total) vs. up to 9.4% for 75+ (fewer total images, but far higher
   signal-to-noise) — the model learned an age-correlated prior from
   genuine training-data imbalance. **Fixed with age-conditional
   thresholds** (independently derived per band via ROC search, unaffected
   by the global threshold correction — see engineering table) that bring
   all age bands to a tight 81.2-90.2% sensitivity range.

4. **ISIC-2020-specific generalization gap**: overall sensitivity on
   ISIC-2020-only data is 70.8% vs. 80.0% on the blended validation set —
   both measured at the same, currently-deployed 0.2385 threshold, so this
   is a true apples-to-apples comparison — confirmed via
   `notebooks/SA02_Ensemble.ipynb`'s split logic: the
   "official" 80% comes from a proportional 15% holdout across all three
   source datasets, and ISIC-2020 images are measurably harder for this
   model than ISIC 2018/2019. The blended metric was masking this.
   Youden-optimal threshold on ISIC-2020 data is 0.037 (sensitivity 90.8%,
   specificity 88.2%) — much lower than either deployed threshold, another
   sign the deployed threshold is tuned to the blended distribution
   specifically, not any one sub-population.

5. **Genuine external validation reveals the most severe gap found this
   session — mechanism now substantially diagnosed.** ISIC 2024 SLICE-3D
   Permissive (2,794-image stratified sample: all 294 malignant cases +
   2,500 benign) comes from institutions with zero overlap with the
   training data (MSKCC, Hospital Clínic de Barcelona, University of
   Queensland, Medical University of Vienna, University of Athens). Result:
   **AUC 0.593** (barely better than chance), **sensitivity 4.8%,
   specificity 98.5%** at the deployed 0.2385 threshold (misses 95 of 100
   real malignancies). Sensitivity is essentially unchanged from the
   pre-correction 0.312 threshold (also 4.8%) — unlike ISIC-2020/PAD-UFES-20,
   where a lower threshold recovers meaningful sensitivity, SLICE-3D's true
   melanoma scores are uniformly low enough that moving the cutoff within
   this range barely helps. That's further evidence this is a genuine
   discriminative failure, not a threshold-calibration one (see the
   case-severity mechanism below).

   **Institution breakdown**: 3 of 4 institutions (94% of the sample, 288/294
   malignant cases: MSKCC, Queensland, ACEMID) cluster in the same poor
   0.576-0.607 AUC range — a broad, consistent collapse, not one anomalous
   population. University of Athens looked much better by AUC (0.921, n=6
   malignant) but this **did not hold up**: only 1 of Athens's 6 malignant
   cases was actually caught at any real operating threshold (16.7%
   sensitivity — consistent with everywhere else); its diagnosis mix
   (33.3% in-situ) is similar to MSKCC/Queensland, not easier. The high AUC
   was a small-sample artifact of AUC being rank-based with only 6 positive
   cases — one well-scored case pulled the ranking up while true
   threshold-based sensitivity remained just as poor. No equipment/protocol
   story survives here.

   **Case-severity mechanism, confirmed with data**: SLICE-3D's malignant
   cases are overwhelmingly early-stage. Of 37 melanomas with recorded
   Breslow thickness, mean is 0.49mm (clinically "thin," T1 territory) and
   the model missed all but 4 of them regardless of exact thickness.
   Melanoma in-situ specifically (60 of 294 malignant cases): flagged only
   6.7% of the time; invasive melanoma (39 cases): 10.3%. This is
   consistent with SLICE-3D being a **screening-detected population**
   (routine full-body imaging catches lesions early, while they're still
   visually subtle) vs. the training data's **clinically-referred**
   population (patients present because a lesion already looks
   concerning — inherently more advanced/obvious). Capture modality (3D
   total-body-photography vs. handheld dermatoscope) likely still
   contributes on top of this — the two explanations aren't mutually
   exclusive and weren't fully separated — but case-severity mismatch is
   now a confirmed, data-backed, independently actionable mechanism, not
   just a caveat. See `docs/RETRAIN_PLAN.md` for what this means for data
   mixing.

6. **Borderline predictions are unstable under image quality alone.** 4/20
   sampled images flipped predicted class under blur perturbation alone, 4/20
   under brightness alone (no JPEG needed for either); one case (baseline
   mel_score=0.414, already borderline) flipped NV→MEL under three different
   independent mild degradations (JPEG, blur, brightness). Test-time
   augmentation (`predict_tta`) roughly halves the blur-induced flip rate
   (4/20 → 2/20) but doesn't fully eliminate it, and doesn't help with JPEG
   compression at all.

7. **One real false-negative case, examined three independent ways**: a true
   melanoma scored as NV (mel_score=0.0037, 98.5% confidence in NV). Grad-CAM++
   attention was correctly on the lesion (no shortcut-learning artifact).
   Three independent uncertainty signals were tested against it — raw softmax
   confidence (98.5%, high), inter-model disagreement across the 3 ONNX
   sub-models (all agreed, low disagreement), and TTA-variance (std=0.0009,
   near zero) — **all three failed to flag it**. Convergent evidence that
   this failure mode is a genuine data/training gap, not something catchable
   by better uncertainty quantification.

8. **No scheduler existed anywhere.** `agent_runner.py`'s docstring claims a
   "scheduled" trigger; no cron/APScheduler/Celery config existed in the repo,
   and `render.yaml` defined only 3 `web` services. Drift/fairness monitoring
   had never run against real traffic.

9. **Skin-tone fairness check attempted (PAD-UFES-20, 2,298 images; DDI, 656
   images) — result is a bigger, different finding than expected.** Overall
   MelAUC is **0.778 (PAD-UFES-20)** and **0.629 (DDI)**, vs. 0.95+ on
   dermoscopy validation. Reason: both datasets are **clinical/smartphone
   photography, not dermoscopy** — the model has never been trained or
   validated on that modality at all, and the modality gap dominates any
   skin-tone-specific signal. Within that degraded regime:
   - PAD-UFES-20's Fitzpatrick distribution is itself skewed (types V-VI:
     11/2,298 images, 0.5%) — too few melanoma cases per skin-tone band
     (3-10-4) to draw a reliable conclusion; the apparent "Dark (IV-VI):
     AUC=0.935, sensitivity=100%" is statistical noise from **4** melanoma
     cases, not a real signal.
   - DDI is properly balanced by design (48-74 malignant cases per tone
     group) and shows a real, differently-shaped disparity, essentially
     unchanged by the double-softmax fix (its metrics are argmax-based,
     which turns out to be invariant to that specific bug): sensitivity is
     *higher* for dark skin (43.8% vs ~24.5% light/medium) but specificity
     is substantially *lower* (73.6% vs 86-93% light/medium) — i.e. more
     false alarms on darker-skinned patients in this clinical-photo
     setting, not more missed cancers.
   - **Net conclusion**: the original question ("is the model biased by skin
     tone on its actual intended use case, dermoscopy") remains unanswered —
     no skin-tone-labeled dermoscopy dataset was found to exist publicly.
     What *is* now confirmed: the model has zero validated performance on
     non-dermoscopic (phone-camera) images, full stop, regardless of skin
     tone. See `outputs/bias_reports/pad_ufes20_scored.csv` and
     `ddi_scored.csv` for the raw per-image data behind this.

10. **Train/serve ensembling-method mismatch — found, measured at scale, and
    the practical impact fixed.** `SA02_Ensemble.ipynb` uses two different
    formulas: one to derive the original `ensemble_metrics.json`
    (geometric/log-space pooling), another baked into the actually-deployed
    ONNX graph (arithmetic/probability-space pooling). Recomputed a
    self-consistent version on the full 10,312-image validation split using
    the real deployed formula: overall AUC barely moved (0.9609 → 0.9540),
    but the threshold that actually delivers 80% sensitivity moved
    substantially (**0.312 → 0.2385**) — the deployed threshold did not hit
    its own design target. `MEL_THRESHOLD` updated accordingly. The notebook
    itself still has both formulas (fixing that is Step 0 of
    `docs/RETRAIN_PLAN.md`, so the next retrain doesn't need this same
    correction pass), but the number actually running in the API is now
    right.

11. **Post-retrain (2026-07-25): the EMB/MILK10k/SLICE-3D data-mixing retrain
    completed, but the external-generalization question from finding #5
    remains genuinely unresolved — not fixed, not confirmed unfixed.** All
    three backbones were retrained on the expanded 73,456-image pool
    (68,472 ISIC + 250 SLICE-3D + 770 EMB + 3,964 MILK10k, per
    `docs/RETRAIN_PLAN.md` Steps 0-3) and re-ensembled via
    `SA02_Ensemble.ipynb`. Two results, pulling in different directions:

    - **Blended internal validation MelAUC moved down slightly**: 0.9540 →
      0.9452 (11,019-image validation split). Not itself concerning — trading
      a little blended-distribution fit for better generalization is exactly
      what mixing in harder external-style data is supposed to look like —
      but it doesn't confirm the trade paid off either.
    - **Genuine external re-test, done properly**: scored the new ensemble
      against only the 44 SLICE-3D malignant cases that were stratified into
      the training holdout and never trained on by any backbone (see
      `build_slice3d_samples()` in `SA01*_Train.ipynb`), paired with
      SLICE-3D's 2,500 benign images (never trained on at all, so
      leakage-free). Scoring against the full 294-malignant-case Permissive
      set — the same set used for finding #5's 0.593 figure — was considered
      and **deliberately rejected**: 250 of those 294 cases are now training
      data, so any AUC computed against the full set would be inflated by
      memorization, not a real measure of generalization. See
      `scripts/dataset_loaders.py::load_slice3d_holdout()`.

      Result: **AUC 0.4586** on the 44-case holdout (vs. finding #5's 0.593
      on the different, 294-case pre-retrain population — not a like-for-like
      comparison). **95% bootstrap CI: [0.348, 0.571]** (2,000 resamples,
      `random_state=42`) — this interval spans both "worse than before" and
      "indistinguishable from chance," and a Mann-Whitney U-test on the same
      split gives p=0.345, confirming the point estimate is not
      statistically distinguishable from noise at n=44. Verified this isn't
      a computation error: cross-checked via an independent Mann-Whitney-based
      AUC calculation (exact match), confirmed no duplicate/missing/corrupted
      rows, confirmed the malignant group's mean mel_score (0.134) is
      directionally higher than benign (0.090) despite the sub-0.5 AUC —
      consistent with a small, noisy sample rather than a broken pipeline.

    **Honest conclusion**: whether this retrain closed any of finding #5's
    external-generalization gap is **not answered** by this test — the
    holdout is too small (n=44 malignant) to distinguish a real effect from
    sampling noise in either direction. A trustworthy answer needs either a
    genuinely larger external malignant test set (e.g. sourcing the full
    non-Permissive SLICE-3D release, ~55 additional metadata columns and
    more institutions, CC-BY-NC — see `docs/KNOWN_GAPS.md`) or accepting this
    as an open question until one becomes available. Raw scored data:
    `outputs/bias_reports/slice3d_holdout_postretrain_scored.csv`.

12. **Finding #11 resolved with a bigger sample: real, statistically
    significant discriminative improvement confirmed — but a new, separate
    calibration problem surfaced.** Sourced and downloaded the full
    non-Permissive SLICE-3D release (401,059 images, 7 institutions vs the
    Permissive subset's 4, CC-BY-NC). Verified it contains 393 malignant
    cases vs Permissive's 294. Built a genuinely-unseen test pool: the 3
    institutions absent from the Permissive subset contribute 99 malignant
    cases never available to training at all, plus the original 44-case
    holdout, for **143 malignant cases total** — confirmed zero overlap with
    the 250 malignant cases in training via an explicit assertion (see
    `scripts/dataset_loaders.py::load_slice3d_full_holdout()`).

    Result: **AUC 0.6131**, 95% CI **[0.5516, 0.6751]** (2,000 bootstrap
    resamples) — this interval no longer spans chance. Mann-Whitney
    cross-check matches exactly, p=5.2×10⁻⁶ (highly significant). Verified
    not a data artifact: 0 nulls, 0 duplicate ids, malignant mean mel_score
    (0.225) clearly above benign (0.086). **This resolves finding #11's
    "inconclusive" verdict for the discrimination question**: at n=143 the
    model does have real, significant ability to rank malignant SLICE-3D
    cases above benign ones — a genuine improvement in kind (not just
    magnitude) over the earlier underpowered n=44 test, and directionally
    better than finding #5's original 0.593 (not a strict like-for-like
    comparison — different, only partly-overlapping populations — but no
    longer contradicted by it either).

    **What it does NOT resolve — a newly-confirmed threshold-transfer
    problem, distinct from the discrimination question above**: the
    currently-deployed threshold (0.119, tuned on the blended internal
    validation set) delivers only **42.7% sensitivity** on this population
    (misses 82 of 143 real malignancies) — clearly below the 80% target.
    Pushing the threshold down to whatever value *does* hit 80% sensitivity
    on this population (0.0199) collapses specificity to **15.6%** (flags
    ~85% of benign cases). This is a real, separate finding: the *ranking*
    of cases (AUC) transfers reasonably well to this external population,
    but the *absolute score scale* the deployed threshold assumes does not —
    consistent with the case-severity/capture-modality distribution shift
    already diagnosed in finding #5. See `docs/RETRAIN_PLAN.md`'s new
    "Path A / Path B" section for the fixes: Path A (population-conditional
    threshold) is implemented and tested (see engineering table below);
    Path B (improving discrimination itself — more real TBP training data,
    metadata/GBDT fusion, segmentation-assisted classification) is planned,
    not yet started.

13. **Authentication added to the by-id patient-data lookup endpoints —
    the top-priority item flagged across two independent audit passes,
    with a deliberately narrower scope than "everything" once the real
    frontend was checked.** `GET /report/{scan_id}`, `/report/{scan_id}/fhir`,
    and `/patient/{patient_id}/history` now require an `X-API-Key` header
    (`api/SA05_api.py`'s `require_api_key`, fails closed — a misconfigured
    deploy with the key unset rejects every request rather than silently
    running with no auth, same lesson as the `MODEL_MODE` default bug).

    **Originally left NOT gated: `/analyze`, `/analyze/batch`,
    `/audit/log`.** Checked every frontend page before deciding this (not
    assumed): the old frontend called these directly from browser
    JavaScript, and this codebase had no backend proxy or session layer to
    hold a secret server-side — embedding `SKINANALYTICA_API_KEY` in
    public page source would have been visible via view-source, not real
    protection. Documented as an open gap rather than left unmentioned.

    **Closed 2026-07-27**, once the frontend was rebuilt: the new
    `assets/auth.js` uses a client-supplied-key pattern instead — the same
    model as Swagger's own "Authorize" button. The user types their key
    into a widget on any page; the browser holds it in `localStorage` only;
    it is never embedded in any shipped page source. That removed the
    actual reason the three endpoints were left open, so all six
    patient-data endpoints are now gated identically. Regression-tested at
    the HTTP layer (`tests/test_auth.py`, 12 tests using
    `fastapi.testclient.TestClient`).

14. **Skin-tone fairness on dermoscopy — finally answered, partially good
    news, with real caveats.** Finding #9 flagged this as unanswered
    because PAD-UFES-20 and DDI turned out to be clinical photography, not
    dermoscopy. MILK10k's own reserved holdout (700 images,
    skin-tone-stratified, genuinely never trained on — see
    `build_milk10k_samples()`) was scored for the first time this session
    against the retrained ensemble
    (`scripts/dataset_loaders.py::load_milk10k_holdout()`,
    `outputs/bias_reports/milk10k_holdout_scored.csv`).

    **Overall**: MelAUC 0.8796, 95% bootstrap CI [0.8340, 0.9213]
    (2,000 resamples) — a real, clearly-above-chance result on genuine
    dermoscopy skin-tone-diverse data, the first of its kind this project
    has had.

    **Per-skin-tone-class breakdown** (0=darkest, 5=lightest;
    `skin_tone_class` from MILK10k's own annotation):

    | Tone | n (mel) | AUC | 95% CI | Sensitivity @ 0.119 |
    |---|---|---|---|---|
    | 0 | 1 (0) | — | — | — (no melanoma cases) |
    | 1 | 15 (3) | 0.8611 | not computed, n too small | 33.3% |
    | 2 | 72 (13) | 0.8449 | [0.6816, 0.9805] | 76.9% |
    | 3 | 424 (28) | 0.8981 | [0.8297, 0.9534] | 64.3% |
    | 4 | 137 (18) | 0.8459 | [0.7329, 0.9417] | 55.6% |
    | 5 | 51 (2) | 1.0000 | not computed, n too small | 100.0% |

    **Honest reading, same discipline as finding #9's DDI/PAD-UFES-20
    read**: tones 0, 1, and 5 have too few melanoma cases (0, 3, 2) to
    trust — the tone-5 "perfect" AUC is the same small-sample artifact
    already seen with Athens (finding #5) and PAD-UFES-20's dark band
    (finding #9), not a real result. Restricting to the three adequately-
    sampled bands (2, 3, 4): their **AUC confidence intervals overlap
    substantially** — no statistically confirmed AUC-level disparity
    across skin tone on this evidence. Their **sensitivity at the deployed
    threshold does spread more** (76.9% / 64.3% / 55.6%), but with only
    13-28 melanoma cases per band this is suggestive, not confirmed — each
    single case shifts the percentage by several points. **This also
    reproduces the broader threshold-transfer pattern from finding #12**:
    overall sensitivity at the deployed threshold is only 64.1% here
    (vs. the 80% target), on data that is genuinely dermoscopic and in the
    product's real scope — a smaller version of the same problem, not
    limited to the TBP/SLICE-3D case.

    **Net conclusion**: no confirmed skin-tone bias at the AUC level on
    this evidence, but (a) the darkest and lightest bands remain
    unanswerable from available data, (b) the sensitivity spread across
    tones 2/3/4 is a real open question worth more data to resolve, and
    (c) the deployed threshold under-delivers on this dermoscopic holdout
    the same way it did on the TBP one, independent of skin tone.

    **Addressed the same day (2026-07-26)**: `SKIN_TONE_THRESHOLDS`
    (`api/SA05_api.py`) gives tones 2/3/4 their own ROC-derived threshold
    for ~80% sensitivity, opt-in via a new `skin_tone_class` parameter on
    `/analyze` — same architecture and same "no caller wires it up yet"
    status as `capture_mode="tbp"` from finding #12/Path A, so this cannot
    silently change behavior for any existing request. Explicitly labeled
    provisional in code and config comments, not validated the way
    `AGE_BAND_THRESHOLDS` is (585 melanoma cases there vs. 13-28 here) —
    revisit once a larger skin-tone-labeled dermoscopy sample exists.

    **Addendum: sex and age breakdown on the same holdout** (no new
    scoring run needed — `age_approx`/`sex` were already in
    `milk10k_holdout_scored.csv`).

    | Split | n (mel) | AUC | 95% CI | Sensitivity @ 0.119 |
    |---|---|---|---|---|
    | Male | 404 (34) | 0.891 | [0.824, 0.950] | 73.5% |
    | Female | 296 (30) | 0.878 | [0.809, 0.941] | 53.3% |
    | Age <40 | 57 (7) | 0.834 | [0.561, 0.992] | too few to trust |
    | Age 40-60 | 200 (23) | 0.895 | [0.803, 0.965] | 69.6% |
    | Age 60+ | 438 (34) | 0.881 | [0.823, 0.933] | 55.9% |

    Same pattern as the skin-tone read: AUC confidence intervals overlap
    for every adequately-sampled split (no confirmed discrimination gap by
    sex or age on this data), but sensitivity-at-threshold spreads more
    than AUC does, and the samples (30-34 melanoma cases per group) are
    too thin to call the sensitivity gap confirmed rather than noise.

    **A more concrete finding**: checked whether the *existing*
    `AGE_BAND_THRESHOLDS` (derived from ISIC-2020, already in production)
    actually close this gap on MILK10k specifically — they don't fully
    transfer. Applying them to this holdout gives the 60-75 band only
    **45.5% sensitivity** (n=22 melanoma cases), worse than the 55.9%
    a flat threshold gave the broader 60+ group above. This is a real,
    if modestly-sampled, sign that `AGE_BAND_THRESHOLDS` — like
    `MEL_THRESHOLD` and `TBP_MEL_THRESHOLD` before it — doesn't
    generalize perfectly across datasets, consistent with the caveat
    already on `AGE_BAND_THRESHOLDS` ("derived from a single,
    not-fully-external ISIC-2020 sample... revisit once further
    validation exists"). **Not re-derived this session** — 22 cases in
    one band isn't a large enough basis to replace an existing,
    already-provisional threshold without more data than that.

15. **Non-dermoscopic-input guardrail added — and a genuine negative
    result found and fixed before shipping.** SkinAnalytica has zero
    validated performance on non-dermoscopic images (finding #9), and no
    way to detect that a wrong-format image was uploaded. A first version
    of `check_dermatoscope_likelihood()` (`src/inference_utils.py`) tried
    two signals: a corner-vs-center brightness "vignette" check (theory:
    a dermatoscope's optical housing darkens image corners) and a
    near-square aspect-ratio check.

    **Both failed real-data validation, run before shipping rather than
    after.** On 15 real ISIC images, the vignette signal swung from +143
    to -85 with no consistent sign — it's dominated by lesion pigmentation
    (a dark lesion reads as a "vignette" even with none), not capture
    equipment. Worse, real dermatoscope-domain images across every dataset
    this project uses span aspect ratios far outside "near-square":
    SLICE-3D crops are 1.00, MILK10k/most ISIC sources are 1.33 (600×450),
    ISIC 2020 goes up to 1.78. The original "near-square" threshold (1.15)
    would have flagged a large fraction of this project's own real
    training images as "not a dermatoscope image."

    **Fixed by recalibrating against real data rather than assumption**:
    dropped the vignette signal entirely, widened the aspect-ratio
    threshold to 2.2 (safely above every real ratio observed). Verified
    with **zero false positives across 1,200 real images sampled from all
    six project datasets** (SLICE-3D, EMB, MILK10k, ISIC 2018/2019/2020,
    200 each) before considering this done. The honest tradeoff: this now
    only catches genuinely extreme cases (a panorama, a heavily
    letterboxed photo) — ordinary 4:3 or 16:9 phone photos overlap real
    dermatoscope aspect ratios almost completely, so this is a narrow,
    low-confidence advisory signal, not a real image-type classifier.
    Surfaced as a warning banner in the new frontend
    (`index.html`) when triggered, never as a hard block on inference.

16. **Annotation-automation rate — computed for the first time, and it
    surfaces a real cost of the sensitivity-tuned thresholds that pure AUC/
    sensitivity numbers don't show.** `docs/RETRAIN_PLAN.md`'s goal
    statement names "annotation automation rate" (the fraction of cases the
    deployed verdict logic — `CANCER_FLAGGED`/`NORMAL` vs.
    `REVIEW_REQUIRED` — can resolve without a human annotator) as the metric
    that actually matters, alongside raw AUC. It had never been computed.
    Replicated `make_verdict()`'s exact routing logic (age-conditional
    threshold where age is available, flat `MEL_THRESHOLD` otherwise)
    against the real scored CSVs already on disk
    (`outputs/bias_reports/*_scored.csv`).

    **On near-training-distribution data** (`isic2020_scored_sample.csv`,
    1,764 images, 33% melanoma — itself an enriched validation sample, not
    natural prevalence): **94.6% auto-resolved**, only 5.4% sent to review.

    **On MILK10k's genuinely-never-trained-on holdout** (700 images, real
    external dermoscopy): **82.9% auto-resolved** as-is — but that holdout
    is itself deliberately enriched (495/700 = 70.7% malignant-class, mostly
    BCC, for statistical power on sensitivity), nothing like a real incoming
    case mix, so this number alone is not a usable estimate of real-world
    automation rate.

    **Resampled MILK10k's holdout (still zero leakage — same never-trained-
    on images, just recombined at different ratios) to approximate a range
    of plausible real screening-population prevalences, and tracked
    flagged-precision (of cases the system confidently flagged as
    `CANCER_FLAGGED`, what fraction were actually malignant) separately from
    automation rate — the two tell very different stories**:

    | Assumed malignant prevalence | n (malignant) | Auto-resolved | Flagged-precision (single draw) | Flagged-recall |
    |---|---|---|---|---|
    | 70.7% (holdout as-is, not representative) | 700 (495) | 82.9% | 82.0% | 63.4% |
    | 20% | 256 (51) | 80.5% | 33.7% | 68.6% |
    | 10% | 228 (23) | 78.9% | 15.9% | 56.5% |
    | 5% | 216 (11) | 77.3% | 8.0% | 54.5% |
    | 2% | 209 (4) | 78.5% | 4.2% | 75.0% |

    ~~The same pattern replicates on ISIC-2020 ... flagged-precision at its
    real natural prevalence (1.76%) drops to 29.8%~~

    **⚠ Correction and rigor upgrade, 2026-07-28 — see finding #27**: the
    single-draw flagged-precision numbers above (this table's rightmost
    numeric column and the 29.8% ISIC-2020 figure) were computed against
    `MEL_THRESHOLD=0.2385`, superseded by finding #26's move to **0.5010**.
    Both were recomputed with proper bootstrap confidence intervals under
    the current threshold — see finding #27 for the refreshed numbers
    (flagged-precision at every prevalence level is now *higher* than
    shown here, a stricter threshold flagging fewer but more-confident
    cases, as expected) and `outputs/bias_reports/isic2020_bootstrap_flagged_precision.json`.
    The qualitative conclusion below is unaffected: flagged-precision at
    realistic prevalence is still far below the enriched-sample numbers,
    just not as low as this now-stale table shows.

    **What this means, stated plainly**: automation rate on its own is a
    misleading headline number. It stays high (77-97%) across every
    prevalence tested, because most cases are confidently `NORMAL` — but at
    realistic low prevalence, the `CANCER_FLAGGED` bucket that's supposed to
    represent "confidently handled" cases is dominated by false alarms
    (originally reported as 4-30% precision, refreshed to **~4-37% [with
    95% CIs as wide as ±7pp] under the current threshold — see finding
    #27**, not 82-91%). Those flagged cases don't go back to an
    annotator for relabeling, so they count as "automated" by the strict
    definition — but in practice they'd still need clinical follow-up to
    resolve, which is real downstream work the automation-rate number
    doesn't capture. This is the same underlying cause as finding #12's
    threshold-transfer problem (sensitivity-tuned thresholds trade away
    specificity), viewed through the annotation-workload lens instead of the
    AUC lens — both point at the same fix (Path B, `docs/RETRAIN_PLAN.md`),
    not two separate problems.

    **Caveats**: no genuinely-natural-prevalence, zero-leakage,
    dermatoscope-domain holdout currently exists in this project — MILK10k's
    resampling is leakage-free but assumption-driven on the target
    prevalence (no authoritative real-world prevalence number for this
    product's actual intended screening population is on record); ISIC-2020's
    version carries a real leakage caveat. At 2-10% assumed prevalence the
    malignant-case counts are very small (4-23 cases) — directionally
    informative, not statistically precise. Raw scored data and the
    resampling script's logic are reproducible from
    `outputs/bias_reports/milk10k_holdout_scored.csv` and
    `outputs/bias_reports/isic2020_scored_sample.csv` plus `make_verdict()`
    in `api/SA05_api.py` — no new scoring run was needed, this is a
    reanalysis of existing scored data.

17. **ISIC-2018 is entirely duplicated inside ISIC-2019's training pool —
    found as a side effect of Path B's embedding extraction, not looked
    for deliberately.** While extracting per-image CNN embeddings for the
    real training pool (`scripts/extract_embeddings.py`, `--split train`;
    see `docs/RETRAIN_PLAN.md` "Workstream 1 -- implementation-ready
    scope"), an integrity check before declaring the extraction done
    found 73,456 total rows but only 63,441 unique `image_id`s — exactly
    10,015 duplicates, matching ISIC-2018's entire sample count.

    **Verified, not assumed**: every one of ISIC-2018's 10,015 images
    (e.g. `ISIC_0024306`) also appears under `source="isic19"` with the
    identical `image_id`. Checked label consistency first (zero
    conflicts — both copies always agree on class) and then checked the
    embeddings themselves: **max absolute difference between the two
    copies' embeddings is 0.0** — these aren't just the same lesion
    re-exported, they're the same JPEG file, byte-for-byte, sampled
    twice into the same training pool under two different source tags.
    This makes sense in retrospect (ISIC-2019's training set is a
    known superset that folds in HAM10000/ISIC-2018 alongside BCN20000
    and MSK), but `build_sample_list()` (ported faithfully from
    `notebooks/SA01_EfficientNet_Train.ipynb` Cell 6 into
    `scripts/build_training_pool.py`, verified to reproduce the exact
    documented 73,456-image pool) treats the two as independent,
    non-overlapping sources.

    **Practical effect**: those 10,015 images got roughly 2x the
    sampling weight of every other image in the ~73K training pool
    during the actual retrain that already happened — on top of
    whatever the existing inverse-class-frequency sampler and
    `young_mel_boost` age-reweighting already do. Not necessarily a
    large effect (10,015 of 73,456 is ~14% of the pool, and the
    duplication doesn't change *which* classes are overrepresented,
    just adds extra weight to a subset already present at full weight
    once), but it's a real, previously-undocumented deviation from the
    "each image seen once per epoch, weighted only by the intended
    sampler" assumption every other finding in this document assumes
    holds.

    **Not fixed at the CNN level this session** — the already-completed
    retrain isn't being redone for this. **Fixed at the fusion level**:
    `scripts/dedup_train_embeddings.py` produces
    `outputs/embeddings/train_embeddings_deduped.npz` (63,441 unique
    rows, one copy per `image_id`, first occurrence in pool order kept)
    from the raw extraction, so Path B's GBDT fusion model (workstream 1
    task 3, not yet started) doesn't inherit the same double-counting.
    The raw, undeduplicated `train_embeddings.npz` is kept as-is
    deliberately — it's the accurate record of what the real training
    pool actually contained, duplication included, not something to
    silently paper over.

18. **Path B workstream 1 (metadata/GBDT fusion) evaluated against both
    external holdouts — mixed result, reported exactly as found, not
    rounded toward the outcome that would look best.** The GBDT trained
    in finding #17's wake (`scripts/train_fusion_gbdt.py`, leakage
    checked against both holdouts before training) was evaluated three
    ways (`scripts/evaluate_fusion.py`), matching the three-part plan in
    `docs/RETRAIN_PLAN.md`'s "Workstream 1" scope:

    **Discrimination on SLICE-3D (the original success criterion)**: AUC
    **0.5755**, 95% CI [0.5203, 0.6331] — *not* an improvement on the
    0.6131 pre-fusion baseline (finding #12); the baseline falls inside
    the fusion model's own CI, so this is statistically indistinguishable
    from no change, not a regression either. **This matches what the
    scoping flagged going in**: SLICE-3D's usable metadata is thin (one
    derived confidence score, no local age/sex/site), so fusion was
    always expected to help ISIC19/20-sourced data more than SLICE-3D
    specifically. Not fixed, not worsened — an honest null result on the
    axis that was least likely to move.

    **Flagged-precision on MILK10k at realistic prevalence (the metric
    that actually activated Path B, per finding #16)**: modestly,
    mostly positive. Re-running finding #16's own resampling methodology
    with the fusion model's probabilities routed through the *same*
    unmodified `make_verdict()` thresholds:

    | Prevalence | Baseline (pre-fusion) | Fusion | Delta |
    |---|---|---|---|
    | 2% | 4.2% | 3.9% | −0.3pp |
    | 5% | 8.0% | 11.9% | +3.9pp |
    | 10% | 15.9% | 22.1% | +6.2pp |
    | 20% | 33.7% | 38.3% | +4.6pp |

    Directionally encouraging at 5/10/20% prevalence, flat (noise-level)
    at 2% — expected, since only 4 malignant cases back that column.
    **This is a single resampling draw, not bootstrap-averaged** (unlike
    the AUC numbers below), so treat these deltas as suggestive, not
    confirmed — the same caution this document applies to every other
    small-sample result.

    **Subgroup check — the result that most needed careful treatment,
    not enthusiasm.** Point-estimate AUCs improved across all three
    adequately-sampled skin-tone bands (tone 2: 0.8449→0.9257, tone 3:
    0.8981→0.9682, tone 4: 0.8459→0.9043, vs. finding #14's original
    numbers) and both adequately-sampled age bands. **But none of these
    clear the OLD confidence intervals** once bootstrap CIs were computed
    for the fusion model's own bands (tone 2: [0.838, 0.984] vs. finding
    #14's original [0.6816, 0.9805] — heavy overlap; tone 3: [0.938,
    0.993] vs. original [0.8297, 0.9534] — close, but the fusion CI's
    lower bound (0.938) doesn't clear the original's upper bound
    (0.9534); tone 4 similar). **Not a confirmed AUC improvement per
    band** at these sample sizes (13-28 melanoma cases per band).

    Sensitivity-at-threshold told a much more dramatic story on its own
    (tone 2 +23.1pp, tone 3 +25.0pp, tone 4 +38.8pp, age 60+ +29.4pp) —
    **deliberately not reported as a fairness win**: those numbers used
    the *unmodified* `SKIN_TONE_THRESHOLDS`/`AGE_BAND_THRESHOLDS`,
    calibrated against the pre-fusion CNN ensemble's score distribution,
    applied unchanged to a differently-shaped fusion-model score
    distribution. Given the AUC-level (threshold-independent) comparison
    shows no statistically confirmed improvement, the much larger
    sensitivity jump is most plausibly a threshold/distribution mismatch
    — the fusion model's scores likely cluster differently near the old
    cutoff — rather than genuinely better subgroup discrimination. Stating
    this plainly rather than leading with the more impressive-looking
    number is the same discipline finding #5's Athens small-sample AUC
    artifact and finding #9's PAD-UFES-20 dark-band noise required.

    **Net read**: Path B's first workstream is a real, modest, mostly
    positive signal on the metric that motivated it (MILK10k
    flagged-precision), a clean null on the metric it was less likely to
    help (SLICE-3D AUC), and a reminder that a promising-looking
    subgroup number needs its own confidence interval before being
    reported as a fairness fix — not a finished result either way.
    **Not deployed**: stays behind the planned opt-in flag
    (`SKINANALYTICA_ENABLE_FUSION`, workstream 1 task 4, not yet wired
    up) pending a decision on whether this result clears the bar to
    ship, informed by this evaluation rather than by the training-time
    internal-validation number alone (0.9854 MelAUC, same-distribution,
    not the real test — see `docs/RETRAIN_PLAN.md`).

19. **Finding #18's flagged-precision result, properly bootstrapped —
    the headline changes.** Finding #18's MILK10k flagged-precision
    deltas (+3.9pp/+6.2pp/+4.6pp at 5%/10%/20% assumed prevalence) came
    from a single resampling draw per prevalence level, explicitly
    flagged there as "suggestive, not confirmed." `scripts/
    bootstrap_flagged_precision.py` reruns it properly: a **paired**
    bootstrap (2,000 resamples per prevalence level, same resampled
    index set scored under both the baseline CNN ensemble and the fusion
    model each draw, so prevalence-sampling noise common to both cancels
    out of the delta — higher power than two independent bootstraps).

    | Prevalence | Fusion precision | Baseline precision | Delta (95% CI) | Real? |
    |---|---|---|---|---|
    | 2% | 4.8% [3.3, 6.1] | 3.6% [1.2, 6.0] | +1.3pp [−1.0, +4.1] | No |
    | 5% | 12.4% [9.9, 15.1] | 9.3% [5.1, 13.4] | +3.1pp [−1.0, +7.5] | No |
    | 10% | 22.7% [19.1, 26.6] | 17.4% [12.3, 22.8] | +5.3pp [−0.2, +10.8] | No (barely) |
    | 20% | 39.4% [35.0, 44.4] | 32.0% [25.7, 38.4] | **+7.4pp [+1.1, +13.8]** | **Yes** |

    **Only the 20% prevalence point has a delta CI that excludes zero.**
    At 2%, 5%, and 10% — the range more representative of a realistic
    screening population, and the range this whole investigation (finding
    #16) was originally motivated by — the improvement is consistently
    positive in direction across all four prevalence levels (never
    negative) but not statistically distinguishable from noise at current
    sample sizes. 10% came closest (CI lower bound −0.2, barely touching
    zero).

    ~~**Revised net read, superseding finding #18's "modestly, mostly
    positive" framing**: Path B's fusion approach shows a real,
    bootstrap-confirmed benefit at higher prevalence...~~

    **⚠ Correction, 2026-07-28 — see finding #27**: the table above was
    computed against `MEL_THRESHOLD=0.2385`, superseded by finding #26.
    Re-run under the current threshold (0.5010), the picture changes in a
    way worth reporting plainly: **no prevalence level's delta CI excludes
    zero anymore**, including 20% (was the one confirmed point above).
    Fusion's precision numbers are directionally still higher than
    baseline's at every level, same as before — but none of that
    difference is statistically confirmed under the new threshold at this
    sample size. This is not evidence fusion stopped helping; it's evidence
    that finding #19's one confirmed data point didn't survive a threshold
    change it was never tested against, which is exactly why it's being
    reported here rather than left standing. See finding #27 for the full
    refreshed table and what this means for Path B.

20. **MRA-MIDAS sourced, loaded, and evaluated as a genuine second
    external holdout — and the result is sobering, not a confirmation.**
    Following up on finding #19's confirmation gap, MRA-MIDAS (Stanford
    AIMI + Cleveland Clinic, non-commercial DUA, real biopsy-confirmed
    labels) was registered for, downloaded, and loaded
    (`scripts/dataset_loaders.py::load_mra_midas()`, dermoscopy-modality
    only, 786 usable images after filtering out 503 control/non-lesion
    images and 260 non-dermoscopy captures). Never touched by any
    training step — the CNNs and the fusion GBDT both predate this
    dataset being sourced — so the full 786-image set serves as a
    holdout directly, no split needed.

    **Both the baseline ensemble and the fusion model were evaluated
    against it, paired (same images, same bootstrap draws):**

    | Metric | Fusion | Baseline | Distinguishable? |
    |---|---|---|---|
    | MelAUC (mel vs rest) | 0.5361 [0.4881, 0.5841] | 0.5793 [0.5328, 0.6262] | No |
    | ~~Malignant-vs-benign AUC~~ | ~~0.3932~~ | ~~0.3947~~ | superseded — see finding #23 |

    **⚠ Correction, 2026-07-28 — see finding #23**: the malignant-vs-benign
    row above used a flawed ranking metric (`mel_score` alone, which
    under-credits true BCC/AKIEC cases). Corrected: **Fusion 0.7375
    [0.7024, 0.7727], Baseline 0.7051 [0.6663, 0.7432]** — real, solid
    discrimination, not the near-chance result originally reported here.
    The MelAUC row above is unaffected by the bug and still accurate.

    **Verified this isn't a pipeline bug before reporting it**: true
    melanoma cases have a *lower* median `mel_score` (0.0685) than true
    nevus cases (0.0921) on this population — checked individual rows,
    not just the aggregate, and the model does correctly score some real
    melanomas very high (0.86-0.95, correctly classified) while missing
    others badly (0.07, misclassified as `bkl`) — a real, inconsistent
    ranking failure, not a labels-swapped artifact.

    **Flagged-precision at realistic assumed prevalence** (same paired
    bootstrap methodology as finding #19): fusion numerically ahead at
    every prevalence tested (+0.4pp to +2.6pp, 2-20%), but every single
    delta's confidence interval includes zero — none distinguishable
    from noise here, a weaker result than even MILK10k's 20%-prevalence
    finding.

    **What this means for finding #19's open question**: MRA-MIDAS does
    **not** provide new evidence that fusion helps — the fusion-vs-baseline
    gap is not distinguishable on either metric, corrected or not. On
    whether the model discriminates *at all* here, see the finding #23
    correction above — melanoma-specific discrimination is genuinely
    weak, but the model can distinguish malignant from benign broadly
    (0.70-0.74) more capably than this finding originally reported. This
    is still a real external-generalization limit, alongside the
    original SLICE-3D collapse in finding #5 and MILK10k's own
    threshold-transfer gap in finding #12 — just a more precisely-scoped
    one than "the model doesn't work here" after the correction. MRA-MIDAS
    uses different capture hardware (Dermlite DL3/4, Canfield Scientific
    VEOS) and a different
    clinical population (Stanford/Cleveland Clinic dermatology referrals)
    than anything in the training data — plausible contributors, not
    confirmed causes. Not something Path B's current approach fixes;
    revisit alongside Path B workstream 2/3 (more real external training
    data, segmentation) rather than treating finding #18/#19's MILK10k
    result as the final word on whether fusion works.

21. **Why external generalization keeps failing, investigated directly
    rather than left as "different institutions" hand-waving — two
    distinct, opposite mechanisms, found by looking at actual
    preprocessed images, not just scores.** Finding #20 left an open
    question: why does discrimination collapse so consistently across
    independent external populations? Rather than speculate, this
    session pulled real example images through the exact production
    preprocessing pipeline (`inference_utils.preprocess_bgr()`) and
    looked at what the model actually sees.

    **MRA-MIDAS: the lesion is a small mark inside a huge frame.**
    Measured across all 786 dermoscopy images: median resolution
    3024x4032 (89% portrait-orientation, matching the source paper's
    description of full-resolution iPhone/iPad photos taken through a
    handheld Dermlite/Canfield dermatoscope attachment). Visually
    confirmed on paired examples: a true melanoma the model scored at
    0.0696 (missed) shows the lesion as a faint ~5-10%-of-frame mark
    inside the dermatoscope's circular field of view, with a visible
    ruler/scale overlay never seen in training data; a true melanoma from
    the *same dataset* scored at 0.9454 (caught) shows a lesion filling a
    visibly larger fraction of the frame. **This within-dataset contrast
    is the important evidence** — it's not just "MRA-MIDAS looks
    different," it's specifically how much of the frame the lesion
    occupies that tracks with the model's success, on the same source,
    same equipment, same population. `preprocess_bgr()`'s resize step is
    a naive `cv2.resize()` straight to 224x224 with no lesion-localizing
    crop first, so a huge image with a small, off-center lesion loses
    almost all of its effective resolution on the actual signal.

    **SLICE-3D: the opposite problem — not enough native resolution to
    begin with.** Measured across 150 sampled holdout images: median
    width 128px (range 87-189px) — these are already-tight lesion crops
    auto-extracted from wide-field 3D-total-body-photography scans by
    SLICE-3D's own upstream pipeline, before this project ever receives
    them. Visually confirmed: the lowest-scored true malignant case in
    the holdout (mel_score 0.0024, 119x119 source) shows a visibly
    blurred, low-detail lesion after the same resize-to-224 step — the
    lesion fills the frame here (good framing, unlike MRA-MIDAS), but
    there was never enough real detail in the source to recover. **Same
    within-dataset check applied here too**: the highest-scored true
    malignant case (mel_score 0.7287, 143x143 source — still small, but
    the largest available) shows a visibly more distinct, higher-contrast
    lesion than the worst case, even though both are well below training
    data's typical resolution. Confirms the mechanism is resolution/detail,
    consistently, not a one-off example.

    **Comfort-zone confirmation on the population that actually works**:
    MILK10k (MelAUC 0.8796, finding #14) sits squarely in the same regime
    as training data — a spot-checked true melanoma (450x600 source,
    landscape) shows the lesion filling roughly two-thirds of the frame
    after preprocessing, sharp and high-contrast, visually similar in
    character to the ISIC training example above. This is the fourth data
    point, not a coincidence: every population that performs well
    (training data, MILK10k) shares the same moderate-resolution,
    pre-cropped-to-the-lesion framing; every population that fails
    (SLICE-3D, MRA-MIDAS) deviates from it in one of the two opposite
    directions.

    **Why this matters more than another score-level number**: SLICE-3D
    and MRA-MIDAS looked like the same generic "external generalization
    problem" from the outside, but they fail for opposite reasons, which
    means they need different fixes. Segmentation-assisted classification
    (Path B workstream 3 — crop to the lesion before classifying) is the
    directly-indicated fix for MRA-MIDAS's failure mode: it would recover
    the effective resolution currently being wasted on background and the
    dermatoscope vignette. It does **not** help SLICE-3D — there's no
    missing crop to recover, the native pixels simply aren't there, so
    that population's ceiling needs either higher-resolution source data
    (if the full, uncropped TBP scans are obtainable) or explicit
    resolution-robustness training (e.g. augmenting training data with
    aggressive downsample/upsample cycles so the model learns to work
    with genuinely blurry input, rather than assuming production-quality
    detail is always available). Metadata fusion (workstream 1, findings
    #18-#20) doesn't address either mechanism, which is consistent with
    why it didn't move either dataset's numbers.

    **Ablation run, 2026-07-27 — confirms the mechanism, with an honest
    "promising, not proven" result.** `scripts/ablation_centercrop_mra_midas.py`
    re-scored all 786 MRA-MIDAS images through the exact same trained CNN
    ensemble, changing only the preprocessing: center-crop to a square
    (shorter side) before the resize-to-224 step, instead of
    `preprocess_bgr()`'s naive full-frame squish. No retraining — if
    this alone moved the numbers, it's a real, immediately actionable
    fix, not just a diagnosis.

    | Metric | Naive resize | Center-crop | Delta | Paired bootstrap 95% CI | Confirmed? |
    |---|---|---|---|---|---|
    | MelAUC | 0.5793 | 0.6002 | +0.0211 | [−0.0023, +0.0450] | No — close, but includes 0 |
    | Malignant-vs-benign AUC | 0.3947 | 0.4367 | +0.0414 | [+0.0185, +0.0645] | Yes — excludes 0 |

    Both metrics moved in the predicted direction on a pure preprocessing
    change, no retraining — real support for the mechanism. MelAUC's
    improvement doesn't clear a 95% CI at n=786 (184 melanoma cases), so
    it's directionally consistent but not confirmed on its own; the
    malignant-vs-benign delta does clear its CI, though that metric
    ranks BCC/AKIEC cases using `mel_score` alone, which isn't really the
    right ranking variable for non-melanoma malignant classes (see the
    MILK10k guardrail check below) — so treat it as corroborating, not
    as the stronger of the two results just because its CI is tighter.

    **Guardrail check, same ablation applied to MILK10k** (the population
    that already works well) — critical before treating center-crop as a
    real candidate fix: does it help MRA-MIDAS at the cost of hurting a
    population that's already fine? MelAUC 0.8796 → 0.8775 (delta
    −0.0020) — negligible, safe. (The same ablation's malignant-vs-benign
    AUC *appeared* to drop further, 0.4293 → 0.3837 — flagged at the time
    as likely a metric artifact, not a real regression, since MILK10k's
    malignant class is 76.8% BCC, the exact case where
    `mel_score`-as-ranking-variable is least appropriate. **Confirmed
    correct, 2026-07-28, finding #23**: with the metric actually fixed,
    the real delta is 0.7905 → 0.7857 — negligible, exactly as predicted,
    not the concerning-looking drop the flawed metric showed.)

    **Net conclusion**: a zero-retraining preprocessing change
    (center-crop before resize) produces a real, mechanistically-explained,
    same-direction improvement on MRA-MIDAS's framing-mismatch failure
    mode, with no measured cost to MILK10k — genuinely promising as a
    cheap, immediate first step, though the modest, not-fully-CI-confirmed
    magnitude (0.58→0.60 MelAUC) means this is a partial mitigation, not
    a fix — MRA-MIDAS's discrimination is still far below MILK10k's 0.88.
    Reasonable next steps, not yet started: (1) a proper lesion-detecting
    crop (not just center-crop, which only helps when the lesion happens
    to be near-center) — the actual Path B workstream 3 proposal; (2)
    fixing the malignant-vs-benign evaluation metric itself to use
    max(mel_score, bcc_score, akiec_score) or similar rather than
    mel_score alone, since this session's evaluations of SLICE-3D,
    MILK10k, and MRA-MIDAS all used the flawed version.

22. **A real lesion-detecting crop was built, unit-tested, visually
    verified, and properly ablated — and it lost to the simpler
    center-crop from finding #21.** Finding #21 flagged "a real
    lesion-detecting crop (not just center-crop, which only helps when
    the lesion happens to be near-center)" as the concrete next step.
    Built it: `inference_utils.detect_lesion_bbox()` — classical CV, no
    trained model — finds the dermatoscope's circular field of view
    (excluding a black vignette if present), then Otsu-thresholds within
    it for a dark, reasonably-central, reasonably-compact region (guards
    against the ruler overlay or hair being mistaken for the lesion).
    Unit-tested against synthetic images (off-center blob, no-lesion
    fallback, circular vignette) — one real bug caught before trusting
    it on real data: the initial `min_area_frac=0.02` threshold rejected
    genuinely small lesions in huge frames, exactly the case this exists
    to handle, fixed to `0.0015`. Visually verified on a real MRA-MIDAS
    example before the full ablation: the detected box landed almost
    exactly on the actual (very subtle) lesion, correctly excluding both
    the ruler markings and the vignette.

    **Then properly ablated, same discipline as finding #21's
    center-crop test — and it didn't win:**

    | Preprocessing | MRA-MIDAS MelAUC | MILK10k MelAUC (guardrail) |
    |---|---|---|
    | Naive resize (baseline) | 0.5793 | 0.8796 |
    | Center-crop (finding #21) | 0.6002 (+0.0211, CI not confirmed) | 0.8775 (−0.0020, safe) |
    | **Lesion-crop (this finding)** | **0.5721 (−0.0071, worse than center-crop)** | **0.8525 (−0.0270; malignant-vs-benign AUC −0.0384, CI [−0.0699, −0.0091], confirmed real regression)** |

    Fails on both counts: doesn't beat center-crop on the population it
    was built for, and measurably hurts the population that already
    worked (MILK10k) — a confirmed regression, not just a noisy wash.

    **Correction, 2026-07-28 — see finding #23**: the malignant-vs-benign
    AUC figures quoted above used the flawed `mel_score`-only metric.
    Corrected, the picture gets *worse* for lesion-crop, not better: on
    MRA-MIDAS, malignant-vs-benign AUC actually **drops** with
    lesion-crop (0.7051 → 0.6857, bootstrap CI [−0.0506, +0.0105] — no
    longer even a confirmed effect either direction, just numerically
    negative); on MILK10k the regression still holds (0.7905 → 0.7612,
    CI [−0.0596, −0.0012], still confirmed real, smaller magnitude than
    originally reported). Net effect of the correction: lesion-crop had
    no metric left in its favor to begin with, and now has one less —
    the verdict below is unchanged, on firmer ground than before.

    **Working explanation, not just "it didn't work"**: dermatoscope
    photography is already centered on the lesion *by construction* — a
    clinician positions the device directly over the lesion before
    capturing. Center-crop silently exploits that reliable human-provided
    framing prior for free. The heuristic detector, when it's right, adds
    nothing center-crop didn't already have; when it's wrong — misled by
    the ruler overlay, vascular skin texture, or a lesion that's lighter
    than surrounding skin rather than darker (Otsu-inverted assumes dark;
    MRA-MIDAS's malignant pool is BCC/AKIEC-heavy, and some of those
    subtypes present as erythematous rather than pigmented) — it actively
    overrides a framing prior that was already good. A classical
    darkness-based heuristic doesn't have enough signal to beat "trust
    the photographer," which is a real, mechanistic explanation for the
    loss, not just a shrug.

    **Practical conclusion**: if a production preprocessing change is
    ever made on the strength of finding #21's mechanism, **center-crop
    is the validated candidate, not this lesion detector** — simpler,
    cheaper, and empirically better on both counts. A genuine improvement
    over center-crop would need a properly trained segmentation model
    (the real Path B workstream 3 investment), not a better classical
    heuristic — this session's attempt at the cheap classical version is
    now a documented dead end, not an open question.

23. **The malignant-vs-benign AUC metric fix, applied everywhere and
    re-run — and it substantially revises finding #20's headline
    conclusion, not just tightens a number.** Findings #18 and #22 both
    flagged the same problem in passing: "malignant-vs-benign AUC" was
    being computed by ranking cases with `mel_score` alone, which
    under-credits true BCC/AKIEC cases (`mel_score` is specifically
    P(melanoma), not a general malignancy score) — flagged as a
    caveat each time, never actually fixed or re-run. Fixed properly
    this session: `scripts/score_dataset.py` now saves `bcc_score`,
    `akiec_score`, and `malignant_score = max(mel, bcc, akiec)` for every
    scoring run; `evaluate_fusion_mra_midas.py` and all three ablation
    scripts (`ablation_centercrop_mra_midas.py`,
    `ablation_centercrop_milk10k.py`, `ablation_lesioncrop.py`) updated
    to rank the malignant-vs-benign comparison with `malignant_score`
    instead. MRA-MIDAS and MILK10k both re-scored, every affected
    evaluation re-run.

    **The correction is large, not cosmetic:**

    | Result | Old (`mel_score` alone) | Corrected (`malignant_score`) |
    |---|---|---|
    | MRA-MIDAS baseline (finding #20) | 0.3947 | **0.7051** |
    | MRA-MIDAS fusion (finding #20) | 0.3932 | **0.7375** [0.7024, 0.7727] |
    | MRA-MIDAS center-crop (finding #21) | 0.4367 | **0.7197** |
    | MRA-MIDAS lesion-crop (finding #22) | 0.4277, CI confirmed real | **0.6857**, CI [−0.0506, +0.0105] — no longer confirmed |
    | MILK10k center-crop guardrail (finding #21) | 0.3837 (Δ −0.046) | **0.7857** (Δ −0.005) — now clearly negligible |
    | MILK10k lesion-crop guardrail (finding #22) | 0.3909 (Δ −0.038, confirmed) | **0.7612** (Δ −0.029, still confirmed) |

    **What this means, read carefully rather than just "the number went
    up"**: finding #20's MelAUC result (0.54-0.58, fusion and baseline
    both weak) is **unaffected by this fix and still stands** — the model
    genuinely struggles to separate melanoma specifically from
    everything else on MRA-MIDAS, verified independently in finding #20
    by checking individual rows. But the **malignant-vs-benign**
    discrimination — can the model tell *any* cancerous lesion
    (mel+bcc+akiec) apart from benign ones — was never actually collapsed
    to chance; it was **0.70-0.74** the whole time, a real, solid signal
    the flawed metric was hiding. The most likely mechanism: MRA-MIDAS's
    malignant pool is roughly balanced across mel (184) / bcc (168) /
    akiec (161), and the model appears to be **confusing the three
    malignant subtypes with each other** more than it confuses malignant
    with benign — e.g. correctly flagging a true BCC case as
    cancer-suspicious via a high `bcc_score`, which `mel_score` alone
    would have completely missed and scored as a "failure."

    **Practical consequences for what's already been decided**:
    - Finding #20's "third population showing external-generalization
      failure" framing needs softening, not retracting — melanoma-vs-rest
      discrimination is genuinely weak there, but "the model doesn't
      work on this population" was an overstatement; "the model can spot
      something's wrong but is worse at naming exactly what" is more
      accurate.
    - Finding #21's MILK10k guardrail scare (−0.046) is confirmed to have
      been a metric artifact all along, exactly as hypothesized when the
      caveat was first written — the real cost of center-crop on MILK10k
      is negligible (−0.005), reinforcing rather than changing the
      "ship center-crop" recommendation.
    - Finding #22's case against the lesion-crop heuristic gets
      **stronger, not weaker**: the one metric that had looked like a
      genuine win for lesion-crop on MRA-MIDAS (the old +0.0330,
      CI-confirmed malignant-vs-benign improvement) is gone with the
      correct metric — direction flips negative, CI no longer excludes
      zero. MILK10k's regression still holds. Lesion-crop had no
      redeeming metric left after this correction; the "documented dead
      end" verdict stands, on firmer ground than before.
    - **No previous recommendation flips as a result of this fix** — ship
      center-crop (opt-in, provisional), don't build the lesion detector
      further, MelAUC-based conclusions throughout findings #18-#22 were
      already correct. What changes is the *severity* of finding #20's
      framing, which is exactly the kind of correction this document
      exists to make visible rather than quietly leave wrong.

24. **Center-crop shipped to production (2026-07-28).** Finding #23
    confirmed no prior recommendation changed, so the "ship center-crop"
    call from findings #21/#23 was acted on: `src/inference_utils.py`'s
    `preprocess_bgr()` — the single source of truth used by the live
    `/analyze` endpoint, `scripts/score_dataset.py`, and
    `scripts/extract_embeddings.py` alike — now center-crops to a square
    (shorter side) before the resize-to-224 step, unconditionally. This
    is a change from the "opt-in, provisional" framing in finding #23:
    on reflection, center-crop isn't a population-conditional choice like
    `capture_mode`/`skin_tone_class` (where different inputs genuinely
    need different handling and the caller must say which) — it's a
    strictly-better default resize behavior, validated safe everywhere it
    was tested, so it ships as the default rather than sitting behind an
    unused flag the way `preprocess_bgr_lesion_crop()` correctly still
    does.

    `api/SA05_api.py`'s `preprocess()` previously duplicated
    `preprocess_bgr()`'s logic inline byte-for-byte (a real drift risk —
    exactly the failure mode `inference_utils.py`'s own module docstring
    warns about) rather than importing it; that duplication is now
    removed, so the API picks up this and any future `preprocess_bgr()`
    change automatically.

    **What was NOT re-validated as part of this ship**: `MEL_THRESHOLD`
    (0.2385), `TBP_MEL_THRESHOLD`, the age-band thresholds, and the
    provisional `skin_tone_class` thresholds were all calibrated against
    scores produced under the OLD naive-resize preprocessing (the
    10,312-image validation split was never re-scored under center-crop).
    AUC is rank-invariant, so the discrimination numbers reported in
    findings #21/#23 are trustworthy — but a fixed probability threshold
    is not rank-invariant, and center-crop changing the actual score
    values for non-square images means the true sensitivity/specificity
    delivered at these thresholds today is unverified, not just
    "probably fine." Re-deriving them means re-scoring the full
    validation split under the new preprocessing (compute/time only, no
    retraining) — not done this session, tracked as an open item in
    `docs/KNOWN_GAPS.md` rather than left implicit. The one concrete
    regression check available (`tests/test_gpu_inference.py`'s pinned
    real-image case) was re-verified by hand: mel_score moved from 0.0037
    to 0.0041 on that image, GPU and ONNX paths still agree to within
    1e-5 under the new preprocessing — the crop is doing what it's
    supposed to, not silently breaking cross-implementation consistency.

25. **Threshold re-derivation attempted for all four threshold families —
    two re-derived cleanly, two left unchanged after a confound was caught
    and ruled out rather than shipped.** Closing finding #24's gap.

    **`AGE_BAND_THRESHOLDS` and `SKIN_TONE_THRESHOLDS`: clean re-derivations,
    updated.** Both were re-scored on the *exact same* image populations as
    their original derivations (verified first: reproducing the published
    values from the existing stored scores byte-for-byte before trusting
    any new number) — so these are real, population-matched measurements
    of center-crop's effect, not artifacts.

    | Threshold | Old | New |
    |---|---|---|
    | Age &lt;30 | 0.0365 | **0.1492** |
    | Age 30-45 | 0.0314 | **0.0728** |
    | Age 45-60 | 0.1179 | **0.1266** |
    | Age 60-75 | 0.2725 | **0.2153** |
    | Age 75+ | 0.1444 | **0.1787** |
    | Skin tone 2 | 0.0423 | **0.0536** |
    | Skin tone 3 | 0.0399 | **0.0627** |
    | Skin tone 4 | 0.0155 | **0.0162** |

    The shifts are large relative to the old values (up to ~4x for the
    youngest band) — plausible, not obviously wrong, given these are
    percentile-based estimates on small per-band melanoma counts (13-190
    cases): a handful of borderline cases reordering near the cutoff moves
    a percentile estimate a lot in a small sample. `tests/test_verdict_logic.py`
    had one assertion tied to the old margin (young age bands sitting
    below *half* of the global threshold); updated to the weaker, still-true
    invariant (below the global threshold, not below half of it) rather than
    reverted — the shrinking margin is itself part of what this finding
    reports, not a test to make pass by any means.

    **`TBP_MEL_THRESHOLD`: confirmed unchanged, exactly as the mechanism
    predicts.** SLICE-3D crops are already square (aspect ratio 1.00), so
    center-crop is a no-op on this population. Re-scoring all 2,643 images
    (143 malignant) gave AUC 0.6131 and threshold 0.0199 — identical to
    4 decimal places. This is a useful sanity check on the whole exercise:
    the one population where center-crop should do *nothing* did nothing.

    **`MEL_THRESHOLD`: NOT changed — a confound was caught before shipping
    a wrong number.** The original 10,312/922-image validation set behind
    0.2385 could not be located as a saved manifest anywhere in the repo
    (computed ad hoc in a prior session, per finding #24's caveat), so the
    first attempt substituted `config/val.csv` (the real, on-disk, fixed
    training validation split — but a different, harder population: 5,978
    images, 960 melanoma). That attempt gave an apparent jump from 0.2385
    to 0.5010 — a result that would have meant center-crop roughly *doubled*
    the threshold needed, wildly out of line with every other center-crop
    effect measured this session (all ±0.01-0.03 AUC-scale).

    Rather than trust a surprising number, it was checked directly: scored
    the *same* 5,978-image `val.csv` population under both old and new
    preprocessing. Result: old-preprocessing threshold on `val.csv` is
    **0.5447**, new is **0.5010** — a small ~8% relative *decrease*, not a
    2x increase. The apparent jump was entirely a population-mismatch
    artifact (`val.csv` needs a much higher threshold than 0.2385 even
    under *unchanged* preprocessing — it is simply a harder population than
    whatever produced 0.2385), not a real effect of center-crop. A 300-image
    matched-pair spot check (150 MEL/150 NV) caught this same pattern
    first, at smaller scale, before the full re-run confirmed it.

    **Decision: leave `MEL_THRESHOLD` at 0.2385.** It was derived on the
    correct (if now unrecoverable) population, and the one clean signal
    available about center-crop's real effect on threshold-for-80%-sensitivity
    points slightly *down*, not up — meaning 0.2385 unchanged is, if
    anything, mildly conservative (favors sensitivity, the safe direction
    for a cancer screen) rather than dangerously stale. Re-deriving this
    one properly would require either recovering the original validation
    manifest or accepting a full retrain-adjacent re-validation effort —
    tracked as open in `docs/KNOWN_GAPS.md`, not silently resolved by this
    finding.

26. **`MEL_THRESHOLD` resolved by adopting a new documented validation
    manifest, rather than leaving it permanently stuck against an
    unrecoverable one.** Finding #25 correctly declined to ship a
    threshold contaminated by a population-mismatch confound, but "leave
    it forever against a validation set nobody can find anymore" is not a
    real long-term answer either. Resolution: `config/val.csv` — the real,
    on-disk, fixed training validation split (5,978 images, 960 melanoma,
    genuinely held out, drawn from ISIC 2018+2019+2020) — is now formally
    adopted as *the* documented, reproducible validation manifest for
    `MEL_THRESHOLD` going forward, replacing the untraceable ad hoc
    10,312/922-image set. `models/production/ensemble/ensemble_metrics_selfconsistent.json`
    was regenerated against it (old values preserved in a
    `superseded_values` key for historical reference, not deleted).

    **The number moved a lot, and that's reported plainly, not
    minimized**: `MEL_THRESHOLD` is now **0.5010** (was 0.2385), at 80%
    sensitivity / **96.5% specificity** (up from 94.2% on the old,
    unrecoverable set) — AUC on the new manifest is 0.9578, in the same
    range as before. This is a real, visible change to production verdict
    routing (a mel_score of, say, 0.35 that used to flag `CANCER_FLAGGED`
    now routes to `REVIEW_REQUIRED` instead), not a rounding correction —
    flagged here exactly so it doesn't ship quietly. One test fixture
    (`tests/test_verdict_logic.py::test_make_verdict_high_mel_score_flags_cancer`)
    had a hardcoded `mel=0.5` probe that the new threshold made ambiguous;
    bumped to 0.65, comfortably clear of both the new threshold and the
    `mel_score > 0.6` priority-1 cutoff in `make_verdict()`.

    **Why this is a legitimate resolution, not a workaround**: unlike the
    val.csv-based re-derivation attempt for the SAME threshold in finding
    #25 (which was rightly rejected because it was being compared against
    a *different, unrelated* population's number, 0.2385, producing a
    misleading apparent 2x jump), this is not a comparison across two
    populations at all — it is deriving the threshold fresh, self-consistently,
    on one clearly-documented population, exactly the way `MEL_THRESHOLD`
    should have been derivable all along if its original validation set
    had been saved. Nothing here claims center-crop caused this
    particular change; the honest framing is that the operational number
    changed because the validation population it is now defined against
    changed — a one-time, clearly-documented, necessary consequence of the
    original manifest being unrecoverable, not evidence the model
    regressed.

27. **Flagged-precision-at-prevalence refreshed under the current
    threshold and given proper confidence intervals for the first time on
    ISIC-2020 — closing findings #16/#19's staleness (introduced by
    finding #26, same session) and rigor gap (open since finding #16)
    together.** Two problems, one fix: (1) `MEL_THRESHOLD` moving to
    0.5010 in finding #26 silently invalidated every flagged-precision
    number in findings #16 and #19, which were computed against the old
    0.2385 — caught by noticing `scripts/evaluate_fusion.py` hardcoded a
    stale copy of the production thresholds (the exact duplication risk
    `src/inference_utils.py`'s module docstring warns about, now confirmed
    to also apply to verdict logic, not just preprocessing) rather than
    importing them; (2) finding #16's ISIC-2020 natural-prevalence number
    (29.8%) was always a single resampling draw, same rigor gap the
    MILK10k numbers had before `bootstrap_flagged_precision.py` existed.

    **Fixed the duplication first**: `scripts/evaluate_fusion.py` and
    `scripts/analyze_bias_report.py`'s hardcoded threshold constants
    updated to match `api/SA05_api.py`'s current values (no shared-import
    path exists for verdict logic the way `inference_utils.preprocess_bgr`
    provides for preprocessing — flagged as a real gap, not fixed here,
    since it's a bigger refactor than this finding's scope. **Closed in
    finding #29**, same session.)

    **MILK10k, re-run under the current threshold** (`scripts/bootstrap_flagged_precision.py`):

    | Prevalence | Fusion precision | Baseline precision | Delta (95% CI) | Real? |
    |---|---|---|---|---|
    | 2% | 5.2% [3.4, 6.7] | 4.5% [1.5, 7.7] | +0.7pp [−2.1, +4.2] | No |
    | 5% | 13.2% [10.3, 16.4] | 11.5% [6.3, 16.9] | +1.8pp [−3.1, +7.0] | No |
    | 10% | 24.1% [20.2, 28.6] | 21.1% [14.8, 27.9] | +3.0pp [−3.0, +9.1] | No |
    | 20% | 41.2% [36.4, 46.8] | 37.4% [30.1, 44.7] | +3.9pp [−2.8, +10.8] | No |

    Every baseline-precision number is now *higher* than the old,
    stale single-draw figures (e.g. 20%: 33.7% → 37.4%) — expected and
    sensible: a stricter threshold (0.5010 vs 0.2385) flags fewer cases,
    and the ones it does flag are more confidently malignant. **No
    prevalence level's delta CI excludes zero anymore** — a real change
    from finding #19, where 20% was confirmed. Fusion is directionally
    ahead at every level, same as before, just not statistically
    confirmed at this sample size under the new threshold.

    **ISIC-2020, given a bootstrap CI for the first time**
    (`scripts/bootstrap_flagged_precision_isic2020.py`, new): flagged-precision
    at the real natural prevalence (1.76% malignant) is **21.8%, 95% CI
    [16.4%, 28.1%]** (n=2,000 valid draws) — down from the old single-draw
    29.8% figure, which was itself computed under the stale threshold, not
    a sign of anything getting worse. The same leakage caveat from finding
    #16 still applies unchanged: ISIC-2020 was almost certainly in the
    training pool, so treat this as an optimistic upper bound, not an
    independent external test — this finding adds a confidence interval to
    an already-caveated number, it does not resolve the leakage issue.

    **What this does and doesn't change**: no prior recommendation flips.
    Path B's fusion approach still shows a consistent directional benefit
    at every prevalence tested, on both the original and refreshed
    numbers — that qualitative story survives. What changes is precision:
    every number in findings #16 and #19 now carries an honest confidence
    interval instead of a single point estimate, and none of them are
    silently wrong relative to today's deployed threshold anymore. The
    remaining open item — no genuinely natural-prevalence, zero-leakage,
    dermatoscope-domain holdout exists to settle the ISIC-2020 leakage
    caveat — is unchanged by this finding; see `docs/KNOWN_GAPS.md`.

28. **Production uncertainty quantification decided: ship the ONNX-only
    disagreement signal, don't add PyTorch.** Finding #7 flagged this as
    "not blocked, but a bigger decision than a code fix" and left it there
    — a real product tradeoff (MC-dropout requires PyTorch in production;
    an ONNX-only inter-model-disagreement proxy exists, is free, but has a
    known blind spot) that shouldn't be decided unilaterally. Put to the
    user directly rather than guessed at.

    **Decision: ship the ONNX-only signal, `SKINANALYTICA_ENABLE_UNCERTAINTY`
    now defaults to `true`** in `api/SA05_api.py`. Rationale: on the one
    real false-negative case tested (finding #7), MC-dropout and the
    disagreement signal fail *identically* — both miss a case where all 3
    backbones agree confidently and are wrong together. Given neither
    approach demonstrably covers the failure mode people actually worry
    about, the free option (already built, zero new dependencies) is the
    right default over committing real infrastructure cost (PyTorch +
    checkpoints in production) for a method that isn't shown to do better
    on the one concrete test available.

    **A real memory cost was caught before this shipped everywhere**:
    `_load_disagreement_backbones()` loads the **fp32** individual backbone
    ONNX files — ViT-L (~1.2GB), ConvNeXt-Large (~785MB), EfficientNetV2-S
    (~81MB), ~2.1GB combined — on top of whatever `MODEL_MODE` already
    loads (~524MB for the int8 ensemble in "full" mode). A ~2.6GB+ total
    footprint would almost certainly OOM-crash the actual deployed
    `skinanalytica-api` service, which runs on Render's **free tier**.
    `render.yaml` now explicitly pins `SKINANALYTICA_ENABLE_UNCERTAINTY=false`
    for that one service, overriding the new code default — the default
    change is real for local development and any adequately-provisioned
    future deployment, but was deliberately NOT allowed to silently change
    the live constrained service's behavior. Actually enabling this live
    would need either a larger Render plan or switching the disagreement
    backbones to the existing int8 files (`onnx_int8/`, ~523MB combined) —
    neither done here, tracked as a follow-up if this is ever prioritized.

29. **Verdict-threshold duplication between `api/SA05_api.py` and
    `scripts/evaluate_fusion.py` closed — the exact gap finding #27 caught
    but didn't fully fix.** Finding #27 found and repaired one instance of
    this drift (a stale hardcoded copy of the thresholds in
    `evaluate_fusion.py` after finding #26 changed `MEL_THRESHOLD`) but
    left the underlying duplication itself in place, flagged as a real
    remaining gap: "no shared-import path exists for verdict logic yet
    ... this file must be checked by hand." That's now fixed the same way
    preprocessing already was (`src/inference_utils.py::preprocess_bgr()`).

    **New module: `src/verdict_logic.py`** — the single source of truth
    for `MEL_THRESHOLD`, `MEL_REVIEW_BAND_FACTOR`, `AGE_BAND_THRESHOLDS`,
    `TBP_MEL_THRESHOLD`, `SKIN_TONE_THRESHOLDS`, `CANCER_CLASSES`,
    `REVIEW_CLASSES`, `get_mel_threshold()`, and the routing decision
    itself (a new `route()` primitive, plus two entry points built on it:
    `make_verdict(probs, ...)` for the full-probability-array shape
    `api/SA05_api.py`'s `/analyze` uses, and `verdict(mel_score,
    pred_class, confidence, ...)` for the pre-extracted-scalar shape the
    evaluation scripts use when bulk-scoring a holdout).

    `api/SA05_api.py`'s ~210-line inline verdict-logic block (all the
    threshold constants plus `get_mel_threshold()`/`make_verdict()`) is
    now a single `from verdict_logic import (...)` — every existing call
    site (`api.MEL_THRESHOLD`, `api.make_verdict(...)`, etc.) keeps
    working unchanged, since Python re-exports imported names onto the
    importing module's namespace. `scripts/evaluate_fusion.py` and
    `scripts/analyze_bias_report.py` now import from the same module
    instead of hand-copying values — the latter's `MEL_THRESHOLD` had
    already been caught stale once this session too (finding #26/#27's
    cleanup pass).

    **Verified as a true no-op, not just "should be fine"**: re-ran
    `evaluate_fusion.py`, `bootstrap_flagged_precision.py`, and
    `analyze_bias_report.py` after the refactor — byte-identical output to
    before on every number. Added `tests/test_verdict_logic_shared.py` (4
    new tests, `tests/conftest.py` extended to put `scripts/` on
    `sys.path`): confirms `api` and `evaluate_fusion` import the literal
    same threshold objects (not just equal values today), and that
    `api.make_verdict()` and `evaluate_fusion.verdict()` agree on routing
    decisions across several probability/age combinations — the actual
    regression guard against this exact drift recurring a third time.

## Engineering changes made this session

| Change | File(s) | Effect (measured) |
|---|---|---|
| Fixed temperature-scaling bug (finding #1) | `src/inference_utils.py`, `api/SA05_api.py`, `agents/verification_agent.py`, both notebooks | Bias report sensitivity: 0.0% (broken) → real numbers |
| Fixed double-softmax bug (finding #2) | `src/inference_utils.py` (`run_session`), same call sites | mel_score on known test case: 0.0589 (wrong) → 0.0037 (verified correct) |
| Capture-modality-conditional threshold — Path A (finding #12) | `api/SA05_api.py` (`TBP_MEL_THRESHOLD`, `capture_mode` param on `/analyze`), `config/skin_config.yaml`, `tests/test_verdict_logic.py` (+8 tests), `tests/test_config_consistency.py` | Sensitivity on the 143-case TBP-style holdout: 42.7% → 80% (at the cost of specificity dropping to 15.6% — a real tradeoff, not a free fix; see finding #12) |
| Auth on by-id patient-data lookups (finding #13) | `api/SA05_api.py` (`require_api_key`), `.env.example`, `render.yaml`, `tests/test_auth.py` (+9 tests) | `/report/{id}`, `/report/{id}/fhir`, `/patient/{id}/history`: open to anyone with an id → require `X-API-Key`, fail closed if unset. `/analyze`/`/analyze/batch`/`/audit/log` deliberately left open — see finding #13 for why |
| Skin-tone-conditional thresholds — provisional (finding #14) | `api/SA05_api.py` (`SKIN_TONE_THRESHOLDS`, `skin_tone_class` param on `/analyze`), `config/skin_config.yaml`, `tests/test_verdict_logic.py` (+16 tests), `tests/test_config_consistency.py` | Sensitivity at the deployed threshold on MILK10k's holdout: 55.6-76.9% (tones 2/3/4) → ~80-85% each, opt-in via `skin_tone_class`, same "no caller yet" status as `capture_mode="tbp"`. Derived from only 13-28 melanoma cases per band — explicitly flagged as provisional, not validated the way the age thresholds are |
| Full auth closure — all 6 patient-data endpoints (finding #13, closed) | `api/SA05_api.py`, `assets/auth.js` (new, client-supplied-key widget), `index.html`/`docs.html`/`status.html`, `tests/test_auth.py` (12 tests) | `/analyze`, `/analyze/batch`, `/audit/log` now gated too, via a Swagger-"Authorize"-style client-held key instead of a server-embedded one — the actual blocker (no way to hold a secret in public page JS) resolved by the frontend rebuild |
| Non-dermoscopic-input guardrail (finding #15) | `src/inference_utils.py` (`check_dermatoscope_likelihood`), `api/SA05_api.py` (`/analyze` response field `image_type_check`), `index.html` (warning banner), `tests/test_inference_utils.py` (+5 tests) | First version's vignette signal disproven on real data (see finding #15) and dropped; recalibrated aspect-ratio threshold verified at 0 false positives / 1,200 real images across all 6 project datasets before shipping |
| Age-conditional melanoma thresholds | `api/SA05_api.py` (`AGE_BAND_THRESHOLDS`) | Sensitivity gap 43.8-82.0% → tight 81.2-90.2% band across all age groups |
| Review-routing band, made relative not flat (`MEL_REVIEW_BAND_FACTOR`) | `api/SA05_api.py` | Fixed a flat 0.15 floor silently becoming a no-op for young patients once age-conditional thresholds dropped below it; now scales with whichever threshold applies |
| Test-time augmentation (`predict_tta`) | `src/inference_utils.py` | Blur-induced flip rate 4/20 → 2/20; brightness 4/20 → 3/20; JPEG unchanged (2/20) |
| Real monitoring scheduler | `agents/scheduled_monitor.py`, `render.yaml` (new `cron` service) | Drift monitoring now runs daily against real `scan_results/`; previously never ran automatically at all |
| Fixed `patient_age`/`patient_sex` not being persisted | `api/SA05_api.py` | Was a silent gap that would have broken the new scheduler's age-based proxy monitoring before it ever produced a real number |
| Opt-in inter-model disagreement uncertainty signal | `api/SA05_api.py` (`SKINANALYTICA_ENABLE_UNCERTAINTY`) | Adds `uncertainty_std` to `/analyze`; off by default (real memory/latency cost — loads all 3 backbones) |
| Self-consistent `MEL_THRESHOLD` correction (finding #10) | `api/SA05_api.py`, `config/skin_config.yaml` | 0.312 (mismatched-pooling calibration) → 0.2385 (matches deployed arithmetic pooling, verified 80% sensitivity on 10,312-image split) |
| Config threshold desync (original bug this session started with) | `config/skin_config.yaml` | 0.45 (stale) → 0.312 → 0.2385, now guarded by `tests/test_config_consistency.py` so it can't silently drift a third time |
| Center-crop shipped as the new default preprocessing (finding #24) | `src/inference_utils.py` (`preprocess_bgr`), `api/SA05_api.py` (`preprocess()` now delegates instead of duplicating), `tests/test_gpu_inference.py` (pinned value re-verified: 0.0037 → 0.0041) | MRA-MIDAS: +0.015 to +0.021 AUC depending on metric; MILK10k guardrail cost negligible (−0.005). Operating thresholds NOT re-derived under the new preprocessing — see finding #24 |
| Operating thresholds re-derived under center-crop (finding #25) | `api/SA05_api.py` (`AGE_BAND_THRESHOLDS`, `SKIN_TONE_THRESHOLDS` updated; `MEL_THRESHOLD`, `TBP_MEL_THRESHOLD` confirmed and left unchanged), `config/skin_config.yaml` (kept in sync), `tests/test_verdict_logic.py` (one assertion loosened from a stale quantitative margin to the still-true qualitative invariant) | Age/skin-tone thresholds updated (clean population-matched re-derivation); TBP confirmed exactly unchanged (no-op on already-square images); global `MEL_THRESHOLD` re-derivation attempt caught a population-mismatch confound and was correctly NOT shipped — see finding #25 |
| `MEL_THRESHOLD` resolved on a new documented manifest (finding #26) | `api/SA05_api.py` (`MEL_THRESHOLD` 0.2385 → 0.5010), `config/skin_config.yaml`, `models/production/ensemble/ensemble_metrics_selfconsistent.json` (regenerated against `config/val.csv`, old values preserved under `superseded_values`), `tests/test_verdict_logic.py` (one fixture bumped past the new threshold) | Sensitivity pinned at 80% by construction; specificity improved 94.2% → 96.5% on the new manifest. A real, visible change to production verdict routing — see finding #26 |
| Flagged-precision refreshed + ISIC-2020 bootstrap CI added (finding #27) | `scripts/evaluate_fusion.py`, `scripts/analyze_bias_report.py` (stale threshold constants re-synced), `scripts/bootstrap_flagged_precision_isic2020.py` (new), `outputs/fusion_model/bootstrap_flagged_precision_results.json` and `outputs/bias_reports/isic2020_bootstrap_flagged_precision.json` (regenerated) | MILK10k baseline precision up at every prevalence (e.g. 20%: 33.7% → 37.4%, stricter threshold flags fewer/more-confident cases); no prevalence level's fusion-vs-baseline delta CI excludes zero anymore (was 1/4 before); ISIC-2020 natural-prevalence precision now has a CI (21.8% [16.4, 28.1]) instead of a single draw — see finding #27 |
| Uncertainty quantification decision shipped (finding #28) | `api/SA05_api.py` (`ENABLE_UNCERTAINTY` default `false`→`true`), `render.yaml` (explicit `SKINANALYTICA_ENABLE_UNCERTAINTY=false` override for the live free-tier service), `.env.example` | `/analyze` now returns `uncertainty_std` by default for any adequately-provisioned deployment (local dev included); the live Render deployment's behavior deliberately unchanged (would OOM on its ~2.6GB combined footprint) — see finding #28 |
| Verdict-threshold duplication closed (finding #29) | `src/verdict_logic.py` (new, single source of truth), `api/SA05_api.py` (~210-line inline block → one import), `scripts/evaluate_fusion.py`, `scripts/analyze_bias_report.py` (both now import instead of hand-copying), `tests/conftest.py` (`scripts/` added to `sys.path`), `tests/test_verdict_logic_shared.py` (new, 4 tests) | Zero behavior change (verified byte-identical script output before/after); closes the exact drift risk finding #27 caught mid-occurrence — see finding #29 |

**Important honesty note on the age-conditional thresholds**: lowering the
threshold from 0.312 to ~0.03-0.12 depending on age band is a real
tradeoff, not a free win — specificity in the under-45 bands drops to
~88-91% (was 97%+ at the old flat threshold), meaning meaningfully more
false positives / review workload for younger patients in exchange for
catching far more of their real melanomas. That's the right tradeoff for a
screening tool, but it's a clinical-policy decision as much as an
engineering one — it's documented inline in `api/SA05_api.py` so it can't
ship silently.

## Known limitations (see [KNOWN_GAPS.md](KNOWN_GAPS.md) for detail)

- **10,015 ISIC-2018 images are byte-for-byte duplicated inside ISIC-2019's
  training pool** (finding #17), giving them ~2x effective sampling weight
  in the already-completed retrain. Not fixed at the CNN level (the retrain
  isn't being redone for this); fixed for Path B's fusion model via
  `scripts/dedup_train_embeddings.py`. Worth checking for in any future
  retrain's `build_sample_list()`, not just accepted as-is.
- ~~`/analyze`, `/analyze/batch`, and `/audit/log` remain unauthenticated~~
  — **closed 2026-07-27** (finding #13). All six patient-data endpoints
  now require `X-API-Key`, via a client-supplied-key widget in the
  rebuilt frontend (`assets/auth.js`) rather than a backend proxy.
- **External validation: discrimination confirmed improved, but the deployed
  threshold does not transfer to this population** (findings #5, #11, #12).
  At n=143 (the largest leakage-free malignant sample available), AUC is
  0.6131 (95% CI [0.5516, 0.6751], p=5.2×10⁻⁶) — a real, statistically
  significant improvement over the earlier underpowered n=44 read, and
  directionally ahead of finding #5's original 0.593. **Still open**: the
  deployed threshold (0.119) only delivers 42.7% sensitivity here, and no
  single threshold hits 80% sensitivity without specificity collapsing to
  15.6% — a calibration-transfer problem, not a discrimination problem.
  Whether this is driven by institution/population shift, capture-modality
  difference (TBP vs. dermatoscope), case-severity mismatch, or some
  combination is diagnosed (finding #5) but not yet fixed.
- **Skin-tone fairness on dermoscopy: partially answered** (findings #9,
  #14). MILK10k's holdout gives overall MelAUC 0.8796 (95% CI [0.834,
  0.921]) with no statistically confirmed AUC disparity across the three
  adequately-sampled skin-tone bands — but the darkest and lightest bands
  still have too few melanoma cases to say anything, and sensitivity at
  the deployed threshold spreads more (55.6%-76.9%) than AUC does, not yet
  confirmed as real vs. small-sample noise. Not fully closed.
- **The model has zero validated performance on non-dermoscopic (clinical/
  phone-camera) images** (finding #9) — MelAUC 0.63-0.78 vs 0.95+ on
  dermoscopy. If this product is ever meant to accept ordinary phone photos
  rather than dermatoscope images, that use case is currently unvalidated.
- **Metrics reflect a single train/val split, single seed.** Bootstrap CIs
  in this document bound sampling noise, not training-process variance —
  real confidence intervals need k-fold or repeated-seed retraining, which
  needs sustained GPU time (the local RTX 4060 found this session
  accelerates inference/scoring well, but full retraining is a much larger
  and longer job than was attempted here).
- ~~Full MC-dropout uncertainty quantification is architecturally absent
  from production... a product decision, not (only) an engineering one~~
  — **decided 2026-07-28** (finding #28): ship the lighter ONNX-only
  substitute (inter-model disagreement) instead of adding PyTorch to
  production. `SKINANALYTICA_ENABLE_UNCERTAINTY` now defaults to `true`
  in `api/SA05_api.py` — but is explicitly pinned back to `false` for the
  live `skinanalytica-api` Render deployment specifically (`render.yaml`),
  since the disagreement backbones' ~2.1GB of fp32 ONNX files would almost
  certainly OOM-crash that free-tier service. Either way, this signal
  **does not catch confidently-wrong-in-unison errors** — on the one real
  false-negative case tested, all 3 backbones agreed confidently and were
  all wrong, the same blind spot MC-dropout has on that case too (see
  finding #7). Not a full solution, a deliberate, documented tradeoff.
- **Age-conditional thresholds were derived from ISIC-2020 data that
  includes training-adjacent images**, not a clean external holdout — treat
  as the best currently-available signal, not a final calibration.
- **Train/serve ensembling-method mismatch** (finding #10) — practical
  impact fixed (threshold corrected to 0.2385), but the notebook itself
  still has both formulas; see `docs/RETRAIN_PLAN.md` Step 0.
- **Flagged-precision at realistic prevalence is low, now with confidence
  intervals** (findings #16, #19, #27): automation rate looks strong
  (77-97%) at every prevalence tested, but `CANCER_FLAGGED` precision
  collapses to roughly 4-45% (95% CI) at realistic low-prevalence
  screening case mixes (vs. 82-91% on the enriched validation samples this
  project has mostly reported against). Bootstrap CIs (finding #27) make
  this a real range with quantified uncertainty rather than a directional
  single-draw estimate, on both MILK10k (external, zero leakage) and
  ISIC-2020 (natural prevalence, but almost certainly training-adjacent —
  treat as an optimistic upper bound). No genuinely natural-prevalence,
  zero-leakage, dermatoscope-domain holdout exists yet to remove that one
  remaining leakage caveat — this finding narrowed the *statistical*
  uncertainty, not the *leakage* uncertainty, and those are different
  problems.
- ~~**Deployed probability thresholds re-derived after the 2026-07-28
  center-crop change — three of four resolved, one still genuinely open**~~
  — **closed 2026-07-28** (finding #26). All four threshold families are
  now resolved: `AGE_BAND_THRESHOLDS` and `SKIN_TONE_THRESHOLDS` were
  cleanly re-derived on their original populations (finding #25);
  `TBP_MEL_THRESHOLD` was confirmed unchanged (SLICE-3D crops are already
  square, center-crop is a no-op there); `MEL_THRESHOLD` was resolved by
  formally adopting `config/val.csv` as the new documented validation
  manifest (the original was unrecoverable) and moving from 0.2385 to
  **0.5010** (finding #26) — a real, visible change to production verdict
  routing (specificity improved 94.2% → 96.5% at the same 80% sensitivity
  target), not a quiet rounding fix.

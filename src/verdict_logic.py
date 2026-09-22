"""
SkinAnalytica — src/verdict_logic.py
Single source of truth for melanoma-verdict threshold selection and
routing (CANCER_FLAGGED / REVIEW_REQUIRED / NORMAL).

Extracted 2026-07-28 (docs/MODEL_CARD.md finding #29) because
scripts/evaluate_fusion.py maintained its own hand-copied version of these
constants and went silently stale within THIS SAME SESSION when
MEL_THRESHOLD/AGE_BAND_THRESHOLDS changed in api/SA05_api.py but not there
(finding #27) — exactly the duplication risk this module's sibling,
inference_utils.preprocess_bgr(), already exists to prevent for
preprocessing. Every place that selects a melanoma threshold or routes a
verdict (the live API, evaluation scripts, future ones) must import from
here instead of re-deriving inline.
"""
from typing import Optional

import numpy as np

UNIFIED_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
MEL_IDX = UNIFIED_CLASSES.index("mel")
CANCER_CLASSES = {"mel", "bcc", "akiec"}
REVIEW_CLASSES = {"vasc"}

# Corrected this session: the original 0.312 was calibrated by
# notebooks/SA02_Ensemble.ipynb using a GEOMETRIC (log-space) pooling
# formula that does not match the ARITHMETIC (probability-space) pooling
# actually baked into the deployed skinanalytica_ensemble.onnx graph (see
# docs/KNOWN_GAPS.md). Re-deriving the "threshold for 80% sensitivity"
# target against the model's real, self-consistent output
# (models/production/ensemble/ensemble_metrics_selfconsistent.json, full
# 10,312-image validation split, 922 melanomas) gives 0.2385, not 0.312 —
# the old value did not actually deliver its own 80%-sensitivity design
# target once evaluated correctly. Overall discriminative performance
# (AUC) barely moved (0.9609 -> 0.9540); only the operating threshold was
# wrong.
#
# UPDATED 2026-07-28 (finding #26), resolving finding #25's open item.
# The original 10,312/922-image validation set above could not be located
# as a saved manifest anywhere in the repo (computed ad hoc in a prior
# session) -- an initial recalibration attempt substituted config/val.csv
# (the real on-disk fixed training validation split, genuinely held out,
# but a DIFFERENT, harder population -- 5,978 images, 960 melanoma) and
# got a misleading-looking apparent jump (0.2385 -> 0.5010) that turned out
# to be almost entirely a population-mismatch artifact, not a center-crop
# effect (verified: scoring val.csv under OLD preprocessing needs 0.5447 to
# hit 80% sensitivity, already far from 0.2385 before center-crop is even
# applied -- see finding #25). Rather than leave MEL_THRESHOLD permanently
# stuck against an unrecoverable population, config/val.csv is now formally
# adopted as the documented, reproducible validation manifest going
# forward (see models/production/ensemble/ensemble_metrics_selfconsistent.json),
# and MEL_THRESHOLD is set to what it actually takes to reach 80%
# sensitivity on that population under the current (center-crop)
# preprocessing: 0.5010, AUC 0.9578, 96.5% specificity (up from 94.2% on
# the old, unrecoverable set) -- a real, visible change to production
# verdict routing, not a rounding correction. See docs/MODEL_CARD.md
# finding #26 and scripts/rederive_thresholds_centercrop.py.
MEL_THRESHOLD = 0.5010

# Stopgap mitigation for the measured ISIC-2020-style generalization gap and
# the borderline-case instability under image-quality perturbation (see
# docs/MODEL_CARD.md findings #4, #6). We can't detect dataset origin at
# inference time, but we CAN widen the review net just below whichever
# cancer threshold actually applies, so borderline cases get a human look
# instead of an unqualified "NORMAL".
#
# This is a FACTOR of the applicable threshold, not a fixed value — a flat
# absolute floor (the original version of this was 0.15) breaks once
# thresholds became age-conditional: for young patients the age threshold
# (~0.03-0.04) already sits below any sensible flat floor, silently making
# the review band a no-op for exactly the group that needs it most, while
# staying active for older patients whose threshold is higher. Scaling by
# the same relative factor keeps the review band consistently meaningful
# across every age band.
MEL_REVIEW_BAND_FACTOR = 0.5

# Age-conditional melanoma thresholds, derived from
# outputs/bias_reports/threshold_ci_age_analysis.json: at the flat 0.312
# threshold, sensitivity ranged from 43.8% (<30) to 82.0% (75+) on ISIC-2020 —
# a single global cutoff structurally can't serve every age band, because
# the model's melanoma-score distribution for true positives shifts with
# age (see docs/KNOWN_GAPS.md for why: far fewer young-patient melanoma
# training examples). These are the per-band thresholds that reached ~80%
# sensitivity in that analysis.
#
# RE-DERIVED 2026-07-28 for the center-crop preprocessing change (finding
# #24/#25): re-scoring the SAME ISIC-2020 sample under the new preprocessing
# is a clean, population-matched before/after comparison. Every band
# shifted upward except 60-75. Old values, for reference: <30 0.0365,
# 30-45 0.0314, 45-60 0.1179, 60-75 0.2725, 75+ 0.1444.
#
# REAL TRADEOFF, not free: this is a clinical-policy decision (more
# false positives / REVIEW_REQUIRED routing for younger patients, in
# exchange for catching far more of their real melanomas), not just a
# code change — see docs/MODEL_CARD.md findings #3, #26.
AGE_BAND_THRESHOLDS = {
    (0, 30): 0.1492,
    (30, 45): 0.0728,
    (45, 60): 0.1266,
    (60, 75): 0.2153,
    (75, 200): 0.1787,
}

# Capture-modality-conditional melanoma threshold — Path A from
# docs/MODEL_CARD.md finding #12. MEL_THRESHOLD (and AGE_BAND_THRESHOLDS)
# were derived against dermatoscope images (ISIC 2018-2020) and do not
# transfer to 3D-total-body-photography-style crops (e.g. ISIC 2024
# SLICE-3D). There is no reliable way to auto-detect TBP-crop-style images
# from pixel data alone at inference time — capture_mode must be
# explicitly supplied by the caller.
#
# CONFIRMED UNCHANGED 2026-07-28 under the center-crop preprocessing change
# (finding #24/#25): SLICE-3D's source crops are already square, so
# center-crop-before-resize is a no-op on this population — re-scoring
# all 2,643 images (143 malignant) gave AUC 0.6131 and threshold 0.0199,
# identical to the pre-change values to 4 decimal places.
TBP_MEL_THRESHOLD = 0.0199

# Skin-tone-conditional melanoma thresholds — from docs/MODEL_CARD.md
# finding #14: MILK10k's dermoscopy skin-tone holdout showed AUC is
# statistically consistent across skin_tone_class 2/3/4 (no confirmed
# discrimination gap), but sensitivity at the global threshold spread
# noticeably by tone. PROVISIONAL — far less statistically solid than
# AGE_BAND_THRESHOLDS (13/28/18 melanoma cases vs 585). Tones 0, 1, 5
# excluded entirely (0/3/2 melanoma cases — nowhere near enough).
# skin_tone_class must be explicitly supplied by the caller; nothing in
# the current product wires this up yet.
#
# RE-DERIVED 2026-07-28 for the center-crop preprocessing change (finding
# #24/#25): same clean population-matched re-scoring as AGE_BAND_THRESHOLDS
# (the exact same 700 MILK10k holdout images). Old values, for reference:
# tone 2 0.0423, tone 3 0.0399, tone 4 0.0155.
SKIN_TONE_THRESHOLDS = {
    2: 0.0536,
    3: 0.0627,
    4: 0.0162,
}


def get_mel_threshold(patient_age: Optional[float], capture_mode: Optional[str] = None,
                       skin_tone_class: Optional[int] = None) -> float:
    """Selects the applicable melanoma-score threshold. Precedence:
    capture_mode="tbp" (a bigger population shift than skin tone within
    dermoscopy) > skin_tone_class > age band > global MEL_THRESHOLD."""
    if capture_mode == "tbp":
        return TBP_MEL_THRESHOLD
    if skin_tone_class in SKIN_TONE_THRESHOLDS:
        return SKIN_TONE_THRESHOLDS[skin_tone_class]
    if patient_age is None or (isinstance(patient_age, float) and np.isnan(patient_age)):
        return MEL_THRESHOLD
    for (lo, hi), t in AGE_BAND_THRESHOLDS.items():
        if lo <= patient_age < hi:
            return t
    return MEL_THRESHOLD


def route(mel_score: float, pred_class: str, confidence: float, mel_threshold: float) -> tuple:
    """The core CANCER_FLAGGED / REVIEW_REQUIRED / NORMAL decision, given an
    already-selected threshold. Returns (verdict, priority)."""
    review_band_low = mel_threshold * MEL_REVIEW_BAND_FACTOR
    if pred_class in CANCER_CLASSES or mel_score >= mel_threshold:
        return "CANCER_FLAGGED", (1 if pred_class == "mel" or mel_score > 0.6 else 2)
    if pred_class in REVIEW_CLASSES or confidence < 0.65 or mel_score >= review_band_low:
        return "REVIEW_REQUIRED", 3
    return "NORMAL", 4


def make_verdict(probs: np.ndarray, pred_class: str, patient_age: Optional[int] = None,
                  capture_mode: Optional[str] = None, skin_tone_class: Optional[int] = None) -> dict:
    """Full-probability-vector entry point — what api/SA05_api.py's
    /analyze endpoint calls."""
    mel_score = float(probs[MEL_IDX])
    confidence = float(probs.max())
    mel_threshold = get_mel_threshold(patient_age, capture_mode, skin_tone_class)
    verdict, priority = route(mel_score, pred_class, confidence, mel_threshold)
    return {"verdict": verdict, "priority": priority,
            "confidence": confidence, "mel_score": mel_score,
            "mel_threshold_used": mel_threshold}


def verdict(mel_score: float, pred_class: str, confidence: float, patient_age: Optional[float] = None,
            capture_mode: Optional[str] = None, skin_tone_class: Optional[int] = None) -> str:
    """Scalar entry point — what the evaluation scripts call when they
    already have mel_score/confidence extracted (bulk-scoring a whole
    holdout, not re-deriving from a raw probs array per call)."""
    mel_threshold = get_mel_threshold(patient_age, capture_mode, skin_tone_class)
    return route(mel_score, pred_class, confidence, mel_threshold)[0]

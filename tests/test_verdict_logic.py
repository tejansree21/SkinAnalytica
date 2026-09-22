"""
Tests for api/SA05_api.py's verdict/threshold logic -- the age-conditional
thresholds and the review-band scaling added this session to close the
age-based sensitivity gap and keep the review net consistent across age
bands. UNIFIED_CLASSES order is [mel, nv, bcc, akiec, bkl, df, vasc].
"""
import numpy as np
import pytest

import SA05_api as api


def make_probs(mel=0.01, nv=0.9, bcc=0.01, akiec=0.01, bkl=0.05, df=0.01, vasc=0.01):
    return np.array([mel, nv, bcc, akiec, bkl, df, vasc])


def test_get_mel_threshold_none_age_falls_back_to_global():
    assert api.get_mel_threshold(None) == api.MEL_THRESHOLD


@pytest.mark.parametrize("age,expected_band", [
    (10, (0, 30)),
    (35, (30, 45)),
    (50, (45, 60)),
    (65, (60, 75)),
    (80, (75, 200)),
])
def test_get_mel_threshold_age_bands(age, expected_band):
    assert api.get_mel_threshold(age) == api.AGE_BAND_THRESHOLDS[expected_band]


def test_young_age_bands_are_lower_than_global_threshold():
    """The age-conditional fix exists because under-45 patients have far
    fewer melanoma-positive training examples (see docs/MODEL_CARD.md
    finding #3), so their threshold needs to be lower to reach the same
    target sensitivity. This is the specific, real invariant -- NOT "every
    age band is below the global threshold": once the global threshold was
    corrected (0.312 -> 0.2385, see MEL_THRESHOLD's comment), the 60-75
    band's own threshold legitimately sat ABOVE the global value for a
    while, because that age band's melanoma-score distribution was
    genuinely more separable than the population average. That's real
    data, not a bug -- don't "fix" this test by reverting to the wrong
    blanket assumption.

    Margin note (2026-07-28, finding #25): re-deriving these under the
    center-crop preprocessing change shrank the young-band margin a lot --
    <30 went from ~6.5x below global to only ~1.6x below (0.0365 vs 0.312
    -> 0.1492 vs 0.2385) -- and flipped the 60-75 band from above global
    back to below it. The assertion below was loosened from "below half of
    global" to "below global" to match: the qualitative pattern (young
    bands need a lower threshold) still holds, the previous quantitative
    margin does not, and that margin shrinking is itself a real, reportable
    finding, not something to paper over by keeping the tighter assertion
    and calling the new values wrong."""
    for band in [(0, 30), (30, 45)]:
        assert api.AGE_BAND_THRESHOLDS[band] < api.MEL_THRESHOLD


def test_all_age_band_thresholds_are_positive_and_below_one():
    for t in api.AGE_BAND_THRESHOLDS.values():
        assert 0 < t < 1


def test_make_verdict_cancer_class_always_flagged_regardless_of_score():
    probs = make_probs(mel=0.01, akiec=0.9)
    decision = api.make_verdict(probs, pred_class="akiec")
    assert decision["verdict"] == "CANCER_FLAGGED"


def test_make_verdict_high_mel_score_flags_cancer():
    probs = make_probs(mel=0.65, nv=0.25)
    decision = api.make_verdict(probs, pred_class="nv", patient_age=None)
    assert decision["verdict"] == "CANCER_FLAGGED"
    assert decision["mel_threshold_used"] == api.MEL_THRESHOLD


def test_make_verdict_review_band_scales_with_age_threshold():
    """A mel_score just above half the applicable threshold, for a class
    that isn't otherwise flagged, should land in REVIEW_REQUIRED -- and the
    boundary should move with the age band, not stay fixed."""
    young_threshold = api.AGE_BAND_THRESHOLDS[(30, 45)]
    just_above_half = young_threshold * api.MEL_REVIEW_BAND_FACTOR + 1e-4
    probs = make_probs(mel=just_above_half, nv=0.9 - just_above_half)
    decision = api.make_verdict(probs, pred_class="nv", patient_age=35)
    assert decision["verdict"] == "REVIEW_REQUIRED"


def test_make_verdict_below_review_band_and_confident_is_normal():
    probs = make_probs(mel=0.001, nv=0.95)
    decision = api.make_verdict(probs, pred_class="nv", patient_age=65)
    assert decision["verdict"] == "NORMAL"


def test_make_verdict_review_class_always_reviewed():
    probs = make_probs(mel=0.001, vasc=0.9)
    decision = api.make_verdict(probs, pred_class="vasc")
    assert decision["verdict"] == "REVIEW_REQUIRED"


def test_make_verdict_low_confidence_triggers_review():
    # no single class dominates -> confidence < 0.65
    probs = np.array([0.02, 0.3, 0.2, 0.18, 0.15, 0.1, 0.05])
    decision = api.make_verdict(probs, pred_class="nv")
    assert decision["confidence"] < 0.65
    assert decision["verdict"] == "REVIEW_REQUIRED"


# ── capture_mode="tbp" (Path A from docs/MODEL_CARD.md finding #12) ────────
# Dermatoscope-tuned thresholds (global or age-conditional) don't transfer to
# 3D-total-body-photography-style crops -- see TBP_MEL_THRESHOLD's comment.
# These bugs are exactly the class the audit flagged as having zero test
# coverage (model-mode default, base-path detection, checkpoint-resume skip
# were all caught by manual review, not automation) -- this closes that gap
# for the new capture_mode logic specifically, rather than repeating the
# pattern a fourth time.

def test_get_mel_threshold_tbp_mode_returns_tbp_threshold():
    assert api.get_mel_threshold(None, capture_mode="tbp") == api.TBP_MEL_THRESHOLD


def test_get_mel_threshold_tbp_mode_overrides_age():
    """capture_mode="tbp" was derived as a population-level fix, not
    stratified by age -- it must take precedence over age-conditional
    thresholds, not be silently overridden by them."""
    assert api.get_mel_threshold(35, capture_mode="tbp") == api.TBP_MEL_THRESHOLD
    assert api.get_mel_threshold(80, capture_mode="tbp") == api.TBP_MEL_THRESHOLD


def test_get_mel_threshold_no_capture_mode_is_unaffected():
    """Omitting capture_mode (the default, dermatoscope path) must behave
    identically to before this change -- this is the regression guard for
    every existing age/global-threshold test above."""
    assert api.get_mel_threshold(None) == api.MEL_THRESHOLD
    assert api.get_mel_threshold(35) == api.AGE_BAND_THRESHOLDS[(30, 45)]


def test_tbp_threshold_is_positive_and_below_one():
    assert 0 < api.TBP_MEL_THRESHOLD < 1


def test_tbp_threshold_is_far_below_global_threshold():
    """The whole point of finding #12's Path A: TBP-style score
    distributions sit much lower than dermatoscope ones, so this threshold
    must be dramatically lower than the global one, same shape of gap as
    the young-age bands."""
    assert api.TBP_MEL_THRESHOLD < api.MEL_THRESHOLD * 0.2


def test_make_verdict_tbp_mode_uses_tbp_threshold():
    probs = make_probs(mel=0.05, nv=0.7)
    decision = api.make_verdict(probs, pred_class="nv", capture_mode="tbp")
    assert decision["mel_threshold_used"] == api.TBP_MEL_THRESHOLD
    assert decision["verdict"] == "CANCER_FLAGGED"  # 0.05 > TBP_MEL_THRESHOLD (0.0199)


def test_make_verdict_tbp_mode_low_score_stays_below_threshold():
    probs = make_probs(mel=0.005, nv=0.9)
    decision = api.make_verdict(probs, pred_class="nv", capture_mode="tbp")
    assert decision["verdict"] != "CANCER_FLAGGED"


# ── skin_tone_class (from docs/MODEL_CARD.md finding #14) ──────────────────
# PROVISIONAL -- derived from 13/28/18 melanoma cases per band (tones 2/3/4
# only), far thinner than the 585 backing AGE_BAND_THRESHOLDS. Same "closed
# the test-coverage gap on the first pass" discipline as the tbp tests above.

@pytest.mark.parametrize("tone", [2, 3, 4])
def test_get_mel_threshold_skin_tone_returns_configured_threshold(tone):
    assert api.get_mel_threshold(None, skin_tone_class=tone) == api.SKIN_TONE_THRESHOLDS[tone]


@pytest.mark.parametrize("tone", [0, 1, 5, 99, None])
def test_get_mel_threshold_unsupported_skin_tone_falls_back(tone):
    """Tones 0/1/5 were deliberately excluded (too few melanoma cases to
    derive anything) -- an unsupported or missing tone must fall back to
    age/global, not silently KeyError or return a wrong value."""
    assert api.get_mel_threshold(None, skin_tone_class=tone) == api.MEL_THRESHOLD


def test_get_mel_threshold_skin_tone_overrides_age():
    """Like capture_mode="tbp", a supported skin_tone_class is a
    population-level fix and must take precedence over age-conditional
    thresholds."""
    assert api.get_mel_threshold(35, skin_tone_class=3) == api.SKIN_TONE_THRESHOLDS[3]


def test_get_mel_threshold_tbp_overrides_skin_tone():
    """Precedence order matters: capture_mode="tbp" reflects a completely
    different capture modality, which is a bigger population shift than
    skin tone within dermoscopy -- it must win if both are somehow
    supplied together."""
    assert api.get_mel_threshold(None, capture_mode="tbp", skin_tone_class=3) == api.TBP_MEL_THRESHOLD


def test_no_skin_tone_class_is_unaffected():
    """Omitting skin_tone_class (the default -- nothing in the current
    product supplies it) must behave identically to before this change."""
    assert api.get_mel_threshold(None) == api.MEL_THRESHOLD
    assert api.get_mel_threshold(35) == api.AGE_BAND_THRESHOLDS[(30, 45)]


def test_all_skin_tone_thresholds_are_positive_and_below_one():
    for t in api.SKIN_TONE_THRESHOLDS.values():
        assert 0 < t < 1


def test_only_tones_2_3_4_are_configured():
    """Tones 0, 1, and 5 had 0, 3, and 2 melanoma cases respectively in
    finding #14 -- nowhere near enough to derive even a provisional
    threshold. This test exists so nobody adds them later without also
    sourcing real data to back them."""
    assert set(api.SKIN_TONE_THRESHOLDS.keys()) == {2, 3, 4}


def test_make_verdict_skin_tone_uses_configured_threshold():
    tone3_threshold = api.SKIN_TONE_THRESHOLDS[3]  # 0.0627 (re-derived 2026-07-28, finding #25)
    probs = make_probs(mel=tone3_threshold + 0.01, nv=0.9 - tone3_threshold - 0.01)
    decision = api.make_verdict(probs, pred_class="nv", skin_tone_class=3)
    assert decision["mel_threshold_used"] == tone3_threshold
    assert decision["verdict"] == "CANCER_FLAGGED"  # mel_score is above the tone-3 threshold


def test_make_verdict_skin_tone_low_score_stays_below_threshold():
    probs = make_probs(mel=0.005, nv=0.9)
    decision = api.make_verdict(probs, pred_class="nv", skin_tone_class=3)
    assert decision["verdict"] != "CANCER_FLAGGED"

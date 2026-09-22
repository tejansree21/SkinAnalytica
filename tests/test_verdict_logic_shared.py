"""
Regression guard for docs/MODEL_CARD.md finding #29: src/verdict_logic.py
is now the single source of truth for verdict thresholds/routing, replacing
duplicated copies in api/SA05_api.py and scripts/evaluate_fusion.py -- one
of which (evaluate_fusion.py) went silently stale within a single session
(finding #27) before this fix. These tests exist so a future threshold
change that updates verdict_logic.py but somehow bypasses it in a call site
would be caught here, not discovered by hand months later.
"""
import numpy as np

import evaluate_fusion
import SA05_api as api
import verdict_logic


def test_api_and_evaluate_fusion_import_the_same_threshold_objects():
    """Not just equal values -- the actual same object, proving both
    modules import from verdict_logic rather than each holding their own
    (possibly-equal-today, drift-prone-tomorrow) copy."""
    assert api.MEL_THRESHOLD is verdict_logic.MEL_THRESHOLD
    assert evaluate_fusion.MEL_THRESHOLD is verdict_logic.MEL_THRESHOLD
    assert api.AGE_BAND_THRESHOLDS is verdict_logic.AGE_BAND_THRESHOLDS
    assert evaluate_fusion.AGE_BAND_THRESHOLDS is verdict_logic.AGE_BAND_THRESHOLDS


def test_api_make_verdict_and_evaluate_fusion_verdict_agree():
    """The two different call shapes (full probs array vs. pre-extracted
    scalars) must still produce the same routing decision for equivalent
    inputs -- this is the actual regression this refactor guards against."""
    cases = [
        # (mel, nv, bcc, akiec, bkl, df, vasc), pred_class, age
        ((0.6, 0.2, 0.05, 0.05, 0.05, 0.025, 0.025), "mel", 35),
        ((0.05, 0.7, 0.1, 0.05, 0.05, 0.025, 0.025), "nv", 65),
        ((0.02, 0.9, 0.03, 0.02, 0.02, 0.005, 0.005), "nv", None),
        ((0.15, 0.6, 0.1, 0.05, 0.05, 0.025, 0.025), "nv", 20),
    ]
    for probs_tuple, pred_class, age in cases:
        probs = np.array(probs_tuple)
        mel_score = probs[verdict_logic.MEL_IDX]
        confidence = probs.max()

        api_result = api.make_verdict(probs, pred_class, patient_age=age)
        fusion_result = evaluate_fusion.verdict(mel_score, pred_class, confidence, age)

        assert api_result["verdict"] == fusion_result, (
            f"api.make_verdict and evaluate_fusion.verdict disagree for "
            f"mel={mel_score}, pred_class={pred_class}, age={age}: "
            f"{api_result['verdict']!r} != {fusion_result!r}"
        )


def test_verdict_logic_route_matches_make_verdict_priority_logic():
    """route() is the lower-level primitive make_verdict() builds on --
    confirm it actually drives the returned priority correctly for the
    two CANCER_FLAGGED priority tiers (mel-or-high-score = 1, else 2)."""
    high_mel = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    result = api.make_verdict(high_mel, pred_class="mel")
    assert result["verdict"] == "CANCER_FLAGGED"
    assert result["priority"] == 1

    flagged_by_class_only = np.array([0.01, 0.02, 0.9, 0.02, 0.02, 0.01, 0.02])
    result2 = api.make_verdict(flagged_by_class_only, pred_class="bcc")
    assert result2["verdict"] == "CANCER_FLAGGED"
    assert result2["priority"] == 2  # bcc, not mel, and mel_score is low


def test_get_mel_threshold_precedence_is_identical_across_modules():
    """capture_mode > skin_tone_class > age band > global -- same precedence
    order verified against both re-exporting modules, not just the source."""
    assert api.get_mel_threshold(35, capture_mode="tbp", skin_tone_class=3) == verdict_logic.TBP_MEL_THRESHOLD
    assert api.get_mel_threshold(35, skin_tone_class=3) == verdict_logic.SKIN_TONE_THRESHOLDS[3]
    assert api.get_mel_threshold(35) == verdict_logic.AGE_BAND_THRESHOLDS[(30, 45)]
    assert api.get_mel_threshold(None) == verdict_logic.MEL_THRESHOLD
    assert evaluate_fusion.get_mel_threshold(35) == verdict_logic.AGE_BAND_THRESHOLDS[(30, 45)]

"""
Tests for src/inference_utils.py — the shared source of truth for logits ->
calibrated probabilities. This module exists specifically because getting
this step wrong (twice, in different ways) silently broke every
threshold-based number this session while leaving AUC looking fine. These
tests exist so it can't happen a third time without a test failing.
"""
import numpy as np
import pytest

import cv2

from inference_utils import (
    softmax_with_temperature, run_session, preprocess_bgr, tta_variants,
    enable_onnx_cuda_dll_dirs, check_dermatoscope_likelihood,
    detect_lesion_bbox, preprocess_bgr_lesion_crop,
)


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Minimal stand-in for onnxruntime.InferenceSession, just enough for run_session()."""
    def __init__(self, output_name, return_value):
        self._output_name = output_name
        self._return_value = return_value

    def get_outputs(self):
        return [FakeOutput(self._output_name)]

    def run(self, output_names, feed_dict):
        return [self._return_value]


def test_softmax_with_temperature_sums_to_one():
    logits = np.array([[1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.0]])
    probs = softmax_with_temperature(logits, temperature=1.0)
    assert probs.shape == (1, 7)
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()


def test_softmax_with_temperature_lower_t_sharpens_distribution():
    logits = np.array([[1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.0]])
    probs_t1 = softmax_with_temperature(logits, temperature=1.0)
    probs_low_t = softmax_with_temperature(logits, temperature=0.2)
    # a temperature below 1 sharpens the distribution: the max should increase
    assert probs_low_t.max() > probs_t1.max()


def test_run_session_applies_softmax_when_output_is_logits():
    """The 3 individual backbone ONNX files output raw logits (output name
    'logits') and genuinely need softmax_with_temperature applied."""
    raw_logits = np.array([[5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sess = FakeSession(output_name="logits", return_value=raw_logits)
    result = run_session(sess, "input", np.zeros((1, 3, 224, 224)), temperature=1.0)
    # softmax must have been applied: values should be probabilities, not raw logits
    assert np.isclose(result.sum(), 1.0)
    assert not np.allclose(result, raw_logits)


def test_run_session_passes_through_when_output_is_probs():
    """The combined skinanalytica_ensemble.onnx graph bakes in per-model
    softmax + temperature + weighted averaging at export time (see
    notebooks/SA02_Ensemble.ipynb's EnsembleModel/export cell) — its output
    (name 'probs') is ALREADY final. Applying softmax_with_temperature to it
    again was the double-softmax bug found and fixed this session."""
    already_probs = np.array([[0.7, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]])
    sess = FakeSession(output_name="probs", return_value=already_probs)
    result = run_session(sess, "input", np.zeros((1, 3, 224, 224)), temperature=0.4095)
    # must be passed through UNCHANGED -- this is the exact bug regression check
    np.testing.assert_array_equal(result, already_probs)


def test_run_session_double_softmax_regression():
    """Concrete regression test pinned to the real numbers from this session:
    for the known false-negative case, the combined ensemble graph's raw
    'probs' output was [0.00375, 0.985, ...]. The buggy code path
    (applying softmax_with_temperature on top) produced mel_score=0.0589 --
    a >15x distortion. run_session must NOT reproduce that."""
    real_probs_output = np.array([[0.00375, 0.9853, 0.00112, 0.0015, 0.0029, 0.004, 0.0014]])
    sess = FakeSession(output_name="probs", return_value=real_probs_output)
    result = run_session(sess, "input", np.zeros((1, 3, 224, 224)), temperature=0.4095)
    assert abs(float(result[0][0]) - 0.00375) < 1e-6, (
        "run_session distorted an already-finished probability vector -- "
        "the double-softmax bug from this session has regressed"
    )


def test_preprocess_bgr_output_shape_and_range():
    img = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    out = preprocess_bgr(img)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32


def test_tta_variants_count_and_shape():
    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    variants = tta_variants(img)
    assert len(variants) == 4  # original, flip, blur, brightness
    for v in variants:
        assert v.shape == img.shape


def test_enable_onnx_cuda_dll_dirs_does_not_crash():
    """Must be safe to call on any machine -- no GPU, no matching CUDA/cuDNN,
    non-Windows -- since scripts/score_dataset.py calls it unconditionally
    before falling back to ONNX. Doesn't assert it returns True: that
    depends on this specific machine's local CUDA/cuDNN install, which
    tests shouldn't assume."""
    result = enable_onnx_cuda_dll_dirs()
    assert isinstance(result, bool)


# ── check_dermatoscope_likelihood -- non-dermoscopic-input guardrail ───────
# Advisory heuristic only (see its docstring for the honest negative result
# on the vignette signal, dropped after failing against this project's own
# real data). Threshold (2.2) is calibrated against real aspect ratios
# sampled from every dataset this project uses: SLICE-3D crops are 1.00,
# MILK10k/most ISIC sources are 1.33, ISIC 2020 goes up to 1.78. These
# tests pin that calibration against regressing back toward the original,
# empirically-disproven "near-square" assumption.

def _flat_bgr(h, w, value=180):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_check_dermatoscope_likelihood_accepts_square_crops():
    """SLICE-3D's real aspect ratio (1.00) must never be flagged."""
    img = _flat_bgr(140, 140)
    result = check_dermatoscope_likelihood(img)
    assert result["is_extreme_aspect_ratio"] is False
    assert result["likely_dermatoscope"] is True


def test_check_dermatoscope_likelihood_accepts_4_3_crops():
    """The most common real format in this project's data (ISIC 2018/2019,
    MILK10k, EMB) -- 600x450 and equivalents, ar=1.33 -- must never be
    flagged. This is a direct regression guard: the original heuristic
    version flagged this exact ratio as non-dermatoscope, which would have
    warned on a large fraction of the project's own real training images."""
    img = _flat_bgr(450, 600)
    result = check_dermatoscope_likelihood(img)
    assert result["is_extreme_aspect_ratio"] is False
    assert result["likely_dermatoscope"] is True


def test_check_dermatoscope_likelihood_accepts_16_9ish_crops():
    """ISIC 2020's real max observed ratio (1.78) must stay under the
    threshold with real margin, not sit right at the boundary."""
    img = _flat_bgr(1053, 1872)  # ar ~= 1.78, a real ISIC 2020 image size
    result = check_dermatoscope_likelihood(img)
    assert result["aspect_ratio"] < 2.2
    assert result["likely_dermatoscope"] is True


def test_check_dermatoscope_likelihood_flags_extreme_panorama():
    """Well outside anything seen in this project's real data -- the one
    case this narrow heuristic is actually meant to catch."""
    img = _flat_bgr(200, 900)  # ar=4.5
    result = check_dermatoscope_likelihood(img)
    assert result["is_extreme_aspect_ratio"] is True
    assert result["likely_dermatoscope"] is False


def test_check_dermatoscope_likelihood_returns_expected_fields():
    img = _flat_bgr(300, 300)
    result = check_dermatoscope_likelihood(img)
    for key in ["likely_dermatoscope", "is_extreme_aspect_ratio", "aspect_ratio"]:
        assert key in result
    assert isinstance(result["aspect_ratio"], float)
    assert result["aspect_ratio"] >= 1.0


def _synthetic_lesion(h, w, blob_cx, blob_cy, blob_r, skin_value=200, blob_value=60):
    """A flat 'skin' background with a dark circular 'lesion' blob at a
    chosen position -- synthetic, but enough to test whether
    detect_lesion_bbox() actually centers on the dark region rather than
    just returning a naive center crop."""
    img = np.full((h, w, 3), skin_value, dtype=np.uint8)
    cv2.circle(img, (blob_cx, blob_cy), blob_r, (blob_value, blob_value, blob_value), -1)
    return img


def test_detect_lesion_bbox_centers_on_an_off_center_blob():
    # 2000x2000 frame, small (r=60) dark blob well off-center -- mirrors
    # finding #21's MRA-MIDAS failure mode (huge frame, small lesion, not
    # centered) far more literally than a real photo would let us assert.
    h, w = 2000, 2000
    blob_cx, blob_cy, blob_r = 500, 1400, 60
    img = _synthetic_lesion(h, w, blob_cx, blob_cy, blob_r)

    x1, y1, x2, y2 = detect_lesion_bbox(img)
    det_cx, det_cy = (x1 + x2) / 2, (y1 + y2) / 2

    # the detected crop's center should land close to the blob's actual
    # center, not the frame's geometric center (1000, 1000) -- that's
    # exactly what a naive center-crop would have returned instead
    assert abs(det_cx - blob_cx) < blob_r * 2
    assert abs(det_cy - blob_cy) < blob_r * 2
    dist_to_frame_center = ((det_cx - w / 2) ** 2 + (det_cy - h / 2) ** 2) ** 0.5
    assert dist_to_frame_center > 300  # meaningfully off the geometric center

    # crop should be square (the model's input is square) -- allow a
    # couple pixels' slack from independent integer rounding when each
    # coordinate is scaled back up from the downscaled detection space;
    # cv2.resize doesn't care about pixel-perfect squareness downstream,
    # only that it isn't wildly non-square like the original full frame
    assert abs((x2 - x1) - (y2 - y1)) <= 2
    # crop should be much smaller than the full 2000x2000 frame -- the
    # whole point is not wasting resolution on background
    assert (x2 - x1) < w * 0.5


def test_detect_lesion_bbox_falls_back_to_center_crop_with_no_lesion():
    # uniform "skin", no dark region at all -- nothing for Otsu to find
    img = _flat_bgr(800, 800, value=190)
    x1, y1, x2, y2 = detect_lesion_bbox(img)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # falls back to a center crop -- same safe fallback the center-crop
    # ablation already validated (docs/MODEL_CARD.md finding #21)
    assert abs(cx - 400) < 50
    assert abs(cy - 400) < 50


def test_detect_lesion_bbox_excludes_a_circular_vignette():
    # black surround (simulating a phone-through-dermatoscope photo's
    # circular field of view) with a bright content circle in the middle,
    # and a small dark lesion inside that -- the vignette itself must not
    # get mistaken for "content area excluded" (i.e. the returned crop
    # should sit inside the bright circle, not span the black corners)
    h, w = 1600, 1200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 550, (210, 210, 210), -1)  # bright FOV
    cv2.circle(img, (w // 2 - 150, h // 2 + 100), 50, (70, 70, 70), -1)  # small lesion, off-center

    x1, y1, x2, y2 = detect_lesion_bbox(img)
    # the crop should not include the pitch-black corners (0,0) is deep
    # in the vignette for this geometry -- x1/y1 should be well inside
    assert x1 > 50 and y1 > 50
    assert x2 < w - 50 and y2 < h - 50


def test_detect_lesion_bbox_returns_valid_coordinates_within_frame():
    img = _flat_bgr(500, 700)
    x1, y1, x2, y2 = detect_lesion_bbox(img)
    assert 0 <= x1 < x2 <= 700
    assert 0 <= y1 < y2 <= 500


def test_preprocess_bgr_lesion_crop_returns_expected_shape():
    img = _synthetic_lesion(1200, 1200, 300, 900, 40)
    out = preprocess_bgr_lesion_crop(img)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32

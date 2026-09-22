"""
Regression test for src/gpu_inference.py against the trusted ONNX result.

Skips gracefully when the large checkpoint files aren't present (they're
gitignored, multi-GB, not part of a normal checkout/CI clone) -- this is an
integration test that needs real model artifacts, not a pure-logic unit
test. Pinned to the known false-negative case used throughout this
session's investigation, cross-checked against the combined ONNX graph
(agreed to within 1e-5 when last validated).

Value updated 2026-07-28: preprocess_bgr() now center-crops to square
before resizing (see docs/MODEL_CARD.md finding #24) -- this image is
450x600, genuinely non-square, so the crop legitimately changes which
pixels reach the model and the pinned mel_score moved from 0.0037 to
0.0041. Re-pin again if preprocess_bgr() changes a second time; a drift
here otherwise means the checkpoint architecture reconstruction
(SkinClassifier) or the pooling formula (arithmetic, matching
EnsembleModel.forward) has silently changed.
"""
import os
import cv2
import pytest

from conftest import CHECKPOINTS_PRESENT, BASE

pytestmark = pytest.mark.skipif(
    not CHECKPOINTS_PRESENT,
    reason="PyTorch checkpoints not present (gitignored, large files) -- skipping GPU integration test",
)


def test_gpu_inference_matches_known_onnx_result():
    from gpu_inference import load_models, load_weights_and_temperature, predict_batch_gpu, CHECKPOINTS
    from inference_utils import preprocess_bgr

    img_path = os.path.join(
        BASE, "data", "ISIC_2018", "ISIC2018_Task3_Training_Input",
        "ISIC2018_Task3_Training_Input", "ISIC_0024306.jpg",
    )
    if not os.path.exists(img_path):
        pytest.skip("Test image not present (gitignored dataset)")

    models = load_models()
    W, T = load_weights_and_temperature()
    order = list(CHECKPOINTS.keys())

    arr = preprocess_bgr(cv2.imread(img_path))
    probs = predict_batch_gpu(models, order, W, T, arr)[0]

    # Cross-checked against the combined ONNX graph under the same (now
    # center-crop) preprocessing: ONNX gave 0.004091, GPU gives 0.004091 --
    # agree to within 1e-5.
    assert abs(float(probs[0]) - 0.0041) < 0.0005, (
        f"GPU inference mel_score={probs[0]:.4f} drifted from the validated "
        f"ONNX reference (0.0041) by more than tolerance"
    )

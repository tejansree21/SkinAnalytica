"""
SkinAnalytica — test_robustness.py
Checks whether predicted verdicts flip under realistic image-quality
degradation (JPEG recompression, blur, brightness/contrast shift) that a
phone-camera capture would introduce vs. a clean dermatoscope image, and
whether test-time augmentation (predict_tta) reduces that flip rate.
Measures internal consistency, not ground-truth accuracy.

Usage:
    python test_robustness.py --image-dir data/sa04_test_sample --limit 20
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference_utils import run_session, load_temperature, preprocess_bgr, predict_tta

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROD = os.path.join(BASE, "models", "production")
CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]


def degrade_jpeg(img, quality=25):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def degrade_blur(img, ksize=7):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def degrade_brightness(img, factor=1.4, shift=25):
    return np.clip(img.astype(np.float32) * factor + shift, 0, 255).astype(np.uint8)


def main(image_dir: str, limit: int):
    onnx_path = os.path.join(PROD, "onnx", "skinanalytica_ensemble.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    T = load_temperature(PROD)

    def predict_single(img_bgr):
        return run_session(sess, inp_name, preprocess_bgr(img_bgr), T)[0]

    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:limit]
    degradations = {"jpeg_q25": degrade_jpeg, "blur_k7": degrade_blur, "bright_1.4x": degrade_brightness}
    flips_single = {k: 0 for k in degradations}
    flips_tta = {k: 0 for k in degradations}

    print(f"Testing {len(paths)} images | Temperature={T}\n")
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        clean_single = CLASSES[predict_single(img).argmax()]
        clean_tta_mean, _ = predict_tta(sess, inp_name, T, img)
        clean_tta = CLASSES[clean_tta_mean.argmax()]

        row = [f"{os.path.basename(path):<20} clean={clean_single}"]
        for name, fn in degradations.items():
            d_img = fn(img.copy())
            d_single = CLASSES[predict_single(d_img).argmax()]
            d_tta_mean, _ = predict_tta(sess, inp_name, T, d_img)
            d_tta = CLASSES[d_tta_mean.argmax()]
            if d_single != clean_single:
                flips_single[name] += 1
            if d_tta != clean_tta:
                flips_tta[name] += 1
            flip_mark = " <-- FLIP" if d_single != clean_single else ""
            row.append(f"{name}={d_single}{flip_mark}")
        print("  ".join(row))

    print("\n=== Summary (single-pass vs TTA flip counts) ===")
    for name in degradations:
        print(f"{name:<14} single={flips_single[name]}/{len(paths)}  tta={flips_tta[name]}/{len(paths)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default=os.path.join(BASE, "data", "sa04_test_sample"))
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    main(args.image_dir, args.limit)

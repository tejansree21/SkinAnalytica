"""
SkinAnalytica -- scripts/ablation_centercrop_milk10k.py
Companion to ablation_centercrop_mra_midas.py -- same center-crop
preprocessing swap (see docs/MODEL_CARD.md finding #21), applied to
MILK10k's holdout instead. This is the critical guardrail check: MILK10k
is one of the populations that already works well (MelAUC 0.8796,
finding #14), sitting in the same "moderate-resolution, pre-cropped"
comfort zone as training data. If center-crop preprocessing helps
MRA-MIDAS but HURTS MILK10k, that rules it out as a real fix -- a change
that trades one population's performance for another's isn't a fix, it's
a different bias. Only worth treating center-crop as a real candidate
production change if it helps (or is at least neutral on) both.

Usage:
  python scripts/ablation_centercrop_milk10k.py
"""
import csv
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference_utils import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from gpu_inference import load_models, load_weights_and_temperature, predict_batch_gpu, CHECKPOINTS  # noqa: E402

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
MEL_IDX = 0
CANCER_CLASSES = {"mel", "bcc", "akiec"}


def preprocess_centercrop(img_bgr: np.ndarray) -> np.ndarray:
    """Identical to ablation_centercrop_mra_midas.py's version -- kept as
    a literal copy rather than a shared import so each ablation script
    stays a self-contained, standalone record of exactly what it tested."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    bhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thr = cv2.threshold(bhat, 10, 255, cv2.THRESH_BINARY)
    img = cv2.inpaint(img_bgr, thr, 1, cv2.INPAINT_TELEA)

    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top:top + side, left:left + side]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def main():
    scored_path = os.path.join(BASE, "outputs", "bias_reports", "milk10k_holdout_scored.csv")
    rows = list(csv.DictReader(open(scored_path, encoding="utf-8")))
    print(f"Re-scoring {len(rows)} MILK10k holdout images with center-crop preprocessing...")

    models = load_models()
    W, T = load_weights_and_temperature()
    order = list(CHECKPOINTS.keys())

    batch_size = 32
    all_probs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        arrays = []
        for r in batch:
            img = cv2.imread(r["path"])
            arrays.append(preprocess_centercrop(img) if img is not None else np.zeros((1, 3, 224, 224), dtype=np.float32))
        batch_np = np.concatenate(arrays, axis=0)
        all_probs.append(predict_batch_gpu(models, order, W, T, batch_np))
        if (i // batch_size) % 5 == 0:
            print(f"  {min(i+batch_size, len(rows))}/{len(rows)}")

    BCC_IDX, AKIEC_IDX = UNIFIED_CLASSES.index("bcc"), UNIFIED_CLASSES.index("akiec")
    probs = np.concatenate(all_probs, axis=0)
    centercrop_mel = probs[:, MEL_IDX]
    centercrop_malignant = probs[:, [MEL_IDX, BCC_IDX, AKIEC_IDX]].max(axis=1)

    y_true_label = np.array([UNIFIED_CLASSES.index(r["label"]) for r in rows])
    y_mel = (y_true_label == MEL_IDX).astype(int)
    true_malignant = np.isin(y_true_label, [UNIFIED_CLASSES.index(c) for c in CANCER_CLASSES]).astype(int)

    naive_mel = np.array([float(r["mel_score"]) for r in rows])
    # malignant_score = max(mel, bcc, akiec) -- 2026-07-27 fix, see
    # docs/MODEL_CARD.md findings #18/#22; MILK10k's malignant pool is
    # 76.8% BCC, exactly the case mel_score-alone under-credits worst
    naive_malignant = np.array([float(r["malignant_score"]) if "malignant_score" in r else float(r["mel_score"]) for r in rows])

    print("\n" + "=" * 60)
    print("Center-crop preprocessing vs. original naive-resize baseline -- MILK10k")
    print("=" * 60)
    for name, y_true, crop_score, naive_score in (
        ("MelAUC (mel vs rest)", y_mel, centercrop_mel, naive_mel),
        ("Malignant-vs-benign AUC", true_malignant, centercrop_malignant, naive_malignant),
    ):
        naive_auc = roc_auc_score(y_true, naive_score)
        crop_auc = roc_auc_score(y_true, crop_score)
        print(f"{name}:")
        print(f"  Naive resize (original):  {naive_auc:.4f}")
        print(f"  Center-crop:              {crop_auc:.4f}")
        print(f"  Delta: {crop_auc - naive_auc:+.4f}\n")

    out_path = os.path.join(BASE, "outputs", "bias_reports", "milk10k_centercrop_scored.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["isic_id", "label", "naive_mel_score", "centercrop_mel_score"])
        for r, nm, cm in zip(rows, naive_mel, centercrop_mel):
            w_.writerow([r["isic_id"], r["label"], nm, cm])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

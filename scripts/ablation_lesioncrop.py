"""
SkinAnalytica -- scripts/ablation_lesioncrop.py
Tests the real lesion-detecting crop (inference_utils.detect_lesion_bbox()
/ preprocess_bgr_lesion_crop()) against both the naive-resize baseline and
the earlier center-crop ablation -- see docs/MODEL_CARD.md finding #21.
Visually verified correct on a real MRA-MIDAS example before this run
(the detected box landed on the actual lesion, not the ruler overlay or
the vignette). No retraining, same CNN ensemble throughout -- only the
preprocessing changes.

Runs on both MRA-MIDAS (the population this exists to help) and MILK10k
(the guardrail: does it cost anything on a population that already works
well?), same discipline as the center-crop ablation.

Usage:
  python scripts/ablation_lesioncrop.py --dataset mra_midas
  python scripts/ablation_lesioncrop.py --dataset milk10k
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference_utils import preprocess_bgr_lesion_crop  # noqa: E402
from gpu_inference import load_models, load_weights_and_temperature, predict_batch_gpu, CHECKPOINTS  # noqa: E402

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
MEL_IDX = 0
CANCER_CLASSES = {"mel", "bcc", "akiec"}

DATASETS = {
    "mra_midas": {
        "scored_csv": "mra_midas_scored.csv",
        "id_col": "midas_file_name",
        "out_csv": "mra_midas_lesioncrop_scored.csv",
    },
    "milk10k": {
        "scored_csv": "milk10k_holdout_scored.csv",
        "id_col": "isic_id",
        "out_csv": "milk10k_lesioncrop_scored.csv",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]

    scored_path = os.path.join(BASE, "outputs", "bias_reports", cfg["scored_csv"])
    rows = list(csv.DictReader(open(scored_path, encoding="utf-8")))
    print(f"Re-scoring {len(rows)} {args.dataset} images with lesion-detecting crop...", flush=True)

    models = load_models()
    W, T = load_weights_and_temperature()
    order = list(CHECKPOINTS.keys())

    batch_size = args.batch_size
    all_probs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        arrays = []
        for r in batch:
            img = cv2.imread(r["path"])
            arrays.append(preprocess_bgr_lesion_crop(img) if img is not None else np.zeros((1, 3, 224, 224), dtype=np.float32))
        batch_np = np.concatenate(arrays, axis=0)
        all_probs.append(predict_batch_gpu(models, order, W, T, batch_np))
        if (i // batch_size) % 5 == 0:
            print(f"  {min(i+batch_size, len(rows))}/{len(rows)}", flush=True)

    BCC_IDX, AKIEC_IDX = UNIFIED_CLASSES.index("bcc"), UNIFIED_CLASSES.index("akiec")
    probs = np.concatenate(all_probs, axis=0)
    crop_mel = probs[:, MEL_IDX]
    crop_malignant = probs[:, [MEL_IDX, BCC_IDX, AKIEC_IDX]].max(axis=1)

    y_true_label = np.array([UNIFIED_CLASSES.index(r["label"]) for r in rows])
    y_mel = (y_true_label == MEL_IDX).astype(int)
    true_malignant = np.isin(y_true_label, [UNIFIED_CLASSES.index(c) for c in CANCER_CLASSES]).astype(int)

    naive_mel = np.array([float(r["mel_score"]) for r in rows])
    # malignant_score = max(mel, bcc, akiec) -- 2026-07-27 fix, see
    # docs/MODEL_CARD.md findings #18/#22
    naive_malignant = np.array([float(r["malignant_score"]) if "malignant_score" in r else float(r["mel_score"]) for r in rows])

    print("\n" + "=" * 60, flush=True)
    print(f"Lesion-crop preprocessing vs. naive-resize baseline -- {args.dataset}", flush=True)
    print("=" * 60, flush=True)
    metrics = (
        ("MelAUC (mel vs rest)", y_mel, crop_mel, naive_mel),
        ("Malignant-vs-benign AUC", true_malignant, crop_malignant, naive_malignant),
    )
    for name, y_true, crop_score, naive_score in metrics:
        naive_auc = roc_auc_score(y_true, naive_score)
        crop_auc = roc_auc_score(y_true, crop_score)
        print(f"{name}:", flush=True)
        print(f"  Naive resize (original):  {naive_auc:.4f}", flush=True)
        print(f"  Lesion-crop:              {crop_auc:.4f}", flush=True)
        print(f"  Delta: {crop_auc - naive_auc:+.4f}\n", flush=True)

    # paired bootstrap on the delta -- same discipline as the center-crop ablation
    rng = np.random.RandomState(42)
    n = len(rows)
    for name, y_true, crop_score, naive_score in metrics:
        deltas = []
        for _ in range(2000):
            idx = rng.randint(0, n, n)
            yt = y_true[idx]
            if len(set(yt.tolist())) < 2:
                continue
            deltas.append(roc_auc_score(yt, crop_score[idx]) - roc_auc_score(yt, naive_score[idx]))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        verdict = "excludes 0 -- real" if (lo > 0 or hi < 0) else "includes 0 -- not confirmed"
        print(f"{name} bootstrap: mean delta={deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}", flush=True)

    out_path = os.path.join(BASE, "outputs", "bias_reports", cfg["out_csv"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow([cfg["id_col"], "label", "naive_mel_score", "lesioncrop_mel_score"])
        for r, nm, cm in zip(rows, naive_mel, crop_mel):
            w_.writerow([r[cfg["id_col"]], r["label"], nm, cm])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

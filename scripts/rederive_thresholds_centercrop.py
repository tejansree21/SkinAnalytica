"""
SkinAnalytica -- scripts/rederive_thresholds_centercrop.py
Re-derives every deployed melanoma-probability threshold
(MEL_THRESHOLD, TBP_MEL_THRESHOLD, AGE_BAND_THRESHOLDS, SKIN_TONE_THRESHOLDS)
under the 2026-07-28 center-crop preprocessing change (docs/MODEL_CARD.md
finding #24), using the exact same "first ROC threshold that reaches 80%
sensitivity" methodology as the original derivations -- validated against
already-published values before this script was trusted: reproduces
TBP_MEL_THRESHOLD=0.0199, all three SKIN_TONE_THRESHOLDS, and all five
AGE_BAND_THRESHOLDS exactly from the existing (naive-resize) scored CSVs.

Four populations, one per threshold family:
  1. config/val.csv (5,978 images, ISIC 2018+2019+2020 blended) -> MEL_THRESHOLD.
     NOTE: this is the real, on-disk, fixed training validation split --
     but the original ensemble_metrics_selfconsistent.json's exact
     "10,312-image, 922-melanoma" set could not be located as a saved,
     reusable manifest anywhere in the repo (computed ad hoc in a prior
     session, never saved to disk). This is the closest available faithful
     reconstruction, not a byte-for-byte reproduction of that exact set --
     see docs/MODEL_CARD.md finding #25 for the honest accounting.
  2. outputs/bias_reports/slice3d_full_holdout_postretrain_scored.csv's
     143-malignant-case holdout (finding #12) -> TBP_MEL_THRESHOLD.
  3. outputs/bias_reports/isic2020_scored_sample.csv's image_id list,
     re-joined to real paths + ISIC-2020 ground-truth age -> AGE_BAND_THRESHOLDS.
  4. outputs/bias_reports/milk10k_holdout_scored.csv's 700-image skin-tone
     holdout (finding #14) -> SKIN_TONE_THRESHOLDS.

Usage:
  python scripts/rederive_thresholds_centercrop.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference_utils import preprocess_bgr  # noqa: E402
from gpu_inference import load_models, load_weights_and_temperature, predict_batch_gpu, CHECKPOINTS  # noqa: E402

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BR = os.path.join(BASE, "outputs", "bias_reports")
UNIFIED_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
MEL_IDX = 0
AGE_BINS = [0, 30, 45, 60, 75, 200]
AGE_LABELS = ["<30", "30-45", "45-60", "60-75", "75+"]


def threshold_for_sensitivity(y_true, score, target=0.80):
    fpr, tpr, th = roc_curve(y_true, score)
    cand = np.where(tpr >= target)[0]
    if len(cand) == 0:
        return None, None, None
    idx = cand[0]
    return float(th[idx]), float(tpr[idx]), float(1 - fpr[idx])


def score_paths(models, order, W, T, paths, batch_size=48, tag=""):
    all_probs = []
    n = len(paths)
    for i in range(0, n, batch_size):
        batch = paths[i:i + batch_size]
        arrays = []
        for p in batch:
            img = cv2.imread(p)
            arrays.append(preprocess_bgr(img) if img is not None else np.zeros((1, 3, 224, 224), dtype=np.float32))
        batch_np = np.concatenate(arrays, axis=0)
        all_probs.append(predict_batch_gpu(models, order, W, T, batch_np))
        done = min(i + batch_size, n)
        if (i // batch_size) % 5 == 0 or done == n:
            print(f"  [{tag}] {done}/{n}", flush=True)
    return np.concatenate(all_probs, axis=0)


def main():
    print("Loading models...", flush=True)
    models = load_models()
    W, T = load_weights_and_temperature()
    order = list(CHECKPOINTS.keys())

    report = {}

    # ---------------------------------------------------------------
    # 1. MEL_THRESHOLD -- config/val.csv
    # ---------------------------------------------------------------
    print("\n=== 1/4: MEL_THRESHOLD (config/val.csv) ===", flush=True)
    val_df = pd.read_csv(os.path.join(BASE, "config", "val.csv"))
    val_df = val_df[val_df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    probs = score_paths(models, order, W, T, val_df["image_path"].tolist(), tag="val")
    mel_score = probs[:, MEL_IDX]
    y_mel = (val_df["label"].str.lower() == "mel").astype(int).values
    auc = roc_auc_score(y_mel, mel_score)
    t, sens, spec = threshold_for_sensitivity(y_mel, mel_score)
    report["MEL_THRESHOLD"] = {
        "old": 0.2385, "new": round(t, 4), "n": len(val_df), "n_mel": int(y_mel.sum()),
        "auc": round(float(auc), 4), "sensitivity_at_new": round(sens, 4), "specificity_at_new": round(spec, 4),
        "population": "config/val.csv (ISIC 2018+2019+2020 blended, on-disk fixed val split -- "
                       "NOT a reproduction of the original ad hoc 10,312-image set, see finding #25)",
    }
    print(f"MEL_THRESHOLD: 0.2385 -> {t:.4f}  (AUC={auc:.4f}, sens={sens:.1%}, spec={spec:.1%}, n={len(val_df)}, n_mel={int(y_mel.sum())})", flush=True)

    # ---------------------------------------------------------------
    # 2. TBP_MEL_THRESHOLD -- SLICE-3D full holdout (143 malignant)
    # ---------------------------------------------------------------
    print("\n=== 2/4: TBP_MEL_THRESHOLD (SLICE-3D 143-malignant holdout) ===", flush=True)
    tbp_df = pd.read_csv(os.path.join(BR, "slice3d_full_holdout_postretrain_scored.csv"))
    tbp_df = tbp_df[tbp_df["path"].apply(os.path.exists)].reset_index(drop=True)
    probs = score_paths(models, order, W, T, tbp_df["path"].tolist(), tag="tbp")
    mel_score = probs[:, MEL_IDX]
    y_mal = tbp_df["malignant"].astype(int).values
    auc = roc_auc_score(y_mal, mel_score)
    t, sens, spec = threshold_for_sensitivity(y_mal, mel_score)
    report["TBP_MEL_THRESHOLD"] = {
        "old": 0.0199, "new": round(t, 4), "n": len(tbp_df), "n_malignant": int(y_mal.sum()),
        "auc": round(float(auc), 4), "sensitivity_at_new": round(sens, 4), "specificity_at_new": round(spec, 4),
    }
    print(f"TBP_MEL_THRESHOLD: 0.0199 -> {t:.4f}  (AUC={auc:.4f}, sens={sens:.1%}, spec={spec:.1%}, n={len(tbp_df)}, n_malignant={int(y_mal.sum())})", flush=True)

    # ---------------------------------------------------------------
    # 3. AGE_BAND_THRESHOLDS -- ISIC-2020 sample
    # ---------------------------------------------------------------
    print("\n=== 3/4: AGE_BAND_THRESHOLDS (ISIC-2020 sample) ===", flush=True)
    age_df = pd.read_csv(os.path.join(BR, "isic2020_scored_sample.csv"))
    img_dir = os.path.join(BASE, "data", "ISIC_2020", "ISIC_2020_Training_JPEG", "train")
    age_df["path"] = age_df["image_id"].apply(lambda iid: os.path.join(img_dir, str(iid) + ".jpg"))
    age_df = age_df[age_df["path"].apply(os.path.exists)].reset_index(drop=True)
    gt = pd.read_csv(os.path.join(BASE, "data", "ISIC_2020", "ISIC_2020_Training_GroundTruth_v2.csv"))
    gt = gt.rename(columns={"image_name": "image_id"})[["image_id", "age_approx"]]
    age_df = age_df.merge(gt, on="image_id", how="left")
    age_df["age_group"] = pd.cut(age_df["age_approx"], bins=AGE_BINS, labels=AGE_LABELS)

    probs = score_paths(models, order, W, T, age_df["path"].tolist(), tag="age")
    age_df["mel_score"] = probs[:, MEL_IDX]

    age_thresholds = {}
    old_vals = {"<30": 0.0365, "30-45": 0.0314, "45-60": 0.1179, "60-75": 0.2725, "75+": 0.1444}
    for ag in AGE_LABELS:
        sub = age_df[age_df["age_group"] == ag]
        y = (sub["label"].str.lower() == "mel").astype(int).values
        m = sub["mel_score"].values
        n_mel = int(y.sum())
        if n_mel < 5 or (1 - y).sum() < 5:
            print(f"  {ag}: insufficient data (n_mel={n_mel})", flush=True)
            continue
        t, sens, spec = threshold_for_sensitivity(y, m)
        age_thresholds[ag] = {"old": old_vals[ag], "new": round(t, 4), "n": len(sub), "n_mel": n_mel,
                               "sensitivity_at_new": round(sens, 4), "specificity_at_new": round(spec, 4)}
        print(f"  {ag}: {old_vals[ag]} -> {t:.4f}  (sens={sens:.1%}, spec={spec:.1%}, n={len(sub)}, n_mel={n_mel})", flush=True)
    report["AGE_BAND_THRESHOLDS"] = age_thresholds

    # ---------------------------------------------------------------
    # 4. SKIN_TONE_THRESHOLDS -- MILK10k holdout
    # ---------------------------------------------------------------
    print("\n=== 4/4: SKIN_TONE_THRESHOLDS (MILK10k holdout) ===", flush=True)
    milk_df = pd.read_csv(os.path.join(BR, "milk10k_holdout_scored.csv"))
    milk_df = milk_df[milk_df["path"].apply(os.path.exists)].reset_index(drop=True)
    probs = score_paths(models, order, W, T, milk_df["path"].tolist(), tag="milk10k")
    milk_df["mel_score"] = probs[:, MEL_IDX]

    tone_thresholds = {}
    old_vals = {2: 0.0423, 3: 0.0399, 4: 0.0155}
    for tone in [2, 3, 4]:
        sub = milk_df[milk_df["skin_tone_class"] == tone]
        y = (sub["label"] == "mel").astype(int).values
        m = sub["mel_score"].values
        n_mel = int(y.sum())
        t, sens, spec = threshold_for_sensitivity(y, m)
        tone_thresholds[tone] = {"old": old_vals[tone], "new": round(t, 4), "n": len(sub), "n_mel": n_mel,
                                  "sensitivity_at_new": round(sens, 4), "specificity_at_new": round(spec, 4)}
        print(f"  Tone {tone}: {old_vals[tone]} -> {t:.4f}  (sens={sens:.1%}, spec={spec:.1%}, n={len(sub)}, n_mel={n_mel})", flush=True)
    report["SKIN_TONE_THRESHOLDS"] = tone_thresholds

    out_path = os.path.join(BR, "rederived_thresholds_centercrop.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()

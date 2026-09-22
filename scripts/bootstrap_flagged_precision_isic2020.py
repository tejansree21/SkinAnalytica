"""
SkinAnalytica -- scripts/bootstrap_flagged_precision_isic2020.py
Same follow-up as scripts/bootstrap_flagged_precision.py, but for
finding #16's ISIC-2020 natural-prevalence number (29.8% at 1.76% malignant
prevalence) -- that number was a single resampling draw, same as the
original MILK10k figures were before scripts/bootstrap_flagged_precision.py
existed. No paired baseline here (there's no fusion-model comparison for
ISIC-2020 in this project), just a proper bootstrap CI on the one number.

Also re-syncs to the CURRENT production verdict thresholds (finding #26
changed MEL_THRESHOLD 0.2385 -> 0.5010, and AGE_BAND_THRESHOLDS with it) --
finding #16's original 29.8% predates that change and is stale.

Leakage caveat carried over unchanged from finding #16: ISIC-2020 was
almost certainly in the training pool, so treat this as an optimistic
upper bound, not an independent external test -- this script only adds a
confidence interval to an existing, already-caveated number, it does not
fix the leakage issue itself.

Usage:
  python scripts/bootstrap_flagged_precision_isic2020.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_fusion import verdict  # noqa: E402 -- current production thresholds

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BR = os.path.join(BASE, "outputs", "bias_reports")
N_BOOT = 2000
NATURAL_PREVALENCE = 0.0176  # per the full 33,126-image ISIC_2020_Training_GroundTruth_v2.csv, finding #16


def main():
    df = pd.read_csv(os.path.join(BR, "isic2020_scored_sample.csv"))
    gt = pd.read_csv(os.path.join(BASE, "data", "ISIC_2020", "ISIC_2020_Training_GroundTruth_v2.csv"))
    gt = gt.rename(columns={"image_name": "image_id"})[["image_id", "age_approx"]]
    df = df.merge(gt, on="image_id", how="left")

    idx_to_class = {0: "mel", 1: "nv", 2: "bcc", 3: "akiec", 4: "bkl", 5: "df", 6: "vasc"}
    y = (df["label"].str.lower() == "mel").astype(int).values
    mel_score = df["mel_score"].values
    pred_class = df["pred_idx"].map(idx_to_class).values
    confidence = df["confidence"].values
    age = df["age_approx"].values

    benign_idx = np.where(y == 0)[0]
    mel_idx = np.where(y == 1)[0]
    n_benign = len(benign_idx)
    n_mel_target = min(int(round(n_benign * NATURAL_PREVALENCE / (1 - NATURAL_PREVALENCE))), len(mel_idx))
    print(f"n_benign={n_benign}  n_mel_available={len(mel_idx)}  n_mel_target_per_draw={n_mel_target}")
    print(f"Running {N_BOOT}-resample bootstrap at natural prevalence {NATURAL_PREVALENCE:.2%}...\n")

    rng = np.random.RandomState(42)
    precisions = []
    for _ in range(N_BOOT):
        b_idx = rng.choice(benign_idx, n_benign, replace=True)
        m_idx = rng.choice(mel_idx, n_mel_target, replace=True)
        idx = np.concatenate([b_idx, m_idx])

        tp, flagged = 0, 0
        for i in idx:
            v = verdict(mel_score[i], pred_class[i], confidence[i], age[i])
            if v == "CANCER_FLAGGED":
                flagged += 1
                if y[i]:
                    tp += 1
        if flagged:
            precisions.append(tp / flagged)

    precisions = np.array(precisions)
    mean, lo, hi = precisions.mean(), *np.percentile(precisions, [2.5, 97.5])
    print(f"Flagged-precision @ {NATURAL_PREVALENCE:.2%} prevalence: {mean:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  (n_valid_draws={len(precisions)})")

    out_path = os.path.join(BASE, "outputs", "bias_reports", "isic2020_bootstrap_flagged_precision.json")
    with open(out_path, "w") as f:
        json.dump({
            "natural_prevalence": NATURAL_PREVALENCE,
            "flagged_precision_mean": float(mean),
            "flagged_precision_ci": [float(lo), float(hi)],
            "n_valid_draws": int(len(precisions)),
            "threshold_used": "current production (finding #26): MEL_THRESHOLD=0.5010",
            "leakage_caveat": "ISIC-2020 was almost certainly in the training pool -- optimistic upper bound, not an independent test",
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""
SkinAnalytica -- scripts/evaluate_fusion_mra_midas.py
MRA-MIDAS as a second external holdout for the Path B fusion model --
follow-up to docs/MODEL_CARD.md finding #19, using an independently-
sourced dataset (different institutions, different population) rather
than re-analyzing the same 700 MILK10k images a third time.

Reuses verdict()/bootstrap_auc_ci()/the class constants from
evaluate_fusion.py -- same discipline as bootstrap_flagged_precision.py,
one implementation of the verdict logic, not a second one that can drift.

Two-part evaluation, same shape as finding #19:
1. Discrimination: MelAUC (mel vs rest) and malignant-vs-benign AUC,
   fusion vs the already-scored baseline (outputs/bias_reports/
   mra_midas_scored.csv), with bootstrap CIs.
2. Flagged-precision at realistic assumed prevalence -- MRA-MIDAS's raw
   holdout is a biopsy-focused clinical population (~65% malignant-class
   by image count), nothing like a real screening intake, so this
   resamples toward plausible prevalences exactly like finding #16/#19
   did for MILK10k. Paired bootstrap (same resampled index set scored
   under both models each draw).

Usage:
  python scripts/evaluate_fusion_mra_midas.py
"""
import csv
import os

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate_fusion import (
    BASE, MODEL_DIR, UNIFIED_CLASSES, MEL_IDX, CANCER_CLASSES,
    verdict, load_features, bootstrap_auc_ci,
)

N_BOOT = 2000
PREVALENCES = (0.02, 0.05, 0.10, 0.20)


def load_baseline_scores() -> dict:
    """Returns image_id -> (mel_score, pred_class, confidence,
    malignant_score). malignant_score = max(mel, bcc, akiec) -- see the
    2026-07-27 fix in scripts/score_dataset.py: the malignant-vs-benign
    comparison needs this, not mel_score alone (mel_score under-credits
    true BCC/AKIEC cases, which is most of MRA-MIDAS's malignant pool)."""
    path = os.path.join(BASE, "outputs", "bias_reports", "mra_midas_scored.csv")
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "malignant_score" in row:
                malignant_score = float(row["malignant_score"])
            else:
                malignant_score = float(row["mel_score"])  # pre-fix CSV, not yet re-scored
            out[row["midas_file_name"]] = (
                float(row["mel_score"]),
                UNIFIED_CLASSES[int(row["pred_idx"])],
                float(row["confidence"]),
                malignant_score,
            )
    return out


def flagged_precision(indices, mel_scores, pred_classes, confidences, ages, true_malignant) -> float:
    tp, flagged = 0, 0
    for i in indices:
        v = verdict(mel_scores[i], pred_classes[i], confidences[i], ages[i])
        if v == "CANCER_FLAGGED":
            flagged += 1
            if true_malignant[i]:
                tp += 1
    return (tp / flagged) if flagged else float("nan")


def main():
    print("Loading fusion model and MRA-MIDAS holdout...")
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_fusion.txt"))
    d = load_features("mra_midas_holdout")
    X, y, image_ids = d["X"], d["y"], d["image_id"]
    feature_names = list(d["feature_names"])
    age_col = feature_names.index("age_approx")
    ages = X[:, age_col]

    BCC_IDX, AKIEC_IDX = UNIFIED_CLASSES.index("bcc"), UNIFIED_CLASSES.index("akiec")

    fusion_probs = booster.predict(X)
    fusion_mel = fusion_probs[:, MEL_IDX]
    fusion_malignant = fusion_probs[:, [MEL_IDX, BCC_IDX, AKIEC_IDX]].max(axis=1)
    fusion_pred_class = [UNIFIED_CLASSES[i] for i in fusion_probs.argmax(axis=1)]
    fusion_conf = fusion_probs.max(axis=1)

    baseline_lookup = load_baseline_scores()
    baseline_mel = np.array([baseline_lookup[iid][0] for iid in image_ids])
    baseline_pred_class = [baseline_lookup[iid][1] for iid in image_ids]
    baseline_conf = np.array([baseline_lookup[iid][2] for iid in image_ids])
    baseline_malignant = np.array([baseline_lookup[iid][3] for iid in image_ids])

    y_mel = (y == MEL_IDX).astype(int)
    true_malignant = np.isin(y, [UNIFIED_CLASSES.index(c) for c in CANCER_CLASSES])

    print(f"\nHoldout: {len(image_ids)} total, {int(y_mel.sum())} melanoma, "
          f"{int(true_malignant.sum())} malignant-class (mel+bcc+akiec)\n")

    print("=" * 70)
    print("1. DISCRIMINATION -- fusion vs baseline, paired holdout")
    print("=" * 70)
    # NOTE (2026-07-27 fix): "Malignant-vs-benign AUC" now ranks with
    # malignant_score = max(mel, bcc, akiec), not mel_score alone --
    # mel_score under-credits true BCC/AKIEC cases, which dominate
    # MRA-MIDAS's malignant pool. See docs/MODEL_CARD.md findings #18/#22.
    for name, y_true, fusion_score, baseline_score in (
        ("MelAUC (mel vs rest)", y_mel, fusion_mel, baseline_mel),
        ("Malignant-vs-benign AUC", true_malignant.astype(int), fusion_malignant, baseline_malignant),
    ):
        f_auc = roc_auc_score(y_true, fusion_score)
        f_lo, f_hi = bootstrap_auc_ci(y_true, fusion_score)
        b_auc = roc_auc_score(y_true, baseline_score)
        b_lo, b_hi = bootstrap_auc_ci(y_true, baseline_score)
        print(f"{name}:")
        print(f"  Fusion:   {f_auc:.4f}  95% CI [{f_lo:.4f}, {f_hi:.4f}]")
        print(f"  Baseline: {b_auc:.4f}  95% CI [{b_lo:.4f}, {b_hi:.4f}]")
        overlap = not (f_lo > b_hi or b_lo > f_hi)
        print(f"  {'CIs overlap -- not distinguishable' if overlap else 'CIs do NOT overlap -- likely real difference'}\n")

    print("=" * 70)
    print("2. FLAGGED-PRECISION AT REALISTIC PREVALENCE (paired bootstrap)")
    print("=" * 70)
    benign_idx = np.where(~true_malignant)[0]
    malignant_idx = np.where(true_malignant)[0]
    n_benign = len(benign_idx)
    print(f"Pool: {len(malignant_idx)} malignant-class, {n_benign} benign\n")

    rng = np.random.RandomState(42)
    print(f"{'Prevalence':<12}{'Fusion precision':<28}{'Baseline precision':<28}{'Delta (paired)':<28}{'CI excludes 0?'}")
    for prevalence in PREVALENCES:
        n_mal_target = min(int(round(n_benign * prevalence / (1 - prevalence))), len(malignant_idx))
        if n_mal_target < 3:
            print(f"{int(prevalence*100)}%{'':<9}too few malignant cases available for this prevalence")
            continue

        fusion_vals, baseline_vals, delta_vals = [], [], []
        for _ in range(N_BOOT):
            b_idx = rng.choice(benign_idx, n_benign, replace=True)
            m_idx = rng.choice(malignant_idx, n_mal_target, replace=True)
            idx = np.concatenate([b_idx, m_idx])

            p_f = flagged_precision(idx, fusion_mel, fusion_pred_class, fusion_conf, ages, true_malignant)
            p_b = flagged_precision(idx, baseline_mel, baseline_pred_class, baseline_conf, ages, true_malignant)
            if np.isnan(p_f) or np.isnan(p_b):
                continue
            fusion_vals.append(p_f)
            baseline_vals.append(p_b)
            delta_vals.append(p_f - p_b)

        fusion_vals, baseline_vals, delta_vals = map(np.array, (fusion_vals, baseline_vals, delta_vals))
        f_mean, f_lo, f_hi = fusion_vals.mean(), *np.percentile(fusion_vals, [2.5, 97.5])
        b_mean, b_lo, b_hi = baseline_vals.mean(), *np.percentile(baseline_vals, [2.5, 97.5])
        d_mean, d_lo, d_hi = delta_vals.mean(), *np.percentile(delta_vals, [2.5, 97.5])
        excludes_zero = (d_lo > 0) or (d_hi < 0)

        print(f"{int(prevalence*100)}%{'':<9}{100*f_mean:>5.1f}% [{100*f_lo:.1f},{100*f_hi:.1f}]{'':<8}"
              f"{100*b_mean:>5.1f}% [{100*b_lo:.1f},{100*b_hi:.1f}]{'':<8}"
              f"{100*d_mean:+.1f}pp [{100*d_lo:+.1f},{100*d_hi:+.1f}]{'':<6}"
              f"{'YES -- likely real' if excludes_zero else 'no -- not distinguishable from noise'}")


if __name__ == "__main__":
    main()

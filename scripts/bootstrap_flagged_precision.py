"""
SkinAnalytica -- scripts/bootstrap_flagged_precision.py
Follow-up to docs/MODEL_CARD.md finding #18: the flagged-precision deltas
reported there (+3.9pp/+6.2pp/+4.6pp at 5%/10%/20% assumed prevalence)
came from a single resampling draw per prevalence level -- not enough to
tell real signal from noise. This runs a proper paired bootstrap instead.

Why PAIRED, not two independent bootstraps: both the baseline (pre-fusion
CNN ensemble, from the already-scored outputs/bias_reports/
milk10k_holdout_scored.csv) and the fusion model's predictions exist for
the exact same 700 MILK10k holdout images. Every bootstrap iteration
resamples the same index set and scores it under BOTH models, so
prevalence-sampling noise that affects both identically cancels out of
the delta -- much higher power to detect a genuine fusion-vs-baseline
difference than comparing two independently-bootstrapped CIs would give.

Reuses verdict()/get_mel_threshold()/the class constants from
evaluate_fusion.py rather than redefining them -- same discipline as
everywhere else in this project (one implementation of the verdict logic,
not two that can quietly drift apart).

Usage:
  python scripts/bootstrap_flagged_precision.py
"""
import csv
import os

import lightgbm as lgb
import numpy as np

from evaluate_fusion import (
    BASE, EMB_DIR, MODEL_DIR, UNIFIED_CLASSES, MEL_IDX, CANCER_CLASSES,
    verdict, load_features, BASELINE_FLAGGED_PRECISION,
)

N_BOOT = 2000
PREVALENCES = (0.02, 0.05, 0.10, 0.20)


def load_baseline_scores() -> dict:
    """image_id -> (mel_score, pred_class, confidence) from the CNN
    ensemble's already-scored CSV -- the same pre-fusion numbers finding
    #16's baseline flagged-precision figures came from."""
    path = os.path.join(BASE, "outputs", "bias_reports", "milk10k_holdout_scored.csv")
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["isic_id"]] = (
                float(row["mel_score"]),
                UNIFIED_CLASSES[int(row["pred_idx"])],
                float(row["confidence"]),
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
    print("Loading fusion model and MILK10k holdout...")
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_fusion.txt"))
    d = load_features("milk10k_holdout")
    X, y, image_ids = d["X"], d["y"], d["image_id"]
    feature_names = list(d["feature_names"])
    age_col = feature_names.index("age_approx")
    ages = X[:, age_col]
    true_malignant = np.isin(y, [UNIFIED_CLASSES.index(c) for c in CANCER_CLASSES])

    fusion_probs = booster.predict(X)
    fusion_mel = fusion_probs[:, MEL_IDX]
    fusion_pred_class = [UNIFIED_CLASSES[i] for i in fusion_probs.argmax(axis=1)]
    fusion_conf = fusion_probs.max(axis=1)

    baseline_lookup = load_baseline_scores()
    baseline_mel = np.array([baseline_lookup[iid][0] for iid in image_ids])
    baseline_pred_class = [baseline_lookup[iid][1] for iid in image_ids]
    baseline_conf = np.array([baseline_lookup[iid][2] for iid in image_ids])

    benign_idx = np.where(~true_malignant)[0]
    malignant_idx = np.where(true_malignant)[0]
    n_benign = len(benign_idx)
    print(f"Holdout: {len(image_ids)} total, {len(malignant_idx)} malignant, {n_benign} benign")
    print(f"Running paired bootstrap, {N_BOOT} resamples per prevalence level...\n")

    rng = np.random.RandomState(42)
    print(f"{'Prevalence':<12}{'Fusion precision':<28}{'Baseline precision':<28}{'Delta (paired)':<28}{'CI excludes 0?'}")

    results = {}
    for prevalence in PREVALENCES:
        n_mal_target = min(int(round(n_benign * prevalence / (1 - prevalence))), len(malignant_idx))

        fusion_vals, baseline_vals, delta_vals = [], [], []
        for _ in range(N_BOOT):
            b_idx = rng.choice(benign_idx, n_benign, replace=True)
            m_idx = rng.choice(malignant_idx, n_mal_target, replace=True)
            idx = np.concatenate([b_idx, m_idx])

            p_fusion = flagged_precision(idx, fusion_mel, fusion_pred_class, fusion_conf, ages, true_malignant)
            p_baseline = flagged_precision(idx, baseline_mel, baseline_pred_class, baseline_conf, ages, true_malignant)

            if np.isnan(p_fusion) or np.isnan(p_baseline):
                continue
            fusion_vals.append(p_fusion)
            baseline_vals.append(p_baseline)
            delta_vals.append(p_fusion - p_baseline)

        fusion_vals, baseline_vals, delta_vals = map(np.array, (fusion_vals, baseline_vals, delta_vals))
        n_valid = len(delta_vals)

        f_mean, f_lo, f_hi = fusion_vals.mean(), *np.percentile(fusion_vals, [2.5, 97.5])
        b_mean, b_lo, b_hi = baseline_vals.mean(), *np.percentile(baseline_vals, [2.5, 97.5])
        d_mean, d_lo, d_hi = delta_vals.mean(), *np.percentile(delta_vals, [2.5, 97.5])
        excludes_zero = (d_lo > 0) or (d_hi < 0)

        print(f"{int(prevalence*100)}%{'':<9}{100*f_mean:>5.1f}% [{100*f_lo:.1f},{100*f_hi:.1f}]{'':<8}"
              f"{100*b_mean:>5.1f}% [{100*b_lo:.1f},{100*b_hi:.1f}]{'':<8}"
              f"{100*d_mean:+.1f}pp [{100*d_lo:+.1f},{100*d_hi:+.1f}]{'':<6}"
              f"{'YES -- likely real' if excludes_zero else 'no -- not distinguishable from noise'}")

        results[prevalence] = {
            "n_valid_draws": n_valid,
            "fusion_precision_mean": float(f_mean), "fusion_precision_ci": [float(f_lo), float(f_hi)],
            "baseline_precision_mean": float(b_mean), "baseline_precision_ci": [float(b_lo), float(b_hi)],
            "delta_mean": float(d_mean), "delta_ci": [float(d_lo), float(d_hi)],
            "ci_excludes_zero": bool(excludes_zero),
            "single_draw_baseline_for_reference": BASELINE_FLAGGED_PRECISION[prevalence],
        }

    import json
    out_path = os.path.join(MODEL_DIR, "bootstrap_flagged_precision_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

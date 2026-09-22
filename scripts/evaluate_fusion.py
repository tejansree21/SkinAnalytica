"""
SkinAnalytica -- scripts/evaluate_fusion.py
Path B, workstream 1 task 5 -- see docs/RETRAIN_PLAN.md
"Workstream 1 -- implementation-ready scope".

Evaluates the trained GBDT fusion model (scripts/train_fusion_gbdt.py)
against the two external holdouts it never saw during training. This is
the real answer to whether Path B helped -- task 3's internal-validation
numbers were a sanity check only, not this.

Deliberately not just AUC, because finding #16 (the thing that activated
Path B in the first place) found a problem AUC alone doesn't show:

1. Discrimination: AUC on the SLICE-3D 143-case holdout, against the
   pre-fusion baseline of 0.6131 (docs/MODEL_CARD.md finding #12).
2. Flagged-precision at realistic prevalence on MILK10k, re-running
   finding #16's own resampling methodology (zero new leakage -- same
   held-out images, recombined at different ratios) with the fusion
   model's probabilities routed through the SAME make_verdict() logic
   (same MEL_THRESHOLD/AGE_BAND_THRESHOLDS as production) -- this
   deliberately does not recalibrate anything, so the comparison isolates
   the effect of swapping the probability source, matching how task 4's
   opt-in wiring would actually work if this ships.
3. Subgroup check: age-band and skin-tone-band sensitivity on MILK10k,
   against finding #14's baseline, to see whether feeding
   skin_tone_class as a real model input (not just a threshold-selector)
   narrows the spread finding #14 flagged.

Usage:
  python scripts/evaluate_fusion.py
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR = os.path.join(BASE, "outputs", "embeddings")
MODEL_DIR = os.path.join(BASE, "outputs", "fusion_model")

# Single source of truth: src/verdict_logic.py (docs/MODEL_CARD.md finding
# #29). This file used to hand-copy MEL_THRESHOLD/AGE_BAND_THRESHOLDS/etc
# from api/SA05_api.py and went silently stale within this same session
# (finding #27) when those changed but this copy didn't -- exactly the
# duplication risk src/inference_utils.py's module docstring warns about
# for preprocessing, now confirmed to apply to verdict logic too.
# get_mel_threshold/verdict re-exported unchanged so existing call sites
# and downstream importers (bootstrap_flagged_precision.py,
# evaluate_fusion_mra_midas.py) keep working without modification.
sys.path.insert(0, os.path.join(BASE, "src"))
from verdict_logic import (  # noqa: E402
    UNIFIED_CLASSES, MEL_IDX, CANCER_CLASSES, REVIEW_CLASSES,
    MEL_THRESHOLD, MEL_REVIEW_BAND_FACTOR, AGE_BAND_THRESHOLDS,
    SKIN_TONE_THRESHOLDS, get_mel_threshold, verdict,
)

# Baselines to compare against, from docs/MODEL_CARD.md findings #12, #14, #16.
BASELINE_SLICE3D_AUC = 0.6131
BASELINE_FLAGGED_PRECISION = {0.02: 0.042, 0.05: 0.080, 0.10: 0.159, 0.20: 0.337}
BASELINE_SKIN_TONE_SENS = {2: 0.769, 3: 0.643, 4: 0.556}  # sensitivity @ flat 0.119, finding #14
BASELINE_AGE_SENS = {"40-60": 0.696, "60+": 0.559}  # finding #14 addendum


def load_features(split: str) -> dict:
    d = np.load(os.path.join(EMB_DIR, f"{split}_fusion_features.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(set(yt.tolist())) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def evaluate_slice3d(booster, feature_names):
    print("=" * 70)
    print("1. DISCRIMINATION -- SLICE-3D 143-case holdout")
    print("=" * 70)
    d = load_features("slice3d_holdout")
    X, y = d["X"], d["y"]
    probs = booster.predict(X)
    mel_score = probs[:, MEL_IDX]
    y_true = (y == MEL_IDX).astype(int)  # matches how the pre-fusion 0.6131 baseline was computed

    auc = roc_auc_score(y_true, mel_score)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true, mel_score)
    stat, pval = mannwhitneyu(mel_score[y_true == 1], mel_score[y_true == 0], alternative="two-sided")
    # cross-check: Mann-Whitney U / (n_pos * n_neg) is an independent AUC estimate
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    auc_crosscheck = stat / (n_pos * n_neg)

    print(f"n = {len(y_true):,} ({n_pos} malignant, {n_neg} benign)")
    print(f"Fusion AUC:        {auc:.4f}  (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"Mann-Whitney cross-check AUC: {auc_crosscheck:.4f} (should match)")
    print(f"Baseline (pre-fusion, finding #12): {BASELINE_SLICE3D_AUC:.4f}")
    delta = auc - BASELINE_SLICE3D_AUC
    print(f"Delta: {delta:+.4f} {'(improvement)' if delta > 0 else '(regression)' if delta < 0 else '(no change)'}")
    print(f"CI vs baseline: {'excludes baseline -- likely real difference' if not (ci_lo <= BASELINE_SLICE3D_AUC <= ci_hi) else 'baseline falls within CI -- not distinguishable from no change'}")
    return {"auc": auc, "ci": [ci_lo, ci_hi], "mannwhitney_p": pval, "n": len(y_true), "n_malignant": int(n_pos)}


def evaluate_flagged_precision(booster, feature_names):
    print()
    print("=" * 70)
    print("2. FLAGGED-PRECISION AT REALISTIC PREVALENCE -- MILK10k holdout")
    print("=" * 70)
    d = load_features("milk10k_holdout")
    X, y, image_id = d["X"], d["y"], d["image_id"]
    age_col = list(feature_names).index("age_approx")
    ages = X[:, age_col]

    probs = booster.predict(X)
    mel_scores = probs[:, MEL_IDX]
    pred_classes = [UNIFIED_CLASSES[i] for i in probs.argmax(axis=1)]
    confidences = probs.max(axis=1)
    true_malignant = np.isin(y, [UNIFIED_CLASSES.index(c) for c in CANCER_CLASSES])

    rows = list(zip(mel_scores, pred_classes, confidences, ages, true_malignant))
    benign_rows = [r for r in rows if not r[4]]
    malignant_rows = [r for r in rows if r[4]]
    print(f"Holdout: {len(rows)} total, {len(malignant_rows)} malignant, {len(benign_rows)} benign\n")

    rng = np.random.RandomState(42)
    results = {}
    print(f"{'Prevalence':<12}{'n(mal)':<8}{'Auto-resolved':<16}{'Flagged-precision':<20}{'Baseline':<12}{'Delta'}")
    for prevalence in (0.02, 0.05, 0.10, 0.20):
        n_benign = len(benign_rows)
        n_mal = min(int(round(n_benign * prevalence / (1 - prevalence))), len(malignant_rows))
        sample_idx = rng.choice(len(malignant_rows), n_mal, replace=False)
        sample = benign_rows + [malignant_rows[i] for i in sample_idx]

        counts = {"CANCER_FLAGGED": 0, "REVIEW_REQUIRED": 0, "NORMAL": 0}
        tp = 0
        for mel_score, pred_class, confidence, age, is_malignant in sample:
            v = verdict(mel_score, pred_class, confidence, age)
            counts[v] += 1
            if v == "CANCER_FLAGGED" and is_malignant:
                tp += 1
        n = len(sample)
        auto = counts["CANCER_FLAGGED"] + counts["NORMAL"]
        flagged = counts["CANCER_FLAGGED"]
        precision = tp / flagged if flagged else float("nan")
        baseline = BASELINE_FLAGGED_PRECISION[prevalence]
        delta = precision - baseline
        results[prevalence] = {"n": n, "n_malignant": n_mal, "auto_resolved_pct": 100 * auto / n,
                                "flagged_precision": precision, "baseline": baseline, "delta": delta}
        print(f"{int(prevalence*100)}%{'':<9}{n_mal:<8}{100*auto/n:>5.1f}%{'':<9}{100*precision:>5.1f}%{'':<13}"
              f"{100*baseline:>5.1f}%{'':<5}{100*delta:+.1f}pp")

    return results


def evaluate_subgroups(booster, feature_names):
    print()
    print("=" * 70)
    print("3. SUBGROUP CHECK -- MILK10k skin-tone and age bands")
    print("=" * 70)
    d = load_features("milk10k_holdout")
    X, y = d["X"], d["y"]
    skin_tone_col = list(feature_names).index("skin_tone_class")
    age_col = list(feature_names).index("age_approx")
    skin_tones = X[:, skin_tone_col]
    ages = X[:, age_col]

    probs = booster.predict(X)
    mel_scores = probs[:, MEL_IDX]
    y_true_mel = (y == MEL_IDX).astype(int)

    # baseline AUC CIs from docs/MODEL_CARD.md finding #14, for a fair
    # "does this exceed the old CI" comparison, not just point estimates
    baseline_auc_ci = {2: (0.6816, 0.9805), 3: (0.8297, 0.9534), 4: (0.7329, 0.9417)}

    print("\nBy skin tone (bands 2/3/4 -- the only adequately-sampled ones per finding #14):")
    print(f"{'Tone':<6}{'n(mel)':<8}{'AUC':<10}{'95% CI':<20}{'Sens @ own thr.':<18}{'Baseline sens':<16}{'Delta'}")
    for tone, threshold in SKIN_TONE_THRESHOLDS.items():
        mask = skin_tones == tone
        n_mel = int(y_true_mel[mask].sum())
        if n_mel < 3:
            print(f"{tone:<6}{n_mel:<8}too few cases to trust")
            continue
        auc = roc_auc_score(y_true_mel[mask], mel_scores[mask])
        ci_lo, ci_hi = bootstrap_auc_ci(y_true_mel[mask], mel_scores[mask])
        old_lo, old_hi = baseline_auc_ci[tone]
        exceeds_old_ci = ci_lo > old_hi
        sens = ((mel_scores[mask] >= threshold) & (y_true_mel[mask] == 1)).sum() / n_mel
        baseline = BASELINE_SKIN_TONE_SENS[tone]
        flag = " *exceeds old CI*" if exceeds_old_ci else ""
        print(f"{tone:<6}{n_mel:<8}{auc:<10.4f}[{ci_lo:.3f},{ci_hi:.3f}]{'':<7}{100*sens:>5.1f}%{'':<11}"
              f"{100*baseline:>6.1f}%{'':<9}{100*(sens-baseline):+.1f}pp{flag}")

    print("\nBy age band:")
    age_bands = {"<40": (0, 40), "40-60": (40, 60), "60+": (60, 200)}
    print(f"{'Band':<10}{'n(mel)':<8}{'AUC':<10}{'Sens @ flat 0.119':<20}{'Baseline sens':<16}{'Delta'}")
    for label, (lo, hi) in age_bands.items():
        mask = (ages >= lo) & (ages < hi) & ~np.isnan(ages)
        n_mel = int(y_true_mel[mask].sum())
        if n_mel < 5:
            print(f"{label:<10}{n_mel:<8}too few cases to trust")
            continue
        auc = roc_auc_score(y_true_mel[mask], mel_scores[mask])
        sens = ((mel_scores[mask] >= 0.119) & (y_true_mel[mask] == 1)).sum() / n_mel
        baseline = BASELINE_AGE_SENS.get(label)
        baseline_str = f"{100*baseline:.1f}%" if baseline is not None else "n/a"
        delta_str = f"{100*(sens-baseline):+.1f}pp" if baseline is not None else ""
        print(f"{label:<10}{n_mel:<8}{auc:<10.4f}{100*sens:>6.1f}%{'':<13}{baseline_str:<16}{delta_str}")


def main():
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_fusion.txt"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), encoding="utf-8") as f:
        feature_names = json.load(f)

    slice3d_result = evaluate_slice3d(booster, feature_names)
    precision_result = evaluate_flagged_precision(booster, feature_names)
    evaluate_subgroups(booster, feature_names)

    out = {"slice3d_discrimination": slice3d_result, "flagged_precision_by_prevalence": precision_result}
    out_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

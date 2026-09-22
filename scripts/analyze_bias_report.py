"""
SkinAnalytica — analyze_bias_report.py
Threshold re-tuning, bootstrap confidence intervals, and age-conditional
threshold derivation from a scored ISIC-2020 sample. Run score_dataset.py
--dataset isic2020 first to produce the input CSV.

Usage:
    python analyze_bias_report.py --scored outputs/bias_reports/isic2020_scored.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
from verdict_logic import MEL_THRESHOLD  # noqa: E402 -- single source of truth, see finding #29
AGE_BINS = [0, 30, 45, 60, 75, 120]
AGE_LABELS = ["<30", "30-45", "45-60", "60-75", "75+"]


def threshold_tradeoff_table(y_true, mel_score):
    fpr, tpr, thresholds = roc_curve(y_true, mel_score)
    j = tpr - fpr
    best_j_idx = np.argmax(j)
    print(f"Youden-optimal threshold: {thresholds[best_j_idx]:.4f}  "
          f"sensitivity={tpr[best_j_idx]:.1%}  specificity={1-fpr[best_j_idx]:.1%}")
    cur_idx = np.argmin(np.abs(thresholds - MEL_THRESHOLD))
    print(f"Current deployed threshold ({MEL_THRESHOLD}): "
          f"sensitivity={tpr[cur_idx]:.1%}  specificity={1-fpr[cur_idx]:.1%}")
    print("\nTradeoff table:")
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, MEL_THRESHOLD, 0.35, 0.40, 0.50]:
        idx = np.argmin(np.abs(thresholds - t))
        print(f"  t={t:<6} sensitivity={tpr[idx]:.1%}  specificity={1-fpr[idx]:.1%}")


def bootstrap_ci(y_true, mel_score, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_auc, boot_sens = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, mb = y_true[idx], mel_score[idx]
        if yb.sum() < 2 or (1 - yb).sum() < 2:
            continue
        boot_auc.append(roc_auc_score(yb, mb))
        mel_rows = mb[yb == 1]
        boot_sens.append((mel_rows >= MEL_THRESHOLD).mean())
    boot_auc, boot_sens = np.array(boot_auc), np.array(boot_sens)
    print(f"MelAUC: {boot_auc.mean():.4f}  95% CI [{np.percentile(boot_auc,2.5):.4f}, {np.percentile(boot_auc,97.5):.4f}]")
    print(f"Sensitivity@{MEL_THRESHOLD}: {boot_sens.mean():.4f}  "
          f"95% CI [{np.percentile(boot_sens,2.5):.4f}, {np.percentile(boot_sens,97.5):.4f}]")


def age_conditional_thresholds(df, target_sensitivity=0.80):
    thresholds = {}
    print(f"{'age_group':<10} {'n_mel':>6} {'current_sens':>14} {'thresh_for_target':>20} {'spec_at_t':>12}")
    for ag in AGE_LABELS:
        sub = df[df["age_group"] == ag]
        y = (sub["label"] == "mel").astype(int).values
        m = sub["mel_score"].values
        n_mel = int(y.sum())
        if n_mel < 5 or (1 - y).sum() < 5:
            print(f"{ag:<10} {n_mel:>6}  insufficient data")
            continue
        cur_sens = (m[y == 1] >= MEL_THRESHOLD).mean()
        fpr, tpr, th = roc_curve(y, m)
        cand = np.where(tpr >= target_sensitivity)[0]
        if len(cand):
            t = float(th[cand[0]])
            spec = 1 - fpr[cand[0]]
            thresholds[ag] = round(t, 4)
            print(f"{ag:<10} {n_mel:>6} {cur_sens:>13.1%} {t:>19.4f} {spec:>11.1%}")
    return thresholds


def main(scored_csv: str, output: str):
    df = pd.read_csv(scored_csv)
    gt_path = os.path.join(BASE, "data", "ISIC_2020", "ISIC_2020_Training_GroundTruth_v2.csv")
    if os.path.exists(gt_path) and "age_approx" not in df.columns:
        gt = pd.read_csv(gt_path).rename(columns={"image_name": "image_id"})[["image_id", "age_approx"]]
        df = df.merge(gt, on="image_id", how="left")
    df["age_group"] = pd.cut(df["age_approx"], bins=AGE_BINS, labels=AGE_LABELS)

    y_true = (df["label"] == "mel").astype(int).values
    mel_score = df["mel_score"].values
    print(f"n={len(df)}  n_mel={y_true.sum()}\n")

    print("=" * 60)
    print("Threshold re-tuning")
    print("=" * 60)
    threshold_tradeoff_table(y_true, mel_score)

    print("\n" + "=" * 60)
    print("Bootstrap confidence intervals")
    print("=" * 60)
    bootstrap_ci(y_true, mel_score)

    print("\n" + "=" * 60)
    print("Age-conditional thresholds")
    print("=" * 60)
    age_thresholds = age_conditional_thresholds(df)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump({"age_conditional_thresholds_80pct_sensitivity": age_thresholds}, f, indent=2)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True, help="Path to a CSV produced by score_dataset.py --dataset isic2020")
    p.add_argument("--output", default="outputs/bias_reports/threshold_ci_age_analysis.json")
    args = p.parse_args()
    main(args.scored, args.output)

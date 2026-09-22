"""
SkinAnalytica -- scripts/train_fusion_gbdt.py
Path B, workstream 1 task 3 -- see docs/RETRAIN_PLAN.md
"Workstream 1 -- implementation-ready scope".

Trains a LightGBM multiclass model on the fused (CNN embedding + metadata)
feature table from scripts/join_fusion_features.py, same 7-class target as
the CNN ensemble.

**Leakage discipline, non-negotiable**: before touching LightGBM at all,
this asserts zero image_id overlap between the training features and
BOTH external holdouts (SLICE-3D's 143-case holdout, MILK10k's 700-image
holdout) reserved for task 5's evaluation. This mirrors the assertions
already in scripts/dataset_loaders.py (e.g.
load_slice3d_full_holdout()'s "assert not (clean_mal & trained_on)") and
exists for the same reason: this project has already shipped a real
leakage bug once (the SA01/SA02 validation-split mismatch, see
docs/RETRAIN_PLAN.md) from skipping exactly this kind of check because a
stage "was just" a smaller/simpler piece of the pipeline. A GBDT fusion
stage is not exempt from that discipline just because it's new.

A separate INTERNAL validation split (stratified, carved out of the
training pool itself, not from either external holdout) is used for
early stopping only -- it never touches task 5's evaluation, and is not a
substitute for it. Reported metrics from this script are a sanity check
that training worked, not the real answer to whether Path B helped;
task 5 answers that.

Usage:
  python scripts/train_fusion_gbdt.py
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR = os.path.join(BASE, "outputs", "embeddings")
MODEL_DIR = os.path.join(BASE, "outputs", "fusion_model")

UNIFIED_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
MEL_IDX = UNIFIED_CLASSES.index("mel")


def load_features(split: str) -> dict:
    path = os.path.join(EMB_DIR, f"{split}_fusion_features.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run scripts/join_fusion_features.py --split {split} first")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def assert_no_leakage(train: dict, slice3d_holdout: dict, milk10k_holdout: dict):
    train_ids = set(train["image_id"].tolist())
    slice3d_ids = set(slice3d_holdout["image_id"].tolist())
    milk10k_ids = set(milk10k_holdout["image_id"].tolist())

    slice3d_overlap = train_ids & slice3d_ids
    milk10k_overlap = train_ids & milk10k_ids

    assert not slice3d_overlap, (
        f"LEAKAGE: {len(slice3d_overlap)} image_ids appear in both the training "
        f"features and the SLICE-3D holdout -- refusing to train. Examples: "
        f"{list(slice3d_overlap)[:5]}"
    )
    assert not milk10k_overlap, (
        f"LEAKAGE: {len(milk10k_overlap)} image_ids appear in both the training "
        f"features and the MILK10k holdout -- refusing to train. Examples: "
        f"{list(milk10k_overlap)[:5]}"
    )
    print(f"Leakage check passed: 0 overlap with SLICE-3D holdout ({len(slice3d_ids):,} ids), "
          f"0 overlap with MILK10k holdout ({len(milk10k_ids):,} ids)")


def find_categorical_indices(feature_names: np.ndarray) -> list:
    # "site" is the only column whose integer code is a real (unordered)
    # category rather than a number with magnitude -- see
    # join_fusion_features.py's docstring for why it's not semantically
    # unified across sources. Everything else (age, sex-as-0/1,
    # skin_tone_class, tbp_confidence, and every *_missing flag) is
    # treated as numeric/ordinal, which LightGBM handles fine including
    # via native NaN splits.
    return [i for i, n in enumerate(feature_names) if n == "site"]


def main():
    print("Loading fused feature tables...")
    train = load_features("train")
    slice3d_holdout = load_features("slice3d_holdout")
    milk10k_holdout = load_features("milk10k_holdout")

    assert_no_leakage(train, slice3d_holdout, milk10k_holdout)

    X, y = train["X"], train["y"]
    feature_names = train["feature_names"]
    print(f"\nTraining pool: {X.shape[0]:,} rows x {X.shape[1]:,} features")

    counts = {UNIFIED_CLASSES[i]: int((y == i).sum()) for i in range(7)}
    print(f"Class distribution: {counts}")

    cat_idx = find_categorical_indices(feature_names)
    print(f"Categorical feature indices: {cat_idx} ({feature_names[cat_idx].tolist() if cat_idx else 'none'})")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    print(f"\nInternal split for early stopping only (NOT the real evaluation -- "
          f"task 5 uses the external holdouts for that): "
          f"{X_train.shape[0]:,} train / {X_val.shape[0]:,} internal-val")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=list(feature_names),
                             categorical_feature=cat_idx, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=list(feature_names),
                           categorical_feature=cat_idx, reference=train_set, free_raw_data=False)

    params = {
        "objective": "multiclass",
        "num_class": 7,
        "metric": "multi_logloss",
        "class_weight": "balanced",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
        "verbosity": -1,
    }

    print(f"\nTraining LightGBM (params: {params})...")
    t0 = time.time()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        valid_names=["internal_val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
    )
    elapsed = time.time() - t0
    print(f"\nTrained in {elapsed:.1f}s, best iteration: {booster.best_iteration}")

    val_probs = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_pred = val_probs.argmax(axis=1)
    val_logloss = log_loss(y_val, val_probs, labels=list(range(7)))
    val_acc = accuracy_score(y_val, val_pred)
    val_mel_auc = roc_auc_score((y_val == MEL_IDX).astype(int), val_probs[:, MEL_IDX])
    print(f"\nInternal validation (sanity check only, not the real evaluation):")
    print(f"  multi_logloss: {val_logloss:.4f}")
    print(f"  accuracy: {val_acc:.4f}")
    print(f"  MelAUC: {val_mel_auc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lgbm_fusion.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    manifest = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train_pool": int(X.shape[0]),
        "n_train_split": int(X_train.shape[0]),
        "n_internal_val": int(X_val.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": counts,
        "categorical_feature_indices": cat_idx,
        "params": params,
        "best_iteration": int(booster.best_iteration),
        "internal_val_multi_logloss": float(val_logloss),
        "internal_val_accuracy": float(val_acc),
        "internal_val_mel_auc": float(val_mel_auc),
        "unified_classes": UNIFIED_CLASSES,
        "leakage_check": "passed -- 0 overlap with slice3d_holdout and milk10k_holdout image_ids",
    }
    manifest_path = os.path.join(MODEL_DIR, "lgbm_fusion_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    feature_names_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names.tolist(), f)

    print(f"\nSaved model: {model_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved feature names: {feature_names_path}")
    print(f"\nNOT evaluated against the real external holdouts yet -- that's task 5, "
          f"not this script. Internal-val numbers above are a training sanity check only.")


if __name__ == "__main__":
    main()

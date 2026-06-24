"""
SkinAnalytica — 00_data_loader.py
===================================
Loads and merges ISIC 2018, 2019, 2020 datasets.
- Patient-level train/val split (no data leakage)
- Handles 7-class (2018), 8-class (2019), binary (2020)
- Merges all into unified 7-class format
- Outputs: train.csv, val.csv, class_weights.json
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica"
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "config")
os.makedirs(OUT, exist_ok=True)

# ── Class Maps ────────────────────────────────────────────────────────────────
# Unified 7-class label set (ISIC 2018 standard)
UNIFIED_CLASSES = [
    "MEL",   # Melanoma
    "NV",    # Melanocytic Nevus
    "BCC",   # Basal Cell Carcinoma
    "AKIEC", # Actinic Keratosis
    "BKL",   # Benign Keratosis
    "DF",    # Dermatofibroma
    "VASC",  # Vascular Lesion
]

CLASS_TO_IDX = {c: i for i, c in enumerate(UNIFIED_CLASSES)}

# ISIC 2018 column → unified label
ISIC2018_MAP = {
    "MEL": "MEL", "NV": "NV", "BCC": "BCC",
    "AKIEC": "AKIEC", "BKL": "BKL", "DF": "DF", "VASC": "VASC"
}

# ISIC 2019 column → unified label (SCC → AKIEC, UNK dropped)
ISIC2019_MAP = {
    "MEL": "MEL", "NV": "NV", "BCC": "BCC",
    "AK": "AKIEC", "BKL": "BKL", "DF": "DF",
    "VASC": "VASC", "SCC": "AKIEC",  # merge SCC into AKIEC
}

# ISIC 2020: only binary (MEL vs benign) → use as MEL/NV
ISIC2020_MAP = {
    "melanoma": "MEL",
    "nevus": "NV",
    "seborrheic_keratosis": "BKL",
    "lentigo_NOS": "BKL",
    "lichenoid_keratosis": "BKL",
    "solar_lentigo": "BKL",
    "cafe_au_lait_macule": "BKL",
    "atypical_melanocytic_proliferation": "MEL",
    "unknown": None,  # drop
}

# ── ISIC 2018 Loader ──────────────────────────────────────────────────────────
def load_isic2018():
    print("Loading ISIC 2018...")

    img_dir = Path(DATA) / "ISIC_2018" / "ISIC2018_Task3_Training_Input" / "ISIC2018_Task3_Training_Input"
    gt_path = list((Path(DATA) / "ISIC_2018" / "ISIC2018_Task3_Training_GroundTruth").rglob("*.csv"))

    if not gt_path:
        raise FileNotFoundError("ISIC 2018 ground truth CSV not found")

    gt = pd.read_csv(gt_path[0])
    print(f"  GT shape: {gt.shape}")
    print(f"  Columns: {gt.columns.tolist()}")

    # One-hot → single label
    label_cols = [c for c in gt.columns if c != "image"]
    gt["label"] = gt[label_cols].idxmax(axis=1)
    gt["label"] = gt["label"].map(ISIC2018_MAP)

    # Build records
    records = []
    for _, row in gt.iterrows():
        img_name = row["image"] + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        if pd.isna(row["label"]):
            continue
        records.append({
            "image_id"  : row["image"],
            "image_path": str(img_path),
            "label"     : row["label"],
            "label_idx" : CLASS_TO_IDX[row["label"]],
            "patient_id": row["image"],  # 2018 has no patient ID — use image as proxy
            "dataset"   : "ISIC_2018",
            "source"    : "HAM10000",
        })

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df)} images")
    print(f"  Label dist: {dict(df['label'].value_counts())}")
    return df


# ── ISIC 2019 Loader ──────────────────────────────────────────────────────────
def load_isic2019():
    print("Loading ISIC 2019...")

    img_dir = Path(DATA) / "ISIC_2019" / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
    gt_path  = Path(DATA) / "ISIC_2019" / "ISIC_2019_Training_GroundTruth.csv"
    meta_path= Path(DATA) / "ISIC_2019" / "ISIC_2019_Training_Metadata.csv"

    if not gt_path.exists():
        raise FileNotFoundError(f"ISIC 2019 GT not found: {gt_path}")

    gt   = pd.read_csv(gt_path)
    meta = pd.read_csv(meta_path) if meta_path.exists() else None

    print(f"  GT shape: {gt.shape}")
    print(f"  Columns: {gt.columns.tolist()}")

    # One-hot → single label
    label_cols = [c for c in gt.columns if c != "image" and c != "UNK"]
    gt["label"] = gt[label_cols].idxmax(axis=1)
    gt["label"] = gt["label"].map(ISIC2019_MAP)

    # Drop UNK
    gt = gt[gt["label"].notna()]

    # Merge metadata if available
    if meta is not None:
        gt = gt.merge(meta, on="image", how="left")

    records = []
    for _, row in gt.iterrows():
        img_name = row["image"] + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        # Extract patient ID from metadata if available
        patient_id = str(row.get("lesion_id", row["image"]))

        records.append({
            "image_id"  : row["image"],
            "image_path": str(img_path),
            "label"     : row["label"],
            "label_idx" : CLASS_TO_IDX[row["label"]],
            "patient_id": patient_id,
            "dataset"   : "ISIC_2019",
            "source"    : "BCN+HAM+MSK",
            "age"       : row.get("age_approx", None),
            "sex"       : row.get("sex", None),
            "site"      : row.get("anatom_site_general", None),
        })

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df)} images")
    print(f"  Label dist: {dict(df['label'].value_counts())}")
    return df


# ── ISIC 2020 Loader ──────────────────────────────────────────────────────────
def load_isic2020():
    print("Loading ISIC 2020...")

    img_dir  = Path(DATA) / "ISIC_2020" / "ISIC_2020_Training_JPEG" / "train"
    gt_v2    = Path(DATA) / "ISIC_2020" / "ISIC_2020_Training_GroundTruth_v2.csv"
    gt_v1    = Path(DATA) / "ISIC_2020" / "ISIC_2020_Training_GroundTruth.csv"
    gt_path  = gt_v2 if gt_v2.exists() else gt_v1

    if not gt_path.exists():
        raise FileNotFoundError(f"ISIC 2020 GT not found")

    gt = pd.read_csv(gt_path)
    print(f"  GT shape: {gt.shape}")
    print(f"  Columns: {gt.columns.tolist()}")

    records = []
    for _, row in gt.iterrows():
        img_name = row["image_name"] + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        # Map diagnosis → unified label
        diagnosis = str(row.get("diagnosis", "")).lower().replace(" ", "_")
        target    = int(row.get("target", -1))

        if "diagnosis" in gt.columns and diagnosis in ISIC2020_MAP:
            label = ISIC2020_MAP[diagnosis]
        elif target == 1:
            label = "MEL"
        elif target == 0:
            label = "NV"
        else:
            continue

        if label is None:
            continue

        records.append({
            "image_id"  : row["image_name"],
            "image_path": str(img_path),
            "label"     : label,
            "label_idx" : CLASS_TO_IDX[label],
            "patient_id": str(row.get("patient_id", row["image_name"])),
            "dataset"   : "ISIC_2020",
            "source"    : "ISIC_2020",
            "age"       : row.get("age_approx", None),
            "sex"       : row.get("sex", None),
            "site"      : row.get("anatom_site_general_challenge", None),
        })

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df)} images")
    print(f"  Label dist: {dict(df['label'].value_counts())}")
    return df


# ── Merge & Split ─────────────────────────────────────────────────────────────
def merge_and_split(df2018, df2019, df2020, val_size=0.2, seed=42):
    print("\nMerging datasets...")

    # Align columns
    cols = ["image_id", "image_path", "label", "label_idx",
            "patient_id", "dataset", "source", "age", "sex", "site"]

    for df in [df2018, df2019, df2020]:
        for c in cols:
            if c not in df.columns:
                df[c] = None

    merged = pd.concat([df2018[cols], df2019[cols], df2020[cols]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id"])
    print(f"Total merged: {len(merged)} images")
    print(f"Label distribution:\n{merged['label'].value_counts()}")

    # ── Patient-level stratified split ────────────────────────────────────────
    # Group by patient_id so same patient never appears in both train and val
    print(f"\nSplitting {(1-val_size)*100:.0f}/{val_size*100:.0f} (patient-level)...")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    groups = merged["patient_id"].values
    X      = merged.index.values

    train_idx, val_idx = next(gss.split(X, groups=groups))
    train_df = merged.iloc[train_idx].copy()
    val_df   = merged.iloc[val_idx].copy()

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Train label dist:\n{train_df['label'].value_counts()}")
    print(f"Val label dist:\n{val_df['label'].value_counts()}")

    # Verify no patient overlap
    train_patients = set(train_df["patient_id"].unique())
    val_patients   = set(val_df["patient_id"].unique())
    overlap        = train_patients & val_patients
    print(f"\nPatient overlap (should be 0): {len(overlap)}")

    return train_df, val_df


# ── Class Weights ─────────────────────────────────────────────────────────────
def compute_class_weights(train_df):
    """Inverse frequency weighting for imbalanced classes."""
    counts  = Counter(train_df["label"])
    total   = sum(counts.values())
    weights = {cls: total / (len(UNIFIED_CLASSES) * cnt) for cls, cnt in counts.items()}

    # Normalize
    max_w = max(weights.values())
    weights = {cls: round(w / max_w, 4) for cls, w in weights.items()}

    print(f"\nClass weights (normalized):")
    for cls, w in weights.items():
        print(f"  {cls:<8} {w:.4f}  (n={counts.get(cls, 0)})")

    return weights


# ── Save Outputs ──────────────────────────────────────────────────────────────
def save_outputs(train_df, val_df, class_weights):
    train_path   = os.path.join(OUT, "train.csv")
    val_path     = os.path.join(OUT, "val.csv")
    weights_path = os.path.join(OUT, "class_weights.json")
    classes_path = os.path.join(OUT, "class_map.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)

    class_map = {
        "ISIC_2018": UNIFIED_CLASSES,
        "ISIC_2019": UNIFIED_CLASSES + ["SCC→AKIEC"],
        "ISIC_2020": ["MEL", "NV (binary)"],
        "unified"  : UNIFIED_CLASSES,
        "idx_to_class": {i: c for i, c in enumerate(UNIFIED_CLASSES)},
        "class_to_idx": CLASS_TO_IDX,
    }
    with open(classes_path, "w") as f:
        json.dump(class_map, f, indent=2)

    thresholds = {
        "ambiguity_flag"   : 2.0,
        "conflict_score"   : 7.0,
        "confidence_low"   : 4.0,
        "confidence_high"  : 7.0,
        "flag_top_two_gap" : 2.0
    }
    with open(os.path.join(OUT, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n✅ Saved:")
    print(f"  train.csv        → {len(train_df)} rows")
    print(f"  val.csv          → {len(val_df)} rows")
    print(f"  class_weights.json")
    print(f"  class_map.json")
    print(f"  thresholds.json")


# ── Dataset Stats ─────────────────────────────────────────────────────────────
def print_stats(train_df, val_df):
    print("\n" + "="*55)
    print("SKINANALYTICA DATASET SUMMARY")
    print("="*55)
    print(f"Total images  : {len(train_df) + len(val_df)}")
    print(f"Train         : {len(train_df)}")
    print(f"Val           : {len(val_df)}")
    print(f"Classes       : {len(UNIFIED_CLASSES)}")
    print(f"\nBy dataset:")
    for ds in ["ISIC_2018", "ISIC_2019", "ISIC_2020"]:
        n = len(train_df[train_df["dataset"] == ds]) + len(val_df[val_df["dataset"] == ds])
        print(f"  {ds:<15} {n}")
    print(f"\nBy class (train):")
    for cls in UNIFIED_CLASSES:
        n = len(train_df[train_df["label"] == cls])
        print(f"  {cls:<8} {n}")
    print("="*55)

    # Check metadata coverage
    meta_cols = ["age", "sex", "site"]
    all_df = pd.concat([train_df, val_df])
    print(f"\nMetadata coverage:")
    for col in meta_cols:
        if col in all_df.columns:
            pct = all_df[col].notna().mean() * 100
            print(f"  {col:<6} {pct:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SkinAnalytica — Data Loader")
    print("="*55)

    # Load each dataset
    df2018 = load_isic2018()
    df2019 = load_isic2019()
    df2020 = load_isic2020()

    # Merge and split
    train_df, val_df = merge_and_split(df2018, df2019, df2020, val_size=0.2)

    # Compute class weights
    class_weights = compute_class_weights(train_df)

    # Save outputs
    save_outputs(train_df, val_df, class_weights)

    # Print summary
    print_stats(train_df, val_df)

    print("\n✅ Data loader complete. Ready for training.")
    print(f"   Next: Run 01_efficientnet_train.py")

"""
SkinAnalytica -- scripts/build_training_pool.py
Ported verbatim from notebooks/SA01_EfficientNet_Train.ipynb's Cell 6 (the
notebook actually used for the real retrain -- see docs/RETRAIN_PLAN.md).

Why this exists as a script and not just notebook cells: this exact logic
already diverged once between SA01 and SA02 (two independent
implementations of "the same" split, silently disagreeing -- see
docs/RETRAIN_PLAN.md's "Found and fixed while training was running" note)
and caused a real risk of validation leakage. Path B's embedding-extraction
script (scripts/extract_embeddings.py) needs the exact training-pool image
list to avoid re-deriving it a third time. This module is the single
source of truth going forward; the notebook's own Cell 6 should eventually
import from here instead of carrying its own copy, but that retrofit is
out of scope for this change.

Returns the same 5-tuple format as the notebook: (path, class_idx, source,
image_id, age_or_None).
"""
import os

import pandas as pd

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
OUT_ROOT = os.path.join(BASE, "outputs")

UNIFIED_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_IDX = {c: i for i, c in enumerate(UNIFIED_CLASSES)}

ISIC18_MAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc"}
ISIC19_MAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl", "DF": "df", "SCC": "akiec", "VASC": "vasc"}
ISIC20_MAP = {
    "melanoma": "mel", "nevus": "nv", "seborrheic_keratosis": "bkl", "lentigo_NOS": "bkl",
    "lichenoid_keratosis": "bkl", "solar_lentigo": "bkl", "cafe_au_lait_macule": "bkl",
    "atypical_melanocytic_proliferation": "mel",
}
MILK10K_MAP = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
    # BEN_OTH, INF, MAL_OTH, SCCKA have no clean home in the 7-class scheme
    # and are dropped rather than guessed -- see docs/KNOWN_GAPS.md.
}

ISIC18_IMG = os.path.join(DATA, "ISIC_2018", "ISIC2018_Task3_Training_Input", "ISIC2018_Task3_Training_Input")
ISIC18_GT = os.path.join(DATA, "ISIC_2018", "ISIC2018_Task3_Training_GroundTruth", "ISIC2018_Task3_Training_GroundTruth", "ISIC2018_Task3_Training_GroundTruth.csv")

ISIC19_IMG = os.path.join(DATA, "ISIC_2019", "ISIC_2019_Training_Input", "ISIC_2019_Training_Input")
ISIC19_GT = os.path.join(DATA, "ISIC_2019", "ISIC_2019_Training_GroundTruth.csv")
ISIC19_META = os.path.join(DATA, "ISIC_2019", "ISIC_2019_Training_Metadata.csv")

ISIC20_IMG = os.path.join(DATA, "ISIC_2020", "ISIC_2020_Training_JPEG", "train")
ISIC20_GT = os.path.join(DATA, "ISIC_2020", "ISIC_2020_Training_GroundTruth_v2.csv")

SLICE3D_IMG = os.path.join(DATA, "ISIC_2024", "images")
SLICE3D_GT = os.path.join(DATA, "ISIC_2024", "ground_truth.csv")
SLICE3D_SUPP = os.path.join(DATA, "ISIC_2024", "supplemental_metadata.csv")
SLICE3D_HOLDOUT_CSV = os.path.join(OUT_ROOT, "slice3d_external_holdout_ids.csv")

EMB_LABELS_CSV = os.path.join(DATA, "EMB", "_repo", "early_melanoma_benchmark_dataset_labels.csv")
EMB_IMG = os.path.join(DATA, "EMB", "images")
EMB_HOLDOUT_CSV = os.path.join(OUT_ROOT, "emb_external_holdout_ids.csv")

MILK10K_META = os.path.join(DATA, "milk10k", "MILK10k_Training_Metadata.csv")
MILK10K_GT = os.path.join(DATA, "milk10k", "MILK10k_Training_GroundTruth.csv")
MILK10K_IMG = os.path.join(DATA, "milk10k", "images", "MILK10k_Training_Input")
MILK10K_HOLDOUT_CSV = os.path.join(OUT_ROOT, "milk10k_external_holdout_ids.csv")


def build_sample_list(verbose: bool = True) -> list:
    samples = []

    if os.path.exists(ISIC18_GT) and os.path.exists(ISIC18_IMG):
        df18 = pd.read_csv(ISIC18_GT)
        ccols = [c for c in df18.columns if c in ISIC18_MAP]
        n_before = len(samples)
        for _, row in df18.iterrows():
            iid = str(row["image"])
            label = None
            for col in ccols:
                if row[col] == 1.0:
                    label = ISIC18_MAP.get(col)
                    break
            if not label:
                continue
            path = os.path.join(ISIC18_IMG, iid + ".jpg")
            if os.path.exists(path):
                samples.append((path, CLASS_IDX[label], "isic18", iid, None))
        if verbose:
            print(f"ISIC 2018: {len(samples)-n_before:,} samples (age: unavailable)")
    elif verbose:
        print(f"ISIC 2018 not found at {ISIC18_GT}")

    if os.path.exists(ISIC19_GT) and os.path.exists(ISIC19_IMG):
        df19 = pd.read_csv(ISIC19_GT)
        ccols = [c for c in df19.columns if c.upper() in ISIC19_MAP]
        age_lookup = {}
        if os.path.exists(ISIC19_META):
            meta19 = pd.read_csv(ISIC19_META)
            age_lookup = dict(zip(meta19["image"].astype(str), meta19["age_approx"]))
        elif verbose:
            print(f"  WARNING: ISIC19 metadata not found at {ISIC19_META} -- age will be None for all ISIC19 samples")
        n_before = len(samples)
        for _, row in df19.iterrows():
            iid = str(row.get("image", row.iloc[0]))
            label = None
            for col in ccols:
                if row[col] == 1.0:
                    label = ISIC19_MAP.get(col.upper())
                    break
            if not label:
                continue
            path = os.path.join(ISIC19_IMG, iid + ".jpg")
            if os.path.exists(path):
                age = age_lookup.get(iid)
                age = None if (age is None or pd.isna(age)) else float(age)
                samples.append((path, CLASS_IDX[label], "isic19", iid, age))
        if verbose:
            print(f"ISIC 2019: {len(samples)-n_before:,} samples")
    elif verbose:
        print(f"ISIC 2019 not found at {ISIC19_GT}")

    if os.path.exists(ISIC20_GT) and os.path.exists(ISIC20_IMG):
        df20 = pd.read_csv(ISIC20_GT)
        n_before = len(samples)
        for _, row in df20.iterrows():
            iid = str(row.get("image_name", row.iloc[0]))
            diagnosis = str(row.get("diagnosis", "")).lower().replace(" ", "_")
            target = int(row.get("target", -1))
            if diagnosis in ISIC20_MAP:
                label = ISIC20_MAP[diagnosis]
            elif target == 1:
                label = "mel"
            elif target == 0:
                label = "nv"
            else:
                continue
            path = os.path.join(ISIC20_IMG, iid + ".jpg")
            if os.path.exists(path):
                age = row.get("age_approx")
                age = None if (age is None or pd.isna(age)) else float(age)
                samples.append((path, CLASS_IDX[label], "isic20", iid, age))
        if verbose:
            print(f"ISIC 2020: {len(samples)-n_before:,} samples")
    elif verbose:
        print(f"ISIC 2020 not found at {ISIC20_GT}")

    if verbose:
        print(f"Total (ISIC 2018-2020): {len(samples):,} samples")
    return samples


def build_slice3d_samples(holdout_frac: float = 0.15, seed: int = 42, verbose: bool = True) -> list:
    if not (os.path.exists(SLICE3D_GT) and os.path.exists(SLICE3D_SUPP) and os.path.exists(SLICE3D_IMG)):
        if verbose:
            print(f"SLICE-3D Permissive not found -- skipping (expected at {SLICE3D_GT})")
        return []

    gt = pd.read_csv(SLICE3D_GT)
    supp = pd.read_csv(SLICE3D_SUPP)[["isic_id", "attribution", "iddx_1", "iddx_full", "mel_thick_mm"]]
    df = gt.merge(supp, on="isic_id", how="left")
    df = df[df["malignant"] == 1.0].copy()

    def map_label(iddx_full):
        s = str(iddx_full).lower()
        if "melanoma" in s:
            return "mel"
        if "basal cell carcinoma" in s:
            return "bcc"
        if "squamous cell carcinoma" in s:
            return "akiec"
        return None

    df["label"] = df["iddx_full"].apply(map_label)
    df = df[df["label"].notna()].copy()

    def severity_bucket(row):
        s = str(row["iddx_full"]).lower()
        if "in situ" in s:
            return "in_situ"
        if pd.notna(row["mel_thick_mm"]) and row["mel_thick_mm"] < 1.0:
            return "thin_invasive"
        return "other"

    df["severity"] = df.apply(severity_bucket, axis=1)
    df["strata"] = df["attribution"].astype(str) + "|" + df["severity"].astype(str)

    holdout_ids = set()
    for strata_val, group in df.groupby("strata"):
        n_hold = max(1, int(round(len(group) * holdout_frac))) if len(group) >= 3 else 0
        if n_hold:
            holdout_ids.update(group.sample(n_hold, random_state=seed)["isic_id"].tolist())

    df["path"] = df["isic_id"].apply(lambda x: os.path.join(SLICE3D_IMG, x + ".jpg"))
    df = df[df["path"].apply(os.path.exists)]

    holdout_df = df[df["isic_id"].isin(holdout_ids)]
    train_df = df[~df["isic_id"].isin(holdout_ids)]

    if verbose:
        print(f"SLICE-3D Permissive: {len(train_df):,} malignant samples for training "
              f"({len(holdout_df):,} held out, matches {SLICE3D_HOLDOUT_CSV})")

    return [
        (row["path"], CLASS_IDX[row["label"]], "slice3d", row["isic_id"], None)
        for _, row in train_df.iterrows()
    ]


def build_emb_samples(holdout_frac: float = 0.15, seed: int = 42, verbose: bool = True) -> list:
    if not (os.path.exists(EMB_LABELS_CSV) and os.path.exists(EMB_IMG)):
        if verbose:
            print(f"EMB not found -- skipping (expected at {EMB_LABELS_CSV})")
        return []

    df = pd.read_csv(EMB_LABELS_CSV)
    df = df[df["source"] == "ISIC"].copy()

    def severity_bucket(row):
        if row["stage_ajcc"] == 0:
            return "in_situ"
        if pd.notna(row["thickness"]) and row["thickness"] < 1.0:
            return "thin_invasive"
        return "other"

    df["severity"] = df.apply(severity_bucket, axis=1)
    df["path"] = df["image"].apply(lambda x: os.path.join(EMB_IMG, x + ".jpg"))
    df = df[df["path"].apply(os.path.exists)]

    holdout_ids = set()
    for sev, group in df.groupby("severity"):
        n_hold = max(1, int(round(len(group) * holdout_frac))) if len(group) >= 3 else 0
        if n_hold:
            holdout_ids.update(group.sample(n_hold, random_state=seed)["image"].tolist())

    holdout_df = df[df["image"].isin(holdout_ids)]
    train_df = df[~df["image"].isin(holdout_ids)]

    if verbose:
        print(f"EMB: {len(train_df):,} thin/early-stage melanoma samples for training "
              f"({len(holdout_df):,} held out, matches {EMB_HOLDOUT_CSV})")

    return [
        (row["path"], CLASS_IDX["mel"], "emb", row["image"], None)
        for _, row in train_df.iterrows()
    ]


def build_milk10k_samples(holdout_frac: float = 0.15, seed: int = 42, verbose: bool = True) -> list:
    if not (os.path.exists(MILK10K_META) and os.path.exists(MILK10K_GT) and os.path.exists(MILK10K_IMG)):
        if verbose:
            print(f"MILK10k not found -- skipping (expected at {MILK10K_META})")
        return []

    meta = pd.read_csv(MILK10K_META)
    gt = pd.read_csv(MILK10K_GT)
    meta = meta[meta["image_type"] == "dermoscopic"].copy()
    df = meta.merge(gt, on="lesion_id", how="inner")

    mapped_cols = list(MILK10K_MAP.keys())

    def row_label(row):
        for col in mapped_cols:
            if row.get(col) == 1.0:
                return MILK10K_MAP[col]
        return None

    df["label"] = df.apply(row_label, axis=1)
    df = df[df["label"].notna()].copy()

    df["path"] = df.apply(lambda r: os.path.join(MILK10K_IMG, r["lesion_id"], r["isic_id"] + ".jpg"), axis=1)
    df = df[df["path"].apply(os.path.exists)]

    holdout_ids = set()
    for tone, group in df.groupby("skin_tone_class"):
        n_hold = max(1, int(round(len(group) * holdout_frac))) if len(group) >= 3 else 0
        if n_hold:
            holdout_ids.update(group.sample(n_hold, random_state=seed)["isic_id"].tolist())

    holdout_df = df[df["isic_id"].isin(holdout_ids)]
    train_df = df[~df["isic_id"].isin(holdout_ids)]

    if verbose:
        print(f"MILK10k: {len(train_df):,} dermoscopic samples for training "
              f"({len(holdout_df):,} held out, matches {MILK10K_HOLDOUT_CSV})")

    def clean_age(v):
        return None if pd.isna(v) else float(v)

    return [
        (row["path"], CLASS_IDX[row["label"]], "milk10k", row["isic_id"], clean_age(row["age_approx"]))
        for _, row in train_df.iterrows()
    ]


def build_full_training_pool(verbose: bool = True) -> list:
    """The exact pool the real retrain used: build_sample_list() +
    build_slice3d_samples() + build_emb_samples() + build_milk10k_samples(),
    same order, same default seed/holdout_frac, as
    notebooks/SA01_EfficientNet_Train.ipynb Cell 6."""
    samples = (
        build_sample_list(verbose)
        + build_slice3d_samples(verbose=verbose)
        + build_emb_samples(verbose=verbose)
        + build_milk10k_samples(verbose=verbose)
    )
    if verbose:
        print(f"\nFull training pool: {len(samples):,} samples")
    return samples


if __name__ == "__main__":
    pool = build_full_training_pool()
    print(f"\nTotal: {len(pool):,} samples (expected ~73,456 per docs/MODEL_CARD.md finding #11)")

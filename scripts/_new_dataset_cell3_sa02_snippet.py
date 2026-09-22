# ---- SA02_Ensemble.ipynb Cell 3 variant: reproduces the same holdout split
# as SA01's build_emb_samples()/build_milk10k_samples() (same seed, same
# input order) but does not re-write the holdout CSVs -- those were already
# saved during training.
EMB_LABELS_CSV = os.path.join(DATA, "EMB", "_repo", "early_melanoma_benchmark_dataset_labels.csv")
EMB_IMG        = os.path.join(DATA, "EMB", "images")
EMB_HOLDOUT_CSV = os.path.join(OUT, "..", "emb_external_holdout_ids.csv")

MILK10K_META    = os.path.join(DATA, "milk10k", "MILK10k_Training_Metadata.csv")
MILK10K_GT      = os.path.join(DATA, "milk10k", "MILK10k_Training_GroundTruth.csv")
MILK10K_IMG     = os.path.join(DATA, "milk10k", "images", "MILK10k_Training_Input")
MILK10K_HOLDOUT_CSV = os.path.join(OUT, "..", "milk10k_external_holdout_ids.csv")

MILK10K_MAP = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
}


def build_emb_samples(holdout_frac=0.15, seed=42):
    if not (os.path.exists(EMB_LABELS_CSV) and os.path.exists(EMB_IMG)):
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
    train_df   = df[~df["image"].isin(holdout_ids)]

    print(f"EMB: {len(train_df):,} thin/early-stage melanoma samples for training "
          f"({len(holdout_df):,} held out -- see {EMB_HOLDOUT_CSV})")
    print(f"  Training severity mix: {train_df['severity'].value_counts().to_dict()}")

    return [
        (row["path"], CLASS_IDX["mel"], "emb", row["image"], None)
        for _, row in train_df.iterrows()
    ]


def build_milk10k_samples(holdout_frac=0.15, seed=42):
    if not (os.path.exists(MILK10K_META) and os.path.exists(MILK10K_GT) and os.path.exists(MILK10K_IMG)):
        print(f"MILK10k not found -- skipping (expected at {MILK10K_META})")
        return []

    meta = pd.read_csv(MILK10K_META)
    gt   = pd.read_csv(MILK10K_GT)
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
    train_df   = df[~df["isic_id"].isin(holdout_ids)]

    print(f"MILK10k: {len(train_df):,} dermoscopic samples for training "
          f"({len(holdout_df):,} held out, skin-tone-stratified -- see {MILK10K_HOLDOUT_CSV})")
    print(f"  Training skin-tone mix: {train_df['skin_tone_class'].value_counts().sort_index().to_dict()}")

    def clean_age(v):
        return None if pd.isna(v) else float(v)

    return [
        (row["path"], CLASS_IDX[row["label"]], "milk10k", row["isic_id"], clean_age(row["age_approx"]))
        for _, row in train_df.iterrows()
    ]

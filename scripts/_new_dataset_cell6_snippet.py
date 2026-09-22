# ---- snippet inserted into Cell 6 of all three training notebooks + SA02_Ensemble.ipynb Cell 3 ----
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
    # BEN_OTH, INF, MAL_OTH, SCCKA have no clean home in the 7-class scheme
    # and are dropped rather than guessed -- see docs/KNOWN_GAPS.md.
}


def build_emb_samples(holdout_frac=0.15, seed=42):
    """
    EMB (Early-Stage Melanoma Benchmark, CC BY 4.0) -- 907 ISIC-sourced
    melanoma images with Breslow-thickness/T-category labels, targeting the
    exact failure mode in docs/MODEL_CARD.md finding #5 (thin/in-situ
    melanoma missed almost entirely). The 197 dermoscopyatlas.com-sourced
    images from the original benchmark are NOT included here -- see
    docs/KNOWN_GAPS.md for why.

    Reserves a severity-stratified holdout, never trained on, same pattern
    as build_slice3d_samples().
    """
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

    os.makedirs(os.path.dirname(EMB_HOLDOUT_CSV), exist_ok=True)
    holdout_df[["image", "severity", "thickness", "stage_ajcc"]].to_csv(EMB_HOLDOUT_CSV, index=False)
    print(f"EMB: {len(train_df):,} thin/early-stage melanoma samples for training "
          f"({len(holdout_df):,} held out for external validation, saved to {EMB_HOLDOUT_CSV})")
    print(f"  Training severity mix: {train_df['severity'].value_counts().to_dict()}")

    return [
        (row["path"], CLASS_IDX["mel"], "emb", row["image"], None)
        for _, row in train_df.iterrows()
    ]


def build_milk10k_samples(holdout_frac=0.15, seed=42):
    """
    MILK10k (CC-BY-NC, non-commercial research use approved this session --
    see docs/KNOWN_GAPS.md) -- 5,240 dermoscopic + 5,240 paired clinical
    close-up images with skin_tone_class (0-5) annotations. Only the
    dermoscopic half is used for training; mixing in the clinical
    close-ups would reintroduce the modality gap from docs/MODEL_CARD.md
    finding #9. Rows whose only positive label is one of the 4 classes with
    no clean 7-class mapping (BEN_OTH/INF/MAL_OTH/SCCKA) are dropped.

    Holdout is stratified by skin_tone_class rather than severity -- this
    becomes the project's first genuine skin-tone-labeled DERMOSCOPY
    validation set (docs/MODEL_CARD.md finding #9 flagged this as
    previously unanswered; PAD-UFES-20/DDI are clinical photography, not
    dermoscopy).
    """
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

    os.makedirs(os.path.dirname(MILK10K_HOLDOUT_CSV), exist_ok=True)
    holdout_df[["isic_id", "lesion_id", "skin_tone_class", "label", "age_approx", "sex"]].to_csv(MILK10K_HOLDOUT_CSV, index=False)
    print(f"MILK10k: {len(train_df):,} dermoscopic samples for training "
          f"({len(holdout_df):,} held out, skin-tone-stratified, for dermoscopy "
          f"skin-tone validation, saved to {MILK10K_HOLDOUT_CSV})")
    print(f"  Training skin-tone mix: {train_df['skin_tone_class'].value_counts().sort_index().to_dict()}")

    def clean_age(v):
        return None if pd.isna(v) else float(v)

    return [
        (row["path"], CLASS_IDX[row["label"]], "milk10k", row["isic_id"], clean_age(row["age_approx"]))
        for _, row in train_df.iterrows()
    ]

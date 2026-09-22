"""
SkinAnalytica — dataset_loaders.py
Loads each external validation/fairness dataset into a common
[path, label, ...extra metadata] DataFrame. Metadata schemas genuinely
differ per dataset (confirmed against the real files via
inspect_dataset_schema.py before writing any of this — do not add a new
dataset here without running that check first, see docs/KNOWN_GAPS.md for
why that matters), so each loader is dataset-specific; the scoring loop
that consumes these (score_dataset.py) is shared.
"""
import os
import pandas as pd

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")

UNIFIED_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]


def load_isic2020(per_class_cap: int = 1000, seed: int = 42) -> pd.DataFrame:
    """ISIC 2020 (dermoscopy) -- same source distribution as training, used
    to show the blended official metric hides a dataset-specific gap."""
    gt_path = os.path.join(DATA, "ISIC_2020", "ISIC_2020_Training_GroundTruth_v2.csv")
    img_dir = os.path.join(DATA, "ISIC_2020", "ISIC_2020_Training_JPEG", "train")
    isic20_map = {
        "melanoma": "mel", "nevus": "nv", "seborrheic_keratosis": "bkl", "lentigo_NOS": "bkl",
        "lichenoid_keratosis": "bkl", "solar_lentigo": "bkl", "cafe_au_lait_macule": "bkl",
        "atypical_melanocytic_proliferation": "mel",
    }
    df = pd.read_csv(gt_path)
    rows = []
    for _, row in df.iterrows():
        iid = str(row["image_name"])
        path = os.path.join(img_dir, iid + ".jpg")
        if not os.path.exists(path):
            continue
        diagnosis = str(row.get("diagnosis", "")).lower().replace(" ", "_")
        target = int(row.get("target", -1))
        if diagnosis in isic20_map:
            label = isic20_map[diagnosis]
        elif target == 1:
            label = "mel"
        elif target == 0:
            label = "nv"
        else:
            continue
        rows.append({"path": path, "label": label, "image_id": iid,
                     "sex": row.get("sex"), "age_approx": row.get("age_approx"),
                     "site": row.get("anatom_site_general_challenge")})
    out = pd.DataFrame(rows)
    return pd.concat([
        out[out["label"] == lbl].sample(min(len(out[out["label"] == lbl]), per_class_cap), random_state=seed)
        for lbl in out["label"].unique()
    ]).reset_index(drop=True)


def load_pad_ufes20() -> pd.DataFrame:
    """PAD-UFES-20 (smartphone clinical photos) -- confirmed schema: the
    Fitzpatrick column is 'fitspatrick' (typo in the source dataset)."""
    diag_map = {"MEL": "mel", "NEV": "nv", "BCC": "bcc", "ACK": "akiec", "SCC": "akiec", "SEK": "bkl"}
    meta = pd.read_csv(os.path.join(DATA, "pad_ufes_20", "metadata.csv"))
    img_dir = os.path.join(DATA, "pad_ufes_20", "images")
    meta["label"] = meta["diagnostic"].map(diag_map)
    meta["path"] = meta["img_id"].apply(lambda x: os.path.join(img_dir, x))
    return meta[meta["path"].apply(os.path.exists)].reset_index(drop=True)


def load_ddi() -> pd.DataFrame:
    """DDI (Diverse Dermatology Images, clinical photos) -- skin_tone codes
    are grouped (12/34/56 = Fitzpatrick I-II/III-IV/V-VI), not a raw scale.
    No per-class diagnosis mapping to the 7-class taxonomy is reliable from
    the free-text `disease` field, so `label` is left unset here -- use
    `malignant` (binary) for scoring instead."""
    meta = pd.read_csv(os.path.join(DATA, "ddi", "metadata.csv"))
    img_dir = os.path.join(DATA, "ddi")
    meta["path"] = meta["DDI_file"].apply(lambda x: os.path.join(img_dir, x))
    return meta[meta["path"].apply(os.path.exists)].reset_index(drop=True)


def load_slice3d_permissive(n_benign_sample: int = 2500, seed: int = 42) -> pd.DataFrame:
    """ISIC 2024 SLICE-3D Permissive (3D total-body-photography crops) --
    true external validation: institutions with zero overlap with training
    data. Ground truth is binary malignant/benign only, 0.135% prevalence
    (294/217,477) -- takes all malignant + a stratified benign sample."""
    gt_path = os.path.join(DATA, "ISIC_2024", "ground_truth.csv")
    img_dir = os.path.join(DATA, "ISIC_2024", "images")
    gt = pd.read_csv(gt_path)
    malignant = gt[gt["malignant"] == 1.0]
    benign = gt[gt["malignant"] == 0.0].sample(n_benign_sample, random_state=seed)
    df = pd.concat([malignant, benign]).reset_index(drop=True)
    df["path"] = df["isic_id"].apply(lambda x: os.path.join(img_dir, x + ".jpg"))
    return df[df["path"].apply(os.path.exists)].reset_index(drop=True)


def load_slice3d_holdout(n_benign_sample: int = 2500, seed: int = 42) -> pd.DataFrame:
    """ISIC 2024 SLICE-3D Permissive -- the GENUINE post-retrain external
    holdout, not the full permissive set. 250 of the 294 malignant cases in
    load_slice3d_permissive() are now in the training data (mixed in via
    build_slice3d_samples(), see docs/RETRAIN_PLAN.md Step 1) -- scoring
    against the full set would be training-data leakage. This loader reads
    only the 44 malignant cases from outputs/slice3d_external_holdout_ids.csv
    that were stratified out and genuinely never trained on by any of the 3
    backbones, paired with a benign sample (SLICE-3D's benign pool was never
    trained on at all, so any benign sample here is leakage-free).

    NOT directly comparable to the pre-retrain 0.593 AUC headline number --
    that was computed on the full 294-malignant-case set. This is a smaller
    (44-case), leakage-free slice of the same population; treat it as
    directionally informative, not a strict apples-to-apples replacement.
    """
    holdout_path = os.path.join(BASE, "outputs", "slice3d_external_holdout_ids.csv")
    gt_path = os.path.join(DATA, "ISIC_2024", "ground_truth.csv")
    img_dir = os.path.join(DATA, "ISIC_2024", "images")

    holdout = pd.read_csv(holdout_path)
    holdout["malignant"] = 1.0

    gt = pd.read_csv(gt_path)
    benign = gt[gt["malignant"] == 0.0].sample(n_benign_sample, random_state=seed)

    df = pd.concat([holdout[["isic_id", "malignant"]], benign[["isic_id", "malignant"]]]).reset_index(drop=True)
    df["path"] = df["isic_id"].apply(lambda x: os.path.join(img_dir, x + ".jpg"))
    return df[df["path"].apply(os.path.exists)].reset_index(drop=True)


def load_slice3d_full_holdout(n_benign_sample: int = 2500, seed: int = 42) -> pd.DataFrame:
    """ISIC 2024 SLICE-3D FULL (non-Permissive) release -- the largest
    leakage-free external test set available to this project, built to
    resolve the inconclusive n=44 result in docs/MODEL_CARD.md finding #11.

    The full release (401,059 images, 7 institutions, CC-BY-NC) contains 393
    malignant cases vs the Permissive subset's 294. Of those 393:
      - 250 are in the training data (mixed in via build_slice3d_samples())
      -  44 are the Permissive-subset holdout (never trained on)
      -  99 come from the 3 institutions absent from the Permissive subset,
         so they were never available to training at all

    This loader returns the 143 genuinely-unseen malignant cases (44 + 99),
    a 3.25x larger positive sample than load_slice3d_holdout()'s 44, paired
    with a benign sample (SLICE-3D's benign pool was never trained on at
    all, so any benign draw is leakage-free). Verified zero overlap with the
    250 trained-on ids.

    Still not a strict apples-to-apples replacement for finding #5's 0.593
    figure (that was 294 malignant cases, 250 of which are now training
    data), but it is the closest valid comparison this project can make.
    """
    full_gt_path = os.path.join(DATA, "ISIC_2024_full", "ISIC_2024_Training_GroundTruth.csv")
    perm_gt_path = os.path.join(DATA, "ISIC_2024", "ground_truth.csv")
    holdout_path = os.path.join(BASE, "outputs", "slice3d_external_holdout_ids.csv")
    img_dir = os.path.join(DATA, "ISIC_2024_full", "images", "ISIC_2024_Training_Input")

    full_gt = pd.read_csv(full_gt_path)
    perm_gt = pd.read_csv(perm_gt_path)
    holdout_ids = set(pd.read_csv(holdout_path)["isic_id"])

    full_mal = set(full_gt[full_gt["malignant"] == 1.0]["isic_id"])
    perm_mal = set(perm_gt[perm_gt["malignant"] == 1.0]["isic_id"])
    trained_on = perm_mal - holdout_ids

    clean_mal = (full_mal - perm_mal) | holdout_ids
    assert not (clean_mal & trained_on), "leakage: trained-on ids in the clean test pool"

    mal_df = pd.DataFrame({"isic_id": sorted(clean_mal), "malignant": 1.0})
    benign = full_gt[full_gt["malignant"] == 0.0].sample(n_benign_sample, random_state=seed)

    df = pd.concat([mal_df, benign[["isic_id", "malignant"]]]).reset_index(drop=True)
    df["path"] = df["isic_id"].apply(lambda x: os.path.join(img_dir, x + ".jpg"))
    return df[df["path"].apply(os.path.exists)].reset_index(drop=True)


def load_milk10k_holdout() -> pd.DataFrame:
    """MILK10k (CC-BY-NC, dermoscopic only) skin-tone-stratified holdout --
    the 700 images reserved by build_milk10k_samples() in
    SA01*_Train.ipynb and genuinely never trained on. This is THE
    dermoscopy skin-tone fairness test this project has been trying to
    answer since PAD-UFES-20/DDI turned out to be clinical photography,
    not dermoscopy (docs/MODEL_CARD.md finding #9) -- reserved during
    training specifically for this, never scored until now.

    Returns 7-class labels (mel/nv/bcc/akiec/bkl/df/vasc) plus
    skin_tone_class (0-5), age_approx, sex -- carried through by
    score_dataset.py unchanged so per-skin-tone breakdown can be computed
    from the scored CSV directly.

    Known limitation carried over from the loader that built this holdout
    (see docs/RETRAIN_PLAN.md): skin_tone_class 0 (darkest) has only ~2
    samples after the original train/holdout split -- too few to draw a
    conclusion for that specific band, same statistical-noise problem
    PAD-UFES-20 had.
    """
    holdout_path = os.path.join(BASE, "outputs", "milk10k_external_holdout_ids.csv")
    img_dir = os.path.join(DATA, "milk10k", "images", "MILK10k_Training_Input")

    df = pd.read_csv(holdout_path)
    df["path"] = df.apply(lambda r: os.path.join(img_dir, r["lesion_id"], r["isic_id"] + ".jpg"), axis=1)
    return df[df["path"].apply(os.path.exists)].reset_index(drop=True)


MRA_MIDAS_FITZPATRICK_MAP = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}


def _mra_midas_map_label(row) -> str:
    """Same substring-matching discipline as build_slice3d_samples()'s
    map_label() -- midas_pathreport is genuine free text (520 distinct
    values among dermoscopy-only, non-control rows alone), not a
    controlled vocabulary, confirmed by direct inspection before writing
    this. midas_melanoma is a clean yes/no flag and is checked first as
    the authoritative source for that one class; everything else is
    substring-matched against midas_pathreport, using the same category
    boundaries already established in ISIC19_MAP/ISIC20_MAP elsewhere in
    this file (SCC/actinic keratosis -> akiec, seborrheic/lichenoid
    keratosis/solar lentigo -> bkl). Returns None (drop the row) for
    anything that doesn't confidently match -- e.g. "verruca vulgaris",
    "fibrous papule", which have no clean home in the 7-class scheme."""
    if row["midas_melanoma"] == "yes":
        return "mel"
    s = str(row["midas_pathreport"]).lower()
    if "basal cell carcinoma" in s:
        return "bcc"
    if "squamous cell carcinoma" in s or "actinic keratosis" in s:
        return "akiec"
    if "seborrheic keratosis" in s or "lichenoid keratosis" in s or "solar lentigo" in s:
        return "bkl"
    if "dermatofibroma" in s:
        return "df"
    if "melanocytic nevus" in s:
        return "nv"
    if "hemangioma" in s or "angioma" in s or "pyogenic granuloma" in s:
        return "vasc"
    return None


def load_mra_midas() -> pd.DataFrame:
    """MRA-MIDAS (Stanford AIMI, non-commercial DUA -- see
    docs/KNOWN_GAPS.md) -- dermoscopy-modality subset only.

    Real, verified findings from setting this loader up (2026-07-27), not
    assumptions:
    - The `release_midas` table (3,416 rows) is smaller than the 3,830
      images the source paper reports -- every non-virtual modality is
      short by exactly 138 records (dscope: 1,049 vs 1,187 published;
      6in: 1,060 vs 1,198; 1ft: 1,047 vs 1,185), while the 260 virtual
      images match exactly. Most likely explanation: Stanford held back a
      private benchmark/test slice from the public "release" table (the
      table's own name), not a data-quality problem -- but not confirmed,
      so this loader works from the 3,416-row release as-is rather than
      assuming it's the full study population.
    - `midas_iscontrol="yes"` flags normal/non-lesion control images (503
      of 3,416) -- excluded here, same reasoning as every other loader in
      this file only wanting real lesion images.
    - `midas_path` is frequently empty (e.g. for every control row
      checked); `midas_file_name` is the reliable join key against the
      actual extracted files.
    - `midas_file_name`'s extension doesn't always match the real file on
      disk (205 of 3,416 rows say `.jpeg` where the real file is `.jpg`,
      confirmed by direct comparison) -- this loader matches by filename
      stem, case-insensitive, not exact filename. 58 of 3,416 rows'
      `midas_file_name` has no matching file at all after that -- dropped
      via the existing os.path.exists filter, consistent with every other
      loader in this file.
    """
    meta_path = os.path.join(DATA, "mra_midas", "release_midas.csv")
    img_dir = os.path.join(DATA, "mra_midas", "images")
    if not os.path.exists(meta_path):
        return pd.DataFrame()

    disk_by_stem = {}
    for fname in os.listdir(img_dir):
        full = os.path.join(img_dir, fname)
        if os.path.isfile(full):
            disk_by_stem.setdefault(os.path.splitext(fname)[0].lower(), fname)

    df = pd.read_csv(meta_path)
    df = df[(df["midas_iscontrol"] == "no") & (df["midas_distance"] == "dscope")].copy()

    df["label"] = df.apply(_mra_midas_map_label, axis=1)
    df = df[df["label"].notna()].copy()

    def resolve_path(fname):
        if not isinstance(fname, str) or not fname:
            return None
        stem = os.path.splitext(fname)[0].lower()
        real = disk_by_stem.get(stem)
        return os.path.join(img_dir, real) if real else None

    df["path"] = df["midas_file_name"].apply(resolve_path)
    df = df[df["path"].notna()].copy()
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    def parse_fitzpatrick(v):
        if not isinstance(v, str) or not v:
            return None
        token = v.strip().split(" ")[0].lower()
        return MRA_MIDAS_FITZPATRICK_MAP.get(token)

    df["fitzpatrick"] = df["midas_fitzpatrick"].apply(parse_fitzpatrick)
    df = df.rename(columns={"midas_age": "age_approx", "midas_gender": "sex", "midas_location": "site"})
    return df[["path", "label", "midas_record_id", "midas_file_name", "age_approx", "sex", "fitzpatrick", "site"]].reset_index(drop=True)


LOADERS = {
    "isic2020": load_isic2020,
    "pad_ufes20": load_pad_ufes20,
    "ddi": load_ddi,
    "slice3d_permissive": load_slice3d_permissive,
    "slice3d_holdout": load_slice3d_holdout,
    "slice3d_full_holdout": load_slice3d_full_holdout,
    "milk10k_holdout": load_milk10k_holdout,
    "mra_midas": load_mra_midas,
}

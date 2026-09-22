"""
SkinAnalytica -- scripts/join_fusion_features.py
Path B, workstream 1 task 2 -- see docs/RETRAIN_PLAN.md
"Workstream 1 -- implementation-ready scope".

Merges per-image CNN embeddings (scripts/extract_embeddings.py's output,
deduplicated per docs/MODEL_CARD.md finding #17) with the structured
metadata available for each source dataset, producing one flat feature
matrix + label vector ready for LightGBM (task 3).

Metadata sources joined (see the table in docs/RETRAIN_PLAN.md's
"Workstream 1" section -- this ports that plan into code):
  ISIC 2019  -- sex, anatom_site_general           (age_approx already in the embeddings npz)
  ISIC 2020  -- sex, anatom_site_general_challenge  (age_approx already in the embeddings npz)
  SLICE-3D   -- tbp_lv_dnn_lesion_confidence
  MILK10k    -- sex, site, skin_tone_class          (age_approx already in the embeddings npz)
  ISIC 2018 / EMB -- no additional metadata locally; all fields NaN + missing-flagged

Every metadata field gets an explicit *_missing indicator column
(LightGBM handles NaN splits natively, but an explicit flag lets it use
"was this even measured" as its own signal, not just the imputed/NaN
value) -- this was an explicit requirement in the workstream-1 scope, not
just a nice-to-have.

`site` is intentionally NOT semantically unified across sources: ISIC19
distinguishes anterior/lateral/posterior torso, ISIC20 collapses those to
one "torso", MILK10k uses yet another vocabulary ("head_neck_face",
"lower_extremity", ...). Merging these into one taxonomy would require a
judgment call this script isn't in a position to make correctly (same
discipline as the diagnosis-mapping work elsewhere in this project) --
instead every distinct site string across every source becomes its own
category. A GBDT can still use this; it just won't assume categories
that were never verified equivalent.

The site-category vocabulary is FIT on the train split and saved
(outputs/embeddings/site_vocab.json) so holdout splits reuse the exact
same encoding -- an unseen site string at eval time maps to a dedicated
"unknown" code rather than silently growing the vocabulary, which would
make train/eval feature meanings diverge.

Usage:
  python scripts/join_fusion_features.py --split train
  python scripts/join_fusion_features.py --split slice3d_holdout
  python scripts/join_fusion_features.py --split milk10k_holdout

Output: outputs/embeddings/<split>_fusion_features.npz containing:
  X              [N, D] float32 -- concatenated CNN embeddings (3840 dims:
                 1280 efficientnetv2-s + 1024 vit-large + 1536 convnext)
                 followed by the structured/metadata feature columns
  y              [N] int64 -- label_idx (7-class target)
  image_id       [N] object
  source         [N] object
  feature_names  [D] object -- column names for X, in order (embedding
                 dims are named "<backbone>_<i>", metadata columns keep
                 their real names)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
EMB_DIR = os.path.join(BASE, "outputs", "embeddings")
VOCAB_PATH = os.path.join(EMB_DIR, "site_vocab.json")

BACKBONE_ORDER = ["efficientnetv2-s", "vit-large-patch16-224", "convnext-large"]
SEX_MAP = {"female": 0, "male": 1}


def _load_isic19_meta():
    path = os.path.join(DATA, "ISIC_2019", "ISIC_2019_Training_Metadata.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {
        str(r["image"]): {"sex": r.get("sex"), "site": r.get("anatom_site_general")}
        for _, r in df.iterrows()
    }


def _load_isic20_meta():
    path = os.path.join(DATA, "ISIC_2020", "ISIC_2020_Training_GroundTruth_v2.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {
        str(r["image_name"]): {"sex": r.get("sex"), "site": r.get("anatom_site_general_challenge")}
        for _, r in df.iterrows()
    }


def _load_slice3d_meta():
    path = os.path.join(DATA, "ISIC_2024", "supplemental_metadata.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {
        str(r["isic_id"]): {"tbp_lv_dnn_lesion_confidence": r.get("tbp_lv_dnn_lesion_confidence")}
        for _, r in df.iterrows()
    }


def _load_milk10k_meta():
    path = os.path.join(DATA, "milk10k", "MILK10k_Training_Metadata.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {
        str(r["isic_id"]): {"sex": r.get("sex"), "site": r.get("site"), "skin_tone_class": r.get("skin_tone_class")}
        for _, r in df.iterrows()
    }


def _load_mra_midas_meta():
    """Keyed by midas_file_name -- the same id extract_embeddings.py uses
    for this source (see its "mra_midas_holdout" split). Deliberately
    does NOT contribute skin_tone_class (MILK10k's own 0-5 scale) even
    though MRA-MIDAS has its own real Fitzpatrick I-VI field
    (midas_fitzpatrick, already parsed to 1-6 by
    dataset_loaders.load_mra_midas()) -- forcing two different clinical
    skin-tone scales into the same trained model feature slot without
    verifying they're equivalent would be the same mistake this project
    already avoided once with `site` (see fit_site_vocab()'s docstring).
    Rather than retrain the model to add a proper fitzpatrick feature,
    this source's skin-tone data is simply absent from the fused
    features for now, correctly reflected by skin_tone_class_missing=1 --
    honest under-use of real data, not a fabricated cross-scale mapping."""
    path = os.path.join(DATA, "mra_midas", "release_midas.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {
        str(r["midas_file_name"]): {"sex": r.get("midas_gender"), "site": r.get("midas_location")}
        for _, r in df.iterrows()
    }


def build_metadata_lookup() -> dict:
    """One dict, image_id -> {sex, site, skin_tone_class, tbp_lv_dnn_lesion_confidence},
    built from all four sources that have anything to offer. ISIC18/EMB
    contribute nothing here (verified: no local metadata files exist for
    either -- see build_training_pool.py) and are simply absent from this
    lookup, which the join step below treats as all-fields-missing."""
    lookup = {}
    for loader in (_load_isic19_meta, _load_isic20_meta, _load_slice3d_meta, _load_milk10k_meta, _load_mra_midas_meta):
        for iid, fields in loader().items():
            lookup.setdefault(iid, {}).update({k: v for k, v in fields.items() if v is not None and not pd.isna(v)})
    return lookup


def fit_site_vocab(site_values: list) -> dict:
    """image_id -> code. 0 is reserved for missing, 1 is reserved for
    "unknown" (a site string seen at eval time that wasn't in the fitted
    train vocabulary). Real categories start at 2."""
    uniq = sorted(set(v for v in site_values if v is not None and not (isinstance(v, float) and pd.isna(v))))
    vocab = {"__missing__": 0, "__unknown__": 1}
    for i, v in enumerate(uniq):
        vocab[v] = i + 2
    return vocab


def encode_site(value, vocab: dict) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return vocab["__missing__"]
    return vocab.get(value, vocab["__unknown__"])


def join_features(split: str, is_train: bool) -> dict:
    emb_name = "train_embeddings_deduped" if split == "train" else f"{split}_embeddings"
    emb_path = os.path.join(EMB_DIR, f"{emb_name}.npz")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"{emb_path} not found -- run scripts/extract_embeddings.py"
            + (" then scripts/dedup_train_embeddings.py" if split == "train" else "") + " first"
        )
    d = np.load(emb_path, allow_pickle=True)
    image_ids = d["image_id"]
    n = len(image_ids)
    print(f"Loaded {emb_path}: {n:,} rows")

    meta_lookup = build_metadata_lookup()
    print(f"Metadata available for {len(meta_lookup):,} distinct image_ids across ISIC19/20, SLICE-3D, MILK10k")

    ages = d["age"]  # already in the embeddings npz, per-image
    sexes, sites, skin_tones, tbp_confs = [], [], [], []
    for iid in image_ids:
        m = meta_lookup.get(str(iid), {})
        sexes.append(m.get("sex"))
        sites.append(m.get("site"))
        skin_tones.append(m.get("skin_tone_class"))
        tbp_confs.append(m.get("tbp_lv_dnn_lesion_confidence"))

    if is_train:
        vocab = fit_site_vocab(sites)
        os.makedirs(EMB_DIR, exist_ok=True)
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2)
        print(f"Fit site vocabulary ({len(vocab)} categories, including missing/unknown) -> {VOCAB_PATH}")
    else:
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"{VOCAB_PATH} not found -- run --split train first to fit the site vocabulary")
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab = json.load(f)
        n_unknown = sum(1 for s in sites if s is not None and not pd.isna(s) if s not in vocab)
        if n_unknown:
            print(f"  {n_unknown} site values not seen in the train vocabulary -- mapped to __unknown__")

    site_codes = np.array([encode_site(s, vocab) for s in sites], dtype=np.float32)
    sex_codes = np.array([SEX_MAP.get(s, np.nan) if s is not None and not pd.isna(s) else np.nan for s in sexes], dtype=np.float32)
    skin_tone_arr = np.array([float(v) if v is not None and not pd.isna(v) else np.nan for v in skin_tones], dtype=np.float32)
    tbp_conf_arr = np.array([float(v) if v is not None and not pd.isna(v) else np.nan for v in tbp_confs], dtype=np.float32)
    age_arr = ages.astype(np.float32)

    meta_cols = {
        "age_approx": age_arr,
        "sex": sex_codes,
        "site": site_codes,
        "skin_tone_class": skin_tone_arr,
        "tbp_lv_dnn_lesion_confidence": tbp_conf_arr,
    }
    # site's own missing/unknown codes already encode "was this measured" for that
    # field, but every OTHER metadata field still needs its own explicit flag --
    # required by the workstream-1 scope, not just for LightGBM's benefit.
    missing_flags = {
        f"{name}_missing": np.isnan(arr).astype(np.float32)
        for name, arr in meta_cols.items() if name != "site"
    }
    missing_flags["site_missing"] = (site_codes == vocab["__missing__"]).astype(np.float32)

    embedding_blocks, embedding_names = [], []
    for backbone in BACKBONE_ORDER:
        arr = d[backbone]
        embedding_blocks.append(arr)
        dim = arr.shape[1]
        embedding_names.extend(f"{backbone}_{i}" for i in range(dim))

    X = np.concatenate(
        embedding_blocks + [meta_cols[k].reshape(-1, 1) for k in meta_cols] + [missing_flags[k].reshape(-1, 1) for k in missing_flags],
        axis=1,
    ).astype(np.float32)
    feature_names = np.array(embedding_names + list(meta_cols.keys()) + list(missing_flags.keys()), dtype=object)

    print(f"Final feature matrix: {X.shape[0]:,} rows x {X.shape[1]:,} columns "
          f"({len(embedding_names)} embedding dims + {len(meta_cols)} metadata + {len(missing_flags)} missing-flags)")

    n_with_age = int((~np.isnan(age_arr)).sum())
    n_with_sex = int((~np.isnan(sex_codes)).sum())
    n_with_site = int((site_codes != vocab["__missing__"]).sum())
    n_with_skin_tone = int((~np.isnan(skin_tone_arr)).sum())
    n_with_tbp = int((~np.isnan(tbp_conf_arr)).sum())
    print(f"  Coverage: age {n_with_age:,}/{n:,}, sex {n_with_sex:,}/{n:,}, site {n_with_site:,}/{n:,}, "
          f"skin_tone_class {n_with_skin_tone:,}/{n:,}, tbp_confidence {n_with_tbp:,}/{n:,}")

    return {
        "X": X,
        "y": d["label_idx"],
        "image_id": image_ids,
        "source": d["source"],
        "feature_names": feature_names,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "slice3d_holdout", "milk10k_holdout", "mra_midas_holdout"])
    ap.add_argument("--out-dir", default=EMB_DIR)
    args = ap.parse_args()

    result = join_features(args.split, is_train=(args.split == "train"))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}_fusion_features.npz")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **result)
    os.replace(tmp_path, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

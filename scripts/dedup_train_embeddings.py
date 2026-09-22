"""
SkinAnalytica -- scripts/dedup_train_embeddings.py
Path B, workstream 1 -- see docs/MODEL_CARD.md finding #17.

extract_embeddings.py's --split train output faithfully reproduces the
real training pool build_full_training_pool() produces (verified against
the documented 73,456 total) -- which turns out to contain 10,015 exact
duplicate images: every ISIC-2018 image is also present, byte-for-byte
identical (embeddings match to 0.0 max abs diff), in the ISIC-2019 pool,
sampled once under source="isic18" and again under source="isic19" with
consistent labels. This isn't a bug in the extraction script; it's a real
property of the training pool the actual retrain used.

This script does NOT modify outputs/embeddings/train_embeddings.npz --
that file stays as the accurate record of what the real training pool
actually contained, duplication included. It writes a separate
train_embeddings_deduped.npz (keeping one copy per image_id, first
occurrence in original pool order) for downstream fusion-model training
(Path B workstream 1 task 3), so the GBDT doesn't inherit the same
~2x sampling weight on those 10,015 images that the CNN retrain did.

Usage:
  python scripts/dedup_train_embeddings.py
"""
import csv
import os

import numpy as np

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_DIR = os.path.join(BASE, "outputs", "embeddings")


def main():
    src_path = os.path.join(EMB_DIR, "train_embeddings.npz")
    d = np.load(src_path, allow_pickle=True)
    ids = d["image_id"]

    seen = set()
    keep_idx = []
    for i, iid in enumerate(ids):
        if iid not in seen:
            seen.add(iid)
            keep_idx.append(i)
    keep_idx = np.array(keep_idx, dtype=np.int64)

    n_before, n_after = len(ids), len(keep_idx)
    print(f"train_embeddings.npz: {n_before:,} rows, {n_after:,} unique image_ids "
          f"({n_before - n_after:,} duplicates removed)")

    deduped = {k: d[k][keep_idx] for k in d.files}

    out_path = os.path.join(EMB_DIR, "train_embeddings_deduped.npz")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **deduped)
    os.replace(tmp_path, out_path)

    manifest_path = os.path.join(EMB_DIR, "train_manifest_deduped.csv")
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "source", "label_idx", "age"])
        for iid, src, lbl, age in zip(deduped["image_id"], deduped["source"], deduped["label_idx"], deduped["age"]):
            w.writerow([iid, src, lbl, "" if np.isnan(age) else age])
    os.replace(tmp_manifest, manifest_path)

    assert len(set(deduped["image_id"].tolist())) == n_after, "dedup didn't actually remove all duplicates"
    print(f"Saved: {out_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

"""
SkinAnalytica -- scripts/extract_embeddings.py
Path B, workstream 1 (metadata/GBDT fusion) -- see docs/RETRAIN_PLAN.md
"Workstream 1 -- implementation-ready scope", task 1.

Extracts each trained CNN backbone's penultimate-layer (pooled, pre-head)
embedding for every image in a chosen split, saved alongside per-image
id/source/label/age so a later feature-joining step (task 2) can merge
these with structured metadata without re-running inference.

Reuses gpu_inference.py's SkinClassifier / checkpoint loading / DEVICE
selection and inference_utils.preprocess_bgr unchanged -- not a new
inference path, just a new thing to do with the existing one (return
backbone(x) instead of head(dropout(backbone(x)))). Preprocessing logic
itself is NOT altered for speed (would make these embeddings not reflect
what the model actually sees in production) -- only *how* it's scheduled
(parallel workers instead of one image at a time) changes.

Why this version exists (2026-07-27): the first version ran single-process
and had no incremental checkpointing. Real image sizes vary wildly across
datasets (ISIC 2018 ~450x600 vs ISIC 2020 up to 4000x6000), and
preprocess_bgr's hair-removal step (cv2.inpaint especially) scales badly
with resolution -- on the real ISIC-2020 segment throughput collapsed to
~1.3 img/s, single-core-bound, which would have taken ~6 hours for that
segment alone, with zero progress saved if it crashed. Fixed here with:
  1. A multiprocessing pool for the CPU-bound preprocessing step (the
     actual bottleneck -- GPU sat at 12-13% utilization the whole time),
     while the GPU forward pass stays single-process in the main process.
  2. Periodic checkpointing to the real output file, so a crash loses at
     most --checkpoint-every images, not the whole run.
  3. Resume support: reruns automatically skip image_ids already present
     in an existing output file, unless --fresh is passed.

Splits:
  train              -- the exact ~73,456-image pool the real retrain used
                         (scripts/build_training_pool.py, ported from
                         notebooks/SA01_EfficientNet_Train.ipynb Cell 6)
  slice3d_holdout    -- the 143-case leakage-free external holdout
                         (scripts/dataset_loaders.py::load_slice3d_full_holdout)
  milk10k_holdout    -- the 700-image leakage-free dermoscopy holdout
                         (scripts/dataset_loaders.py::load_milk10k_holdout)

Usage:
  python scripts/extract_embeddings.py --split slice3d_holdout
  python scripts/extract_embeddings.py --split milk10k_holdout
  python scripts/extract_embeddings.py --split train
  python scripts/extract_embeddings.py --split train --limit 200        # smoke test
  python scripts/extract_embeddings.py --split train                    # reruns resume automatically
  python scripts/extract_embeddings.py --split train --fresh            # ignore existing partial output

Output: outputs/embeddings/<split>_embeddings.npz, containing one
[N, feat_dim] float32 array per backbone (keys = backbone names from
gpu_inference.CHECKPOINTS) plus parallel arrays image_id/source/label_idx/age
(age is NaN where unknown). A human-readable outputs/embeddings/<split>_manifest.csv
is also written for spot-checking without loading the .npz. Both are
overwritten atomically (written to a .tmp path, then renamed) on every
checkpoint, so a crash mid-write can't corrupt the last good checkpoint.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "outputs", "embeddings")


def _read_and_preprocess(path: str):
    """Worker-process function -- must stay top-level (picklable) for
    ProcessPoolExecutor on Windows (spawn start method). Each worker
    process imports cv2/inference_utils itself; no torch/GPU state is
    touched here, so this never contends with the main process's CUDA
    context."""
    import cv2
    from inference_utils import preprocess_bgr
    img = cv2.imread(path)
    if img is None:
        return None
    return preprocess_bgr(img)


def load_backbones(base: str = BASE) -> dict:
    """Same checkpoints as gpu_inference.load_models(), but returns the
    backbone submodule directly (no dropout, no classification head) --
    that's the penultimate-layer embedding this script exists to extract."""
    import torch
    from gpu_inference import CHECKPOINTS, SkinClassifier, DEVICE
    backbones = {}
    for name, (timm_name, ckpt_path) in CHECKPOINTS.items():
        m = SkinClassifier(timm_name, n_classes=7, pretrained=False)
        ckpt = torch.load(os.path.join(base, ckpt_path), map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        m.to(DEVICE)
        backbones[name] = m.backbone  # skip m.dropout / m.head entirely
    return backbones


def load_split(split: str, limit: int = None) -> list:
    """Returns a list of dicts: {path, image_id, source, label_idx, age}."""
    if split == "train":
        from build_training_pool import build_full_training_pool
        samples = build_full_training_pool(verbose=True)
        rows = [
            {"path": p, "image_id": iid, "source": src, "label_idx": cls_idx, "age": age if age is not None else float("nan")}
            for (p, cls_idx, src, iid, age) in samples
        ]
    elif split in ("slice3d_holdout", "milk10k_holdout"):
        from dataset_loaders import LOADERS, UNIFIED_CLASSES
        loader_name = "slice3d_full_holdout" if split == "slice3d_holdout" else "milk10k_holdout"
        df = LOADERS[loader_name]()
        rows = []
        for _, r in df.iterrows():
            if split == "slice3d_holdout":
                label_idx = UNIFIED_CLASSES.index("mel") if r["malignant"] == 1.0 else -1  # benign has no single 7-class label here
                iid = r["isic_id"]
                age = float("nan")
            else:
                label_idx = UNIFIED_CLASSES.index(r["label"]) if r["label"] in UNIFIED_CLASSES else -1
                iid = r["isic_id"]
                age = r.get("age_approx", float("nan"))
                age = float("nan") if age is None else float(age)
            rows.append({"path": r["path"], "image_id": iid, "source": split, "label_idx": label_idx, "age": age})
    elif split == "mra_midas_holdout":
        # Never touched by training (CNNs or the GBDT fusion model both
        # predate this dataset being sourced), so the entire loader output
        # serves as the holdout directly -- no train/holdout split needed
        # here, unlike SLICE-3D/MILK10k which had to carve one out of a
        # pool that was otherwise used for training.
        from dataset_loaders import LOADERS, UNIFIED_CLASSES
        df = LOADERS["mra_midas"]()
        rows = []
        for _, r in df.iterrows():
            label_idx = UNIFIED_CLASSES.index(r["label"]) if r["label"] in UNIFIED_CLASSES else -1
            age = r.get("age_approx", float("nan"))
            age = float("nan") if age is None else float(age)
            rows.append({"path": r["path"], "image_id": r["midas_file_name"], "source": split, "label_idx": label_idx, "age": age})
    else:
        raise ValueError(f"unknown split: {split}")

    if limit:
        rows = rows[:limit]
    return rows


def load_existing(out_path: str):
    """Returns a dict in the same shape as extract()'s result, or None if
    no prior output exists. Used both to resume (skip already-embedded
    ids) and to merge with new results at each checkpoint."""
    if not os.path.exists(out_path):
        return None
    d = np.load(out_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def merge_results(old: dict, new: dict) -> dict:
    if old is None:
        return new
    if new is None or len(new.get("image_id", [])) == 0:
        return old
    merged = {}
    for key in old:
        merged[key] = np.concatenate([old[key], new[key]], axis=0)
    return merged


def save_checkpoint(result: dict, out_dir: str, split: str):
    """Atomic-ish save: write to a .tmp path then rename, so a crash
    mid-write never corrupts the last good checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_embeddings.npz")
    tmp_path = out_path + ".tmp"
    # np.savez_compressed silently appends ".npz" to any path that doesn't
    # already end with it -- pass an open file handle instead of the tmp
    # path string so it writes exactly where we tell it to.
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **result)
    os.replace(tmp_path, out_path)

    manifest_path = os.path.join(out_dir, f"{split}_manifest.csv")
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "source", "label_idx", "age"])
        for iid, src, lbl, age in zip(result["image_id"], result["source"], result["label_idx"], result["age"]):
            w.writerow([iid, src, lbl, "" if np.isnan(age) else age])
    os.replace(tmp_manifest, manifest_path)
    return out_path, manifest_path


def extract(rows: list, backbones: dict, batch_size: int, workers: int,
            checkpoint_every: int, out_dir: str, split: str, prior: dict) -> dict:
    """Parallel-preprocess on `workers` CPU processes (the actual
    bottleneck -- hair-removal/inpaint cost scales with source image
    resolution, which varies a lot across datasets), feed the GPU in the
    main process as results arrive in order, checkpoint periodically."""
    import torch
    from gpu_inference import DEVICE

    order = list(backbones.keys())
    embeddings = {name: [] for name in order}
    kept_meta = {"image_id": [], "source": [], "label_idx": [], "age": []}

    n = len(rows)
    t0 = time.time()
    done = 0
    since_checkpoint = 0
    paths = [r["path"] for r in rows]

    def flush_batch(arrays, valid_meta):
        if not arrays:
            return
        x = torch.from_numpy(np.concatenate(arrays, axis=0)).to(DEVICE)
        with torch.no_grad():
            for name in order:
                feat = backbones[name](x)
                embeddings[name].append(feat.cpu().numpy())
        for r in valid_meta:
            kept_meta["image_id"].append(r["image_id"])
            kept_meta["source"].append(r["source"])
            kept_meta["label_idx"].append(r["label_idx"])
            kept_meta["age"].append(r["age"])

    arrays_buf, meta_buf = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r, arr in zip(rows, pool.map(_read_and_preprocess, paths, chunksize=8)):
            done += 1
            if arr is None:
                print(f"  WARNING: unreadable image, skipping: {r['path']}")
            else:
                arrays_buf.append(arr)
                meta_buf.append(r)

            if len(arrays_buf) >= batch_size or done == n:
                n_batch = len(arrays_buf)
                flush_batch(arrays_buf, meta_buf)
                since_checkpoint += n_batch
                arrays_buf, meta_buf = [], []

            if done % (batch_size * 5) == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else float("inf")
                print(f"  {done:,}/{n:,} ({100*done/n:.1f}%) -- {rate:.1f} img/s, ETA {eta/60:.1f} min")

            if since_checkpoint >= checkpoint_every or done == n:
                partial = {name: (np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 0), dtype=np.float32))
                           for name, arrs in embeddings.items()}
                partial["image_id"] = np.array(kept_meta["image_id"], dtype=object)
                partial["source"] = np.array(kept_meta["source"], dtype=object)
                partial["label_idx"] = np.array(kept_meta["label_idx"], dtype=np.int64)
                partial["age"] = np.array(kept_meta["age"], dtype=np.float64)
                merged = merge_results(prior, partial)
                out_path, _ = save_checkpoint(merged, out_dir, split)
                since_checkpoint = 0
                print(f"  checkpoint saved: {len(merged['image_id']):,} total images -> {out_path}")

    result = {name: (np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 0), dtype=np.float32))
              for name, arrs in embeddings.items()}
    result["image_id"] = np.array(kept_meta["image_id"], dtype=object)
    result["source"] = np.array(kept_meta["source"], dtype=object)
    result["label_idx"] = np.array(kept_meta["label_idx"], dtype=np.int64)
    result["age"] = np.array(kept_meta["age"], dtype=np.float64)
    return merge_results(prior, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "slice3d_holdout", "milk10k_holdout", "mra_midas_holdout"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                     help="parallel CPU processes for preprocessing (default: cpu_count - 1)")
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                     help="save progress to disk after this many newly-processed images")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N samples (smoke test)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing partial output and start over")
    args = ap.parse_args()

    from gpu_inference import DEVICE
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("  WARNING: no CUDA device found -- this will be very slow for the full training pool, "
              "consider --limit for a smoke test on CPU")
    print(f"CPU preprocessing workers: {args.workers} (of {os.cpu_count()} cores)")

    print(f"\nLoading split: {args.split}")
    rows = load_split(args.split, limit=args.limit)
    print(f"  {len(rows):,} samples in split")

    out_path = os.path.join(args.out_dir, f"{args.split}_embeddings.npz")
    prior = None if args.fresh else load_existing(out_path)
    if prior is not None:
        already_done = set(prior["image_id"].tolist())
        before = len(rows)
        rows = [r for r in rows if r["image_id"] not in already_done]
        print(f"  Resuming: {len(already_done):,} already embedded (from {out_path}), "
              f"{len(rows):,} remaining (of {before:,})")
        if not rows:
            print("  Nothing left to do -- already complete.")
            return
    elif os.path.exists(out_path):
        print(f"  --fresh passed: ignoring existing {out_path}")

    print("\nLoading backbones (embedding-only, no classification head)...")
    backbones = load_backbones()
    for name, bb in backbones.items():
        n_params = sum(p.numel() for p in bb.parameters())
        print(f"  {name}: {n_params:,} backbone params")

    print(f"\nExtracting embeddings (batch_size={args.batch_size}, workers={args.workers}, "
          f"checkpoint every {args.checkpoint_every:,} images)...")
    result = extract(rows, backbones, batch_size=args.batch_size, workers=args.workers,
                      checkpoint_every=args.checkpoint_every, out_dir=args.out_dir,
                      split=args.split, prior=prior)

    out_path, manifest_path = save_checkpoint(result, args.out_dir, args.split)

    n_out = len(result["image_id"])
    feat_dims = {name: result[name].shape[1] if result[name].ndim == 2 and n_out else 0 for name in backbones}
    print(f"\nDone. {n_out:,} total images embedded.")
    print(f"  Feature dims: {feat_dims}")
    print(f"  Saved: {out_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

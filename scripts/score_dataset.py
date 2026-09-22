"""
SkinAnalytica — score_dataset.py
Shared scoring tool for every external validation / fairness dataset. Uses
GPU inference (src/gpu_inference.py) when PyTorch + a CUDA device are
available (13-21x faster, see docs/KNOWN_GAPS.md), falling back to CPU ONNX
otherwise -- both paths produce the same numbers (see
tests/test_gpu_inference.py), so which one runs is a speed decision, not a
correctness one.

Usage:
    python score_dataset.py --dataset isic2020 --output outputs/bias_reports/isic2020_scored.csv
    python score_dataset.py --dataset pad_ufes20 --output outputs/bias_reports/pad_ufes20_scored.csv
    python score_dataset.py --dataset ddi --output outputs/bias_reports/ddi_scored.csv
    python score_dataset.py --dataset slice3d_permissive --output outputs/bias_reports/slice3d_scored.csv

Re-run this (all four) after any retrain to refresh docs/MODEL_CARD.md's
findings -- that's the whole point of this living in scripts/ instead of a
throwaway notebook cell.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset_loaders import LOADERS

BASE = os.environ.get("SKINANALYTICA_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROD = os.path.join(BASE, "models", "production")


def _try_gpu_backend():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from gpu_inference import load_models, load_weights_and_temperature, predict_batch_gpu, CHECKPOINTS
        models = load_models()
        W, T = load_weights_and_temperature()
        order = list(CHECKPOINTS.keys())

        def predict(batch):
            return predict_batch_gpu(models, order, W, T, batch)
        return predict, "GPU (PyTorch)"
    except Exception as e:
        print(f"GPU backend unavailable ({e}), falling back to CPU ONNX")
        return None


def _onnx_backend(precision: str = "fp32"):
    import onnxruntime as ort
    from inference_utils import run_session, load_temperature, enable_onnx_cuda_dll_dirs
    enable_onnx_cuda_dll_dirs()  # no-op if not on Windows / no matching CUDA+cuDNN found
    if precision == "int8":
        onnx_path = os.path.join(PROD, "onnx_int8", "skinanalytica_ensemble_int8.onnx")
    else:
        onnx_path = os.path.join(PROD, "onnx", "skinanalytica_ensemble.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    T = load_temperature(PROD)
    active = sess.get_providers()[0]

    def predict(batch):
        return run_session(sess, inp_name, batch, T)
    return predict, f"ONNX ({active})"


def score(dataset: str, output: str, batch_size: int = 64, precision: str = "fp32"):
    from inference_utils import preprocess_bgr

    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}', choose from {list(LOADERS.keys())}")

    print(f"Loading dataset: {dataset}")
    df = LOADERS[dataset]()
    print(f"Loaded {len(df)} rows")

    # GPU backend (src/gpu_inference.py) runs the real PyTorch checkpoints,
    # not the ONNX files, so precision only applies to the CPU ONNX fallback.
    predict, backend_name = _try_gpu_backend() or _onnx_backend(precision)
    print(f"Scoring with backend: {backend_name}" + (f" [{precision}]" if "ONNX" in backend_name else ""))

    paths = df["path"].tolist()
    all_probs = []
    t0 = time.time()
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        arrays = []
        for p in batch_paths:
            img = cv2.imread(p)
            arrays.append(preprocess_bgr(img) if img is not None else np.zeros((1, 3, 224, 224), dtype=np.float32))
        batch = np.concatenate(arrays, axis=0)
        all_probs.append(predict(batch))
        if (i // batch_size) % 10 == 0:
            print(f"  {min(i+batch_size,len(paths))}/{len(paths)}  ({time.time()-t0:.0f}s)")

    probs_arr = np.concatenate(all_probs, axis=0)
    df["mel_score"] = probs_arr[:, 0]
    # bcc_score/akiec_score/malignant_score added 2026-07-27 -- see
    # docs/MODEL_CARD.md findings #18/#22: "malignant-vs-benign AUC" was
    # being computed using mel_score alone as the ranking variable, which
    # under-credits BCC/AKIEC cases (mel_score is specifically P(melanoma),
    # not a general malignancy score). malignant_score = max(mel, bcc,
    # akiec) is the correct ranking variable for that comparison.
    df["bcc_score"] = probs_arr[:, 2]
    df["akiec_score"] = probs_arr[:, 3]
    df["malignant_score"] = probs_arr[:, [0, 2, 3]].max(axis=1)
    df["pred_idx"] = probs_arr.argmax(axis=1)
    df["confidence"] = probs_arr.max(axis=1)
    df["model_flagged_cancer"] = df["pred_idx"].isin({0, 2, 3})  # mel, bcc, akiec

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nSaved: {output}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--precision", choices=["fp32", "int8"], default="fp32",
                    help="Which combined ONNX ensemble file to use when the GPU/PyTorch "
                         "backend isn't available. int8 is several times faster on CPU.")
    args = p.parse_args()
    score(args.dataset, args.output, args.batch_size, args.precision)

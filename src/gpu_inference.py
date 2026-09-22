"""
SkinAnalytica — gpu_inference.py
GPU-accelerated ensemble inference via PyTorch, for local research/validation
use (bootstrap CIs, external-validation runs, bias/fairness re-checks after a
retrain) — NOT for the deployed API, which serves from ONNX on Render and has
no GPU regardless of anything here.

Why this exists: onnxruntime-gpu's CUDA execution provider doesn't work in
this environment — it wants cublasLt64_13.dll (CUDA 13.x), but the only
cuDNN 9.x available (bundled with PyTorch) is built against CUDA 12.1. See
docs/KNOWN_GAPS.md for the full diagnosis. This loads the real training
checkpoints directly instead, giving a 13-21x speedup over CPU ONNX for
scoring runs.

Architecture reconstructed from notebooks/SA01_Model_Training.ipynb's
`SkinClassifier` — confirmed against the actual checkpoint state_dict keys
(head.weight/head.bias only, feat_dim 1280/1024/1536 matching known timm
values), NOT the alternate, more complex head class found in the separate
SA01_EfficientNet_Train.ipynb notebook, which does not match the real
checkpoints.

Validated against the trusted ONNX path (tests/test_gpu_inference.py) before
being used for any real scoring — do not skip that check after modifying
this file.
"""
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm

from inference_utils import preprocess_bgr

BASE = os.environ.get("SKINANALYTICA_BASE", str(Path(__file__).resolve().parent.parent))
PROD = os.path.join(BASE, "models", "production")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "efficientnetv2-s": ("tf_efficientnetv2_s", "models/checkpoints/tf_efficientnetv2_s/skin_efficientnetv2-s_best.pth"),
    "vit-large-patch16-224": ("vit_large_patch16_224", "models/checkpoints/vit_large_patch16_224/skin_vit-large-patch16-224_best.pth"),
    "convnext-large": ("convnext_large", "models/checkpoints/convnext_large/skin_convnext-large_best.pth"),
}


class SkinClassifier(nn.Module):
    def __init__(self, model_name, n_classes=7, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))


def load_models(base: str = BASE) -> dict:
    models = {}
    for name, (timm_name, ckpt_path) in CHECKPOINTS.items():
        m = SkinClassifier(timm_name, n_classes=7, pretrained=False)
        ckpt = torch.load(os.path.join(base, ckpt_path), map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        m.to(DEVICE)
        models[name] = m
    return models


def load_weights_and_temperature(prod_dir: str = PROD):
    W = json.load(open(os.path.join(prod_dir, "ensemble", "ensemble_weights.json")))["weights"]
    T = json.load(open(os.path.join(prod_dir, "ensemble", "temperature.json")))["temperature"]
    return W, T


@torch.no_grad()
def predict_batch_gpu(models: dict, order: list, W: list, T: float, batch_np: np.ndarray) -> np.ndarray:
    """
    batch_np: [N,3,H,W] float32, from preprocess_bgr (or a concatenation of
    several calls to it). Returns calibrated probs [N,7] (numpy).

    Matches EnsembleModel.forward() in notebooks/SA02_Ensemble.ipynb exactly:
    per-model softmax(logits/T), then weighted average in probability space —
    the same arithmetic pooling actually baked into the deployed
    skinanalytica_ensemble.onnx graph. See docs/KNOWN_GAPS.md for why this
    matters (a *different*, geometric-pooling formula was used to derive the
    originally-published ensemble_metrics.json).
    """
    x = torch.from_numpy(batch_np).to(DEVICE)
    probs_per_model = []
    for name in order:
        logits = models[name](x)
        probs_per_model.append(torch.softmax(logits / T, dim=1))
    stacked = torch.stack(probs_per_model, dim=0)  # [n_models, N, 7]
    w = torch.tensor(W, dtype=torch.float32, device=DEVICE).view(-1, 1, 1)
    weighted = (stacked * w).sum(dim=0)  # [N, 7]
    return weighted.cpu().numpy()


def score_image_paths(paths: list, batch_size: int = 64) -> np.ndarray:
    """Convenience wrapper: load models once, score a list of image paths, return [N,7] probs."""
    import cv2
    models = load_models()
    W, T = load_weights_and_temperature()
    order = list(CHECKPOINTS.keys())

    all_probs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        arrays = []
        for p in batch_paths:
            img = cv2.imread(p)
            arrays.append(preprocess_bgr(img) if img is not None else np.zeros((1, 3, 224, 224), dtype=np.float32))
        batch = np.concatenate(arrays, axis=0)
        all_probs.append(predict_batch_gpu(models, order, W, T, batch))
    return np.concatenate(all_probs, axis=0)

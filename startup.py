"""
SkinAnalytica — startup.py
Downloads ONNX models from HuggingFace Hub on first run.
Called by Render before uvicorn starts.

Mode is set by SKINANALYTICA_MODEL_MODE env var:
  efficientnet  → only downloads skin_efficientnetv2-s.onnx (76.9MB) — Render free tier
  full          → downloads all 3 individual + ensemble (default for HF Spaces)
  int8          → downloads INT8 ensemble only (500MB)
"""

import os, sys, requests, hashlib
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
HF_REPO    = os.environ.get("HF_MODEL_REPO", "tejansree/skinanalytica-models")
HF_BASE    = f"https://huggingface.co/{HF_REPO}/resolve/main"
MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "/opt/render/project/src/models/production/onnx"))
MODE       = os.environ.get("SKINANALYTICA_MODEL_MODE", "efficientnet")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
INT8_DIR   = MODEL_DIR.parent / "onnx_int8"
INT8_DIR.mkdir(parents=True, exist_ok=True)

# ── Model manifests ───────────────────────────────────────────────
ALL_MODELS = {
    "efficientnet": [
        {"file": "skin_efficientnetv2-s.onnx",        "dir": MODEL_DIR,  "size_mb": 76.9},
    ],
    "full": [
        {"file": "skin_efficientnetv2-s.onnx",        "dir": MODEL_DIR,  "size_mb": 76.9},
        {"file": "skin_vit-large-patch16-224.onnx",   "dir": MODEL_DIR,  "size_mb": 1157},
        {"file": "skin_convnext-large.onnx",          "dir": MODEL_DIR,  "size_mb": 748},
        {"file": "skinanalytica_ensemble.onnx",       "dir": MODEL_DIR,  "size_mb": 1983},
    ],
    "int8": [
        {"file": "skinanalytica_ensemble_int8.onnx",  "dir": INT8_DIR,   "size_mb": 500},
    ],
}

# Also pull ensemble JSON configs regardless of mode
CONFIG_FILES = [
    "ensemble_weights.json",
    "temperature.json",
    "ensemble_metrics.json",
]

def download_file(url: str, dest: Path, size_mb: float):
    if dest.exists():
        print(f"  ✓ {dest.name} already present ({dest.stat().st_size//1024//1024}MB)")
        return True
    print(f"  ↓ Downloading {dest.name} (~{size_mb:.0f}MB)...")
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192*16):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    if done % (1024*1024*20) < 8192*16:
                        print(f"    {pct:.0f}%  ({done//1024//1024}MB / {total//1024//1024}MB)")
        print(f"  ✅ {dest.name} downloaded")
        return True
    except Exception as e:
        print(f"  ❌ Failed to download {dest.name}: {e}")
        return False


def download_config(name: str):
    cfg_dir = MODEL_DIR.parent / "ensemble"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg_dir / name
    if dest.exists():
        print(f"  ✓ {name} present")
        return
    url = f"{HF_BASE}/ensemble/{name}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_text(r.text, encoding="utf-8")
            print(f"  ✅ {name} downloaded")
        else:
            print(f"  ⚠ {name} not found on HF Hub (status {r.status_code}) — will use defaults")
    except Exception as e:
        print(f"  ⚠ {name} download failed: {e}")


def main():
    print(f"\n{'='*50}")
    print(f"SkinAnalytica Model Startup")
    print(f"Mode       : {MODE}")
    print(f"HF Repo    : {HF_REPO}")
    print(f"Model dir  : {MODEL_DIR}")
    print(f"{'='*50}\n")

    models = ALL_MODELS.get(MODE, ALL_MODELS["efficientnet"])
    success = 0

    print("Downloading models:")
    for m in models:
        url  = f"{HF_BASE}/{m['file']}"
        dest = Path(m["dir"]) / m["file"]
        if download_file(url, dest, m["size_mb"]):
            success += 1

    print("\nDownloading config files:")
    for cfg in CONFIG_FILES:
        download_config(cfg)

    print(f"\n{success}/{len(models)} models ready")
    if success == 0:
        print("WARNING: No models downloaded — API will start but /analyze will fail")
    print()


if __name__ == "__main__":
    main()

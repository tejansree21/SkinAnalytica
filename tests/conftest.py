"""Shared pytest fixtures. Adds src/ and api/ to sys.path so tests can
import project modules without installing the project as a package."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "api"))
sys.path.insert(0, os.path.join(BASE, "agents"))
sys.path.insert(0, os.path.join(BASE, "delivery"))
sys.path.insert(0, os.path.join(BASE, "assistant"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

os.environ.setdefault("SKINANALYTICA_BASE", BASE)

PROD_DIR = os.path.join(BASE, "models", "production")
MODEL_FILES_PRESENT = os.path.exists(os.path.join(PROD_DIR, "onnx", "skinanalytica_ensemble.onnx"))
CHECKPOINTS_PRESENT = os.path.exists(
    os.path.join(BASE, "models", "checkpoints", "tf_efficientnetv2_s", "skin_efficientnetv2-s_best.pth")
)

"""
SkinAnalytica — verification_agent.py
Runs QA on a batch — annotation conflicts, confidence scoring, clinical flags.
Can be triggered automatically (on upload) or manually.
"""

import os, json
import numpy as np
import pandas as pd
import onnxruntime as ort
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from base_agent import BaseAgent, BASE

PROD           = os.path.join(BASE, "models", "production")
OUT_DIR        = os.path.join(BASE, "outputs", "verification_reports")
UNIFIED_CLASSES= ["mel","nv","bcc","akiec","bkl","df","vasc"]
IMG_SIZE       = 224
IMAGENET_MEAN  = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD   = np.array([0.229,0.224,0.225], dtype=np.float32)

# Annotation QA thresholds
AMBIGUOUS_GAP      = 2.0   # confidence gap < 2.0 → AMBIGUOUS
CONFLICT_SCORE     = 7.0   # score > 7.0 + human disagrees → ANNOTATION_CONFLICT
CONFIDENCE_FLOOR   = 0.65  # below this → LOW_CONFIDENCE flag

class VerificationAgent(BaseAgent):
    """
    Autonomous annotation QA agent.
    Trigger: auto (on batch upload) or manual.
    Produces: structured QA report with per-image flags.
    """

    def __init__(self):
        super().__init__("verification_agent")
        os.makedirs(OUT_DIR, exist_ok=True)
        self._load_model()

    def _load_model(self):
        int8_p = os.path.join(PROD, "onnx_int8", "skinanalytica_ensemble_int8.onnx")
        fp32_p = os.path.join(PROD, "onnx", "skinanalytica_ensemble.onnx")
        path   = int8_p if os.path.exists(int8_p) else fp32_p
        T_path = os.path.join(PROD, "ensemble", "temperature.json")
        self.T = 0.4095
        if os.path.exists(T_path):
            with open(T_path) as f:
                self.T = json.load(f)["temperature"]
        self.sess     = ort.InferenceSession(path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        self.logger.info(f"Model loaded: {os.path.basename(path)}  T={self.T}")

    def _preprocess(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None: return np.zeros((1,3,IMG_SIZE,IMG_SIZE), dtype=np.float32)
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
        bhat   = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _,thr  = cv2.threshold(bhat, 10, 255, cv2.THRESH_BINARY)
        img    = cv2.inpaint(img, thr, 1, cv2.INPAINT_TELEA)
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img    = cv2.resize(img, (IMG_SIZE,IMG_SIZE)).astype(np.float32)/255.0
        img    = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img.transpose(2,0,1)[None]

    def _confidence_score(self, probs: np.ndarray) -> float:
        """Convert max probability to 1-10 scale."""
        return round(float(probs.max()) * 10, 2)

    def _annotation_flag(self, probs: np.ndarray, human_label: str = None) -> str:
        sorted_p = np.sort(probs)[::-1]
        gap      = (sorted_p[0] - sorted_p[1]) * 10
        score    = self._confidence_score(probs)
        pred_cls = UNIFIED_CLASSES[int(probs.argmax())]

        if human_label and human_label != pred_cls and score >= CONFLICT_SCORE:
            return "ANNOTATION_CONFLICT"
        if gap < AMBIGUOUS_GAP:
            return "AMBIGUOUS"
        if probs.max() < CONFIDENCE_FLOOR:
            return "LOW_CONFIDENCE"
        return "OK"

    def _run(self, image_folder: str, labels_csv: str = None,
             label_col: str = "dx", image_col: str = "image",
             output_name: str = "verification",
             trigger: str = "manual") -> dict:

        folder    = Path(image_folder)
        img_files = sorted([f for f in folder.rglob("*")
                            if f.suffix.lower() in {".jpg",".jpeg",".png"}])
        self.logger.info(f"Verifying {len(img_files)} images — trigger={trigger}")

        # Load human labels if provided
        human_labels = {}
        if labels_csv and os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            if image_col in df.columns and label_col in df.columns:
                human_labels = dict(zip(df[image_col].astype(str),
                                       df[label_col].astype(str)))

        records = []
        flags   = {"OK":0,"AMBIGUOUS":0,"LOW_CONFIDENCE":0,
                   "ANNOTATION_CONFLICT":0,"ERROR":0}
        t0 = datetime.now()

        for fp in img_files:
            try:
                arr    = self._preprocess(str(fp))
                logits = self.sess.run(None, {self.inp_name: arr})[0][0]
                exp    = np.exp(logits - logits.max())
                probs  = exp / exp.sum()

                pred_idx  = int(probs.argmax())
                pred_cls  = UNIFIED_CLASSES[pred_idx]
                score     = self._confidence_score(probs)
                stem      = fp.stem
                human_lbl = human_labels.get(stem) or human_labels.get(fp.name)
                flag      = self._annotation_flag(probs, human_lbl)

                rec = {
                    "image_id"       : stem,
                    "path"           : str(fp),
                    "pred_class"     : pred_cls,
                    "confidence_score": score,
                    "flag"           : flag,
                    "human_label"    : human_lbl,
                    "mel_prob"       : round(float(probs[0]), 4),
                    "probs"          : {c: round(float(p),4)
                                        for c,p in zip(UNIFIED_CLASSES, probs)},
                }
                records.append(rec)
                flags[flag] = flags.get(flag, 0) + 1

            except Exception as e:
                records.append({"image_id": fp.stem, "path": str(fp),
                                 "flag": "ERROR", "error": str(e)})
                flags["ERROR"] += 1

        elapsed = (datetime.now() - t0).total_seconds()

        # Top uncertain cases (for active learning)
        uncertain = sorted(
            [r for r in records if r.get("confidence_score") is not None],
            key=lambda x: x["confidence_score"]
        )[:50]

        # Annotation conflict details
        conflicts = [r for r in records if r["flag"] == "ANNOTATION_CONFLICT"]

        result = {
            "trigger"       : trigger,
            "image_folder"  : image_folder,
            "total_images"  : len(img_files),
            "flags"         : flags,
            "records"       : records,
            "top_uncertain" : uncertain,
            "conflicts"     : conflicts,
            "elapsed_s"     : round(elapsed, 1),
        }

        # Save report
        report_path = os.path.join(OUT_DIR, f"{output_name}_verification.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        # Print summary
        print(f"\nVERIFICATION REPORT: {output_name}")
        print("=" * 50)
        print(f"  Trigger        : {trigger}")
        print(f"  Images         : {len(img_files)}")
        print(f"  Time           : {elapsed:.1f}s")
        print(f"\n  Flag breakdown:")
        for flag, count in flags.items():
            bar = "█" * min(20, count // max(1, len(img_files) // 20))
            print(f"    {flag:<25} {count:>5}  {bar}")
        print(f"\n  Annotation conflicts : {len(conflicts)}")
        print(f"  Top-50 uncertain     : saved for active learning")
        print(f"  Report saved         : {report_path}")

        return result


if __name__ == "__main__":
    agent  = VerificationAgent()
    result = agent.run(
        image_folder = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\data\sa04_test_sample",
        output_name  = "test_verification",
        trigger      = "manual",
    )

"""
SkinAnalytica — SA05_api.py (deployment version)
Cross-platform path detection: Windows local vs Linux (Render/HF Spaces)
Drop-in replacement for SA05_api.py
"""

import os, sys, json, uuid, logging, time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

# ── Cross-platform base path ──────────────────────────────────────
def _detect_base() -> str:
    """Auto-detect environment and return correct base path."""
    # 1. Explicit env var always wins
    if os.environ.get("SKINANALYTICA_BASE"):
        return os.environ["SKINANALYTICA_BASE"]
    # 2. Render / Linux deployment
    if os.path.exists("/app"):
        return "/app"
    # 3. HuggingFace Spaces
    if os.path.exists("/home/user/app"):
        return "/home/user/app"
    # 4. Windows local dev
    win_path = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica"
    if os.path.exists(win_path):
        return win_path
    # 5. Current directory fallback
    return str(Path(__file__).parent)

BASE    = _detect_base()
PROD    = os.path.join(BASE, "models", "production")
OUT_DIR = os.path.join(BASE, "outputs")

# ── Model mode ────────────────────────────────────────────────────
MODEL_MODE = os.environ.get("SKINANALYTICA_MODEL_MODE", "efficientnet")

# ── Allowed origins ───────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    # Vercel deployments
    "https://skinanalytica.vercel.app",
    "https://*.vercel.app",
    # HuggingFace Spaces
    "https://*.hf.space",
    # Render services
    "https://*.onrender.com",
]
# Add custom domain if set
if os.environ.get("SKINANALYTICA_FRONTEND_URL"):
    ALLOWED_ORIGINS.append(os.environ["SKINANALYTICA_FRONTEND_URL"])

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("skinanalytica.api")

app = FastAPI(
    title       = "SkinAnalytica API",
    description = "ISIC-grade dermoscopy AI platform",
    version     = "1.1.0",
)

app.add_middleware(CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "OPTIONS", "PATCH"],
    allow_headers     = ["*"],
)

# ── Classes & constants ───────────────────────────────────────────
UNIFIED_CLASSES = ["mel","nv","bcc","akiec","bkl","df","vasc"]
CLASS_FULL = {
    "mel":"Melanoma","nv":"Melanocytic Nevus","bcc":"Basal Cell Carcinoma",
    "akiec":"Actinic Keratosis","bkl":"Benign Keratosis",
    "df":"Dermatofibroma","vasc":"Vascular Lesion"
}
ICD10 = {"mel":"C43.9","nv":"D22.9","bcc":"C44.91",
         "akiec":"L57.0","bkl":"L82.1","df":"D23.9","vasc":"D18.01"}
SNOMED = {"mel":"372244006","nv":"400122008","bcc":"254701007",
          "akiec":"202820008","bkl":"21238008","df":"432328008","vasc":"400210000"}
NCCN = {
    "mel":   ["Urgent dermatology referral","Wide local excision (WLE)","Sentinel lymph node biopsy if Breslow >1mm","Dermoscopy-guided margin assessment"],
    "nv":    ["Routine monitoring — annual skin check","Reassure patient: benign lesion","Photograph for baseline if atypical features"],
    "bcc":   ["Dermatology referral within 2 weeks","Mohs surgery (facial/high-risk sites)","Excision with 4mm margins (low-risk)"],
    "akiec": ["Topical 5-fluorouracil or imiquimod","Cryotherapy for isolated lesions","Dermatology review — pre-malignant potential"],
    "bkl":   ["No treatment required","Reassure patient: benign keratosis","Cryotherapy if symptomatic"],
    "df":    ["No treatment required","Excision if symptomatic or uncertain diagnosis"],
    "vasc":  ["Dermatology review","Laser therapy if cosmetically concerning","Rule out angiosarcoma in elderly patients"],
}
IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)
IMG_SIZE      = 224

# ── Model registry ────────────────────────────────────────────────
class ModelRegistry:
    def __init__(self):
        self.session    = None
        self.inp_name   = None
        self.loaded     = False
        self.temperature= 0.4095
        self.weights    = [0.0511, 0.8346, 0.1143]
        self.model_name = "none"

    def _try_load(self, path: str, label: str) -> bool:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(path,
                providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            self.session  = sess
            self.inp_name = sess.get_inputs()[0].name
            self.model_name = label
            logger.info(f"Loaded: {label} ({os.path.getsize(path)//1024//1024}MB) "
                        f"via {sess.get_providers()[0]}")
            return True
        except Exception as e:
            logger.warning(f"Could not load {label}: {e}")
            return False

    def load(self):
        onnx_dir  = os.path.join(PROD, "onnx")
        int8_dir  = os.path.join(PROD, "onnx_int8")

        # Priority order based on MODEL_MODE
        candidates = []
        if MODEL_MODE == "efficientnet":
            candidates = [
                (os.path.join(onnx_dir, "skin_efficientnetv2-s.onnx"), "EfficientNetV2-S"),
            ]
        elif MODEL_MODE == "int8":
            candidates = [
                (os.path.join(int8_dir, "skinanalytica_ensemble_int8.onnx"), "Ensemble-INT8"),
                (os.path.join(onnx_dir, "skin_efficientnetv2-s.onnx"), "EfficientNetV2-S"),
            ]
        else:  # full
            candidates = [
                (os.path.join(int8_dir, "skinanalytica_ensemble_int8.onnx"), "Ensemble-INT8"),
                (os.path.join(onnx_dir, "skinanalytica_ensemble.onnx"),      "Ensemble-FP32"),
                (os.path.join(onnx_dir, "skin_efficientnetv2-s.onnx"),       "EfficientNetV2-S"),
            ]

        for path, label in candidates:
            if os.path.exists(path) and self._try_load(path, label):
                self.loaded = True
                break

        # Load temperature
        T_path = os.path.join(PROD, "ensemble", "temperature.json")
        if os.path.exists(T_path):
            with open(T_path) as f:
                self.temperature = json.load(f).get("temperature", 0.4095)

        # Load weights
        W_path = os.path.join(PROD, "ensemble", "ensemble_weights.json")
        if os.path.exists(W_path):
            with open(W_path) as f:
                self.weights = json.load(f).get("weights", self.weights)

        if self.loaded:
            logger.info(f"Model ready: {self.model_name}  T={self.temperature:.4f}")
        else:
            logger.error("No ONNX model loaded — /analyze will return 503")

    def predict(self, arr: np.ndarray) -> np.ndarray:
        if not self.session:
            raise RuntimeError("No model loaded")
        logits = self.session.run(None, {self.inp_name: arr})[0]
        logits = logits / self.temperature
        exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

model_reg = ModelRegistry()

# ── Preprocessing ─────────────────────────────────────────────────
def preprocess(img_bytes: bytes) -> np.ndarray:
    import cv2
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    bhat   = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _,thr  = cv2.threshold(bhat, 10, 255, cv2.THRESH_BINARY)
    img    = cv2.inpaint(img, thr, 1, cv2.INPAINT_TELEA)
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img    = cv2.resize(img, (IMG_SIZE,IMG_SIZE)).astype(np.float32)/255.0
    img    = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2,0,1)[None].astype(np.float32)

# ── Verdict logic ─────────────────────────────────────────────────
CANCER_CLASSES  = {"mel","bcc","akiec"}
REVIEW_CLASSES  = {"vasc"}
MEL_THRESHOLD   = 0.312

def make_verdict(probs: np.ndarray, pred_class: str) -> dict:
    mel_score  = float(probs[0])
    confidence = float(probs.max())
    if pred_class in CANCER_CLASSES or mel_score >= MEL_THRESHOLD:
        verdict  = "CANCER_FLAGGED"
        priority = 1 if pred_class == "mel" or mel_score > 0.6 else 2
    elif pred_class in REVIEW_CLASSES or confidence < 0.65:
        verdict  = "REVIEW_REQUIRED"
        priority = 3
    else:
        verdict  = "NORMAL"
        priority = 4
    return {"verdict":verdict,"priority":priority,
            "confidence":confidence,"mel_score":mel_score}

# ── Persistence ───────────────────────────────────────────────────
OUT_SCANS = os.path.join(OUT_DIR, "scan_results")
OUT_R     = os.path.join(OUT_DIR, "research_reports")
os.makedirs(OUT_SCANS, exist_ok=True)
os.makedirs(OUT_R,     exist_ok=True)

def _save_scan(data: dict):
    try:
        path = os.path.join(OUT_SCANS, f"{data['scan_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not save scan: {e}")

# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info(f"SkinAnalytica API v1.1.0 starting...")
    logger.info(f"BASE={BASE}  MODE={MODEL_MODE}")
    model_reg.load()

# ── Routes ────────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(status_code=204, content=None)

@app.get("/health")
async def health():
    return {
        "status"   : "ok",
        "version"  : "1.1.0",
        "model"    : "loaded" if model_reg.loaded else "not_loaded",
        "model_name": model_reg.model_name,
        "mode"     : MODEL_MODE,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/models")
async def models_info():
    metrics_path = os.path.join(PROD, "ensemble", "ensemble_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {
        "ensemble"  : f"SkinAnalytica ({model_reg.model_name})",
        "n_classes" : 7,
        "classes"   : UNIFIED_CLASSES,
        "mode"      : MODEL_MODE,
        "metrics"   : metrics,
        "loaded"    : model_reg.loaded,
    }

@app.post("/analyze")
async def analyze(
    file        : UploadFile = File(...),
    cancer_type : Optional[str] = None,
    patient_id  : Optional[str] = None,
    patient_age : Optional[int] = None,
    patient_sex : Optional[str] = None,
):
    if not model_reg.loaded:
        raise HTTPException(503, "Model not loaded — check startup logs")

    t0        = time.time()
    img_bytes = await file.read()
    if len(img_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 20MB)")

    try:
        arr   = preprocess(img_bytes)
    except Exception as e:
        raise HTTPException(400, f"Image preprocessing failed: {e}")

    try:
        probs = model_reg.predict(arr)[0]
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    pred_idx   = int(probs.argmax())
    pred_class = UNIFIED_CLASSES[pred_idx]
    decision   = make_verdict(probs, pred_class)
    scan_id    = str(uuid.uuid4())[:8]
    latency_ms = int((time.time() - t0) * 1000)

    result = {
        "scan_id"       : scan_id,
        "verdict"       : decision["verdict"],
        "priority"      : decision["priority"],
        "pred_class"    : pred_class,
        "pred_class_full": CLASS_FULL[pred_class],
        "confidence"    : round(decision["confidence"], 4),
        "mel_score"     : round(decision["mel_score"], 4),
        "cancer_prob"   : round(float(probs[0]), 4),
        "icd10"         : ICD10.get(pred_class, "L98.9"),
        "snomed"        : SNOMED.get(pred_class),
        "treatment_recs": NCCN.get(pred_class, []),
        "plain_summary" : (f"{decision['verdict'].replace('_',' ').title()} — "
                           f"{CLASS_FULL[pred_class]} detected with "
                           f"{decision['confidence']:.1%} confidence."),
        "probs"         : {c: round(float(p),4) for c,p in zip(UNIFIED_CLASSES, probs)},
        "model_used"    : model_reg.model_name,
        "latency_ms"    : latency_ms,
        "timestamp"     : datetime.now().isoformat(),
        "patient_id"    : patient_id,
    }
    _save_scan(result)
    return result

@app.post("/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    if not model_reg.loaded:
        raise HTTPException(503, "Model not loaded")
    results = []
    for file in files:
        img_bytes = await file.read()
        try:
            arr        = preprocess(img_bytes)
            probs      = model_reg.predict(arr)[0]
            pred_idx   = int(probs.argmax())
            pred_class = UNIFIED_CLASSES[pred_idx]
            decision   = make_verdict(probs, pred_class)
            scan_id    = str(uuid.uuid4())[:8]
            results.append({
                "scan_id"    : scan_id,
                "filename"   : file.filename,
                "verdict"    : decision["verdict"],
                "priority"   : decision["priority"],
                "pred_class" : pred_class,
                "confidence" : round(decision["confidence"], 4),
                "mel_score"  : round(decision["mel_score"], 4),
                "icd10"      : ICD10.get(pred_class, "L98.9"),
                "probs"      : {c: round(float(p),4) for c,p in zip(UNIFIED_CLASSES,probs)},
                "timestamp"  : datetime.now().isoformat(),
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    flagged = sum(1 for r in results if r.get("verdict") == "CANCER_FLAGGED")
    review  = sum(1 for r in results if r.get("verdict") == "REVIEW_REQUIRED")
    return {"total":len(results),"flagged_count":flagged,
            "review_count":review,"results":results,
            "timestamp":datetime.now().isoformat()}

@app.get("/report/{scan_id}")
async def get_report(scan_id: str):
    path = os.path.join(OUT_SCANS, f"{scan_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"Scan {scan_id} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@app.get("/report/{scan_id}/fhir")
async def fhir_export(scan_id: str):
    path = os.path.join(OUT_SCANS, f"{scan_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"Scan {scan_id} not found")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "resourceType"   : "DiagnosticReport",
        "id"             : scan_id,
        "status"         : "final",
        "code"           : {"text": "Dermoscopy AI Analysis — SkinAnalytica v1.1.0"},
        "conclusion"     : f"{d.get('verdict')}: {d.get('pred_class_full')} ({d.get('confidence',0):.1%})",
        "conclusionCode" : [{"coding":[{
            "system": "http://hl7.org/fhir/sid/icd-10",
            "code"  : d.get("icd10","L98.9"),
            "display": d.get("pred_class_full",""),
        }]}],
        "issued"         : d.get("timestamp"),
        "extension"      : [{
            "url"         : "https://skinanalytica.ai/fhir/melanoma-score",
            "valueDecimal": d.get("mel_score"),
        }],
    }

@app.get("/patient/{patient_id}/history")
async def patient_history(patient_id: str):
    scans = []
    if os.path.exists(OUT_SCANS):
        for fname in sorted(os.listdir(OUT_SCANS)):
            if not fname.endswith(".json"): continue
            try:
                with open(os.path.join(OUT_SCANS,fname), encoding="utf-8") as f:
                    r = json.load(f)
                if r.get("patient_id") == patient_id:
                    scans.append(r)
            except: continue
    if not scans:
        raise HTTPException(404, f"No scans found for patient {patient_id}")
    return {"patient_id":patient_id,"scan_count":len(scans),
            "scans":sorted(scans, key=lambda x: x.get("timestamp",""), reverse=True)}

@app.get("/research/sessions")
async def list_sessions():
    import glob
    sessions = []
    for fname in sorted(glob.glob(os.path.join(OUT_R,"*.csv"))):
        sessions.append({
            "session_name": os.path.basename(fname).replace(".csv",""),
            "size_bytes"  : os.path.getsize(fname),
            "created"     : datetime.fromtimestamp(os.path.getctime(fname)).isoformat(),
        })
    return {"sessions":sessions,"count":len(sessions)}

@app.get("/audit/log")
async def audit_log(limit: int = 50):
    scans = []
    if os.path.exists(OUT_SCANS):
        for fname in sorted(os.listdir(OUT_SCANS), reverse=True):
            if not fname.endswith(".json"): continue
            try:
                with open(os.path.join(OUT_SCANS,fname), encoding="utf-8") as f:
                    r = json.load(f)
                scans.append({"scan_id":r.get("scan_id"),"verdict":r.get("verdict"),
                               "pred_class":r.get("pred_class"),"confidence":r.get("confidence"),
                               "timestamp":r.get("timestamp")})
            except: continue
            if len(scans) >= limit: break
    return {"count":len(scans),"entries":scans}

@app.post("/webhooks/tula")
async def tula_webhook(payload: dict):
    event = payload.get("event","")
    logger.info(f"Tula webhook: {event}")
    return {"received":True,"event":event,"timestamp":datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("SA05_api:app", host="0.0.0.0",
                port=int(os.environ.get("PORT",8001)), reload=False)

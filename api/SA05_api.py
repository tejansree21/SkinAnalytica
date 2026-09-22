"""
SkinAnalytica — SA05_api.py (deployment version)
Cross-platform path detection: Windows local vs Linux (Render/HF Spaces)
Drop-in replacement for SA05_api.py
"""

import os, sys, json, uuid, logging, time, secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference_utils import run_session, check_dermatoscope_likelihood, preprocess_bgr

# ── Cross-platform base path ──────────────────────────────────────
def _detect_base() -> str:
    """
    Auto-detect environment and return correct base path.

    Branches on sys.platform first, not on path existence, because
    os.path.exists("/app") is not a reliable Linux/container signal --
    on at least one Windows dev machine used for this project, "/app"
    resolves to something that exists outside any container (a Git
    Bash/MSYS artifact), which previously made this function silently
    resolve BASE to "/app" on Windows, so ModelRegistry.load() found zero
    model files and /analyze 503'd regardless of MODEL_MODE. That failure
    was masked in the test suite because tests/conftest.py explicitly sets
    SKINANALYTICA_BASE, so it only showed up running the API directly.
    """
    # 1. Explicit env var always wins
    if os.environ.get("SKINANALYTICA_BASE"):
        return os.environ["SKINANALYTICA_BASE"]

    if sys.platform == "win32":
        # 2. Windows local dev
        win_path = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica"
        if os.path.exists(win_path):
            return win_path
    else:
        # 2. Render / Linux deployment
        if os.path.exists("/app"):
            return "/app"
        # 3. HuggingFace Spaces
        if os.path.exists("/home/user/app"):
            return "/home/user/app"

    # 4. Current directory fallback
    return str(Path(__file__).parent)

BASE    = _detect_base()
PROD    = os.path.join(BASE, "models", "production")
OUT_DIR = os.path.join(BASE, "outputs")

# ── Model mode ────────────────────────────────────────────────────
# "full" loads the 3-model ensemble (int8, falling back to fp32, falling
# back to EfficientNet-alone if neither ONNX file is present) -- this is
# the mode every threshold/calibration constant below (MEL_THRESHOLD,
# AGE_BAND_THRESHOLDS, the 0.4095 temperature) was actually derived
# against (see docs/MODEL_CARD.md finding #10). Do not default this back
# to "efficientnet": that mode serves a single backbone whose own natural
# operating threshold differs from the ensemble-calibrated ones used here,
# so specificity/sensitivity in production would silently stop matching
# anything documented in the model card.
MODEL_MODE = os.environ.get("SKINANALYTICA_MODEL_MODE", "full")

# Inter-model disagreement as an uncertainty signal (prototyped earlier this
# session as a cheap ONNX-only alternative to MC-dropout — no PyTorch needed
# in production).
#
# DECIDED 2026-07-28 (docs/MODEL_CARD.md finding #28): ship this over adding
# PyTorch to production. MC-dropout is the "correct" method, but on the one
# real false-negative case tested, BOTH approaches failed identically (all 3
# backbones agreed confidently and were all wrong) — so the heavier option
# doesn't demonstrably buy more protection against the failure people
# actually worry about, while committing real infra cost. This signal is
# still useful for the different, real case of backbones genuinely
# disagreeing, so it ships as the default rather than staying an unused
# opt-in flag.
#
# Default ON here -- but the actual deployed skinanalytica-api service
# (render.yaml) explicitly PINS THIS FALSE, overriding the default. Reason:
# _load_disagreement_backbones() loads the fp32 individual backbone files
# (ViT-L alone is ~1.2GB, ConvNeXt-Large ~785MB, EfficientNetV2-S ~81MB --
# ~2.1GB combined) IN ADDITION to whatever MODEL_MODE already loads (the
# int8 ensemble, ~524MB, in "full" mode's first candidate) -- a ~2.6GB+
# total footprint that would almost certainly OOM-crash Render's free tier.
# Turning this on for real on that service would need either a larger Render
# plan or switching _load_disagreement_backbones() to the existing int8
# backbone files (onnx_int8/, ~523MB combined) -- neither done here; this
# default change is for local dev and any future adequately-provisioned
# deployment, not a silent change to the live constrained one.
#
# Known limitation either way, tested this session (see docs/MODEL_CARD.md
# finding #7): on the one real false-negative case tried, all 3 backbones
# agreed confidently and were all wrong — this signal will NOT catch that
# class of error. It's still useful for cases where backbones genuinely
# disagree, just not a safety net for confidently-wrong-in-unison mistakes.
ENABLE_UNCERTAINTY = os.environ.get("SKINANALYTICA_ENABLE_UNCERTAINTY", "true").lower() == "true"

# ── Auth ──────────────────────────────────────────────────────────
# GET /report/{scan_id}, /report/{scan_id}/fhir, and
# /patient/{patient_id}/history were previously unauthenticated --
# /patient/{patient_id}/history would return a patient's full scan history
# to anyone who knew or guessed an id, with no credential required at all.
# This was the top-priority item flagged across two independent audit
# passes; gated here.
#
# Deliberately fails CLOSED, not open: if SKINANALYTICA_API_KEY is unset,
# every protected request is rejected (503) rather than silently allowed
# through. The opposite behavior -- a misconfigured deploy quietly running
# with no auth at all -- is exactly the gap this is meant to close, and a
# fail-open design would let that happen again silently, the same failure
# shape as MODEL_MODE's old wrong default (see docs/MODEL_CARD.md finding
# #10) and _detect_base()'s old fragile path check.
#
# UPDATE (2026-07-27): /analyze, /analyze/batch, and /audit/log are now
# gated too. The original reasoning for leaving them open was real -- the
# old frontend called them directly from browser JS with no way to hold a
# server secret client-side, and embedding SKINANALYTICA_API_KEY in public
# page source would have been fake security, not real protection. The new
# frontend (index.html/docs.html/status.html) resolves this the way
# Swagger's own "Authorize" flow does: the API key is a CLIENT-SUPPLIED
# credential the user types in and the browser holds in localStorage for
# that browser only -- never shipped in page source, never a secret this
# codebase owns or embeds. See the key-entry UI in index.html/status.html.
# A misconfigured deploy still fails closed exactly as below.
API_KEY = os.environ.get("SKINANALYTICA_API_KEY")

async def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """FastAPI dependency gating every patient-data endpoint (analyze,
    analyze/batch, report, fhir, patient history, audit log) -- see the
    scope note above for the history of why some of these were exempted
    and how that got closed. Uses secrets.compare_digest for a
    timing-safe comparison rather than `==` -- naive string equality on a
    secret is a real (if narrow) side-channel, and there's no reason to
    accept that risk when the fix is free."""
    if not API_KEY:
        raise HTTPException(503, "Server misconfigured: SKINANALYTICA_API_KEY not set")
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Missing or invalid API key (X-API-Key header required)")

# ── Allowed origins ───────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    # New static frontend (index.html/docs.html/status.html), served via
    # `python -m http.server 8080` in local dev -- see .claude/launch.json.
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    # Vercel — all variants
    "https://skinanalytica.vercel.app",
    "https://skin-analytica.vercel.app",
    "https://skinanalytica-git-main-tejansree21.vercel.app",
    # HuggingFace Spaces
    "https://tejansree-neuroscope-ai.hf.space",
    # Render services
    "https://skinanalytica-api.onrender.com",
    "https://skinanalytica-assistant.onrender.com",
    "https://skinanalytica-delivery.onrender.com",
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
    allow_origin_regex= r"https://.*\.vercel\.app|https://.*\.hf\.space",
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

        self.disagreement_sessions = {}
        if ENABLE_UNCERTAINTY:
            self._load_disagreement_backbones()

    def _load_disagreement_backbones(self):
        import onnxruntime as ort
        onnx_dir = os.path.join(PROD, "onnx")
        backbones = {
            "efficientnetv2-s": "skin_efficientnetv2-s.onnx",
            "vit-large-patch16-224": "skin_vit-large-patch16-224.onnx",
            "convnext-large": "skin_convnext-large.onnx",
        }
        for name, fname in backbones.items():
            path = os.path.join(onnx_dir, fname)
            if not os.path.exists(path):
                logger.warning(f"Uncertainty backbone missing, skipping: {fname}")
                continue
            try:
                sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                self.disagreement_sessions[name] = sess
            except Exception as e:
                logger.warning(f"Could not load uncertainty backbone {fname}: {e}")
        if self.disagreement_sessions:
            logger.info(f"Uncertainty signal enabled: {len(self.disagreement_sessions)} backbones loaded")

    def predict(self, arr: np.ndarray) -> np.ndarray:
        if not self.session:
            raise RuntimeError("No model loaded")
        return run_session(self.session, self.inp_name, arr, self.temperature)

    def predict_disagreement(self, arr: np.ndarray) -> Optional[float]:
        """Std of mel_score across the 3 individual backbones — an uncertainty
        proxy that doesn't need MC-dropout/PyTorch. Returns None if disabled
        or backbones failed to load. See ENABLE_UNCERTAINTY's docstring for
        this signal's known blind spot."""
        if not self.disagreement_sessions:
            return None
        mel_scores = []
        for sess in self.disagreement_sessions.values():
            inp_name = sess.get_inputs()[0].name
            probs = run_session(sess, inp_name, arr, self.temperature)
            mel_scores.append(float(probs[0][0]))
        return float(np.std(mel_scores))

model_reg = ModelRegistry()

# ── Preprocessing ─────────────────────────────────────────────────
def preprocess(img_bytes: bytes) -> np.ndarray:
    """Decode + delegate to inference_utils.preprocess_bgr(), the single
    source of truth for hair-removal/center-crop/resize/normalize -- this
    used to duplicate that logic inline, which is exactly the kind of
    per-call-site drift the module docstring on inference_utils.py warns
    against. Picks up the 2026-07-28 center-crop change automatically."""
    import cv2
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return preprocess_bgr(img)

# -- Verdict logic --------------------------------------------------
# Single source of truth: src/verdict_logic.py (docs/MODEL_CARD.md finding
# #29) -- this used to be a ~210-line inline block duplicated (and once
# already silently desynced, finding #27) in scripts/evaluate_fusion.py.
# Every threshold/routing constant below is re-exported here unchanged so
# existing call sites (api.MEL_THRESHOLD, api.get_mel_threshold(...),
# api.make_verdict(...), etc.) keep working without modification.
from verdict_logic import (
    CANCER_CLASSES, REVIEW_CLASSES, MEL_THRESHOLD, MEL_REVIEW_BAND_FACTOR,
    AGE_BAND_THRESHOLDS, TBP_MEL_THRESHOLD, SKIN_TONE_THRESHOLDS,
    get_mel_threshold, make_verdict,
)


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
    # self-consistent (arithmetic-pooling) metrics -- matches the deployed
    # ensemble's actual math, not the originally-published geometric-pooling
    # numbers in ensemble_metrics.json. See docs/MODEL_CARD.md finding #10.
    metrics_path = os.path.join(PROD, "ensemble", "ensemble_metrics_selfconsistent.json")
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

@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze(
    file        : UploadFile = File(...),
    cancer_type : Optional[str] = None,
    patient_id  : Optional[str] = None,
    patient_age : Optional[int] = None,
    patient_sex : Optional[str] = None,
    capture_mode: Optional[str] = None,  # "tbp" for 3D-total-body-photography-style
                                          # crops -- see TBP_MEL_THRESHOLD. Defaults to
                                          # the standard dermatoscope-calibrated
                                          # thresholds when omitted.
    skin_tone_class: Optional[int] = None,  # 0-5 (MILK10k scale), only used if in
                                             # SKIN_TONE_THRESHOLDS (2/3/4) -- provisional,
                                             # see its comment. Omit to use age/global.
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

    # Non-dermoscopic-input guardrail (see check_dermatoscope_likelihood's
    # docstring for what this is and isn't -- an advisory heuristic, never
    # a hard gate). Decodes the raw bytes a second time deliberately,
    # rather than threading the array out of preprocess(), to avoid
    # touching that function's existing contract/tests for a check that
    # can fail open safely.
    image_type_check = None
    try:
        import cv2
        raw_arr = np.frombuffer(img_bytes, np.uint8)
        raw_img = cv2.imdecode(raw_arr, cv2.IMREAD_COLOR)
        if raw_img is not None:
            image_type_check = check_dermatoscope_likelihood(raw_img)
    except Exception as e:
        logger.warning(f"Dermatoscope-likelihood check failed (non-fatal): {e}")

    try:
        probs = model_reg.predict(arr)[0]
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    pred_idx   = int(probs.argmax())
    pred_class = UNIFIED_CLASSES[pred_idx]
    decision   = make_verdict(probs, pred_class, patient_age=patient_age, capture_mode=capture_mode,
                               skin_tone_class=skin_tone_class)
    scan_id    = str(uuid.uuid4())[:8]

    disagreement = None
    if ENABLE_UNCERTAINTY:
        try:
            disagreement = model_reg.predict_disagreement(arr)
        except Exception as e:
            logger.warning(f"Disagreement computation failed: {e}")

    latency_ms = int((time.time() - t0) * 1000)

    result = {
        "scan_id"       : scan_id,
        "verdict"       : decision["verdict"],
        "priority"      : decision["priority"],
        "pred_class"    : pred_class,
        "pred_class_full": CLASS_FULL[pred_class],
        "confidence"    : round(decision["confidence"], 4),
        "mel_score"     : round(decision["mel_score"], 4),
        "mel_threshold_used": decision["mel_threshold_used"],
        "uncertainty_std": round(disagreement, 4) if disagreement is not None else None,
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
        "patient_age"   : patient_age,
        "patient_sex"   : patient_sex,
        "capture_mode"  : capture_mode,
        "skin_tone_class": skin_tone_class,
        "image_type_check": image_type_check,
    }
    _save_scan(result)
    return result

@app.post("/analyze/batch", dependencies=[Depends(require_api_key)])
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

@app.get("/report/{scan_id}", dependencies=[Depends(require_api_key)])
async def get_report(scan_id: str):
    path = os.path.join(OUT_SCANS, f"{scan_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"Scan {scan_id} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@app.get("/report/{scan_id}/fhir", dependencies=[Depends(require_api_key)])
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

@app.get("/patient/{patient_id}/history", dependencies=[Depends(require_api_key)])
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

@app.get("/audit/log", dependencies=[Depends(require_api_key)])
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

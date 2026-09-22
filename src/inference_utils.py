"""
SkinAnalytica — inference_utils.py
Single source of truth for ONNX logits -> calibrated probabilities,
image preprocessing, and test-time augmentation.

Every place that runs the ensemble (API, agents, notebooks) must use
softmax_with_temperature() instead of re-deriving softmax inline —
skipping the temperature divide silently breaks any fixed-probability
threshold (e.g. the melanoma cutoff, api/SA05_api.py's MEL_THRESHOLD) while leaving AUC unaffected,
since AUC only depends on rank order, not the temperature-scaled scale.
"""
import os
import sys
import json
import glob
import numpy as np
import cv2

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def enable_onnx_cuda_dll_dirs() -> bool:
    """
    Windows only. Registers the DLL directories onnxruntime-gpu's CUDA
    execution provider needs (torch's bundled cuDNN 9.x/cuBLAS, plus the
    system CUDA 12.x toolkit's cudart/cublas) so InferenceSession(...,
    providers=["CUDAExecutionProvider", ...]) actually finds GPU support
    instead of silently falling back to CPU.

    Why this is needed instead of just setting PATH: Windows Python 3.8+
    no longer honors PATH for a native extension's own internal
    LoadLibrary calls (onnxruntime's CUDA provider .dll loading its own
    dependencies) — only os.add_dll_directory() reliably works. This was
    diagnosed by getting the same generic "Error 126: module not found"
    from three different onnxruntime-gpu versions (1.27.0 wanting
    CUDA 13.x/cuDNN 9.x, 1.18.1 wanting CUDA 11/12 + cuDNN 8.x, 1.20.1
    wanting CUDA 12.x/cuDNN 9.x — the one actually satisfiable here) before
    realizing PATH itself was the problem, not the DLLs' absence.

    Requires onnxruntime-gpu (not the plain onnxruntime the deployed API
    uses) pinned to a version targeting CUDA 12.x + cuDNN 9.x — 1.20.1 was
    confirmed working locally. Safe to call on a machine without a GPU or
    without these exact paths: silently does nothing if they don't exist.
    Call this once before creating any onnxruntime InferenceSession that
    should use CUDA. Not relevant to the deployed Render API (CPU-only
    regardless, plain onnxruntime, no Windows DLL search involved).
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return False
    candidates = []
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            candidates.append(torch_lib)
    except ImportError:
        pass
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    for bin_dir in sorted(glob.glob(os.path.join(cuda_root, "v12*", "bin")), reverse=True):
        candidates.append(bin_dir)

    added_any = False
    for d in candidates:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                added_any = True
            except OSError:
                pass
    return added_any


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling then softmax. logits: [N, C] or [C]."""
    scaled = logits / temperature
    exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def load_temperature(prod_dir: str, default: float = 0.4095) -> float:
    """Load the calibrated ensemble temperature from models/production/ensemble/temperature.json."""
    path = os.path.join(prod_dir, "ensemble", "temperature.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("temperature", default)
    return default


def run_session(session, inp_name: str, arr: np.ndarray, temperature: float) -> np.ndarray:
    """
    Run an ONNX session and return calibrated probabilities, regardless of
    whether the loaded graph outputs raw logits (the individual backbone
    files: skin_efficientnetv2-s.onnx, skin_vit-large-patch16-224.onnx,
    skin_convnext-large.onnx) or already-finished probabilities (the
    combined skinanalytica_ensemble.onnx / *_int8.onnx graphs).

    The combined graph bakes in per-model softmax, temperature division,
    AND weighted averaging at export time (see notebooks/SA02_Ensemble.ipynb
    cell 13/14's EnsembleModel.forward and torch.onnx.export(...,
    output_names=["probs"])) — its raw output already sums to ~1.0.
    Applying softmax_with_temperature to that output a second time silently
    distorts every threshold-based number (sensitivity, verdict routing)
    while leaving AUC looking fine (rank-invariant) — the same failure
    shape as the original bias-report bug this session started with.

    Detecting via the output tensor name is deliberate: every prior fix in
    this codebase assumed callers would remember which file needs which
    treatment, and that assumption is exactly what broke here. Do not add
    a new call site that reimplements this decision inline.
    """
    out = session.run(None, {inp_name: arr})[0]
    out_name = session.get_outputs()[0].name
    if out_name == "probs":
        return out
    return softmax_with_temperature(out, temperature)


def check_dermatoscope_likelihood(img_bgr: np.ndarray) -> dict:
    """
    Lightweight heuristic guardrail -- NOT a trained classifier -- for
    flagging images that are extremely unlikely to be dermatoscope
    captures, so a user doesn't get a confident-looking classification on
    a badly out-of-scope input silently. SkinAnalytica is dermatoscope-only
    (see docs/MODEL_CARD.md finding #9 -- zero validated performance on
    other modalities) but has no reliable way to actually detect that at
    inference time -- this checks only aspect ratio, and only flags
    genuinely extreme cases, not "looks unusual."

    HONEST NEGATIVE RESULT this heuristic used to include and no longer
    does: a corner-vs-center brightness "vignette" check, on the theory
    that a dermatoscope's optical housing darkens image corners. Verified
    against this project's own real data before shipping and found
    UNRELIABLE -- the signal is dominated by lesion pigmentation (a dark
    lesion in the center reads as a "vignette" even with no vignette at
    all), not by capture equipment, and swung both directions (+143 to
    -85 on a 15-image real ISIC sample) with no consistent sign. Dropped
    entirely rather than shipped as a false signal.

    The aspect-ratio threshold below (2.2) was calibrated against actual
    images from every dataset this project uses, not assumed: SLICE-3D
    crops are 1.00 (square), MILK10k/most ISIC sources are 1.33 (4:3),
    ISIC 2020 goes up to 1.78 (some capture devices produce near-16:9
    frames). 2.2 sits safely above all of that, so this will not
    false-positive on any real dermoscopy image sampled from this
    project's own data -- but it also means this only catches genuinely
    extreme cases (a wide panorama, an unusually letterboxed photo), not
    "an ordinary 4:3 or 16:9 phone photo," since those aspect ratios
    overlap real dermatoscope output almost completely. This is a narrow,
    low-confidence check, not a real image-type classifier -- treat the
    output as an advisory hint in the UI, never as a hard gate on
    inference itself.
    """
    h, w = img_bgr.shape[:2]
    aspect_ratio = max(h, w) / max(1, min(h, w))
    is_extreme_aspect_ratio = aspect_ratio > 2.2

    likely_dermatoscope = not is_extreme_aspect_ratio

    return {
        "likely_dermatoscope": likely_dermatoscope,
        "is_extreme_aspect_ratio": is_extreme_aspect_ratio,
        "aspect_ratio": round(aspect_ratio, 2),
    }


def preprocess_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Hair removal + center-crop + resize + ImageNet normalize. Returns [1,3,H,W] float32.

    Center-crop-to-square (shorter side) before the resize-to-224 step was
    shipped to production 2026-07-28 -- see docs/MODEL_CARD.md finding #24
    and scripts/ablation_centercrop_mra_midas.py / ablation_centercrop_milk10k.py
    for the validation behind it: a real gain on MRA-MIDAS's huge,
    small-lesion-in-frame photos (+0.0146 to +0.0209 AUC depending on
    metric) with negligible cost on MILK10k's already-well-performing
    population (-0.002 to -0.005 AUC). A no-op for already-square input
    (SLICE-3D's 1.00 aspect ratio crops). Every caller of this function
    (the live /analyze endpoint, score_dataset.py, extract_embeddings.py)
    picks this up automatically -- do not re-derive preprocessing inline
    elsewhere (see module docstring). NOTE: MEL_THRESHOLD/TBP_MEL_THRESHOLD/
    skin_tone_class thresholds in api/SA05_api.py were calibrated under the
    OLD naive-resize behavior and have not been re-derived against this
    center-crop -- see docs/KNOWN_GAPS.md, flagged there as an open gap,
    not silently resolved by this change.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    bhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thr = cv2.threshold(bhat, 10, 255, cv2.THRESH_BINARY)
    img = cv2.inpaint(img_bgr, thr, 1, cv2.INPAINT_TELEA)

    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top:top + side, left:left + side]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def detect_lesion_bbox(img_bgr: np.ndarray, downscale_to: int = 512,
                        padding_frac: float = 0.20, min_area_frac: float = 0.0015,
                        max_area_frac: float = 0.85) -> tuple:
    """
    Locates the lesion within img_bgr, returns a SQUARE bounding box
    (x1, y1, x2, y2) in ORIGINAL image coordinates, padded around the
    detected lesion. Classical CV, no trained model -- see
    docs/MODEL_CARD.md finding #21 for why this exists: MRA-MIDAS's
    dermatoscope-attachment photos are huge (median 3024x4032) with the
    lesion occupying a small fraction of the frame inside the
    dermatoscope's circular field of view; a naive full-frame resize
    wastes almost all the available detail on background. A simple
    center-crop ablation already confirmed the mechanism (+0.021 MelAUC,
    not fully CI-confirmed at n=786) -- this aims to do better than
    "assume the lesion is near-center" by actually finding it.

    Two-stage classical approach:
    1. Find the "content area" -- excludes a dark circular vignette if
       one is present (common in phone-through-dermatoscope photos), a
       no-op if the frame has no such border (e.g. ISIC-style images).
    2. Within the content area, Otsu-threshold on grayscale (lesions are
       typically darker/more pigmented than surrounding skin), then pick
       the largest contour that's reasonably central and reasonably
       compact -- guards against a ruler/scale overlay or a hair clump
       being mistaken for the lesion (both tend to sit near an edge and/or
       be thin and elongated, scored low by the centrality/compactness
       terms below).

    Falls back to a center square crop of the content area -- the same
    fallback the center-crop ablation already validated as safe -- when
    no contour clears the area/shape thresholds. Detection runs on a
    downscaled copy for speed (the bbox is scaled back up to full
    resolution); on a huge source image this is also considerably faster
    than the original pipeline, since the expensive hair-removal inpaint
    step downstream now runs on a small crop instead of the full frame.

    NOT wired into preprocess_bgr() or any production path -- see
    preprocess_bgr_lesion_crop() below for the paired preprocessing
    function, itself not yet used by api/SA05_api.py. Opt-in only, same
    discipline as capture_mode/skin_tone_class -- nothing calls this yet.
    """
    h0, w0 = img_bgr.shape[:2]
    scale = downscale_to / max(h0, w0) if max(h0, w0) > downscale_to else 1.0
    small = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale))) if scale < 1.0 else img_bgr.copy()
    hs, ws = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bright_mask = gray > 15  # excludes a near-black vignette/background, if present
    if bright_mask.sum() < 0.05 * hs * ws:
        cx1, cy1, cx2, cy2 = 0, 0, ws, hs  # degenerate (near-all-black) -- use the whole frame
    else:
        ys, xs = np.where(bright_mask)
        cx1, cy1, cx2, cy2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    content = small[cy1:cy2, cx1:cx2]
    ch, cw = content.shape[:2]

    cgray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    cgray = cv2.GaussianBlur(cgray, (5, 5), 0)
    _, mask = cv2.threshold(cgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = ch * cw
    frame_cx, frame_cy = cw / 2, ch / 2
    max_dist = (frame_cx ** 2 + frame_cy ** 2) ** 0.5 or 1.0

    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        area_frac = area / frame_area if frame_area else 0
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bx, by = x + w / 2, y + h / 2
        centrality = 1 - (((bx - frame_cx) ** 2 + (by - frame_cy) ** 2) ** 0.5) / max_dist
        compactness = min(w, h) / max(w, h) if max(w, h) else 0
        score = area_frac * 0.4 + centrality * 0.4 + compactness * 0.2
        if score > best_score:
            best_score, best = score, (x, y, x + w, y + h)

    if best is None:
        side = min(ch, cw)
        fx1, fy1 = (cw - side) // 2, (ch - side) // 2
        best = (fx1, fy1, fx1 + side, fy1 + side)

    bx1, by1, bx2, by2 = best
    pad_w, pad_h = int((bx2 - bx1) * padding_frac), int((by2 - by1) * padding_frac)
    bx1, by1 = max(0, bx1 - pad_w), max(0, by1 - pad_h)
    bx2, by2 = min(cw, bx2 + pad_w), min(ch, by2 + pad_h)

    side = max(bx2 - bx1, by2 - by1)  # force square, same reasoning as the center-crop ablation
    ccx, ccy = (bx1 + bx2) // 2, (by1 + by2) // 2
    bx1, by1 = max(0, ccx - side // 2), max(0, ccy - side // 2)
    bx2, by2 = min(cw, bx1 + side), min(ch, by1 + side)
    bx1, by1 = max(0, bx2 - side), max(0, by2 - side)  # re-clamp if the frame edge was hit

    inv_scale = 1.0 / scale
    ox1 = max(0, int((cx1 + bx1) * inv_scale))
    oy1 = max(0, int((cy1 + by1) * inv_scale))
    ox2 = min(w0, int((cx1 + bx2) * inv_scale))
    oy2 = min(h0, int((cy1 + by2) * inv_scale))
    return ox1, oy1, ox2, oy2


def preprocess_bgr_lesion_crop(img_bgr: np.ndarray) -> np.ndarray:
    """detect_lesion_bbox() + the same hair-removal/resize/normalize as
    preprocess_bgr() -- but applied to the detected crop, not the full
    frame. Not called by anything in api/SA05_api.py yet; exists for
    ablation testing (scripts/ablation_lesioncrop_mra_midas.py) before any
    production decision. See preprocess_bgr()'s docstring for why this is
    a separate function rather than a change to it."""
    x1, y1, x2, y2 = detect_lesion_bbox(img_bgr)
    cropped = img_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = img_bgr
    return preprocess_bgr(cropped)


def tta_variants(img_bgr: np.ndarray) -> list:
    """
    Generate mild augmented copies of a raw BGR image for test-time
    augmentation. Perturbations are deliberately small (sub-clinical) —
    the goal is to smooth out prediction instability from incidental
    capture conditions (mild blur, compression, lighting), not to test
    against them. See docs/KNOWN_GAPS.md for the difference vs the
    robustness stress-test degradations.
    """
    variants = [img_bgr]
    variants.append(cv2.flip(img_bgr, 1))  # horizontal flip
    variants.append(cv2.GaussianBlur(img_bgr, (3, 3), 0))  # mild blur
    bright = np.clip(img_bgr.astype(np.float32) * 1.08 + 5, 0, 255).astype(np.uint8)
    variants.append(bright)
    return variants


def predict_tta(session, inp_name: str, temperature: float, img_bgr: np.ndarray):
    """
    Run the ensemble over TTA variants of one image and average the
    calibrated probabilities. Returns (mean_probs, std_probs) — std_probs
    is a per-class measure of how much the verdict wobbles under mild
    input perturbation, usable as an uncertainty signal distinct from
    inter-model disagreement or MC-dropout.
    """
    variants = tta_variants(img_bgr)
    batch = np.concatenate([preprocess_bgr(v) for v in variants], axis=0)
    probs = run_session(session, inp_name, batch, temperature)
    return probs.mean(axis=0), probs.std(axis=0)

"""
SkinAnalytica — data_layer.py
Queries saved JSON outputs to answer data-grounded questions.
"""

import os, json, glob
from datetime import datetime, timedelta
from pathlib import Path

BASE    = os.environ.get("SKINANALYTICA_BASE",
          r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica")
PROD    = os.path.join(BASE, "models", "production")
OUTPUTS = os.path.join(BASE, "outputs")


def _load_latest(folder: str, pattern: str = "*.json") -> dict:
    """Load the most recently modified JSON file matching pattern."""
    files = sorted(
        glob.glob(os.path.join(folder, pattern)),
        key=os.path.getmtime, reverse=True
    )
    if not files:
        return {}
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


def _load_all(folder: str, pattern: str = "*.json") -> list:
    files = sorted(glob.glob(os.path.join(folder, pattern)),
                   key=os.path.getmtime, reverse=True)
    results = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                results.append(json.load(f))
        except:
            continue
    return results


# ── Q01: Melanoma flagged count ───────────────────────────────────
def get_melanoma_flagged_count(days_back: int = 7) -> dict:
    scans_dir = os.path.join(OUTPUTS, "scan_results")
    if not os.path.exists(scans_dir):
        return {"count": 0, "note": "No scan results found"}
    cutoff = datetime.now() - timedelta(days=days_back)
    count  = 0
    scans  = []
    for fp in glob.glob(os.path.join(scans_dir, "*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                r = json.load(f)
            ts = datetime.fromisoformat(r.get("timestamp","2000-01-01"))
            if ts >= cutoff and r.get("verdict") == "CANCER_FLAGGED":
                count += 1
                scans.append({"scan_id": r.get("scan_id"),
                               "pred_class": r.get("pred_class"),
                               "confidence": r.get("confidence"),
                               "timestamp": r.get("timestamp")})
        except:
            continue
    return {"count": count, "days_back": days_back, "scans": scans[:10]}


# ── Q02: False negative rate from last batch ─────────────────────
def get_last_batch_fnr() -> dict:
    metrics_path = os.path.join(PROD, "ensemble", "ensemble_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        return {
            "fnr"        : m.get("fnr"),
            "sensitivity": m.get("sensitivity"),
            "source"     : "ensemble_metrics",
        }
    ver = _load_latest(os.path.join(OUTPUTS, "verification_reports"))
    if ver:
        flags = ver.get("flags", {})
        total = ver.get("total_images", 1)
        return {"flags": flags, "total": total, "source": "verification_report"}
    return {"note": "No batch results found"}


# ── Q03: Confidence by anatomical site ───────────────────────────
def get_confidence_by_site() -> dict:
    bias = _load_latest(os.path.join(OUTPUTS, "bias_reports"), "*bias*.json")
    if not bias:
        return {"note": "No bias report found — run SA02b_BiasReport first"}
    by_site = bias.get("fairness_results", {}).get("by_site",
              bias.get("by_site", {}))
    if not by_site:
        return {"note": "No site data in bias report"}
    sorted_sites = sorted(
        [(k,v) for k,v in by_site.items() if isinstance(v,dict) and v.get("mel_auc")],
        key=lambda x: x[1].get("mel_auc", 1.0)
    )
    return {"by_site": dict(sorted_sites), "lowest": sorted_sites[:3] if sorted_sites else []}


# ── Q04: Annotation conflicts ─────────────────────────────────────
def get_annotation_conflicts() -> dict:
    reports = _load_all(os.path.join(OUTPUTS, "verification_reports"))
    all_conflicts = []
    for r in reports:
        conflicts = r.get("conflicts", [])
        all_conflicts.extend(conflicts)
    return {"total_conflicts": len(all_conflicts),
            "conflicts": all_conflicts[:20]}


# ── Q05: Low confidence rate ──────────────────────────────────────
def get_low_confidence_rate() -> dict:
    report = _load_latest(os.path.join(OUTPUTS, "verification_reports"))
    if not report:
        return {"note": "No verification report found"}
    flags = report.get("flags", {})
    total = report.get("total_images", 1)
    low   = flags.get("LOW_CONFIDENCE", 0)
    return {
        "low_confidence_count": low,
        "total": total,
        "rate": round(low / max(total,1), 4),
        "flags": flags,
    }


# ── Q06: Current ensemble metrics ────────────────────────────────
def get_ensemble_metrics() -> dict:
    path = os.path.join(PROD, "ensemble", "ensemble_metrics.json")
    if not os.path.exists(path):
        return {"note": "ensemble_metrics.json not found — run SA02"}
    with open(path) as f:
        return json.load(f)


# ── Q07: Drift warnings ───────────────────────────────────────────
def get_drift_status() -> dict:
    report = _load_latest(os.path.join(OUTPUTS, "drift_reports"))
    if not report:
        return {"note": "No drift report found — run AgentRunner first"}
    return {
        "severity": report.get("severity"),
        "flags"   : report.get("flags", []),
        "kl_div"  : report.get("kl_divergence"),
        "conf_mean": report.get("conf_mean"),
    }


# ── Q08: Active learning queue ────────────────────────────────────
def get_active_learning_queue() -> dict:
    report = _load_latest(os.path.join(OUTPUTS, "active_learning"))
    if not report:
        return {"note": "No active learning report found"}
    return {
        "top_n"          : report.get("top_n"),
        "total_scored"   : report.get("total_scored"),
        "est_auc_gain"   : report.get("estimated_auc_gain"),
        "class_breakdown": report.get("class_breakdown"),
        "top_10"         : report.get("top_uncertain", [])[:10],
    }


# ── Q09: Calibration status ───────────────────────────────────────
def get_calibration_status() -> dict:
    report = _load_latest(os.path.join(OUTPUTS, "calibration_reports"))
    if not report:
        return {"note": "No calibration report found — run CalibrationAgent"}
    return {
        "ece"           : report.get("ece"),
        "overconfidence": report.get("overconfidence"),
        "needs_recal"   : report.get("needs_recal"),
        "recommendation": report.get("recommendation"),
        "current_T"     : report.get("current_T"),
        "recommended_T" : report.get("recommended_T"),
    }


# ── Q10: Sex disparity ────────────────────────────────────────────
def get_sex_disparity() -> dict:
    bias = _load_latest(os.path.join(OUTPUTS, "bias_reports"), "*bias*.json")
    if not bias:
        return {"note": "No bias report found — run SA02b_BiasReport"}
    by_sex = bias.get("by_sex", {})
    if not by_sex:
        by_sex = bias.get("fairness_results", {}).get("by_sex", {})
    return {"by_sex": by_sex}


# ── Router: map question to data function ─────────────────────────
DATA_ROUTES = {
    "Q01": get_melanoma_flagged_count,
    "Q02": get_last_batch_fnr,
    "Q03": get_confidence_by_site,
    "Q04": get_annotation_conflicts,
    "Q05": get_low_confidence_rate,
    "Q06": get_ensemble_metrics,
    "Q07": get_drift_status,
    "Q08": get_active_learning_queue,
    "Q09": get_calibration_status,
    "Q10": get_sex_disparity,
}

def query_data(question_id: str, **kwargs) -> dict:
    """Route a question ID to its data function."""
    fn = DATA_ROUTES.get(question_id)
    if not fn:
        return {"error": f"No data route for {question_id}"}
    return fn(**kwargs)


if __name__ == "__main__":
    print("Q06 — Ensemble metrics:")
    print(json.dumps(get_ensemble_metrics(), indent=2))
    print("\nQ07 — Drift status:")
    print(json.dumps(get_drift_status(), indent=2))

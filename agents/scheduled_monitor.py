"""
SkinAnalytica — scheduled_monitor.py
Runs drift + proxy-fairness monitoring against real accumulated production
scan results. Nothing in this repo previously ran continuously — every
agent (drift_monitor_agent, fairness_monitor_agent, etc.) only ran against
manually-supplied demo batches via its own `if __name__ == "__main__"` block.
This script is meant to be invoked on a schedule (see the `cron` service in
render.yaml) against outputs/scan_results/, which SA05_api.py's /analyze
endpoint writes to on every real request.
"""
import os
import sys
import json
from datetime import datetime, timedelta

BASE = os.environ.get(
    "SKINANALYTICA_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, os.path.join(BASE, "agents"))
from drift_monitor_agent import DriftMonitorAgent

SCAN_DIR = os.path.join(BASE, "outputs", "scan_results")
OUT_DIR = os.path.join(BASE, "outputs", "scheduled_monitoring")
os.makedirs(OUT_DIR, exist_ok=True)

AGE_BANDS = [(0, 30, "<30"), (30, 45, "30-45"), (45, 60, "45-60"),
             (60, 75, "60-75"), (75, 200, "75+")]


def load_recent_scans(lookback_hours: int = 24) -> list:
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    records = []
    if not os.path.exists(SCAN_DIR):
        return records
    for fname in os.listdir(SCAN_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SCAN_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        ts = r.get("timestamp")
        if ts:
            try:
                if datetime.fromisoformat(ts) < cutoff:
                    continue
            except ValueError:
                pass
        records.append(r)
    return records


def to_drift_record(scan: dict) -> dict:
    """Adapt SA05_api.py's scan_results schema to what drift_monitor_agent expects."""
    return {
        "pred_class": scan.get("pred_class"),
        "confidence_score": (scan.get("confidence") or 0) * 10,
    }


def predicted_verdict_rate_by_age(records: list) -> dict:
    """
    Proxy fairness signal for LIVE unlabeled traffic.

    fairness_monitor_agent.py needs ground-truth human_label to compute
    real sensitivity per subgroup — that doesn't exist for real patient
    scans (there's no annotator confirming the true diagnosis at inference
    time). What CAN be computed continuously, without labels, is whether
    the CANCER_FLAGGED rate diverges sharply across age bands. A gap here
    is not proof of a sensitivity gap (that requires the labeled bias
    report in outputs/bias_reports/) — it's an early-warning proxy that's
    at least computable on real traffic instead of nothing at all.
    """
    buckets = {label: {"n": 0, "flagged": 0} for _, _, label in AGE_BANDS}
    for r in records:
        age = r.get("patient_age")
        if age is None:
            continue
        for lo, hi, label in AGE_BANDS:
            if lo <= age < hi:
                buckets[label]["n"] += 1
                if r.get("verdict") == "CANCER_FLAGGED":
                    buckets[label]["flagged"] += 1
                break
    out = {}
    for label, d in buckets.items():
        out[label] = {
            "n": d["n"],
            "flagged_rate": round(d["flagged"] / d["n"], 4) if d["n"] else None,
        }
    return out


def main(lookback_hours: int = 24) -> dict:
    records = load_recent_scans(lookback_hours)
    print(f"Loaded {len(records)} scans from the last {lookback_hours}h")

    if not records:
        report = {"status": "no_data", "timestamp": datetime.now().isoformat()}
        print("No recent scans — nothing to monitor")
    else:
        drift_records = [to_drift_record(r) for r in records]
        agent = DriftMonitorAgent()
        drift_result = agent.run(records=drift_records, output_name="scheduled")
        age_rates = predicted_verdict_rate_by_age(records)

        report = {
            "timestamp": datetime.now().isoformat(),
            "lookback_hours": lookback_hours,
            "n_scans": len(records),
            "drift": drift_result,
            "predicted_cancer_flagged_rate_by_age": age_rates,
            "note": ("Age-band rates are a proxy signal only — true sensitivity "
                     "by subgroup requires labeled validation data "
                     "(outputs/bias_reports/), not live unlabeled traffic."),
        }
        if drift_result.get("severity") not in (None, "OK"):
            print(f"*** DRIFT ALERT: severity={drift_result.get('severity')} "
                  f"flags={drift_result.get('flags')}")

    out_path = os.path.join(OUT_DIR, f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lookback-hours", type=int, default=24)
    args = p.parse_args()
    main(args.lookback_hours)

"""
SkinAnalytica — drift_monitor_agent.py
Detects model drift and OOD batches after each inference run.
"""

import os, json
import numpy as np
from datetime import datetime
from base_agent import BaseAgent, BASE

OUT_DIR        = os.path.join(BASE, "outputs", "drift_reports")
UNIFIED_CLASSES= ["mel","nv","bcc","akiec","bkl","df","vasc"]

# Reference distribution from training (ISIC 2018+2019+2020 merged)
TRAIN_DIST = {
    "mel":0.0909, "nv":0.7587, "bcc":0.0561,
    "akiec":0.0266, "bkl":0.0571, "df":0.0052, "vasc":0.0058
}
TRAIN_CONF_MEAN = 0.82
TRAIN_CONF_STD  = 0.14

class DriftMonitorAgent(BaseAgent):
    """
    Runs after each batch. Detects:
      1. Distribution shift (KL divergence vs training dist)
      2. Confidence distribution shift
      3. OOD signals (unusual class frequencies)
      4. Melanoma rate anomalies
    """

    def __init__(self):
        super().__init__("drift_monitor_agent")
        os.makedirs(OUT_DIR, exist_ok=True)

    def _kl_divergence(self, p: dict, q: dict) -> float:
        eps = 1e-8
        kl  = 0.0
        for cls in UNIFIED_CLASSES:
            pi = p.get(cls, eps)
            qi = q.get(cls, eps)
            if pi > 0:
                kl += pi * np.log(pi / max(qi, eps))
        return round(float(kl), 4)

    def _run(self, verification_report_path: str = None,
             records: list = None,
             output_name: str = "drift_check") -> dict:

        if records is None:
            if verification_report_path and os.path.exists(verification_report_path):
                with open(verification_report_path) as f:
                    data = json.load(f)
                records = data.get("records", [])
            else:
                raise ValueError("Provide records or verification_report_path")

        n = len(records)
        if n == 0:
            return {"status": "no_records"}

        self.logger.info(f"Checking drift on {n} images")

        # Batch class distribution
        class_counts = {c: 0 for c in UNIFIED_CLASSES}
        conf_scores  = []

        for rec in records:
            cls = rec.get("pred_class")
            if cls in class_counts:
                class_counts[cls] += 1
            cs = rec.get("confidence_score")
            if cs is not None:
                conf_scores.append(cs / 10.0)

        batch_dist = {c: class_counts[c] / n for c in UNIFIED_CLASSES}

        # KL divergence from training distribution
        kl_div = self._kl_divergence(batch_dist, TRAIN_DIST)

        # Confidence stats
        conf_mean = float(np.mean(conf_scores)) if conf_scores else 0.0
        conf_std  = float(np.std(conf_scores))  if conf_scores else 0.0
        conf_shift= abs(conf_mean - TRAIN_CONF_MEAN) / max(TRAIN_CONF_STD, 1e-6)

        # Melanoma rate
        mel_rate   = batch_dist.get("mel", 0.0)
        mel_ref    = TRAIN_DIST["mel"]
        mel_anomaly= abs(mel_rate - mel_ref) > 0.10

        # Flags
        flags   = []
        severity= "OK"

        if kl_div > 0.5:
            flags.append(f"HIGH_DISTRIBUTION_SHIFT: KL={kl_div:.3f}")
            severity = "HIGH"
        elif kl_div > 0.2:
            flags.append(f"MODERATE_DISTRIBUTION_SHIFT: KL={kl_div:.3f}")
            severity = "MEDIUM" if severity == "OK" else severity

        if conf_shift > 2.0:
            flags.append(f"CONFIDENCE_SHIFT: batch_mean={conf_mean:.2f} vs train={TRAIN_CONF_MEAN:.2f}")
            severity = "HIGH" if kl_div > 0.2 else "MEDIUM"

        if mel_anomaly:
            flags.append(f"MEL_RATE_ANOMALY: batch={mel_rate:.3f} vs train={mel_ref:.3f}")

        low_conf_pct = sum(1 for c in conf_scores if c < 0.65) / max(len(conf_scores),1)
        if low_conf_pct > 0.30:
            flags.append(f"HIGH_LOW_CONFIDENCE_RATE: {low_conf_pct:.1%} below 65%")

        result = {
            "severity"          : severity,
            "flags"             : flags,
            "kl_divergence"     : kl_div,
            "batch_distribution": batch_dist,
            "train_distribution": TRAIN_DIST,
            "conf_mean"         : round(conf_mean, 4),
            "conf_std"          : round(conf_std, 4),
            "conf_shift_z"      : round(conf_shift, 2),
            "mel_rate"          : round(mel_rate, 4),
            "mel_anomaly"       : mel_anomaly,
            "n_images"          : n,
        }

        # Save
        report_path = os.path.join(OUT_DIR, f"{output_name}_drift.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        icon = {"OK":"✅","MEDIUM":"🟡","HIGH":"🔴"}.get(severity,"⚠️")
        print(f"\nDRIFT MONITOR: {output_name}")
        print("=" * 50)
        print(f"  Severity    : {icon} {severity}")
        print(f"  KL div      : {kl_div:.4f}  (>0.2=medium, >0.5=high)")
        print(f"  Conf mean   : {conf_mean:.3f}  (train ref: {TRAIN_CONF_MEAN:.3f})")
        print(f"  Mel rate    : {mel_rate:.3f}  (train ref: {mel_ref:.3f})")
        if flags:
            print(f"\n  Flags:")
            for flag in flags:
                print(f"    ⚠ {flag}")
        else:
            print("  No drift detected ✅")
        print(f"\n  Report: {report_path}")

        return result


if __name__ == "__main__":
    agent  = DriftMonitorAgent()
    result = agent.run(
        verification_report_path = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\outputs\verification_reports\test_verification_verification.json",
        output_name = "test_drift",
    )

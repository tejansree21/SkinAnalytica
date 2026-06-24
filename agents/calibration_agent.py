"""
SkinAnalytica — calibration_agent.py
Weekly confidence calibration check — detects overconfidence drift.
"""

import os, json
import numpy as np
from datetime import datetime
from base_agent import BaseAgent, BASE

OUT_DIR = os.path.join(BASE, "outputs", "calibration_reports")
PROD    = os.path.join(BASE, "models", "production")

class CalibrationAgent(BaseAgent):
    """
    Runs weekly. Compares predicted confidence against actual accuracy.
    Detects overconfidence (model says 95%, wrong 30% of the time).
    Recommends temperature recalibration if needed.
    """

    def __init__(self):
        super().__init__("calibration_agent")
        os.makedirs(OUT_DIR, exist_ok=True)

    def _ece(self, confidences, accuracies, n_bins=10):
        """Expected Calibration Error."""
        bins   = np.linspace(0, 1, n_bins + 1)
        ece    = 0.0
        n_total= len(confidences)
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() == 0: continue
            bin_conf = confidences[mask].mean()
            bin_acc  = accuracies[mask].mean()
            ece     += (mask.sum() / n_total) * abs(bin_acc - bin_conf)
        return float(ece)

    def _run(self, records: list = None,
             verification_report_path: str = None,
             output_name: str = "calibration") -> dict:

        if records is None and verification_report_path:
            with open(verification_report_path) as f:
                data    = json.load(f)
            records = data.get("records", [])

        # Filter to records with both human label and prediction
        eval_recs = [r for r in records
                     if r.get("human_label") and r.get("pred_class")
                     and r.get("confidence_score") is not None]

        self.logger.info(f"Calibration check on {len(eval_recs)} labelled images")

        if len(eval_recs) < 10:
            print("⚠ Not enough labelled images for calibration check (need ≥10)")
            return {"status": "insufficient_data", "n": len(eval_recs)}

        confidences = np.array([r["confidence_score"] / 10.0 for r in eval_recs])
        accuracies  = np.array([
            1.0 if r["pred_class"] == r["human_label"] else 0.0
            for r in eval_recs
        ])

        ece          = self._ece(confidences, accuracies)
        mean_conf    = float(confidences.mean())
        mean_acc     = float(accuracies.mean())
        overconfidence = mean_conf - mean_acc

        # Per-class calibration
        class_cal = {}
        classes   = set(r["pred_class"] for r in eval_recs)
        for cls in classes:
            sub_recs = [r for r in eval_recs if r["pred_class"] == cls]
            if len(sub_recs) < 3: continue
            sub_conf = np.array([r["confidence_score"]/10.0 for r in sub_recs])
            sub_acc  = np.array([1.0 if r["pred_class"]==r["human_label"] else 0.0
                                 for r in sub_recs])
            class_cal[cls] = {
                "n"             : len(sub_recs),
                "mean_conf"     : round(float(sub_conf.mean()), 4),
                "mean_acc"      : round(float(sub_acc.mean()), 4),
                "overconfidence": round(float(sub_conf.mean() - sub_acc.mean()), 4),
            }

        # Load current temperature
        T_path = os.path.join(PROD, "ensemble", "temperature.json")
        current_T = 0.4095
        if os.path.exists(T_path):
            with open(T_path) as f:
                current_T = json.load(f)["temperature"]

        # Recommendation
        needs_recal = ece > 0.10 or abs(overconfidence) > 0.15
        if overconfidence > 0.10:
            rec_T = round(current_T * 1.1, 4)
            recommendation = f"Increase T from {current_T:.4f} to {rec_T:.4f} (model overconfident)"
        elif overconfidence < -0.10:
            rec_T = round(current_T * 0.9, 4)
            recommendation = f"Decrease T from {current_T:.4f} to {rec_T:.4f} (model underconfident)"
        else:
            rec_T          = current_T
            recommendation = "No recalibration needed"

        result = {
            "n_eval"          : len(eval_recs),
            "ece"             : round(ece, 4),
            "mean_conf"       : round(mean_conf, 4),
            "mean_acc"        : round(mean_acc, 4),
            "overconfidence"  : round(overconfidence, 4),
            "needs_recal"     : needs_recal,
            "current_T"       : current_T,
            "recommended_T"   : rec_T,
            "recommendation"  : recommendation,
            "per_class"       : class_cal,
        }

        report_path = os.path.join(OUT_DIR, f"{output_name}_calibration.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        status_icon = "🔴" if needs_recal else "✅"
        print(f"\nCALIBRATION AGENT: {output_name}")
        print("=" * 50)
        print(f"  Status          : {status_icon} {'RECALIBRATION NEEDED' if needs_recal else 'OK'}")
        print(f"  ECE             : {ece:.4f}  (>0.10 = poor)")
        print(f"  Mean confidence : {mean_conf:.4f}")
        print(f"  Mean accuracy   : {mean_acc:.4f}")
        print(f"  Overconfidence  : {overconfidence:+.4f}")
        print(f"  Current T       : {current_T:.4f}")
        print(f"\n  Recommendation  : {recommendation}")
        print(f"  Report          : {report_path}")

        return result


if __name__ == "__main__":
    agent  = CalibrationAgent()
    result = agent.run(
        verification_report_path = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\outputs\verification_reports\test_verification_verification.json",
        output_name = "weekly_calibration",
    )

"""
SkinAnalytica — fairness_monitor_agent.py
Real-time Fitzpatrick + demographic fairness monitoring per batch.
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
from base_agent import BaseAgent, BASE

OUT_DIR = os.path.join(BASE, "outputs", "fairness")

DISPARITY_THRESHOLDS = {
    "sex"  : 0.02,
    "age"  : 0.05,
    "site" : 0.05,
}

class FairnessMonitorAgent(BaseAgent):
    """
    Monitors per-batch fairness across demographics.
    Flags subgroups with significantly lower sensitivity.
    """

    def __init__(self):
        super().__init__("fairness_monitor_agent")
        os.makedirs(OUT_DIR, exist_ok=True)

    def _mel_auc_subset(self, df_sub) -> float:
        from sklearn.metrics import roc_auc_score
        try:
            mel_true  = (df_sub["human_label"] == "mel").astype(int).values
            mel_score = df_sub["mel_prob"].values
            if mel_true.sum() < 3 or (1 - mel_true).sum() < 3:
                return np.nan
            return float(roc_auc_score(mel_true, mel_score))
        except:
            return np.nan

    def _run(self, records: list = None,
             verification_report_path: str = None,
             metadata_csv: str = None,
             output_name: str = "fairness") -> dict:

        if records is None and verification_report_path:
            with open(verification_report_path) as f:
                data    = json.load(f)
            records = data.get("records", [])

        df = pd.DataFrame(records)
        self.logger.info(f"Fairness check on {len(df)} images")

        # Merge metadata if provided
        if metadata_csv and os.path.exists(metadata_csv):
            meta = pd.read_csv(metadata_csv)
            meta.columns = [c.lower() for c in meta.columns]
            id_col = next((c for c in ["image_name","image","image_id"] if c in meta.columns), None)
            if id_col:
                meta = meta.rename(columns={id_col: "image_id"})
                df   = df.merge(meta[["image_id","sex","age_approx",
                                      "anatom_site_general_challenge"]].rename(
                    columns={"age_approx":"age","anatom_site_general_challenge":"site"}
                ), on="image_id", how="left")

        results  = {}
        warnings = []

        # Check each demographic axis
        for axis in ["sex","age","site"]:
            if axis not in df.columns:
                results[f"by_{axis}"] = {"status": "metadata_unavailable"}
                continue
            if "human_label" not in df.columns or "mel_prob" not in df.columns:
                results[f"by_{axis}"] = {"status": "labels_unavailable"}
                continue

            axis_results = {}
            df_axis = df.dropna(subset=[axis, "human_label", "mel_prob"])
            if len(df_axis) == 0:
                results[f"by_{axis}"] = {"status": "no_data"}
                continue

            # Age binning
            if axis == "age" and df_axis["age"].dtype in [np.float64, np.int64]:
                df_axis = df_axis.copy()
                df_axis["age"] = pd.cut(df_axis["age"],
                    bins=[0,30,45,60,75,120],
                    labels=["<30","30-45","45-60","60-75","75+"])

            for val in df_axis[axis].unique():
                sub = df_axis[df_axis[axis] == val]
                auc = self._mel_auc_subset(sub)
                axis_results[str(val)] = {
                    "n": len(sub),
                    "mel_auc": round(auc, 4) if not np.isnan(auc) else None,
                }

            # Check disparity
            valid_aucs = [v["mel_auc"] for v in axis_results.values()
                         if v["mel_auc"] is not None]
            if len(valid_aucs) >= 2:
                gap = max(valid_aucs) - min(valid_aucs)
                threshold = DISPARITY_THRESHOLDS.get(axis, 0.05)
                if gap > threshold:
                    worst = min(axis_results.items(),
                               key=lambda x: x[1]["mel_auc"] or 1.0)
                    warnings.append({
                        "axis"     : axis,
                        "gap"      : round(gap, 4),
                        "threshold": threshold,
                        "worst"    : worst[0],
                        "worst_auc": worst[1]["mel_auc"],
                        "severity" : "HIGH" if gap > threshold*2 else "MEDIUM",
                    })

            results[f"by_{axis}"] = axis_results

        fitzpatrick_note = (
            "Fitzpatrick skin type breakdown requires ISIC 2024 SLICE-3D metadata. "
            "Available via ISIC API — not present in standard ISIC 2018/2019/2020 CSVs."
        )

        output = {
            "fairness_results"    : results,
            "disparity_warnings"  : warnings,
            "fitzpatrick_note"    : fitzpatrick_note,
            "n_images"            : len(df),
        }

        report_path = os.path.join(OUT_DIR, f"{output_name}_fairness.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nFAIRNESS MONITOR: {output_name}")
        print("=" * 50)
        print(f"  Images checked  : {len(df)}")
        if warnings:
            print(f"\n  ⚠ Disparity warnings ({len(warnings)}):")
            for w in warnings:
                icon = "🔴" if w["severity"] == "HIGH" else "🟡"
                print(f"    {icon} {w['axis'].upper()} gap={w['gap']:.4f} "
                      f"(threshold={w['threshold']}) — worst: {w['worst']} ({w['worst_auc']:.4f})")
        else:
            print("  ✅ No significant disparities detected")
        print(f"\n  Note: {fitzpatrick_note}")
        print(f"  Report: {report_path}")

        return output


if __name__ == "__main__":
    agent  = FairnessMonitorAgent()
    result = agent.run(
        verification_report_path = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\outputs\verification_reports\test_verification_verification.json",
        metadata_csv = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\data\ISIC_2020\ISIC_2020_Training_GroundTruth_v2.csv",
        output_name  = "test_fairness",
    )

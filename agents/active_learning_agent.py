"""
SkinAnalytica — active_learning_agent.py
Identifies top-N most uncertain images and sends to researcher annotation queue.
"""

import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from base_agent import BaseAgent, BASE

OUT_DIR = os.path.join(BASE, "outputs", "active_learning")

class ActiveLearningAgent(BaseAgent):
    """
    After each batch, identifies images that would most improve
    model performance if relabelled.
    Uses: low confidence + high class confusion + prediction entropy.
    """

    def __init__(self):
        super().__init__("active_learning_agent")
        os.makedirs(OUT_DIR, exist_ok=True)

    def _entropy(self, probs: dict) -> float:
        p = np.array(list(probs.values()))
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def _margin(self, probs: dict) -> float:
        sorted_p = sorted(probs.values(), reverse=True)
        return float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

    def _run(self, verification_report_path: str = None,
             records: list = None,
             top_n: int = 50,
             output_name: str = "active_learning") -> dict:

        # Load from verification report or accept records directly
        if records is None:
            if verification_report_path and os.path.exists(verification_report_path):
                with open(verification_report_path) as f:
                    data = json.load(f)
                records = data.get("records", [])
            else:
                raise ValueError("Provide either records or verification_report_path")

        self.logger.info(f"Scoring {len(records)} images for active learning")

        scored = []
        for rec in records:
            if "probs" not in rec or rec.get("flag") == "ERROR":
                continue
            probs   = rec["probs"]
            entropy = self._entropy(probs)
            margin  = self._margin(probs)
            # Uncertainty score: high entropy + low margin = most uncertain
            uncertainty = entropy * (1 - margin)

            scored.append({
                "image_id"        : rec.get("image_id"),
                "path"            : rec.get("path"),
                "pred_class"      : rec.get("pred_class"),
                "confidence_score": rec.get("confidence_score"),
                "flag"            : rec.get("flag"),
                "human_label"     : rec.get("human_label"),
                "entropy"         : round(entropy, 4),
                "margin"          : round(margin, 4),
                "uncertainty_score": round(uncertainty, 4),
                "mel_prob"        : rec.get("mel_prob"),
            })

        # Sort by uncertainty descending
        scored.sort(key=lambda x: x["uncertainty_score"], reverse=True)
        top_uncertain = scored[:top_n]

        # Estimate AUC gain (simplified heuristic)
        avg_uncertainty = np.mean([s["uncertainty_score"] for s in top_uncertain]) if top_uncertain else 0
        est_auc_gain    = round(avg_uncertainty * 0.01, 4)

        # Class breakdown of uncertain cases
        class_counts = {}
        for s in top_uncertain:
            cls = s.get("pred_class","unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1

        result = {
            "total_scored"       : len(scored),
            "top_n"              : top_n,
            "top_uncertain"      : top_uncertain,
            "class_breakdown"    : class_counts,
            "avg_uncertainty"    : round(avg_uncertainty, 4),
            "estimated_auc_gain" : est_auc_gain,
        }

        # Save annotation queue
        queue_path = os.path.join(OUT_DIR, f"{output_name}_annotation_queue.json")
        with open(queue_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        # CSV for easy review
        csv_path = os.path.join(OUT_DIR, f"{output_name}_annotation_queue.csv")
        pd.DataFrame(top_uncertain).to_csv(csv_path, index=False)

        print(f"\nACTIVE LEARNING QUEUE: {output_name}")
        print("=" * 50)
        print(f"  Total scored        : {len(scored)}")
        print(f"  Top-{top_n} selected  : {len(top_uncertain)}")
        print(f"  Est. AUC gain       : +{est_auc_gain}")
        print(f"\n  Class breakdown of uncertain cases:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<10} {count}")
        print(f"\n  Annotation queue    : {queue_path}")
        print(f"  CSV for review      : {csv_path}")

        return result


if __name__ == "__main__":
    agent  = ActiveLearningAgent()
    result = agent.run(
        verification_report_path = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\outputs\verification_reports\test_verification_verification.json",
        top_n       = 50,
        output_name = "test_al",
    )

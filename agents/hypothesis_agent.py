"""
SkinAnalytica — hypothesis_agent.py
Researcher states a hypothesis in plain English → agent tests it automatically.
"""

import os, json, re
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from base_agent import BaseAgent, BASE

OUT_DIR = os.path.join(BASE, "outputs", "hypothesis_tests")

# Pre-built question patterns
PATTERNS = [
    (r"worse.*(back|torso|trunk)",      "site", "torso"),
    (r"worse.*(face|head|neck)",         "site", "head/neck"),
    (r"worse.*(palm|sole|acral)",        "site", "palms/soles"),
    (r"worse.*(male|men|man)",           "sex",  "male"),
    (r"worse.*(female|women|woman)",     "sex",  "female"),
    (r"worse.*(old|elder|senior|75)",    "age_group", "75+"),
    (r"worse.*(young|youth|under.?30)",  "age_group", "<30"),
    (r"low.*(confidence|certain)",       "flag", "LOW_CONFIDENCE"),
    (r"(melanoma|mel).*(miss|false.?neg)","pred_class","nv"),
]

class HypothesisAgent(BaseAgent):
    """
    Accepts a hypothesis in plain English, tests it against batch results,
    returns structured answer with statistical significance.

    Example:
        agent.run("Does the model perform worse on lesions from the back?",
                  verification_report_path=...)
    """

    def __init__(self):
        super().__init__("hypothesis_agent")
        os.makedirs(OUT_DIR, exist_ok=True)

    def _parse_hypothesis(self, hypothesis: str):
        """Match hypothesis to a testable pattern."""
        h = hypothesis.lower()
        for pattern, axis, value in PATTERNS:
            if re.search(pattern, h):
                return axis, value
        return None, None

    def _test_hypothesis(self, df: pd.DataFrame, axis: str, value: str) -> dict:
        """Two-sample test: group matching value vs rest."""
        if axis not in df.columns:
            return {"error": f"Column '{axis}' not in data — need metadata CSV"}

        group_a = df[df[axis].astype(str).str.lower() == value.lower()]
        group_b = df[df[axis].astype(str).str.lower() != value.lower()]

        if len(group_a) < 5 or len(group_b) < 5:
            return {"error": f"Not enough samples: group_a={len(group_a)}, group_b={len(group_b)}"}

        score_a = group_a["confidence_score"].dropna().values / 10.0
        score_b = group_b["confidence_score"].dropna().values / 10.0

        t_stat, p_val = stats.ttest_ind(score_a, score_b)

        mel_auc_a = mel_auc_b = np.nan
        if "human_label" in df.columns and "mel_prob" in df.columns:
            from sklearn.metrics import roc_auc_score
            for grp, name in [(group_a, "a"), (group_b, "b")]:
                try:
                    mt = (grp["human_label"] == "mel").astype(int).values
                    ms = grp["mel_prob"].values
                    if mt.sum() >= 3 and (1-mt).sum() >= 3:
                        auc = roc_auc_score(mt, ms)
                        if name == "a": mel_auc_a = auc
                        else:           mel_auc_b = auc
                except: pass

        direction = "lower" if score_a.mean() < score_b.mean() else "higher"
        significant = p_val < 0.05

        return {
            "group_value"      : value,
            "n_group"          : len(group_a),
            "n_rest"           : len(group_b),
            "mean_conf_group"  : round(float(score_a.mean()), 4),
            "mean_conf_rest"   : round(float(score_b.mean()), 4),
            "direction"        : direction,
            "mel_auc_group"    : round(mel_auc_a, 4) if not np.isnan(mel_auc_a) else None,
            "mel_auc_rest"     : round(mel_auc_b, 4) if not np.isnan(mel_auc_b) else None,
            "t_statistic"      : round(float(t_stat), 4),
            "p_value"          : round(float(p_val), 4),
            "significant"      : significant,
        }

    def _run(self, hypothesis: str,
             records: list = None,
             verification_report_path: str = None,
             metadata_csv: str = None,
             output_name: str = None) -> dict:

        if records is None and verification_report_path:
            with open(verification_report_path) as f:
                data    = json.load(f)
            records = data.get("records", [])

        df = pd.DataFrame(records)
        self.logger.info(f"Testing: '{hypothesis}' on {len(df)} records")

        # Merge metadata
        if metadata_csv and os.path.exists(metadata_csv):
            meta = pd.read_csv(metadata_csv)
            meta.columns = [c.lower() for c in meta.columns]
            id_col = next((c for c in ["image_name","image","image_id"]
                          if c in meta.columns), None)
            if id_col:
                meta = meta.rename(columns={id_col:"image_id"})
                cols = [c for c in ["image_id","sex","age_approx",
                                    "anatom_site_general_challenge"]
                        if c in meta.columns]
                df   = df.merge(meta[cols].rename(
                    columns={"age_approx":"age",
                             "anatom_site_general_challenge":"site"}),
                    on="image_id", how="left")

        axis, value = self._parse_hypothesis(hypothesis)

        if axis is None:
            answer = ("I couldn't automatically interpret this hypothesis. "
                      "Try phrasing it as: 'Does the model perform worse on [X]?' "
                      "where X is a site, sex, or age group.")
            return {"hypothesis": hypothesis, "answer": answer, "tested": False}

        test_result = self._test_hypothesis(df, axis, value)

        if "error" in test_result:
            return {"hypothesis": hypothesis, "answer": test_result["error"],
                    "tested": False}

        # Plain English answer
        direction  = test_result["direction"]
        p_val      = test_result["p_value"]
        sig        = test_result["significant"]
        conf_group = test_result["mean_conf_group"]
        conf_rest  = test_result["mean_conf_rest"]
        diff       = abs(conf_group - conf_rest)

        if sig:
            answer = (
                f"Yes — the model shows {direction} confidence on {value} "
                f"({conf_group:.1%} vs {conf_rest:.1%}, Δ={diff:.1%}, "
                f"p={p_val:.3f}, statistically significant). "
                f"Based on {test_result['n_group']} {value} images vs "
                f"{test_result['n_rest']} others."
            )
            if test_result.get("mel_auc_group") and test_result.get("mel_auc_rest"):
                auc_diff = test_result["mel_auc_group"] - test_result["mel_auc_rest"]
                answer += (f" MelAUC: {test_result['mel_auc_group']:.4f} vs "
                           f"{test_result['mel_auc_rest']:.4f} (Δ={auc_diff:+.4f}).")
        else:
            answer = (
                f"No significant evidence — confidence on {value} "
                f"({conf_group:.1%}) is not meaningfully different from "
                f"the rest ({conf_rest:.1%}, p={p_val:.3f}, not significant)."
            )

        result = {
            "hypothesis"  : hypothesis,
            "axis_tested" : axis,
            "value_tested": value,
            "answer"      : answer,
            "tested"      : True,
            "test_result" : test_result,
        }

        out_name   = output_name or f"hypothesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_path= os.path.join(OUT_DIR, f"{out_name}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"\nHYPOTHESIS TEST")
        print("=" * 55)
        print(f"  Q: {hypothesis}")
        print(f"  A: {answer}")
        print(f"  Report: {report_path}")

        return result


if __name__ == "__main__":
    agent  = HypothesisAgent()
    result = agent.run(
        hypothesis = "Does the model perform worse on lesions from the head and neck?",
        verification_report_path = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\outputs\verification_reports\test_verification_verification.json",
        metadata_csv = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\data\ISIC_2020\ISIC_2020_Training_GroundTruth_v2.csv",
    )

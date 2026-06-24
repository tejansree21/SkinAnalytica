"""
SkinAnalytica — agent_runner.py
Orchestrates all 8 agents in pipeline order.
Can be triggered: on_upload | manual | scheduled
"""

import os, json
from datetime import datetime
from base_agent import BASE

class AgentRunner:
    """
    Runs agents in the correct pipeline order:
    1. SelfHealingAgent      — clean data first
    2. VerificationAgent     — QA + annotation flags
    3. ActiveLearningAgent   — uncertain cases
    4. DriftMonitorAgent     — distribution shift
    5. FairnessMonitorAgent  — demographic bias
    6. CalibrationAgent      — confidence calibration
    7. HypothesisAgent       — optional researcher query
    8. LiteratureScoutAgent  — optional weekly scan
    """

    def __init__(self):
        # Lazy imports so individual agents can be used standalone
        from self_healing_agent   import SelfHealingAgent
        from verification_agent   import VerificationAgent
        from active_learning_agent import ActiveLearningAgent
        from drift_monitor_agent  import DriftMonitorAgent
        from fairness_monitor_agent import FairnessMonitorAgent
        from calibration_agent    import CalibrationAgent
        from hypothesis_agent     import HypothesisAgent
        from literature_scout_agent import LiteratureScoutAgent

        self.agents = {
            "self_healing"   : SelfHealingAgent(),
            "verification"   : VerificationAgent(),
            "active_learning": ActiveLearningAgent(),
            "drift_monitor"  : DriftMonitorAgent(),
            "fairness_monitor": FairnessMonitorAgent(),
            "calibration"    : CalibrationAgent(),
            "hypothesis"     : HypothesisAgent(),
            "literature_scout": LiteratureScoutAgent(),
        }
        self.pipeline_log = []

    def run_pipeline(self,
                     image_folder   : str,
                     metadata_csv   : str = None,
                     labels_csv     : str = None,
                     label_col      : str = "dx",
                     hypothesis     : str = None,
                     trigger        : str = "manual",
                     skip_literature: bool = True,
                     auto_fix       : bool = False,
                     session_name   : str = None) -> dict:

        session = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n{'='*60}")
        print(f"SKINANALYTICA AGENT PIPELINE: {session}")
        print(f"Trigger: {trigger}")
        print(f"{'='*60}")

        results = {}
        t0      = datetime.now()

        # ── Step 1: Self-Healing ──────────────────────────────────
        print(f"\n[1/8] Self-Healing Data Agent")
        r = self.agents["self_healing"].run(
            image_folder=image_folder, auto_fix=auto_fix,
            output_name=session
        )
        results["self_healing"] = r
        if r.get("ready_count", 1) == 0:
            print("⚠ No valid images after healing — pipeline aborted")
            return results

        # ── Step 2: Verification ─────────────────────────────────
        print(f"\n[2/8] Verification Agent")
        r = self.agents["verification"].run(
            image_folder=image_folder,
            labels_csv=labels_csv, label_col=label_col,
            output_name=session, trigger=trigger
        )
        results["verification"] = r
        records = r.get("records", [])

        # ── Step 3: Active Learning ───────────────────────────────
        print(f"\n[3/8] Active Learning Agent")
        r = self.agents["active_learning"].run(
            records=records, top_n=50, output_name=session
        )
        results["active_learning"] = r

        # ── Step 4: Drift Monitor ─────────────────────────────────
        print(f"\n[4/8] Drift Monitor Agent")
        r = self.agents["drift_monitor"].run(
            records=records, output_name=session
        )
        results["drift_monitor"] = r

        # ── Step 5: Fairness Monitor ──────────────────────────────
        print(f"\n[5/8] Fairness Monitor Agent")
        r = self.agents["fairness_monitor"].run(
            records=records, metadata_csv=metadata_csv,
            output_name=session
        )
        results["fairness_monitor"] = r

        # ── Step 6: Calibration ───────────────────────────────────
        print(f"\n[6/8] Calibration Agent")
        r = self.agents["calibration"].run(
            records=records, output_name=session
        )
        results["calibration"] = r

        # ── Step 7: Hypothesis (optional) ────────────────────────
        if hypothesis:
            print(f"\n[7/8] Hypothesis Agent")
            r = self.agents["hypothesis"].run(
                hypothesis=hypothesis, records=records,
                metadata_csv=metadata_csv
            )
            results["hypothesis"] = r
        else:
            print(f"\n[7/8] Hypothesis Agent — skipped (no hypothesis provided)")

        # ── Step 8: Literature Scout (optional, weekly) ──────────
        if not skip_literature:
            print(f"\n[8/8] Literature Scout Agent")
            r = self.agents["literature_scout"].run(
                days_back=7, output_name=session
            )
            results["literature_scout"] = r
        else:
            print(f"\n[8/8] Literature Scout — skipped (set skip_literature=False to run)")

        elapsed = (datetime.now() - t0).total_seconds()

        # ── Pipeline summary ──────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE: {session}")
        print(f"Total time : {elapsed:.1f}s")
        print(f"\nKey findings:")

        flags = results.get("verification",{}).get("flags",{})
        if flags:
            conflicts = flags.get("ANNOTATION_CONFLICT", 0)
            ambiguous = flags.get("AMBIGUOUS", 0)
            low_conf  = flags.get("LOW_CONFIDENCE", 0)
            print(f"  Annotation conflicts : {conflicts}")
            print(f"  Ambiguous cases      : {ambiguous}")
            print(f"  Low confidence       : {low_conf}")

        drift_sev = results.get("drift_monitor",{}).get("severity","N/A")
        print(f"  Drift severity       : {drift_sev}")

        cal_rec = results.get("calibration",{}).get("recommendation","N/A")
        print(f"  Calibration          : {cal_rec}")

        al_gain = results.get("active_learning",{}).get("estimated_auc_gain","N/A")
        print(f"  Est. AUC gain if top-50 relabelled: +{al_gain}")

        # Save pipeline summary
        summary_path = os.path.join(BASE, "outputs", "agent_logs",
                                    f"{session}_pipeline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"session": session, "elapsed_s": elapsed,
                       "trigger": trigger, "results_summary": {
                           k: v.get("status","done") if isinstance(v,dict) else "done"
                           for k,v in results.items()
                       }}, f, indent=2)
        print(f"\n  Summary: {summary_path}")
        print(f"{'='*60}")

        return results

    def status(self) -> dict:
        """Show status of all agents."""
        return {name: agent.summary() for name, agent in self.agents.items()}


# ── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    runner = AgentRunner()

    results = runner.run_pipeline(
        image_folder  = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\data\sa04_test_sample",
        metadata_csv  = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\data\ISIC_2020\ISIC_2020_Training_GroundTruth_v2.csv",
        trigger       = "manual",
        hypothesis    = "Does the model perform worse on lesions from the head and neck?",
        skip_literature = True,
        session_name  = "demo_pipeline",
    )

"""
SkinAnalytica — base_agent.py
Base class all agents inherit from.
"""

import os, json, logging
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE    = os.environ.get("SKINANALYTICA_BASE",
          r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica")
LOG_DIR = os.path.join(BASE, "outputs", "agent_logs")
os.makedirs(LOG_DIR, exist_ok=True)

class BaseAgent:
    """
    Shared base for all SkinAnalytica agents.
    Provides: logging, result persistence, status tracking, run history.
    """

    def __init__(self, name: str):
        self.name       = name
        self.status     = "idle"   # idle | running | done | error
        self.last_run   = None
        self.last_result= None
        self.run_count  = 0

        # Per-agent log file
        self.log_path = os.path.join(LOG_DIR, f"{name}.jsonl")
        self.logger   = logging.getLogger(f"skinanalytica.agents.{name}")
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )

    # ── Lifecycle ────────────────────────────────────────────────────

    def run(self, *args, **kwargs) -> dict:
        """Entry point — subclasses implement _run()."""
        self.status   = "running"
        self.last_run = datetime.now().isoformat()
        self.run_count += 1
        self.logger.info(f"Starting run #{self.run_count}")

        try:
            result = self._run(*args, **kwargs)
            result["agent"]     = self.name
            result["run_count"] = self.run_count
            result["timestamp"] = self.last_run
            result["status"]    = "done"
            self.status         = "done"
            self.last_result    = result
            self._persist(result)
            self.logger.info(f"Run #{self.run_count} complete")
            return result
        except Exception as e:
            self.status = "error"
            self.logger.error(f"Run #{self.run_count} failed: {e}")
            err = {
                "agent"    : self.name,
                "status"   : "error",
                "error"    : str(e),
                "timestamp": self.last_run,
            }
            self._persist(err)
            return err

    def _run(self, *args, **kwargs) -> dict:
        """Override in subclass."""
        raise NotImplementedError

    # ── Persistence ──────────────────────────────────────────────────

    def _persist(self, result: dict):
        """Append result to agent's JSONL log."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception as e:
            self.logger.warning(f"Could not persist result: {e}")

    def get_history(self, last_n: int = 10) -> list:
        """Return last N run results from log."""
        if not os.path.exists(self.log_path):
            return []
        lines = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    lines.append(json.loads(line))
                except:
                    continue
        return lines[-last_n:]

    def summary(self) -> dict:
        """Agent status summary."""
        return {
            "name"       : self.name,
            "status"     : self.status,
            "run_count"  : self.run_count,
            "last_run"   : self.last_run,
        }

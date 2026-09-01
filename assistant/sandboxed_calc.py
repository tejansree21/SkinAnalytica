"""Gives the Research Assistant's Claude calls a real computation tool
instead of letting it approximate arithmetic in prose (e.g. "what's the
AUC delta between the two highest-confidence sites"). Sandboxed by the
universal-sandbox project (a sibling repo) under the 'skinanalytica'
project policy (extends 'healthcare-imaging'): no network, no file writes
without human approval, memory/timeout capped. See
../../universal-sandbox/ for the sandbox itself; this module just wires
SkinAnalytica's assistant to it.
"""
import sys
import uuid
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parents[2] / "universal-sandbox"
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SANDBOX_ROOT))

from sandbox.executor import ApprovalRequired, run  # noqa: E402

POLICY_NAME = "skinanalytica"
WORKSPACE = SANDBOX_ROOT / "workspace" / "skinanalytica"

CALC_TOOL = {
    "name": "run_calculation",
    "description": (
        "Run Python code to compute an exact numeric answer (e.g. AUC "
        "deltas, weighted averages, ranking by a metric) over numbers "
        "already given to you in the platform data context. Only use this "
        "for real arithmetic you'd otherwise have to approximate — not for "
        "anything requiring file access or network calls, which this tool "
        "cannot do. Print the final result with print(). Only numpy and "
        "the standard library are available."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source. Must print() its result.",
            }
        },
        "required": ["code"],
    },
}


def run_calculation(code: str) -> dict:
    """Execute `code` through the universal sandbox. Returns a dict with
    either {"result": stdout} or {"error": message} — never raises, so the
    caller can always feed something back to Claude."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    script_path = WORKSPACE / f"calc_{uuid.uuid4().hex[:8]}.py"
    script_path.write_text(code, encoding="utf-8")

    try:
        outcome = run(str(script_path), POLICY_NAME, approved=False,
                       workspace=str(WORKSPACE), runtime="docker")
    except ApprovalRequired as e:
        return {
            "error": (
                "This calculation touches a file, network, or subprocess action "
                "that requires human approval and was not run. "
                f"Flagged actions: {e.plan['needs_approval']}"
            )
        }
    except (PermissionError, SyntaxError) as e:
        return {"error": f"Blocked: {e}"}
    finally:
        script_path.unlink(missing_ok=True)

    if outcome.get("reason"):
        return {"error": f"Execution stopped: {outcome['reason']}"}
    if outcome["returncode"] != 0:
        return {"error": f"Code raised an error:\n{outcome['stderr']}"}
    return {"result": outcome["stdout"].strip()}

"""Gives the Research Assistant's Claude calls a real computation tool
instead of letting it approximate arithmetic in prose. The actual
execution logic lives in the universal-sandbox project (a sibling repo,
sandbox/client.py) — shared across every project that wires in a calc
tool, not duplicated here.
"""
import sys
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parents[2] / "universal-sandbox"
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SANDBOX_ROOT))

from sandbox.client import make_calc_tool  # noqa: E402

CALC_TOOL, run_calculation = make_calc_tool(
    policy_name="skinanalytica",
    runtime="docker",
    description=(
        "Run Python code to compute an exact numeric answer (e.g. AUC "
        "deltas, weighted averages, ranking by a metric) over numbers "
        "already given to you in the platform data context. Only use this "
        "for real arithmetic you'd otherwise have to approximate — not for "
        "anything requiring file access or network calls, which this tool "
        "cannot do. Print the final result with print(). Only numpy and "
        "the standard library are available."
    ),
)

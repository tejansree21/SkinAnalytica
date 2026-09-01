"""Tests for assistant/sandboxed_calc.py — the run_calculation tool that
lets the Research Assistant execute real Python (via the sibling
universal-sandbox project) instead of approximating arithmetic in prose."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "assistant"))

from sandboxed_calc import CALC_TOOL, run_calculation


def test_run_calculation_returns_correct_result():
    result = run_calculation("import numpy as np\nprint(np.mean([1, 2, 3, 4, 5]))")
    assert result == {"result": "3.0"}


def test_run_calculation_supports_plain_python():
    result = run_calculation("print(2 + 2)")
    assert result == {"result": "4"}


def test_run_calculation_blocks_file_write_pending_approval():
    result = run_calculation('open("leak.txt", "w").write("x")')
    assert "error" in result
    assert "approval" in result["error"].lower()


def test_run_calculation_denies_socket_import():
    result = run_calculation("import socket")
    assert "error" in result
    assert "blocked" in result["error"].lower()


def test_run_calculation_surfaces_runtime_errors_without_raising():
    result = run_calculation("1 / 0")
    assert "error" in result
    assert "ZeroDivisionError" in result["error"]


def test_calc_tool_schema_matches_anthropic_tool_shape():
    assert CALC_TOOL["name"] == "run_calculation"
    assert "code" in CALC_TOOL["input_schema"]["properties"]
    assert CALC_TOOL["input_schema"]["required"] == ["code"]

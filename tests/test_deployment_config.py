"""
Regression tests for the deployment-path bugs that shipped invisibly
earlier in this project's life -- caught by manual review each time, never
by an automated test, which an independent audit flagged as a pattern
worth closing rather than repeating a fourth time (see docs/MODEL_CARD.md
finding #10 and the "Auth" section's comment history in api/SA05_api.py).

Two distinct bugs are covered here:
  1. MODEL_MODE silently defaulting to "efficientnet" (single weakest
     backbone) while every threshold in the codebase was calibrated
     against the ensemble -- see finding #10.
  2. _detect_base() resolving BASE to "/app" on a Windows dev machine
     where that path spuriously exists (a Git Bash/MSYS artifact), so
     ModelRegistry.load() found zero model files and /analyze 503'd.

conftest.py explicitly sets SKINANALYTICA_BASE before any test imports
SA05_api, which is exactly what masked bug #2 originally (it only showed
up running the API directly, not under pytest) -- these tests call
_detect_base() directly, bypassing that env-var override, specifically to
re-create the conditions that let it slip through the first time.
"""
import importlib
import os
import sys

import pytest

import SA05_api as api


def test_model_mode_defaults_to_full_not_efficientnet(monkeypatch):
    """Regression guard for finding #10: a clean environment with no
    SKINANALYTICA_MODEL_MODE set must default to the ensemble, not the
    single weakest backbone whose score distribution doesn't match any
    threshold in this codebase."""
    monkeypatch.delenv("SKINANALYTICA_MODEL_MODE", raising=False)
    importlib.reload(api)
    try:
        assert api.MODEL_MODE == "full"
    finally:
        importlib.reload(api)  # restore normal test-suite state


def test_detect_base_respects_explicit_env_var_on_any_platform(monkeypatch):
    monkeypatch.setenv("SKINANALYTICA_BASE", r"C:\some\explicit\path")
    assert api._detect_base() == r"C:\some\explicit\path"


def test_detect_base_on_windows_ignores_spurious_app_path(monkeypatch):
    """The exact bug: on this dev machine, os.path.exists("/app") returns
    True even outside any container. A Windows branch that checked path
    existence instead of platform would silently resolve BASE to "/app"
    here -- this test fails if that regresses, by asserting the win32
    branch never even calls os.path.exists("/app") in the first place."""
    monkeypatch.delenv("SKINANALYTICA_BASE", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    checked_paths = []
    real_exists = os.path.exists

    def spy_exists(path):
        checked_paths.append(path)
        if path == r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica":
            return True  # simulate being on the real dev machine
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", spy_exists)
    result = api._detect_base()

    assert "/app" not in checked_paths, (
        "_detect_base()'s win32 branch checked os.path.exists('/app') -- "
        "this is exactly the regression that broke local runs before, "
        "since that path can spuriously exist outside any container"
    )
    assert result == r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica"


def test_detect_base_on_linux_checks_app_paths(monkeypatch):
    """The non-Windows branch is still allowed (expected, even) to check
    /app and /home/user/app -- those are real, meaningful signals on an
    actual Render/HF Spaces container. Only the Windows branch trusting
    them was the bug."""
    monkeypatch.delenv("SKINANALYTICA_BASE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: True if p == "/app" else real_exists(p))
    assert api._detect_base() == "/app"


def test_clean_checkout_base_resolves_to_a_real_project_directory():
    """The actual end-to-end regression this whole class of bug caused:
    ModelRegistry.load() found zero files because BASE pointed at a
    directory with no models/production/onnx underneath it. Assert the
    currently-resolved BASE (as the real test suite runs it) actually has
    that directory, not just that _detect_base() returns *some* string."""
    onnx_dir = os.path.join(api.PROD, "onnx")
    assert os.path.isdir(onnx_dir), (
        f"api.BASE resolved to {api.BASE!r}, which has no models/production/onnx "
        "directory -- this is precisely the silent-503 failure mode from finding #10"
    )


@pytest.mark.skipif(
    not os.path.exists(os.path.join(api.PROD, "onnx_int8", "skinanalytica_ensemble_int8.onnx")),
    reason="INT8 ensemble ONNX file not present -- gitignored, not part of a fresh checkout",
)
def test_model_registry_actually_loads_a_model_with_full_mode(monkeypatch):
    """The single most direct regression test for 'does a clean checkout
    actually load a model': instantiate a real ModelRegistry with
    MODEL_MODE=full pointed at the real on-disk production files, and
    assert it reports loaded=True with a non-'none' model name -- not a
    mock, the real onnxruntime.InferenceSession path."""
    monkeypatch.setattr(api, "MODEL_MODE", "full")
    reg = api.ModelRegistry()
    reg.load()
    assert reg.loaded is True
    assert reg.model_name != "none"
    assert "Ensemble" in reg.model_name, (
        f"MODEL_MODE='full' loaded {reg.model_name!r} instead of an ensemble variant -- "
        "this is exactly finding #10's failure mode, just caught by a real load() call "
        "instead of manual review this time"
    )

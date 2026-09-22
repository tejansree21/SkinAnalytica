"""
Tests for api/SA05_api.py's patient-data authentication (require_api_key).

This closes the top-priority item flagged across two independent audit
passes: GET /patient/{patient_id}/history would return a patient's full
scan history to anyone who knew or guessed an id, with zero credential
required.

SCOPE (updated 2026-07-27): ALL patient-data endpoints are now gated --
/analyze, /analyze/batch, /report/{id}, /report/{id}/fhir,
/patient/{id}/history, /audit/log. Earlier this session, /analyze and
/audit/log were deliberately left open because the old frontend called
them directly from browser JS with no way to hold a secret client-side.
The new frontend (index.html/docs.html/status.html) resolves that with a
client-supplied-key pattern (assets/auth.js) -- the same model as
Swagger's own "Authorize" button: the user types the key in, the browser
holds it in localStorage, and it's never embedded in any page source. That
removed the reason to leave anything open, so the scope was tightened.

Uses fastapi.testclient.TestClient for real HTTP-level checks (status
codes, header handling) rather than calling the dependency function
directly -- this is exactly the kind of deployment-path behavior that
model-mode default, base-path detection, and checkpoint-resume skip all
slipped through unguarded; testing it at the HTTP layer is the point.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_key(monkeypatch):
    """A configured server (SKINANALYTICA_API_KEY set before import) --
    reimports the module so the API_KEY module-level constant picks up the
    env var, since it's read once at import time, not per-request."""
    monkeypatch.setenv("SKINANALYTICA_API_KEY", "test-secret-key")
    import SA05_api as api
    importlib.reload(api)
    return TestClient(api.app), api


@pytest.fixture
def client_unconfigured(monkeypatch):
    """A server with no API key configured at all -- must fail closed."""
    monkeypatch.delenv("SKINANALYTICA_API_KEY", raising=False)
    import SA05_api as api
    importlib.reload(api)
    return TestClient(api.app), api


def test_patient_history_rejects_missing_key(client_with_key):
    client, _ = client_with_key
    r = client.get("/patient/some-patient-id/history")
    assert r.status_code == 401


def test_patient_history_rejects_wrong_key(client_with_key):
    client, _ = client_with_key
    r = client.get("/patient/some-patient-id/history", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


def test_patient_history_accepts_correct_key(client_with_key):
    client, _ = client_with_key
    r = client.get("/patient/nonexistent-patient/history", headers={"X-API-Key": "test-secret-key"})
    # 404 (no scans for this patient) proves auth passed and the real
    # handler ran -- 401 would mean auth rejected the request first.
    assert r.status_code == 404


def test_audit_log_rejects_missing_key(client_with_key):
    client, _ = client_with_key
    r = client.get("/audit/log")
    assert r.status_code == 401


def test_audit_log_accepts_correct_key(client_with_key):
    client, _ = client_with_key
    r = client.get("/audit/log", headers={"X-API-Key": "test-secret-key"})
    assert r.status_code == 200


def test_report_endpoint_rejects_missing_key(client_with_key):
    client, _ = client_with_key
    r = client.get("/report/some-scan-id")
    assert r.status_code == 401


def test_fhir_export_rejects_missing_key(client_with_key):
    client, _ = client_with_key
    r = client.get("/report/some-scan-id/fhir")
    assert r.status_code == 401


def test_analyze_rejects_missing_key(client_with_key):
    client, _ = client_with_key
    r = client.post("/analyze", files={"file": ("x.jpg", b"not a real image", "image/jpeg")})
    assert r.status_code == 401


def test_analyze_accepts_correct_key_and_reaches_handler(client_with_key, monkeypatch):
    """Force model_reg.loaded so a real request without a model wouldn't
    mask the auth check behind an unrelated 503. A bad-image 400 (not 401)
    proves the request reached the real handler."""
    client, api = client_with_key
    monkeypatch.setattr(api.model_reg, "loaded", True)
    r = client.post(
        "/analyze",
        files={"file": ("x.jpg", b"not a real image", "image/jpeg")},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert r.status_code != 401


def test_analyze_batch_rejects_missing_key(client_with_key):
    client, _ = client_with_key
    r = client.post("/analyze/batch", files={"files": ("x.jpg", b"x", "image/jpeg")})
    assert r.status_code == 401


def test_unconfigured_server_fails_closed_not_open(client_unconfigured):
    """The critical property: if SKINANALYTICA_API_KEY is never set (e.g. a
    misconfigured deploy), protected endpoints must reject every request
    (503) rather than silently allowing them through with no auth at all --
    that silent-allow behavior is exactly the original bug."""
    client, _ = client_unconfigured
    r = client.get("/patient/some-patient-id/history")
    assert r.status_code == 503
    r2 = client.get("/patient/some-patient-id/history", headers={"X-API-Key": "anything"})
    assert r2.status_code == 503


def test_health_and_models_stay_public(client_with_key):
    """/health and /models carry no patient data and must remain
    unauthenticated -- render.yaml's healthCheckPath depends on /health
    being reachable with no credential."""
    client, _ = client_with_key
    assert client.get("/health").status_code == 200
    assert client.get("/models").status_code == 200

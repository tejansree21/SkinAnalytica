"""
Tests for delivery/delivery_api.py -- flagged as an untested gap in the
independent audit. Scoped to endpoints that are safe to exercise without
side effects (no real email/PDF generation triggered, no network calls) --
the email/report-generation endpoints queue a BackgroundTask that writes
real files to outputs/emails|reports/, which these tests deliberately
don't invoke, to stay hermetic.

Also confirms directly what earlier findings only implied: this API has
NO authentication on any route -- worth having on record as an explicit
test, not just an assumption.
"""
import pytest
from fastapi.testclient import TestClient

import delivery_api


@pytest.fixture
def client():
    delivery_api._notifications.clear()
    return TestClient(delivery_api.app)


def test_delivery_health(client):
    r = client.get("/delivery/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_delivery_endpoints_have_no_authentication(client):
    """Documents the current state plainly rather than assuming it --
    unlike api/SA05_api.py's patient-data endpoints (finding #13), this
    API has no X-API-Key requirement on any route. Not asserting this is
    wrong (this service doesn't handle patient data directly, and there's
    no evidence in this repo that it's mounted into the public deployment
    rather than run internally) -- just recording it as a fact so a future
    change here is deliberate, not accidental."""
    r = client.get("/delivery/health")
    assert r.status_code == 200
    assert "www-authenticate" not in r.headers


def test_push_notification_queue_add_list_and_mark_read(client):
    r = client.post("/delivery/notify", json={"title": "Test", "body": "hello", "severity": "info"})
    assert r.status_code == 200
    notif_id = r.json()["notification_id"]

    r = client.get("/delivery/notifications")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["notifications"][0]["id"] == notif_id
    assert body["notifications"][0]["read"] is False

    r = client.patch(f"/delivery/notifications/{notif_id}/read")
    assert r.status_code == 200

    r = client.get("/delivery/notifications?unread_only=true")
    assert r.json()["count"] == 0


def test_mark_read_on_unknown_notification_id_returns_404(client):
    r = client.patch("/delivery/notifications/does-not-exist/read")
    assert r.status_code == 404


def test_notifications_list_returns_most_recent_first(client):
    for i in range(3):
        client.post("/delivery/notify", json={"title": f"n{i}", "body": "x"})
    r = client.get("/delivery/notifications")
    titles = [n["title"] for n in r.json()["notifications"]]
    assert titles == ["n2", "n1", "n0"]


def test_download_report_404_for_missing_file(client):
    r = client.get("/delivery/report/download/does-not-exist.pdf")
    assert r.status_code == 404


def test_generate_scan_report_404_for_missing_scan(client):
    r = client.post("/delivery/report/scan/nonexistent-scan-id")
    assert r.status_code == 404


def test_preview_email_404_for_missing_file(client):
    r = client.get("/delivery/emails/does-not-exist.html")
    assert r.status_code == 404


def test_list_emails_handles_missing_directory_gracefully(client, monkeypatch, tmp_path):
    monkeypatch.setattr(delivery_api, "OUT_DIR", str(tmp_path / "nonexistent"))
    r = client.get("/delivery/emails")
    assert r.status_code == 200
    assert r.json() == {"emails": [], "count": 0}

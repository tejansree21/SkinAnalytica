"""
Tests for assistant/assistant_api.py -- flagged as an untested gap in the
independent audit. Deliberately scoped to endpoints that never construct
a real anthropic.Anthropic() client or make a network call:
`/assistant/ask` and `/assistant/ask/prebuilt` route to the Claude API for
non-data-grounded questions (research_assistant.py's `_call_claude()`),
and are NOT exercised here -- doing so in a test would either hit a real
API (flaky, costs money, needs a key) or silently pass/fail depending on
environment. `/assistant/data/{id}` calls the data layer directly,
bypassing the assistant/Claude path entirely, and is safe to test.
`/assistant/history/{id}` only constructs ResearchAssistant.__init__,
which is confirmed cheap (session file load only, no Claude client
constructed until an actual question is asked) -- safe.
"""
import pytest
from fastapi.testclient import TestClient

import assistant_api


@pytest.fixture
def client():
    assistant_api._sessions.clear()
    return TestClient(assistant_api.app)


def test_assistant_health(client):
    r = client.get("/assistant/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "claude_available" in body
    assert body["active_sessions"] == 0


def test_list_questions_returns_the_full_library(client):
    from question_library import QUESTION_LIBRARY
    r = client.get("/assistant/questions")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(QUESTION_LIBRARY)
    assert len(body["questions"]) == len(QUESTION_LIBRARY)


def test_ask_rejects_empty_question(client):
    r = client.post("/assistant/ask", json={"question": "   ", "session_id": "test"})
    assert r.status_code == 400


def test_data_route_404_for_unknown_question_id(client):
    r = client.get("/assistant/data/Q99999-not-real")
    assert r.status_code == 404
    assert "Q99999-not-real" in r.json()["detail"]


def test_data_route_200_for_a_real_question_id(client):
    from data_layer import DATA_ROUTES
    qid = next(iter(DATA_ROUTES))
    r = client.get(f"/assistant/data/{qid}")
    # data_layer's own loaders return {} gracefully when no local run
    # data exists yet -- this just confirms the route wiring works and
    # doesn't 500, not that specific data is present in this environment
    assert r.status_code == 200


def test_history_for_a_fresh_session_is_empty(client):
    r = client.get("/assistant/history/brand-new-session-id")
    assert r.status_code == 200
    body = r.json()
    assert body["history"] == []
    assert body["count"] == 0


def test_clear_history_on_never_created_session_does_not_error(client):
    # session_id was never asked anything -- _sessions won't contain it,
    # clear_history() should be a safe no-op, not a KeyError
    r = client.delete("/assistant/history/never-existed")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"

from fastapi.testclient import TestClient

from app.api import chat as chat_module
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_agent_response(monkeypatch):
    def fake_generate_response(message: str, session_id: str):
        return {
            "response": f"Test reply to: {message}",
            "session_id": session_id,
            "tool_calls": [],
        }

    monkeypatch.setattr(
        chat_module.agent,
        "generate_response",
        fake_generate_response,
    )

    response = client.post(
        "/chat",
        json={"message": "Hello Bloxi", "session_id": "api-test"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Test reply to: Hello Bloxi",
        "session_id": "api-test",
        "tool_calls": [],
    }


def test_chat_endpoint_generates_session_id(monkeypatch):
    captured = {}

    def fake_generate_response(message: str, session_id: str):
        captured["session_id"] = session_id
        return {
            "response": "Hello from Bloxi.",
            "session_id": session_id,
            "tool_calls": [],
        }

    monkeypatch.setattr(
        chat_module.agent,
        "generate_response",
        fake_generate_response,
    )

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    assert captured["session_id"]
    assert response.json()["session_id"] == captured["session_id"]


def test_chat_endpoint_rejects_whitespace_message():
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_endpoint_handles_agent_failure(monkeypatch):
    def failing_generate_response(message: str, session_id: str):
        raise RuntimeError("Internal provider details")

    monkeypatch.setattr(
        chat_module.agent,
        "generate_response",
        failing_generate_response,
    )

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Bloxi could not generate a response. Please try again."
    )
    assert "Internal provider details" not in response.text

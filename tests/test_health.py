from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model"] == "claude-sonnet-4-5-20250929"
    assert data["llm_enabled"] is False
    assert data["database"] is False
    assert "version" in data

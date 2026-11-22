"""
Tests for backend health endpoint and application startup.
"""

from fastapi import status


def test_root_endpoint(client):
    """Verify root endpoint responds with metadata."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["health"] == "/api/health"


def test_health_endpoint(client):
    """Verify /health and /api/health return status ok and AI readiness."""
    for path in ["/health", "/api/health"]:
        response = client.get(path)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "ai_engine_ready" in data
        assert "approved_studio_exists" in data
        assert "weights_available" in data
        assert isinstance(data["weights_available"], list)

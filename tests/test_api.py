from fastapi.testclient import TestClient

from app.main import app


def test_health_and_claims() -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.get("/claims")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_validate_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/claims/SIN-2026-0042/validate")
    assert response.status_code == 200
    payload = response.json()
    assert "final_repair_invoice" in payload["missing_documents"]

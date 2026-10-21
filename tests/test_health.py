from fastapi.testclient import TestClient
from ai_document_intelligence.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
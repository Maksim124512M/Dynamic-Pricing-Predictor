from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_invalid_payload():
    response = client.post("/predict/", json={"price": "abc"})

    assert response.status_code == 422
import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_upload_sensor_data():
    response = client.post(
        "/upload-sensor-data/",
        json={
            "timestamp": "2024-01-01T12:00:00Z",
            "machine_id": "crane1",
            "energy_consumption": 200.5
        }
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success", "result": "energy data processed"}

def test_get_recommendations():
    response = client.get("/recommendations/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Should return a list of recommendations

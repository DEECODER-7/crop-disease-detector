"""
Basic tests for the API. Run with:
    pytest tests/test_api.py
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_classes"] == 38


def test_predict_rejects_non_image_file():
    fake_file = ("not_an_image.txt", b"this is not an image", "text/plain")
    response = client.post("/predict", files={"file": fake_file})
    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes():
    fake_file = ("fake.jpg", b"definitely not real jpeg bytes", "image/jpeg")
    response = client.post("/predict", files={"file": fake_file})
    assert response.status_code == 400
import pytest
from fastapi.testclient import TestClient
from src.api import app
import io
from PIL import Image
import numpy as np

client = TestClient(app)

def test_health_check():
    """Verify the API health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # Note: inference_engine might be None if lifespan didn't run, 
    # but TestClient handles lifespan in modern versions.
    assert "status" in response.json()

def test_predict_invalid_metadata():
    """Ensure the API rejects non-image files."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image."

# Note: We skip the full prediction test in CI here because loading the 94.25% 
# ResNet weights requires the .pth file and significant memory, which might 
# fail on small GH Runners without specific setup. 
# We focus on the API layer and the Security logic.

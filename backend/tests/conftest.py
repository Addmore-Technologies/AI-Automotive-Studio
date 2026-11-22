"""
Test fixtures and configuration for backend tests.
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings

SAMPLE_DIR = os.path.abspath(os.path.join(settings.PROJECT_ROOT, "sample_data", "input"))


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_car_image_path():
    """Return path to sample sedan image."""
    path = os.path.join(SAMPLE_DIR, "car_sedan_front.jpg")
    if not os.path.exists(path):
        pytest.skip("Sample image car_sedan_front.jpg not found.")
    return path


@pytest.fixture
def sample_car_bytes(sample_car_image_path):
    """Return bytes of sample sedan image."""
    with open(sample_car_image_path, "rb") as f:
        return f.read()

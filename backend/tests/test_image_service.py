"""
Integration tests for image processing API endpoints and service.
"""

import io
import cv2
import numpy as np
from fastapi import status


def test_process_car_image_success(client, sample_car_image_path):
    """
    Test full end-to-end image processing with a real vehicle photograph:
    upload -> detection -> segmentation -> plate replacement -> compositing -> result retrieval.
    """
    with open(sample_car_image_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/image/process",
        files={"file": ("sedan.jpg", file_bytes, "image/jpeg")},
        data={"fictional_plate_text": "STUDIO 88"},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["success"] is True
    assert data["status"] == "completed"
    assert "job_id" in data
    assert data["output_format"] == "png"
    assert data["result_url"].startswith("/api/results/studio_")

    # Verify result retrieval endpoint
    res_response = client.get(data["result_url"])
    assert res_response.status_code == status.HTTP_200_OK
    assert res_response.headers["content-type"] == "image/png"
    assert "inline" in res_response.headers.get("content-disposition", "")

    # Verify download parameter
    dl_response = client.get(data["download_url"])
    assert dl_response.status_code == status.HTTP_200_OK
    assert "attachment" in dl_response.headers.get("content-disposition", "")

    # Verify generated studio image dimensions (1920x1080)
    img_arr = np.frombuffer(res_response.content, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (1080, 1920, 3)


def test_process_image_no_vehicle_detected(client):
    """
    Verify that an image with no vehicles returns HTTP 422 with NO_VEHICLE_DETECTED.
    """
    # Create a blank white image
    blank = np.full((300, 400, 3), 255, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", blank)
    blank_bytes = encoded.tobytes()

    response = client.post(
        "/api/image/process",
        files={"file": ("blank.png", blank_bytes, "image/png")},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = response.json().get("detail", {})
    assert detail.get("status") == "NO_VEHICLE_DETECTED"
    assert "No vehicle detected" in detail.get("error", "")


def test_frontend_compat_route_process_image(client, sample_car_bytes):
    """Verify alias route /process/image matches /api/image/process."""
    response = client.post(
        "/process/image",
        files={"file": ("sedan.jpg", sample_car_bytes, "image/jpeg")},
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True

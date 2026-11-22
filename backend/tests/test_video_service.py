"""
Integration tests for video processing, frame extraction, reconstruction, and job tracking.
"""

import os
import tempfile
import cv2
import numpy as np
from fastapi import status

from backend.app.services.video_service import job_manager


def _create_synthetic_car_video(sample_image_path: str, num_frames: int = 5) -> str:
    """Helper to create a temporary MP4 video from a car image with slight camera jitter."""
    img = cv2.imread(sample_image_path)
    h, w = img.shape[:2]
    # Resize to modest resolution for rapid testing
    test_w, test_h = 640, 360
    img_resized = cv2.resize(img, (test_w, test_h))

    tmp_dir = tempfile.gettempdir()
    video_path = os.path.join(tmp_dir, f"test_car_{num_frames}f.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 24.0, (test_w, test_h))

    for i in range(num_frames):
        frame = img_resized.copy()
        if i > 0:
            noise = np.random.randint(-2, 3, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


def test_video_process_end_to_end(client, sample_car_image_path):
    """
    Test full end-to-end video processing:
    upload -> validation -> frame extraction -> AI frame processing with EMA smoothing ->
    reconstruction -> result retrieval.
    """
    video_path = _create_synthetic_car_video(sample_car_image_path, num_frames=5)
    assert os.path.exists(video_path)

    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        response = client.post(
            "/api/video/process",
            files={"file": ("test_clip.mp4", video_bytes, "video/mp4")},
            data={"enable_temporal_smoothing": True},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["success"] is True
        assert data["status"] == "completed"
        assert "job_id" in data
        job_id = data["job_id"]
        assert data["result_url"].startswith("/api/results/studio_")

        # Verify job status via /api/jobs/{job_id}
        job_resp = client.get(f"/api/jobs/{job_id}")
        assert job_resp.status_code == status.HTTP_200_OK
        job_data = job_resp.json()
        assert job_data["status"] == "completed"
        assert job_data["progress_pct"] == 100.0
        assert job_data["total_frames"] == 5

        # Verify result video retrieval
        res_resp = client.get(data["result_url"])
        assert res_resp.status_code == status.HTTP_200_OK
        assert res_resp.headers["content-type"] == "video/mp4"

        # Verify downloaded video is valid and has expected dimensions (1920x1080)
        temp_dl = os.path.join(tempfile.gettempdir(), f"downloaded_{job_id}.mp4")
        with open(temp_dl, "wb") as f:
            f.write(res_resp.content)

        cap = cv2.VideoCapture(temp_dl)
        assert cap.isOpened()
        out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert out_frames == 5
        assert out_w == 1920
        assert out_h == 1080

        if os.path.exists(temp_dl):
            os.remove(temp_dl)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def test_video_no_vehicle_detected(client):
    """
    Verify that a video with no cars triggers HTTP 422 with NO_VEHICLE_DETECTED.
    """
    tmp_path = os.path.join(tempfile.gettempdir(), "blank_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 24.0, (320, 240))
    for _ in range(3):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    try:
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()

        response = client.post(
            "/api/video/process",
            files={"file": ("blank.mp4", video_bytes, "video/mp4")},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        detail = response.json().get("detail", {})
        assert detail.get("status") == "NO_VEHICLE_DETECTED"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_job_not_found(client):
    """Verify 404 for non-existent job ID."""
    response = client.get("/api/jobs/unknown-uuid-999")
    assert response.status_code == status.HTTP_404_NOT_FOUND

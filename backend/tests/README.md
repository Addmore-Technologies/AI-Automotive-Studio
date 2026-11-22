# Backend Test Suite — AI Automotive Studio

This directory contains automated unit and integration tests for the FastAPI backend, media orchestration, upload validation, AI pipeline integration, and video reconstruction.

## Test Structure

- `conftest.py`: Shared test fixtures (FastAPI `TestClient`, sample car image fixtures).
- `test_health.py`: Verifies application startup, metadata, and `/api/health` system readiness.
- `test_upload_validation.py`: Verifies:
  - Valid image content inspection (JPEG, PNG).
  - Rejection of unsupported extensions (e.g., `.pdf`, `.txt`, `.avi`).
  - Rejection of corrupted files with fake magic bytes.
  - File-size limit enforcement (HTTP 413).
  - Filename sanitization and path traversal prevention.
- `test_image_service.py`:
  - End-to-end car image upload, vehicle detection, segmentation, plate replacement, studio compositing, and result retrieval.
  - Verification of 1920x1080 studio image dimensions.
  - Handling of non-car images (HTTP 422 `NO_VEHICLE_DETECTED`).
  - Frontend route alias compatibility (`/process/image`).
- `test_video_service.py`:
  - End-to-end video upload, frame extraction, sequential AI processing with EMA temporal smoothing, video reconstruction, and playable MP4 output verification.
  - Real-time job polling via `/api/jobs/{job_id}`.
  - Video stream result delivery and download headers.
  - Non-vehicle video rejection (HTTP 422 `NO_VEHICLE_DETECTED`).

## Running Tests

From the repository root (`AI-Automotive-Studio`):

```bash
# Run all backend tests
pytest backend/tests -v

# Run with coverage
pytest backend/tests --cov=backend/app -v
```

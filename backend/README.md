# AI Automotive Studio — Backend & Video Processing Service

The FastAPI backend for **AI Automotive Studio**, responsible for media ingestion, validation, AI/ML pipeline orchestration, video frame extraction, temporal consistency stabilization, video reconstruction, and studio result delivery.

---

## 1. Architecture & Responsibilities

The backend acts as the bridge connecting Arya's React interface with the Computer Vision pipeline:

```
┌─────────────────────────┐          HTTP Multipart / JSON          ┌───────────────────────────┐
│     Arya's Frontend     │ ◄─────────────────────────────────────► │      FastAPI Backend      │
│ (React + TanStack UI)   │                                         │ (Routes, Services, Files) │
└─────────────────────────┘                                         └─────────────┬─────────────┘
                                                                                  │ Python API
                                                                                  ▼
                                                                    ┌───────────────────────────┐
                                                                    │      AI Engine Pipeline   │
                                                                    │ (YOLOv8, Seg, Plate, Comp)│
                                                                    └───────────────────────────┘
```

- **FastAPI Application**: Lifespan management, CORS support, centralized Pydantic settings, standardized error handling.
- **Upload Validation & Security**: Magic bytes header inspection, PIL/OpenCV decode verification, file size enforcement, filename sanitization, and path traversal guards.
- **AI Pipeline Service**: Imports reusable functions from `ai_engine.pipeline` (`process_image` and `process_frame`); zero duplicate AI algorithms in backend.
- **Video Orchestration Service**: Sequential frame extraction, inter-frame temporal smoothing via exponential moving averages (EMA), audio preservation where feasible, video reconstruction (`cv2.VideoWriter`), and asynchronous job tracking.
- **Results Delivery**: High-performance static and streaming delivery with `Accept-Ranges: bytes` support and optional file download attachments.

---

## 2. Directory Structure

```text
backend/
├── app/
│   ├── main.py                     # FastAPI application factory and lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Centralized configuration and path management
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py               # System health & AI readiness check
│   │   ├── image.py                # Image upload and studio compositing endpoints
│   │   ├── video.py                # Video upload, processing & job status endpoints
│   │   └── results.py              # Result media delivery and downloads
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py             # Pydantic request models
│   │   └── responses.py            # Pydantic response models
│   └── services/
│       ├── __init__.py
│       ├── file_service.py         # File validation, magic bytes, temp workspace
│       ├── ai_pipeline_service.py  # Wrapper interfacing with ai_engine
│       ├── image_service.py        # Image processing workflow orchestration
│       └── video_service.py        # Video frame extraction, EMA smoothing, encoder
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Test fixtures and test client
│   ├── test_health.py              # Health and startup tests
│   ├── test_upload_validation.py   # Magic bytes, formats, size limits, security tests
│   ├── test_image_service.py       # End-to-end image processing & error tests
│   ├── test_video_service.py       # Video extraction, reconstruction & job tests
│   └── README.md
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- PyTorch & Torchvision
- OpenCV (`opencv-python`)
- Optional: `ffmpeg` installed in PATH for audio stream extraction and muxing.

### Setup Instructions

1. **Navigate to the repository root**:
   ```bash
   cd AI-Automotive-Studio
   ```

2. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   pip install -r ai_engine/requirements.txt
   ```

3. **Verify model weights and assets**:
   ```bash
   python scripts/setup_assets.py
   ```
   Ensure `ai_engine/weights/` has `yolov8n.pt`, `yolov8n-seg.pt`, and `studio_assets/backgrounds/approved_studio.png` exists.

4. **Environment Configuration**:
   ```bash
   cp backend/.env.example backend/.env
   ```

---

## 4. Running the Application

### Start the development server
From the repository root:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or using Python module syntax:
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

- Interactive OpenAPI Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 5. API Endpoints & Request/Response Contract

### `GET /api/health`
Check backend and computer vision pipeline readiness.
- **Response `200 OK`**:
  ```json
  {
    "status": "ok",
    "version": "1.0.0",
    "app_name": "AI Automotive Studio API",
    "ai_engine_ready": true,
    "approved_studio_exists": true,
    "weights_available": ["yolov8m-seg.pt", "yolov8n-seg.pt", "yolov8n.pt"],
    "timestamp": 1732257000.0
  }
  ```

---

### `POST /api/image/process` (alias: `/process/image`)
Upload a vehicle image (JPG, JPEG, PNG) to place it in the LED studio.
- **Form Data**:
  - `file` *(required)*: Binary image file.
  - `fictional_plate_text` *(optional)*: Custom text for plate (default: `"STUDIO 01"`).
  - `enable_shadows` *(optional, bool)*: Ground contact shadow (default: `true`).
  - `enable_reflections` *(optional, bool)*: Floor reflection (default: `true`).
  - `vehicle_scale_pct` *(optional, float)*: Vehicle scale in studio (default: `0.72`).
- **Response `200 OK`**:
  ```json
  {
    "job_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "completed",
    "success": true,
    "result_url": "/api/results/studio_9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d.png",
    "download_url": "/api/results/studio_9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d.png?download=true",
    "cutout_url": "/api/results/studio_9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d_cutout.png",
    "output_format": "png",
    "message": "Studio image generated successfully.",
    "metadata": {
      "original_filename": "car_sedan.jpg",
      "input_dimensions": {"width": 1280, "height": 853},
      "output_dimensions": {"width": 1920, "height": 1080},
      "detection": {"has_detection": true, "count": 1},
      "segmentation": {"coverage_pct": 28.5}
    },
    "execution_time_ms": 350.2
  }
  ```
- **Error Response `422 Unprocessable Entity`** (No vehicle detected):
  ```json
  {
    "status": "NO_VEHICLE_DETECTED",
    "error": "No vehicle detected in the provided image.",
    "detail": "No vehicle was detected in the provided media. Ensure the car is clearly visible."
  }
  ```

---

### `POST /api/video/process` (alias: `/process/video`)
Upload a vehicle video (MP4, MOV) to process frame-by-frame with temporal consistency into the studio environment.
- **Form Data & Query Parameters**:
  - `file` *(required)*: Binary video file.
  - `fictional_plate_text` *(optional)*: Custom text for plate.
  - `enable_temporal_smoothing` *(optional, bool)*: EMA box & mask stabilization (default: `true`).
  - `async_mode` *(query parameter, bool)*: If `true`, returns `202 Accepted` with initial job response for polling. Defaults to `false` for direct synchronous completion as expected by Arya's frontend.
- **Response `200 OK`**:
  ```json
  {
    "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "status": "completed",
    "success": true,
    "result_url": "/api/results/studio_f47ac10b-58cc-4372-a567-0e02b2c3d479.mp4",
    "download_url": "/api/results/studio_f47ac10b-58cc-4372-a567-0e02b2c3d479.mp4?download=true",
    "output_format": "mp4",
    "message": "Video processing completed successfully.",
    "progress_pct": 100.0,
    "metadata": {
      "fps": 24.0,
      "frame_count": 120,
      "dimensions": {"width": 1920, "height": 1080},
      "detected_frames": 120,
      "audio_preserved": false
    },
    "execution_time_ms": 4200.5
  }
  ```

---

### `GET /api/jobs/{job_id}` (alias: `/jobs/{job_id}`)
Retrieve real-time processing status, progress percentage, and result URLs for a job.
- **Response `200 OK`**:
  ```json
  {
    "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "kind": "video",
    "status": "processing",
    "progress_pct": 65.4,
    "current_frame": 78,
    "total_frames": 120,
    "result_url": null,
    "download_url": null,
    "message": "Processing frame 78 of 120 (65.4%)",
    "error": null
  }
  ```

---

### `GET /api/results/{filename}` (alias: `/results/{filename}`)
Stream or download generated images and videos.
- **Parameters**: `download=true` (optional) to trigger file download attachment.
- **Features**: `Accept-Ranges: bytes` header for streaming video player scrubbing.

---

## 6. Video Processing & Media Orchestration Details

1. **Validation & Metadata Inspection**:
   - Upload streamed directly to disk to prevent memory exhaustion.
   - Headers validated for ISO base media file format (`ftyp`, `moov`, `mdat`).
   - `cv2.VideoCapture` extracts metadata: frame count, fps, dimensions, duration.
2. **Sequential Frame Extraction**:
   - Frames extracted in strict sequential order.
   - Each frame fed into `ai_engine.pipeline.process_frame(frame, state=state)`.
3. **Temporal Stabilization**:
   - Inter-frame `state` dictionary passed across frames.
   - Bounding box coordinates smoothed via Exponential Moving Average (EMA, `alpha=0.65`).
   - Segmentation masks smoothed across frames via EMA (`alpha=0.75`), preventing boundary jitter.
4. **Reconstruction**:
   - OpenCV `VideoWriter` encodes processed frames at original frame rate (default codec `mp4v` / `avc1`).
   - Standard output resolution: 1920×1080 (matching studio background).
5. **Audio Preservation**:
   - If FFmpeg is present, audio streams are extracted from the original video and remuxed into the final studio MP4 without quality loss.
   - If FFmpeg is absent, video reconstruction completes cleanly with original fps and video timing.
6. **Safe Temporary File Cleanup**:
   - All intermediate frames and upload files in `temp/<job_id>` are automatically purged in `finally` blocks upon completion or failure.

---

## 7. Running Backend Tests

Run all unit and integration tests:
```bash
pytest backend/tests -v
```

Run both AI engine tests and backend tests together:
```bash
pytest tests backend/tests -v
```

All 40 tests (25 AI engine + 15 backend) pass cleanly.

---

## 8. Known Limitations & Recommendations

- **H.264 Encoder**: On Windows without Cisco OpenH264 binaries or FFmpeg CLI in PATH, OpenCV falls back to `mp4v` (MPEG-4 Part 2). Most modern browsers and media players play `mp4v` inside an MP4 container, but for broad browser streaming compatibility in production, installing FFmpeg or providing `openh264-*.dll` enables AVC1/H.264 muxing.
- **GPU Acceleration**: CUDA is utilized automatically when available on the host machine; on CPU-only machines, YOLOv8 nano/medium models run using optimized PyTorch CPU kernels.

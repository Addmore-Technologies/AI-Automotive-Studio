# API Contract — AI Automotive Studio (Aparna & Arya Agreement)

**Agreed and Finalized by Aparna (Backend) and Arya (Frontend)**

## Base Configuration
- Development backend URL: `http://localhost:8000`
- Frontend environment variable: `VITE_STUDIO_API_URL=http://localhost:8000`
- Supported MIME & Extensions:
  - Images: `image/jpeg` (`.jpg`, `.jpeg`), `image/png` (`.png`)
  - Videos: `video/mp4` (`.mp4`), `video/quicktime` (`.mov`)

---

## Endpoints Specification

### 1. Health Check
- **Path**: `GET /api/health` (alias: `GET /health`)
- **Status**: `200 OK`
- **Response JSON**:
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

### 2. Image Processing
- **Path**: `POST /api/image/process` (alias: `POST /process/image`)
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `file` *(required)*: Binary file.
  - `fictional_plate_text` *(optional, string)*: Custom license plate text.
  - `enable_shadows` *(optional, boolean, default: true)*.
  - `enable_reflections` *(optional, boolean, default: true)*.
- **Success Response (`200 OK`)**:
  ```json
  {
    "job_id": "uuid-v4-string",
    "status": "completed",
    "success": true,
    "result_url": "/api/results/studio_<job_id>.png",
    "download_url": "/api/results/studio_<job_id>.png?download=true",
    "cutout_url": "/api/results/studio_<job_id>_cutout.png",
    "output_format": "png",
    "message": "Studio image generated successfully.",
    "execution_time_ms": 320.5
  }
  ```

---

### 3. Video Processing
- **Path**: `POST /api/video/process` (alias: `POST /process/video`)
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `file` *(required)*: Binary video file.
  - `fictional_plate_text` *(optional, string)*.
  - `enable_temporal_smoothing` *(optional, boolean, default: true)*.
  - `async_mode` *(query param, boolean, default: false)*: Set to `true` for background jobs.
- **Success Response (`200 OK`)**:
  ```json
  {
    "job_id": "uuid-v4-string",
    "status": "completed",
    "success": true,
    "result_url": "/api/results/studio_<job_id>.mp4",
    "download_url": "/api/results/studio_<job_id>.mp4?download=true",
    "output_format": "mp4",
    "message": "Video processing completed successfully.",
    "progress_pct": 100.0,
    "execution_time_ms": 4120.0
  }
  ```

---

### 4. Job Status & Progress Polling
- **Path**: `GET /api/jobs/{job_id}` (alias: `GET /jobs/{job_id}`)
- **Success Response (`200 OK`)**:
  ```json
  {
    "job_id": "uuid-v4-string",
    "kind": "video",
    "status": "processing",
    "progress_pct": 75.0,
    "current_frame": 90,
    "total_frames": 120,
    "result_url": null,
    "download_url": null,
    "message": "Processing frame 90 of 120 (75.0%)",
    "error": null
  }
  ```

---

### 5. Result Delivery & Downloads
- **Path**: `GET /api/results/{filename}` (alias: `GET /results/{filename}`)
- **Query Parameter**: `download=true` (forces `attachment` header)
- **Headers Returned**:
  - `Content-Type`: `image/png`, `image/jpeg`, `video/mp4`, etc.
  - `Content-Disposition`: `inline; filename="..."` or `attachment; filename="..."`
  - `Accept-Ranges`: `bytes` (enables scrubbing/seeking in `<video>` tags)

---

### 6. Standardized Error Responses
- **No Vehicle Detected (`422 Unprocessable Entity`)**:
  ```json
  {
    "status": "NO_VEHICLE_DETECTED",
    "error": "No vehicle detected in the provided image.",
    "detail": "No vehicle was detected in the provided media. Ensure the car is clearly visible."
  }
  ```
  *(Matches Arya's regex `/no vehicle|vehicle not (found|detected)/i`)*

- **Unsupported Format (`415 Unsupported Media Type`)**:
  ```json
  {
    "status": "UNSUPPORTED_FORMAT",
    "error": "Unsupported format '.xyz'. Accepted formats: JPG, PNG, MP4, MOV."
  }
  ```

- **Corrupted File (`400 Bad Request`)**:
  ```json
  {
    "status": "CORRUPTED_FILE",
    "error": "File header does not match a valid media container or cannot be decoded."
  }
  ```

- **File Too Large (`413 Payload Too Large`)**:
  ```json
  {
    "status": "FILE_TOO_LARGE",
    "error": "Media file exceeds maximum allowed size limit."
  }
  ```

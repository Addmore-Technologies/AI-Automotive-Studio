# API Contract — Jointly Complete

Aparna and Arya must agree on this before frontend/backend integration.

- Development base URL: to be confirmed.
- `POST /api/image/process` (proposed), multipart field `file`.
- `POST /api/video/process` (proposed), multipart field `file`.
- `GET /api/jobs/{job_id}` (proposed), status and result URL.
- `GET /api/results/{filename}` (proposed), generated result access.

Decide whether image processing returns immediately or uses a job ID. Define exact response JSON, error shape/status codes, CORS origins, and output URL access. These paths are proposals, not finalized contracts.

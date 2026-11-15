# Backend Assignment — Aparna

## Goal
Build the FastAPI service that receives uploads from Arya's frontend, calls the shared AI/ML pipeline, coordinates video processing, and returns results.

## Responsibilities
1. Create FastAPI app, configuration, routes, schemas, services, and tests.
2. Accept JPG/JPEG/PNG images and MP4/MOV videos.
3. Validate actual media type and size; manage temporary files safely.
4. Call reusable functions exposed by `ai_engine/`; do not reimplement AI algorithms.
5. Extract video frames using FFmpeg/OpenCV, call the frame pipeline, and reconstruct a playable video. Preserve frame rate/dimensions and audio where feasible.
6. Expose result URLs/download endpoints and processing status if jobs are asynchronous.
7. Document setup, API requests/responses, and errors.

## Files to create
```text
backend/
├── app/
│   ├── main.py
│   ├── routes/{health.py,image.py,video.py,results.py}
│   ├── schemas/{requests.py,responses.py}
│   ├── services/{image_service.py,video_service.py,ai_pipeline_service.py,file_service.py}
│   └── core/config.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Proposed endpoints — confirm with Arya
- `GET /api/health`
- `POST /api/image/process` — multipart field `file`
- `POST /api/video/process` — multipart field `file`
- `GET /api/jobs/{job_id}` — status and result URL
- `GET /api/results/{filename}` — generated result

## Suggested dependencies
FastAPI, Uvicorn, Pydantic, `python-multipart`; FFmpeg/OpenCV if needed for video handling. Keep model-specific dependencies in the AI engine unless directly needed here.

## Work order
Build health route → upload validation → agree AI function contract → integrate image processing → connect Arya's frontend → add video processing → test failure cases.

## Done when
The documented backend accepts supported uploads, calls the shared AI pipeline, returns usable results, and has basic route/validation tests.

## Boundaries
Aparna owns API orchestration and video extraction/reconstruction. AI/ML owns detection/segmentation/compositing functions. Arya owns React UI.

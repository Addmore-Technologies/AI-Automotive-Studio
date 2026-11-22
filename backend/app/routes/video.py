"""
Video processing and job tracking endpoints.
"""

import os
from typing import Optional, Union
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.requests import VideoProcessOptions
from backend.app.schemas.responses import JobProgressResponse, VideoProcessResponse
from backend.app.services.video_service import video_service, job_manager
from backend.app.services.file_service import (
    FileService,
    UnsupportedFormatError,
    CorruptedFileError,
    FileTooLargeError,
)
from backend.app.services.ai_pipeline_service import (
    NoVehicleDetectedException,
    AiPipelineException,
)

router = APIRouter(tags=["Video Processing"])


@router.post(
    "/api/video/process",
    response_model=Union[VideoProcessResponse, JobProgressResponse],
    status_code=status.HTTP_200_OK,
    summary="Upload and process car video into studio render",
)
@router.post(
    "/process/video",
    response_model=Union[VideoProcessResponse, JobProgressResponse],
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def process_video_endpoint(
    file: UploadFile = File(..., description="Multipart vehicle video file (MP4, MOV)"),
    fictional_plate_text: Optional[str] = Form(None, description="Custom text for fictional license plate"),
    enable_temporal_smoothing: bool = Form(True, description="Enable inter-frame temporal smoothing"),
    async_mode: bool = Query(False, description="Whether to process in background and poll /api/jobs/{job_id}"),
) -> Union[VideoProcessResponse, JobProgressResponse]:
    """
    Accept an uploaded vehicle video, extract frames, process them sequentially
    with temporal smoothing, preserve audio where feasible, and reconstruct a playable studio video.
    """
    filename = file.filename or "uploaded_video.mp4"
    job_id, job_dir = FileService.create_job_workspace()
    temp_input_path = os.path.join(job_dir, f"input_{FileService.sanitize_filename(filename)}")

    try:
        # Stream upload directly to disk
        file_size = 0
        with open(temp_input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > settings.MAX_VIDEO_SIZE_BYTES:
                    max_mb = settings.MAX_VIDEO_SIZE_BYTES // (1024 * 1024)
                    raise FileTooLargeError(f"Video file exceeds maximum allowed size of {max_mb} MB.")
                buffer.write(chunk)

        # Validate video headers, stream, and frame decodability
        meta = FileService.validate_video_file(temp_input_path, filename)

        options = VideoProcessOptions(
            fictional_plate_text=fictional_plate_text,
            enable_temporal_smoothing=enable_temporal_smoothing,
            async_mode=async_mode,
        )

        # Initialize job in manager
        job = job_manager.create_job(job_id=job_id, kind="video", total_frames=meta.get("total_frames"))

        # Start processing
        res = video_service.start_video_processing(
            file_path=temp_input_path,
            original_filename=filename,
            metadata=meta,
            options=options,
            job_id=job_id,
        )

        if async_mode:
            return res

        return VideoProcessResponse(
            job_id=job_id,
            status=res.status.value if hasattr(res.status, "value") else str(res.status),
            success=True,
            result_url=res.result_url,
            download_url=res.download_url,
            output_format="mp4",
            message=res.message,
            progress_pct=res.progress_pct,
            metadata=res.metadata,
            execution_time_ms=res.execution_time_ms,
        )

    except UnsupportedFormatError as e:
        FileService.cleanup_job_workspace(job_id)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"status": "UNSUPPORTED_FORMAT", "error": str(e)},
        )
    except FileTooLargeError as e:
        FileService.cleanup_job_workspace(job_id)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"status": "FILE_TOO_LARGE", "error": str(e)},
        )
    except CorruptedFileError as e:
        FileService.cleanup_job_workspace(job_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "CORRUPTED_FILE", "error": str(e)},
        )
    except NoVehicleDetectedException as e:
        FileService.cleanup_job_workspace(job_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "NO_VEHICLE_DETECTED", "error": str(e)},
        )
    except AiPipelineException as e:
        FileService.cleanup_job_workspace(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "AI_PIPELINE_ERROR", "error": str(e)},
        )
    except Exception as e:
        FileService.cleanup_job_workspace(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "INTERNAL_ERROR", "error": f"An error occurred during video processing: {e}"},
        )
    finally:
        await file.close()


@router.get(
    "/api/jobs/{job_id}",
    response_model=JobProgressResponse,
    summary="Retrieve video processing job status and progress",
)
@router.get(
    "/jobs/{job_id}",
    response_model=JobProgressResponse,
    include_in_schema=False,
)
def get_job_status(job_id: str) -> JobProgressResponse:
    """
    Retrieve real-time processing status, progress percentage, and result URLs for a job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"status": "JOB_NOT_FOUND", "error": f"Job with ID '{job_id}' was not found."},
        )
    return job

"""
Response schemas for the FastAPI backend.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("ok", description="Overall service status")
    version: str = Field(..., description="API version")
    app_name: str = Field(..., description="Application name")
    ai_engine_ready: bool = Field(..., description="Whether AI models and background assets are ready")
    approved_studio_exists: bool = Field(..., description="Whether the approved studio background is present")
    weights_available: List[str] = Field(default_factory=list, description="Available YOLO model weight files")
    timestamp: float = Field(..., description="Current UNIX timestamp")


class ImageProcessResponse(BaseModel):
    """Image processing response."""
    job_id: str = Field(..., description="Unique processing job ID")
    status: str = Field("completed", description="Processing status (completed, failed)")
    success: bool = Field(True, description="Whether processing was successful")
    result_url: str = Field(..., description="Relative or absolute URL to view the processed studio image")
    download_url: str = Field(..., description="URL to download the processed studio image")
    cutout_url: Optional[str] = Field(None, description="URL to download the transparent RGBA vehicle cutout")
    output_format: str = Field("png", description="Output media format")
    message: Optional[str] = Field("Studio image generated successfully", description="Status message")
    error: Optional[str] = Field(None, description="Error description if processing failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Detection, segmentation and compositing metadata")
    execution_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")


class VideoProcessResponse(BaseModel):
    """Video processing response."""
    job_id: str = Field(..., description="Unique processing job ID")
    status: str = Field(..., description="Processing status (completed, processing, failed)")
    success: bool = Field(True, description="Whether processing succeeded or is actively running")
    result_url: Optional[str] = Field(None, description="URL to view/stream the processed studio video")
    download_url: Optional[str] = Field(None, description="URL to download the processed studio video")
    output_format: str = Field("mp4", description="Output media format")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error description if processing failed")
    progress_pct: float = Field(0.0, description="Processing completion percentage (0 - 100)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Video metadata (fps, dimensions, frame counts)")
    execution_time_ms: Optional[float] = Field(None, description="Total execution time in milliseconds")


class JobProgressResponse(BaseModel):
    """Job status response for polling."""
    job_id: str = Field(..., description="Unique job identifier")
    kind: str = Field("video", description="Media kind (image, video)")
    status: JobStatus = Field(..., description="Current job status")
    success: bool = Field(True, description="Whether the job is currently successful or running without fatal error")
    progress_pct: float = Field(0.0, description="Completion percentage (0.0 - 100.0)")
    stage: Optional[str] = Field(None, description="Current processing stage identifier")
    current_frame: Optional[int] = Field(None, description="Current frame being processed")
    total_frames: Optional[int] = Field(None, description="Total video frames")
    result_url: Optional[str] = Field(None, description="URL to view processed result")
    download_url: Optional[str] = Field(None, description="URL to download processed result")
    cutout_url: Optional[str] = Field(None, description="URL for transparent cutout")
    output_format: Optional[str] = Field(None, description="Output format (png, mp4)")
    message: Optional[str] = Field(None, description="Status or progress message")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: Optional[float] = Field(None, description="Total execution time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    status: str = Field("error", description="Error status code identifier")
    error: str = Field(..., description="User-facing error message")
    detail: Optional[str] = Field(None, description="Technical detail or stack description")

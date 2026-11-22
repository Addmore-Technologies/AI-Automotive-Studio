"""
Request schemas for the FastAPI backend.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ImageProcessOptions(BaseModel):
    """Optional parameters for image processing."""
    fictional_plate_text: Optional[str] = Field(None, description="Custom text for fictional license plate")
    enable_shadows: bool = Field(True, description="Enable realistic contact shadow underneath vehicle")
    enable_reflections: bool = Field(True, description="Enable realistic floor reflection underneath vehicle")
    vehicle_scale_pct: Optional[float] = Field(None, ge=0.3, le=0.95, description="Scale percentage of vehicle in studio (0.3 to 0.95)")


class VideoProcessOptions(BaseModel):
    """Optional parameters for video processing."""
    fictional_plate_text: Optional[str] = Field(None, description="Custom text for fictional license plate")
    enable_temporal_smoothing: bool = Field(True, description="Enable EMA temporal smoothing across frames")
    async_mode: bool = Field(False, description="Process video asynchronously in background and poll job status")

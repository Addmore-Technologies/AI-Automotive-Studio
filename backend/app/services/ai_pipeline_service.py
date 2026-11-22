"""
AI Pipeline Service - High-level wrapper integrating with ai_engine.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from backend.app.core.config import settings
from ai_engine.pipeline import (
    AutomotiveStudioPipeline,
    get_pipeline,
)
from ai_engine.common.types import PipelineConfig, MultiVehiclePolicy


class AiPipelineException(Exception):
    """Base exception for AI/ML pipeline errors."""
    pass


class NoVehicleDetectedException(AiPipelineException):
    """Raised when no vehicle is detected in the input image or video."""
    pass


class SegmentationFailedException(AiPipelineException):
    """Raised when vehicle segmentation mask cannot be generated."""
    pass


class AiPipelineService:
    """Wrapper service interfacing between FastAPI backend and ai_engine pipeline."""

    def __init__(self):
        self._pipeline: Optional[AutomotiveStudioPipeline] = None

    def get_pipeline_instance(self, config_override: Optional[PipelineConfig] = None) -> AutomotiveStudioPipeline:
        """Get or initialize pipeline instance."""
        if config_override is not None:
            return AutomotiveStudioPipeline(config_override)
        if self._pipeline is None:
            config = PipelineConfig(
                studio_background_path=settings.STUDIO_BACKGROUND_PATH,
                fictional_plate_text=settings.DEFAULT_PLATE_TEXT,
                fictional_plate_subtext=settings.DEFAULT_PLATE_SUBTEXT,
                vehicle_scale_pct=settings.DEFAULT_VEHICLE_SCALE,
                save_cutout=True,
            )
            self._pipeline = AutomotiveStudioPipeline(config)
        return self._pipeline

    def check_readiness(self) -> Tuple[bool, List[str]]:
        """
        Check if model weights and studio background exist.
        Returns (is_ready, available_weights_list).
        """
        weights_found = []
        if os.path.exists(settings.WEIGHTS_DIR):
            for f in os.listdir(settings.WEIGHTS_DIR):
                if f.endswith(".pt") or f.endswith(".onnx"):
                    weights_found.append(f)

        bg_exists = os.path.exists(settings.STUDIO_BACKGROUND_PATH)
        ready = bg_exists and len(weights_found) > 0
        return ready, weights_found

    def process_image(
        self,
        input_path: str,
        output_path: str,
        fictional_plate_text: Optional[str] = None,
        enable_shadows: bool = True,
        enable_reflections: bool = True,
        vehicle_scale_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a car image through the AI/ML pipeline and write to output_path.
        """
        config = PipelineConfig(
            studio_background_path=settings.STUDIO_BACKGROUND_PATH,
            fictional_plate_text=fictional_plate_text or settings.DEFAULT_PLATE_TEXT,
            fictional_plate_subtext=settings.DEFAULT_PLATE_SUBTEXT,
            enable_plate_replacement=True,
            enable_shadows=enable_shadows,
            enable_reflections=enable_reflections,
            vehicle_scale_pct=vehicle_scale_pct or settings.DEFAULT_VEHICLE_SCALE,
            save_cutout=True,
        )

        pipeline = self.get_pipeline_instance(config)
        result = pipeline.process_image(
            input_path=input_path,
            output_path=output_path,
            config_override={"fictional_plate_text": fictional_plate_text} if fictional_plate_text else None,
        )

        if not result.get("success"):
            status = result.get("status", "UNKNOWN_ERROR")
            error_msg = result.get("error", "AI pipeline processing failed.")

            if status == "NO_VEHICLE_DETECTED":
                raise NoVehicleDetectedException(error_msg)
            elif status == "SEGMENTATION_EMPTY":
                raise SegmentationFailedException(error_msg)
            else:
                raise AiPipelineException(f"Pipeline error [{status}]: {error_msg}")

        return result

    def process_video_frame(
        self,
        frame: np.ndarray,
        state: Optional[Dict[str, Any]] = None,
        studio_background: Optional[np.ndarray] = None,
        fictional_plate_text: Optional[str] = None,
        enable_temporal_smoothing: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single BGR video frame through the AI pipeline with temporal smoothing.
        """
        pipeline = self.get_pipeline_instance()
        pipeline.config.enable_temporal_smoothing = enable_temporal_smoothing
        if fictional_plate_text:
            pipeline.config.fictional_plate_text = fictional_plate_text

        return pipeline.process_frame(
            frame=frame,
            studio_background=studio_background,
            state=state,
        )


ai_pipeline_service = AiPipelineService()

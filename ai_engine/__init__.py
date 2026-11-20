"""
AI Engine - Automotive Studio Vision System.
"""

from ai_engine.common.types import (
    BoundingBox,
    DetectionResult,
    SegmentationResult,
    PlateResult,
    PipelineConfig,
    MultiVehiclePolicy,
    ColorOrder,
    VehicleClass,
)
from ai_engine.detection.vehicle_detector import VehicleDetector
from ai_engine.segmentation.vehicle_segmenter import VehicleSegmenter
from ai_engine.background_removal.remover import BackgroundRemover
from ai_engine.number_plate.plate_processor import PlateProcessor
from ai_engine.compositing.shadow_reflection import ShadowReflectionGenerator
from ai_engine.compositing.studio_compositor import StudioCompositor
from ai_engine.pipeline import (
    AutomotiveStudioPipeline,
    process_image,
    process_frame,
    get_pipeline,
)

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "SegmentationResult",
    "PlateResult",
    "PipelineConfig",
    "MultiVehiclePolicy",
    "ColorOrder",
    "VehicleClass",
    "VehicleDetector",
    "VehicleSegmenter",
    "BackgroundRemover",
    "PlateProcessor",
    "ShadowReflectionGenerator",
    "StudioCompositor",
    "AutomotiveStudioPipeline",
    "process_image",
    "process_frame",
    "get_pipeline",
]

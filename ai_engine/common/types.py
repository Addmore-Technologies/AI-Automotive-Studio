"""
Common types, data structures, and configurations for AI Automotive Studio.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class VehicleClass(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    UNKNOWN = "vehicle"


class MultiVehiclePolicy(str, Enum):
    LARGEST = "largest"          # Primary vehicle with the largest bounding box area
    CENTER = "center"            # Vehicle closest to image center
    HIGHEST_CONF = "highest_conf"# Vehicle with highest detection confidence
    ALL = "all"                  # Composite all detected vehicles


class ColorOrder(str, Enum):
    BGR = "BGR"  # OpenCV default
    RGB = "RGB"  # PIL / Matplotlib / PyTorch default


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    class_id: int = 2
    class_name: str = "car"

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": round(float(self.confidence), 4),
            "class_name": self.class_name,
        }


@dataclass
class DetectionResult:
    """Detection output containing detected vehicles."""
    boxes: List[BoundingBox] = field(default_factory=list)
    primary_index: Optional[int] = None
    image_shape: Tuple[int, int] = (0, 0)  # (height, width)
    execution_time_ms: float = 0.0

    @property
    def primary_box(self) -> Optional[BoundingBox]:
        if self.primary_index is not None and 0 <= self.primary_index < len(self.boxes):
            return self.boxes[self.primary_index]
        return self.boxes[0] if self.boxes else None

    @property
    def has_detection(self) -> bool:
        return len(self.boxes) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_detection": self.has_detection,
            "count": len(self.boxes),
            "primary_box": self.primary_box.to_dict() if self.primary_box else None,
            "all_boxes": [b.to_dict() for b in self.boxes],
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class SegmentationResult:
    """Segmentation output containing vehicle mask and refined edges."""
    mask: np.ndarray  # 2D uint8 array (0 or 255), shape (H, W)
    confidence: float = 1.0
    execution_time_ms: float = 0.0
    refined: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        h, w = self.mask.shape[:2] if self.mask is not None else (0, 0)
        coverage = float(np.count_nonzero(self.mask > 0)) / (h * w) if (h * w) > 0 else 0.0
        return {
            "mask_shape": [h, w],
            "coverage_pct": round(coverage * 100, 2),
            "confidence": round(float(self.confidence), 4),
            "refined": self.refined,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "details": self.details,
        }


@dataclass
class PlateResult:
    """Number plate detection and fictional replacement result."""
    found: bool = False
    box: Optional[BoundingBox] = None
    quadrilateral: Optional[List[Tuple[int, int]]] = None  # 4 corner points for perspective
    original_region: Optional[np.ndarray] = None
    replaced_text: str = ""
    confidence: float = 0.0
    status_note: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "box": self.box.to_dict() if self.box else None,
            "quadrilateral": self.quadrilateral,
            "replaced_text": self.replaced_text,
            "confidence": round(float(self.confidence), 4),
            "status_note": self.status_note,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class PipelineConfig:
    """Configuration settings for the processing pipeline."""
    # Weights and paths
    detector_weights: Optional[str] = None
    segmenter_weights: Optional[str] = None
    studio_background_path: Optional[str] = None

    # Detection & Segmentation options
    confidence_threshold: float = 0.35
    multi_vehicle_policy: MultiVehiclePolicy = MultiVehiclePolicy.LARGEST
    refine_edges: bool = True
    edge_blur_kernel: int = 3
    hole_fill_threshold: int = 2500

    # Fictional plate options
    enable_plate_replacement: bool = True
    fictional_plate_text: str = "STUDIO 01"
    fictional_plate_subtext: str = "AUTO-AI"

    # Compositing & Studio lighting options
    enable_shadows: bool = True
    shadow_intensity: float = 0.65
    shadow_blur_radius: int = 15
    enable_reflections: bool = True
    reflection_opacity: float = 0.28
    reflection_falloff_height: float = 0.35
    reflection_blur_kernel: int = 11

    # Positioning
    ground_offset_pct: float = 0.02
    vehicle_scale_pct: float = 0.72  # Vehicle takes approx 72% of studio width

    # Video stabilization options
    enable_temporal_smoothing: bool = True
    mask_ema_alpha: float = 0.75  # Higher = faster adaptation, lower = smoother

    # Output options
    color_order: ColorOrder = ColorOrder.BGR
    save_cutout: bool = True

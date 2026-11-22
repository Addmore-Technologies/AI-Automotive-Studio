"""
Vehicle Detection Module using YOLO.
Detects cars, trucks, buses, and motorcycles in images and video frames.
Handles multi-vehicle selection policies and no-detection edge cases.
"""

import os
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure ultralytics writes its config/settings within the project
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(".ultralytics"))

from ultralytics import YOLO
from ai_engine.common.types import (
    BoundingBox,
    DetectionResult,
    MultiVehiclePolicy,
    VehicleClass,
    ColorOrder,
)


# COCO class IDs for vehicles
VEHICLE_CLASS_MAP = {
    2: VehicleClass.CAR.value,
    3: VehicleClass.MOTORCYCLE.value,
    5: VehicleClass.BUS.value,
    7: VehicleClass.TRUCK.value,
}


class VehicleDetector:
    """
    Vehicle detector utilizing YOLO models.
    Supports single-frame images and video frame streams.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        multi_vehicle_policy: MultiVehiclePolicy = MultiVehiclePolicy.LARGEST,
        device: Optional[str] = None,
    ):
        """
        Initialize the VehicleDetector.

        Args:
            weights_path: Path to YOLO weights (.pt). If None, searches common local paths.
            confidence_threshold: Minimum confidence score to accept detection.
            multi_vehicle_policy: How to pick the primary vehicle when multiple are present.
            device: 'cpu', 'cuda', or None for auto.
        """
        self.confidence_threshold = confidence_threshold
        self.multi_vehicle_policy = multi_vehicle_policy
        self.weights_path = self._resolve_weights(weights_path)
        
        # Load YOLO model
        self.model = YOLO(self.weights_path)
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        print(f"[VehicleDetector] Loaded YOLO model from {self.weights_path} on {self.device}")

    def _resolve_weights(self, specified_path: Optional[str]) -> str:
        if specified_path and os.path.exists(specified_path):
            return specified_path
        
        candidates = [
            os.path.abspath("ai_engine/weights/yolov8n.pt"),
            os.path.abspath("ai_engine/weights/yolov8n-seg.pt"),
            os.path.abspath("ai_engine/weights/yolov8m-seg.pt"),
            "yolov8n.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "yolov8n.pt"

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def detect(
        self,
        image_or_path: Union[str, np.ndarray],
        color_order: ColorOrder = ColorOrder.BGR,
    ) -> DetectionResult:
        """
        Detect vehicles in an image or video frame.

        Args:
            image_or_path: Filepath string or uint8 numpy array.
            color_order: BGR (default OpenCV) or RGB.

        Returns:
            DetectionResult containing detected bounding boxes and primary vehicle index.
        """
        start_time = time.perf_counter()

        # Load or normalize image array
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"Image not found at: {image_or_path}")
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Failed to decode image from: {image_or_path}")
        else:
            image = image_or_path

        h, w = image.shape[:2]

        # Convert RGB to BGR for YOLO if input is RGB
        if color_order == ColorOrder.RGB and isinstance(image_or_path, np.ndarray):
            yolo_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            yolo_input = image

        # Run YOLO inference filtering for vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
        vehicle_class_ids = list(VEHICLE_CLASS_MAP.keys())
        results = self.model.predict(
            source=yolo_input,
            classes=vehicle_class_ids,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        boxes: List[BoundingBox] = []
        if results and len(results) > 0:
            det = results[0]
            if det.boxes is not None and len(det.boxes) > 0:
                xyxy = det.boxes.xyxy.cpu().numpy()
                confs = det.boxes.conf.cpu().numpy()
                cls_ids = det.boxes.cls.cpu().numpy().astype(int)

                for box_coords, conf, cls_id in zip(xyxy, confs, cls_ids):
                    x1 = max(0, min(int(round(box_coords[0])), w - 1))
                    y1 = max(0, min(int(round(box_coords[1])), h - 1))
                    x2 = max(x1 + 1, min(int(round(box_coords[2])), w))
                    y2 = max(y1 + 1, min(int(round(box_coords[3])), h))
                    
                    class_name = VEHICLE_CLASS_MAP.get(int(cls_id), "vehicle")
                    boxes.append(
                        BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=class_name,
                        )
                    )

        # Resolve primary vehicle index based on policy
        primary_idx = self._resolve_primary_vehicle(boxes, w, h)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DetectionResult(
            boxes=boxes,
            primary_index=primary_idx,
            image_shape=(h, w),
            execution_time_ms=elapsed_ms,
        )

    def _resolve_primary_vehicle(
        self, boxes: List[BoundingBox], img_w: int, img_h: int
    ) -> Optional[int]:
        if not boxes:
            return None
        if len(boxes) == 1:
            return 0

        if self.multi_vehicle_policy == MultiVehiclePolicy.LARGEST:
            # Vehicle with largest area
            return int(np.argmax([b.area for b in boxes]))
        elif self.multi_vehicle_policy == MultiVehiclePolicy.HIGHEST_CONF:
            return int(np.argmax([b.confidence for b in boxes]))
        elif self.multi_vehicle_policy == MultiVehiclePolicy.CENTER:
            # Vehicle closest to image center
            center_x, center_y = img_w / 2.0, img_h / 2.0
            dists = [
                (b.center[0] - center_x) ** 2 + (b.center[1] - center_y) ** 2
                for b in boxes
            ]
            return int(np.argmin(dists))
        else:
            # Default to largest
            return int(np.argmax([b.area for b in boxes]))

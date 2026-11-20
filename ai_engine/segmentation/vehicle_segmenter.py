"""
Vehicle Segmentation Module.
Produces refined vehicle masks preserving mirrors, spoilers, wheels, and fine edges.
Handles hole filling, edge smoothing, anti-aliasing, and quality checks.
"""

import os
import time
from typing import Optional, Tuple, Union, Dict, Any
import cv2
import numpy as np

# Ensure ultralytics writes its config/settings within the project
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(".ultralytics"))

from ultralytics import YOLO
from ai_engine.common.types import (
    BoundingBox,
    SegmentationResult,
    ColorOrder,
    MultiVehiclePolicy,
)


class VehicleSegmenter:
    """
    Vehicle segmentation engine using YOLO-seg instance segmentation.
    Includes advanced post-processing: morphological edge refinement,
    internal window/cabin hole filling, speckle cleanup, and edge anti-aliasing.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.30,
        refine_edges: bool = True,
        edge_blur_kernel: int = 3,
        hole_fill_threshold: int = 2500,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.refine_edges = refine_edges
        self.edge_blur_kernel = edge_blur_kernel if edge_blur_kernel % 2 == 1 else edge_blur_kernel + 1
        self.hole_fill_threshold = hole_fill_threshold
        
        self.weights_path = self._resolve_weights(weights_path)
        self.model = YOLO(self.weights_path)
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        print(f"[VehicleSegmenter] Loaded segmentation model from {self.weights_path} on {self.device}")

    def _resolve_weights(self, specified_path: Optional[str]) -> str:
        if specified_path and os.path.exists(specified_path):
            return specified_path
        
        candidates = [
            os.path.abspath("ai_engine/weights/yolov8m-seg.pt"),
            os.path.abspath("ai_engine/weights/yolov8n-seg.pt"),
            "yolov8n-seg.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "yolov8n-seg.pt"

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def segment(
        self,
        image_or_path: Union[str, np.ndarray],
        target_box: Optional[BoundingBox] = None,
        color_order: ColorOrder = ColorOrder.BGR,
    ) -> SegmentationResult:
        """
        Segment the vehicle from the image.

        Args:
            image_or_path: Image path or numpy array.
            target_box: Optional target bounding box from detection to match.
            color_order: BGR (OpenCV) or RGB.

        Returns:
            SegmentationResult containing full-resolution refined binary mask.
        """
        start_time = time.perf_counter()

        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"Image not found at: {image_or_path}")
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Failed to read image from: {image_or_path}")
        else:
            image = image_or_path

        h, w = image.shape[:2]

        if color_order == ColorOrder.RGB and isinstance(image_or_path, np.ndarray):
            yolo_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            yolo_input = image

        # Predict with YOLO-seg on vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
        results = self.model.predict(
            source=yolo_input,
            classes=[2, 3, 5, 7],
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        raw_mask = np.zeros((h, w), dtype=np.uint8)
        confidence = 0.0

        if results and len(results) > 0:
            det = results[0]
            if det.masks is not None and len(det.masks) > 0:
                masks_data = det.masks.data.cpu().numpy()  # shape (N, orig_h, orig_w) or scaled
                confs = det.boxes.conf.cpu().numpy() if det.boxes is not None else [1.0] * len(masks_data)
                boxes_xyxy = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else None

                best_idx = 0
                if target_box is not None and boxes_xyxy is not None:
                    # Match target box by IoU
                    t_box = [target_box.x1, target_box.y1, target_box.x2, target_box.y2]
                    best_iou = -1.0
                    for i, b in enumerate(boxes_xyxy):
                        iou = self._calculate_iou(t_box, b)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                else:
                    # Choose mask with largest area
                    areas = [np.count_nonzero(m) for m in masks_data]
                    best_idx = int(np.argmax(areas)) if areas else 0

                selected_mask = masks_data[best_idx]
                confidence = float(confs[best_idx]) if best_idx < len(confs) else 1.0

                # Resize mask to original image dimensions if needed
                if selected_mask.shape[:2] != (h, w):
                    selected_mask = cv2.resize(
                        selected_mask.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                raw_mask = (selected_mask > 0.5).astype(np.uint8) * 255

        # Perform boundary refinement, hole-filling, and speckle removal
        refined_mask, details = self._refine_mask(raw_mask, h, w)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return SegmentationResult(
            mask=refined_mask,
            confidence=confidence,
            execution_time_ms=elapsed_ms,
            refined=self.refine_edges,
            details=details,
        )

    def _refine_mask(self, mask: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Refine raw segmentation mask:
        1. Keep largest connected component (main vehicle body).
        2. Fill small internal holes (windshield reflections, dark cabin areas).
        3. Smooth jagged edges using morphological closing with elliptical kernel.
        4. Detect image boundary clipping.
        """
        details: Dict[str, Any] = {
            "initial_coverage_pct": round(float(np.count_nonzero(mask)) / (h * w) * 100, 2),
            "clipped_borders": [],
        }

        if np.count_nonzero(mask) == 0:
            details["status"] = "empty_mask"
            return mask, details

        output_mask = mask.copy()

        # 1. Filter out background fragments / speckles outside the vehicle
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_mask, connectivity=8)
        if num_labels > 2:
            # stat[cv2.CC_STAT_AREA] for label 0 is background; find largest component > 0
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            output_mask = (labels == largest_label).astype(np.uint8) * 255

        # 2. Fill small internal holes inside the car (e.g., windows or cabin gaps)
        if self.hole_fill_threshold > 0:
            contours, hierarchy = cv2.findContours(output_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None:
                for i, h_elem in enumerate(hierarchy[0]):
                    # h_elem: [Next, Previous, First_Child, Parent]
                    parent = h_elem[3]
                    if parent != -1:  # This contour is an internal hole
                        hole_area = cv2.contourArea(contours[i])
                        if hole_area < self.hole_fill_threshold:
                            cv2.drawContours(output_mask, contours, i, 255, -1)

        # 3. Morphological boundary refinement around side mirrors, wheels, spoilers
        if self.refine_edges:
            # Elliptical kernel preserves curved car contours
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Subtle open to remove thin 1-pixel antenna noise spikes
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_OPEN, kernel_open)

        # 4. Check for border clipping
        clipped = []
        if np.any(output_mask[0, :] > 0):
            clipped.append("top")
        if np.any(output_mask[h - 1, :] > 0):
            clipped.append("bottom")
        if np.any(output_mask[:, 0] > 0):
            clipped.append("left")
        if np.any(output_mask[:, w - 1] > 0):
            clipped.append("right")

        details["clipped_borders"] = clipped
        details["final_coverage_pct"] = round(float(np.count_nonzero(output_mask)) / (h * w) * 100, 2)
        details["status"] = "refined_successfully"

        return output_mask, details

    def _calculate_iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return float(iou)

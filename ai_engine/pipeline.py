"""
AI Automotive Studio - End-to-End Vision Pipeline.
Provides stable image and video-frame entry points for backend integration.
Orchestrates vehicle detection, segmentation, background removal,
fictional number plate replacement, and studio compositing.
"""

import os
import time
from typing import Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np

from ai_engine.common.types import (
    BoundingBox,
    DetectionResult,
    SegmentationResult,
    PlateResult,
    PipelineConfig,
    ColorOrder,
    MultiVehiclePolicy,
)
from ai_engine.detection.vehicle_detector import VehicleDetector
from ai_engine.segmentation.vehicle_segmenter import VehicleSegmenter
from ai_engine.background_removal.remover import BackgroundRemover
from ai_engine.number_plate.plate_processor import PlateProcessor
from ai_engine.compositing.studio_compositor import StudioCompositor


class AutomotiveStudioPipeline:
    """
    Unified Computer Vision Pipeline for automotive isolation,
    fictional plate replacement, and studio composition.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize pipeline modules
        self.detector = VehicleDetector(
            weights_path=self.config.detector_weights,
            confidence_threshold=self.config.confidence_threshold,
            multi_vehicle_policy=self.config.multi_vehicle_policy,
        )
        self.segmenter = VehicleSegmenter(
            weights_path=self.config.segmenter_weights,
            confidence_threshold=self.config.confidence_threshold,
            refine_edges=self.config.refine_edges,
            edge_blur_kernel=self.config.edge_blur_kernel,
            hole_fill_threshold=self.config.hole_fill_threshold,
        )
        self.remover = BackgroundRemover(
            feather_radius=self.config.edge_blur_kernel,
            despill_edges=True,
        )
        self.plate_processor = PlateProcessor(
            default_fictional_text=self.config.fictional_plate_text,
            default_subtext=self.config.fictional_plate_subtext,
        )
        self.compositor = StudioCompositor(
            approved_background_path=self.config.studio_background_path,
            vehicle_scale_pct=self.config.vehicle_scale_pct,
            enable_shadows=self.config.enable_shadows,
            enable_reflections=self.config.enable_reflections,
            shadow_intensity=self.config.shadow_intensity,
            reflection_opacity=self.config.reflection_opacity,
        )

    def process_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single vehicle image end-to-end.

        Args:
            input_path: Path to the input car image (JPG, PNG, WebP).
            output_path: Path to save the final composited image. If None, auto-generates.
            config_override: Optional parameter overrides for this execution.

        Returns:
            Structured dictionary with metadata, paths, and status.
        """
        start_total = time.perf_counter()

        if not os.path.exists(input_path):
            return {
                "success": False,
                "status": "ERROR_FILE_NOT_FOUND",
                "error": f"Input file not found at: {input_path}",
                "input_path": input_path,
            }

        # 1. Read input image (BGR)
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            return {
                "success": False,
                "status": "ERROR_DECODE_FAILED",
                "error": f"Failed to read image file: {input_path}",
                "input_path": input_path,
            }

        h, w = image_bgr.shape[:2]

        # 2. Vehicle Detection
        det_res = self.detector.detect(image_bgr, color_order=ColorOrder.BGR)
        if not det_res.has_detection:
            return {
                "success": False,
                "status": "NO_VEHICLE_DETECTED",
                "error": "No vehicle detected in the provided image.",
                "input_path": input_path,
                "detection": det_res.to_dict(),
            }

        primary_box = det_res.primary_box

        # 3. Vehicle Segmentation
        seg_res = self.segmenter.segment(image_bgr, target_box=primary_box, color_order=ColorOrder.BGR)
        if np.count_nonzero(seg_res.mask) == 0:
            return {
                "success": False,
                "status": "SEGMENTATION_EMPTY",
                "error": "Failed to generate vehicle segmentation mask.",
                "input_path": input_path,
                "detection": det_res.to_dict(),
                "segmentation": seg_res.to_dict(),
            }

        # 4. Fictional Number Plate Replacement (performed on vehicle image before cutout)
        plate_res = PlateResult(found=False, status_note="Plate replacement disabled")
        processed_vehicle_img = image_bgr.copy()
        
        if self.config.enable_plate_replacement:
            custom_text = config_override.get("fictional_plate_text") if config_override else None
            processed_vehicle_img, plate_res = self.plate_processor.process(
                image=processed_vehicle_img,
                vehicle_box=primary_box,
                fictional_text=custom_text or self.config.fictional_plate_text,
                color_order=ColorOrder.BGR,
            )

        # 5. Background Removal (produce RGBA cutout)
        foreground_rgba = self.remover.extract_foreground(
            image_or_path=processed_vehicle_img,
            mask=seg_res.mask,
            color_order=ColorOrder.BGR,
        )

        # 6. Fixed Studio Compositing
        composited_bgr, comp_meta = self.compositor.composite(
            foreground_rgba=foreground_rgba,
            color_order=ColorOrder.BGR,
        )

        # 7. Save outputs
        if output_path is None:
            base_dir = "outputs"
            os.makedirs(base_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(base_dir, f"{stem}_studio.png")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, composited_bgr)

        # Save transparent cutout if requested
        cutout_path = None
        if self.config.save_cutout:
            base, ext = os.path.splitext(output_path)
            cutout_path = f"{base}_cutout.png"
            self.remover.save_cutout(foreground_rgba, cutout_path)

        total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0

        return {
            "success": True,
            "status": "COMPLETED",
            "input_path": os.path.abspath(input_path),
            "output_path": os.path.abspath(output_path),
            "cutout_path": os.path.abspath(cutout_path) if cutout_path else None,
            "image_dimensions": {"width": w, "height": h},
            "detection": det_res.to_dict(),
            "segmentation": seg_res.to_dict(),
            "number_plate": plate_res.to_dict(),
            "compositing": comp_meta,
            "execution_time_ms": round(total_elapsed_ms, 2),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        studio_background: Optional[np.ndarray] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single video frame with temporal stabilization.

        Args:
            frame: Input video frame as BGR numpy array (shape HxWx3).
            studio_background: Optional custom studio background (BGR array).
            state: Mutable state dictionary passed across consecutive frames
                   for temporal smoothing and flicker reduction.

        Returns:
            Tuple of (composited_frame_bgr, metadata_dict).
        """
        start_time = time.perf_counter()
        h, w = frame.shape[:2]

        # 1. Detection
        det_res = self.detector.detect(frame, color_order=ColorOrder.BGR)
        primary_box = det_res.primary_box

        # Temporal Box Smoothing for video
        if state is not None and "prev_box" in state and primary_box is not None:
            prev_b = state["prev_box"]
            # Smooth box coordinates using EMA
            alpha_b = 0.65
            smooth_x1 = int(round(alpha_b * primary_box.x1 + (1 - alpha_b) * prev_b.x1))
            smooth_y1 = int(round(alpha_b * primary_box.y1 + (1 - alpha_b) * prev_b.y1))
            smooth_x2 = int(round(alpha_b * primary_box.x2 + (1 - alpha_b) * prev_b.x2))
            smooth_y2 = int(round(alpha_b * primary_box.y2 + (1 - alpha_b) * prev_b.y2))
            primary_box = BoundingBox(
                x1=smooth_x1,
                y1=smooth_y1,
                x2=smooth_x2,
                y2=smooth_y2,
                confidence=primary_box.confidence,
                class_name=primary_box.class_name,
            )
            state["prev_box"] = primary_box
        elif state is not None and primary_box is not None:
            state["prev_box"] = primary_box

        # 2. Segmentation
        seg_res = self.segmenter.segment(frame, target_box=primary_box, color_order=ColorOrder.BGR)
        current_mask = seg_res.mask

        # Temporal Mask Stabilization for video
        if self.config.enable_temporal_smoothing and state is not None:
            if "prev_mask" in state and state["prev_mask"] is not None and state["prev_mask"].shape == (h, w):
                prev_mask_f = state["prev_mask"].astype(np.float32)
                curr_mask_f = current_mask.astype(np.float32)
                
                # Exponential moving average on mask to prevent boundary flickering
                alpha_m = self.config.mask_ema_alpha
                smoothed_mask_f = alpha_m * curr_mask_f + (1.0 - alpha_m) * prev_mask_f
                stabilized_mask = (smoothed_mask_f > 120).astype(np.uint8) * 255
                state["prev_mask"] = smoothed_mask_f
                current_mask = stabilized_mask
            else:
                state["prev_mask"] = current_mask.astype(np.float32)

        # 3. Plate Replacement (optional on video frames)
        plate_res = PlateResult(found=False, status_note="Skipped for video frame")
        processed_frame = frame.copy()
        if self.config.enable_plate_replacement and primary_box is not None:
            processed_frame, plate_res = self.plate_processor.process(
                image=processed_frame,
                vehicle_box=primary_box,
                color_order=ColorOrder.BGR,
            )

        # 4. Background Removal
        fg_rgba = self.remover.extract_foreground(
            image_or_path=processed_frame,
            mask=current_mask,
            color_order=ColorOrder.BGR,
        )

        # 5. Studio Compositing
        comp_bgr, comp_meta = self.compositor.composite(
            foreground_rgba=fg_rgba,
            background=studio_background,
            color_order=ColorOrder.BGR,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        meta = {
            "success": True,
            "has_detection": det_res.has_detection,
            "box": primary_box.to_dict() if primary_box else None,
            "plate_replaced": plate_res.found,
            "compositing": comp_meta,
            "execution_time_ms": round(elapsed_ms, 2),
        }
        return comp_bgr, meta


# Stable singleton instance for quick calls
_DEFAULT_PIPELINE: Optional[AutomotiveStudioPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> AutomotiveStudioPipeline:
    """Get or initialize the AutomotiveStudioPipeline instance."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None or config is not None:
        _DEFAULT_PIPELINE = AutomotiveStudioPipeline(config)
    return _DEFAULT_PIPELINE


def process_image(
    input_path: str,
    output_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Primary stable entry point for Aparna's backend to process an image.

    Args:
        input_path: Absolute or relative path to the input car image.
        output_path: Optional destination path for the composited studio image.
        config: Optional PipelineConfig.

    Returns:
        Structured result dictionary.
    """
    pipeline = get_pipeline(config)
    return pipeline.process_image(input_path, output_path)


def process_frame(
    frame: np.ndarray,
    studio_background: Optional[np.ndarray] = None,
    state: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Primary stable entry point for Aparna's backend to process a video frame.

    Args:
        frame: Single frame as BGR uint8 numpy array (from OpenCV/FFmpeg).
        studio_background: Optional background BGR array. If None, uses approved studio.
        state: State dictionary preserved across calls for temporal smoothing.
        config: Optional PipelineConfig.

    Returns:
        Tuple of (composited_frame_bgr, metadata_dict).
    """
    pipeline = get_pipeline(config)
    return pipeline.process_frame(frame, studio_background=studio_background, state=state)

"""
Fictional Number Plate Replacement Module.
Locates visible number plates using morphological/contour computer vision analysis,
generates a clearly fictional identifier, and warps it with perspective alignment.
Handles obscured, small, or non-visible plates safely.
"""

import os
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from ai_engine.common.types import BoundingBox, PlateResult, ColorOrder


class PlateProcessor:
    """
    Detects visible vehicle number plates and replaces real registrations
    with a photorealistic fictional studio plate aligned to the vehicle's perspective.
    """

    def __init__(
        self,
        default_fictional_text: str = "STUDIO 01",
        default_subtext: str = "AI AUTOMOTIVE",
        min_plate_aspect: float = 1.8,
        max_plate_aspect: float = 6.0,
        min_plate_area_ratio: float = 0.0003,
        max_plate_area_ratio: float = 0.08,
    ):
        self.default_fictional_text = default_fictional_text
        self.default_subtext = default_subtext
        self.min_plate_aspect = min_plate_aspect
        self.max_plate_aspect = max_plate_aspect
        self.min_plate_area_ratio = min_plate_area_ratio
        self.max_plate_area_ratio = max_plate_area_ratio

    def process(
        self,
        image: np.ndarray,
        vehicle_box: Optional[BoundingBox] = None,
        fictional_text: Optional[str] = None,
        color_order: ColorOrder = ColorOrder.BGR,
    ) -> Tuple[np.ndarray, PlateResult]:
        """
        Detect and replace number plate in the image.

        Args:
            image: Input image (BGR or RGB uint8).
            vehicle_box: Optional bounding box of the vehicle to restrict search to bumper.
            fictional_text: Custom fictional string (e.g. 'STUDIO 01').
            color_order: BGR (OpenCV) or RGB.

        Returns:
            Tuple of (modified_image, PlateResult).
        """
        start_time = time.perf_counter()
        plate_text = fictional_text or self.default_fictional_text
        h, w = image.shape[:2]

        # Convert to BGR for consistent processing if RGB
        if color_order == ColorOrder.RGB:
            bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_img = image.copy()

        # 1. Locate number plate candidate region
        plate_box, quad_pts, conf, note = self._locate_plate(bgr_img, vehicle_box)

        if plate_box is None or quad_pts is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            out_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) if color_order == ColorOrder.RGB else bgr_img
            return out_img, PlateResult(
                found=False,
                status_note=note,
                replaced_text="",
                confidence=0.0,
                execution_time_ms=elapsed_ms,
            )

        # 2. Extract original plate patch to match environmental lighting
        x1, y1, x2, y2 = plate_box.x1, plate_box.y1, plate_box.x2, plate_box.y2
        orig_roi = bgr_img[y1:y2, x1:x2].copy()

        # 3. Generate high-resolution fictional plate
        plate_w = max(240, plate_box.width * 2)
        plate_h = max(60, int(plate_w / 4.5))
        fictional_plate_img = self._generate_fictional_plate(
            width=plate_w,
            height=plate_h,
            text=plate_text,
            subtext=self.default_subtext,
            reference_roi=orig_roi,
        )

        # 4. Warp and composite fictional plate onto vehicle with perspective
        composed_bgr = self._warp_and_blend_plate(
            bgr_img, fictional_plate_img, quad_pts
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        out_img = cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGB) if color_order == ColorOrder.RGB else composed_bgr

        result = PlateResult(
            found=True,
            box=plate_box,
            quadrilateral=quad_pts,
            original_region=orig_roi,
            replaced_text=plate_text,
            confidence=conf,
            status_note="Plate detected and replaced with fictional identifier",
            execution_time_ms=elapsed_ms,
        )
        return out_img, result

    def _locate_plate(
        self, bgr_img: np.ndarray, vehicle_box: Optional[BoundingBox]
    ) -> Tuple[Optional[BoundingBox], Optional[List[Tuple[int, int]]], float, str]:
        """
        Locate plate using Sobel gradient, morphological rectangle closing,
        and geometric contour filtering within the vehicle bumper region.
        """
        img_h, img_w = bgr_img.shape[:2]

        # Determine search region (bumper zone)
        if vehicle_box is not None:
            # Plates are predominantly located in the lower 60% of the vehicle
            vx1, vy1, vx2, vy2 = vehicle_box.x1, vehicle_box.y1, vehicle_box.x2, vehicle_box.y2
            vh = vy2 - vy1
            vw = vx2 - vx1
            
            roi_y1 = max(0, int(vy1 + vh * 0.40))
            roi_y2 = min(img_h, vy2)
            roi_x1 = max(0, int(vx1 + vw * 0.10))
            roi_x2 = min(img_w, int(vx2 - vw * 0.10))
        else:
            roi_y1 = int(img_h * 0.4)
            roi_y2 = img_h
            roi_x1 = int(img_w * 0.1)
            roi_x2 = int(img_w * 0.9)

        if (roi_y2 - roi_y1) < 20 or (roi_x2 - roi_x1) < 40:
            return None, None, 0.0, "Vehicle ROI too small for plate detection"

        roi = bgr_img[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_h, roi_w = roi.shape[:2]
        roi_area = roi_w * roi_h
        total_img_area = img_w * img_h

        # Grayscale & Preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # High-pass filter via Sobel-X (captures vertical stroke edges of license plate characters)
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        abs_sobel = cv2.convertScaleAbs(sobel_x)
        blurred = cv2.GaussianBlur(abs_sobel, (5, 5), 0)

        # Morphological closing to merge letters into a solid plate bounding box
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, rect_kernel)

        # Thresholding
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (total_img_area * self.min_plate_area_ratio) or area > (total_img_area * self.max_plate_area_ratio):
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < 5 or rh < 5:
                continue

            # Ensure aspect ratio is width / height
            if rw < rh:
                rw, rh = rh, rw
                angle += 90.0

            aspect = rw / float(rh)
            if self.min_plate_aspect <= aspect <= self.max_plate_aspect:
                # Check bounding box points
                box_pts = cv2.boxPoints(rect)
                box_pts = np.int32(box_pts)
                
                # Verify brightness contrast of the candidate area in original ROI
                mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                cv2.drawContours(mask, [box_pts], 0, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                # Preference for high-contrast, moderately bright regions (plates)
                score = area * (1.0 + (mean_val / 255.0))
                candidates.append((score, box_pts, rw, rh, aspect))

        if not candidates:
            # Secondary pass: color thresholding for bright white/yellow plate background
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # White plate mask
            white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 50, 255]))
            # Yellow plate mask (e.g. UK rear plates)
            yellow_mask = cv2.inRange(hsv, np.array([15, 70, 120]), np.array([35, 255, 255]))
            combined_color = cv2.bitwise_or(white_mask, yellow_mask)
            
            kernel_color = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
            color_closed = cv2.morphologyEx(combined_color, cv2.MORPH_CLOSE, kernel_color)
            color_cnts, _ = cv2.findContours(color_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in color_cnts:
                area = cv2.contourArea(cnt)
                if area < (total_img_area * self.min_plate_area_ratio) or area > (total_img_area * self.max_plate_area_ratio):
                    continue
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect
                if rw < rh:
                    rw, rh = rh, rw
                aspect = rw / float(rh + 1e-5)
                if self.min_plate_aspect <= aspect <= self.max_plate_aspect:
                    box_pts = cv2.boxPoints(rect)
                    box_pts = np.int32(box_pts)
                    candidates.append((area * 0.8, box_pts, rw, rh, aspect))

        if not candidates:
            return None, None, 0.0, "No visible number plate detected in bumper region"

        # Select highest-scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidate = candidates[0]
        box_pts = best_candidate[1]

        # Convert coordinates back from ROI to full image
        global_quad = []
        for pt in box_pts:
            global_quad.append((int(pt[0] + roi_x1), int(pt[1] + roi_y1)))

        # Order quadrilateral points: top-left, top-right, bottom-right, bottom-left
        ordered_quad = self._order_quad_points(global_quad)

        # Compute axis-aligned bounding box
        xs = [p[0] for p in ordered_quad]
        ys = [p[1] for p in ordered_quad]
        bx1 = max(0, min(xs))
        by1 = max(0, min(ys))
        bx2 = min(img_w, max(xs))
        by2 = min(img_h, max(ys))

        plate_box = BoundingBox(
            x1=bx1,
            y1=by1,
            x2=bx2,
            y2=by2,
            confidence=0.88,
            class_id=1,
            class_name="number_plate",
        )

        return plate_box, ordered_quad, 0.88, "Plate successfully located"

    def _order_quad_points(self, pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
        pts_arr = np.array(pts, dtype=np.float32)
        # Sum of coords: top-left has smallest sum, bottom-right has largest sum
        s = pts_arr.sum(axis=1)
        tl = pts_arr[np.argmin(s)]
        br = pts_arr[np.argmax(s)]

        # Difference of coords (y - x): top-right has smallest diff, bottom-left has largest diff
        diff = np.diff(pts_arr, axis=1)
        tr = pts_arr[np.argmin(diff)]
        bl = pts_arr[np.argmax(diff)]

        return [
            (int(tl[0]), int(tl[1])),
            (int(tr[0]), int(tr[1])),
            (int(br[0]), int(br[1])),
            (int(bl[0]), int(bl[1])),
        ]

    def _generate_fictional_plate(
        self,
        width: int,
        height: int,
        text: str,
        subtext: str,
        reference_roi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Synthesize a realistic fictional studio license plate:
        - Reflective acrylic plate with embossed border
        - European/studio deep blue left band with studio logo
        - Bold black fictional characters
        - Subtext & mounting screws
        - Environmental lighting modulation
        """
        plate = np.ones((height, width, 3), dtype=np.uint8) * 242

        # 1. Subtle horizontal specular gradient
        xx = np.linspace(0.92, 1.05, width, dtype=np.float32)
        plate = np.clip(plate * xx[None, :, None], 0, 255).astype(np.uint8)

        # 2. Outer black border and thin bevel
        border_thickness = max(2, int(height * 0.04))
        cv2.rectangle(
            plate,
            (border_thickness, border_thickness),
            (width - border_thickness, height - border_thickness),
            (20, 20, 20),
            border_thickness,
        )

        # 3. Studio Left Band (Navy Blue European/Showroom style)
        band_w = int(width * 0.12)
        band_bgr = (140, 60, 20)  # Rich deep studio blue
        cv2.rectangle(
            plate,
            (border_thickness, border_thickness),
            (band_w, height - border_thickness),
            band_bgr,
            -1,
        )

        # Studio crest/circle on the blue band
        circle_cx = band_w // 2
        circle_cy = height // 2 - int(height * 0.10)
        circle_r = max(4, int(height * 0.18))
        cv2.circle(plate, (circle_cx, circle_cy), circle_r, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(
            plate,
            "AI",
            (circle_cx - int(circle_r * 0.65), circle_cy + int(circle_r * 0.45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 * (height / 60.0),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 4. Main Fictional Text (e.g. "STUDIO 01")
        font_scale = 1.15 * (height / 60.0)
        thickness = max(2, int(2.5 * (height / 60.0)))
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Center text horizontally between right side of blue band and right edge
        avail_w = width - band_w - border_thickness
        text_x = band_w + (avail_w - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2 - int(height * 0.04)

        # Embossed shadow under text
        cv2.putText(
            plate,
            text,
            (text_x + 1, text_y + 1),
            font,
            font_scale,
            (160, 160, 160),
            thickness,
            cv2.LINE_AA,
        )
        # Main bold text
        cv2.putText(
            plate,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (18, 18, 18),
            thickness,
            cv2.LINE_AA,
        )

        # 5. Subtext at bottom
        if subtext:
            sub_scale = 0.32 * (height / 60.0)
            sub_size, _ = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
            sub_x = band_w + (avail_w - sub_size[0]) // 2
            sub_y = height - int(border_thickness * 1.5) - 2
            cv2.putText(
                plate,
                subtext,
                (sub_x, sub_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                sub_scale,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )

        # 6. Mounting screws / caps
        screw_y = height // 2
        for screw_x in [band_w + int(avail_w * 0.06), width - border_thickness - int(avail_w * 0.06)]:
            cv2.circle(plate, (screw_x, screw_y), max(2, int(height * 0.05)), (110, 110, 110), -1)
            cv2.circle(plate, (screw_x, screw_y), max(2, int(height * 0.05)), (40, 40, 40), 1)

        # 7. Ambient lighting adaptation
        if reference_roi is not None and reference_roi.size > 0:
            mean_bgr = cv2.mean(reference_roi)[:3]
            # Modulate plate tone to match car's ambient brightness and tint
            ambient_factor = np.mean(mean_bgr) / 180.0
            ambient_factor = np.clip(ambient_factor, 0.55, 1.25)
            plate = np.clip(plate.astype(np.float32) * ambient_factor, 0, 255).astype(np.uint8)

        return plate

    def _warp_and_blend_plate(
        self,
        base_img: np.ndarray,
        fictional_plate: np.ndarray,
        dest_quad: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Warp fictional plate to match destination quad perspective and blend into base image.
        """
        ph, pw = fictional_plate.shape[:2]
        src_pts = np.array(
            [[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32
        )
        dst_pts = np.array(dest_quad, dtype=np.float32)

        # Perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        h, w = base_img.shape[:2]

        # Warp fictional plate onto image plane
        warped_plate = cv2.warpPerspective(
            fictional_plate,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Generate smooth mask for blending
        plate_mask = np.ones((ph, pw), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            plate_mask,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Feather warped mask edges to prevent hard seam
        feathered_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0).astype(np.float32) / 255.0
        mask_3ch = feathered_mask[:, :, None]

        # Blend
        blended = (
            base_img.astype(np.float32) * (1.0 - mask_3ch)
            + warped_plate.astype(np.float32) * mask_3ch
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

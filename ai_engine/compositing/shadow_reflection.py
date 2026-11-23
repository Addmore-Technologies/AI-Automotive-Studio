"""
Shadow and Reflection Generation Module.
Synthesizes realistic contact shadows, ambient occlusion floorpan shadows,
and polished studio floor reflections for automotive staging.
"""

from typing import Tuple
import cv2
import numpy as np


class ShadowReflectionGenerator:
    """
    Generates realistic contact shadows and floor reflections for isolated vehicles.
    Simulates high-end showroom studio lighting and polished epoxy floors.
    """

    def __init__(
        self,
        shadow_intensity: float = 0.70,
        shadow_blur_radius: int = 15,
        reflection_opacity: float = 0.28,
        reflection_falloff_height: float = 0.35,
        reflection_blur_kernel: int = 11,
    ):
        self.shadow_intensity = shadow_intensity
        self.shadow_blur_radius = shadow_blur_radius if shadow_blur_radius % 2 == 1 else shadow_blur_radius + 1
        self.reflection_opacity = reflection_opacity
        self.reflection_falloff_height = reflection_falloff_height
        self.reflection_blur_kernel = reflection_blur_kernel if reflection_blur_kernel % 2 == 1 else reflection_blur_kernel + 1

    def create_contact_shadow(
        self,
        vehicle_mask: np.ndarray,
        canvas_shape: Tuple[int, int],
        vehicle_offset: Tuple[int, int],
    ) -> np.ndarray:
        """
        Synthesize multi-tier contact and ambient shadows on the studio floor:
        1. Tight tyre contact patches (high density ambient occlusion).
        2. Diffused underbody chassis shadow.
        3. Soft ambient cast shadow fading outward.

        Args:
            vehicle_mask: Binary mask of the vehicle (uint8 0 or 255).
            canvas_shape: (height, width) of the studio background.
            vehicle_offset: (x, y) placement of top-left corner of vehicle on canvas.

        Returns:
            Float32 shadow attenuation map (shape HxW, values 0.0 to 1.0,
            where 0.0 = full shadow absorption, 1.0 = no shadow).
        """
        canvas_h, canvas_w = canvas_shape[:2]
        vh, vw = vehicle_mask.shape[:2]
        vx, vy = vehicle_offset

        shadow_alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        if np.count_nonzero(vehicle_mask) == 0:
            return 1.0 - shadow_alpha

        # Find bottom profile of the vehicle mask (where tyres contact ground)
        bottom_profile = np.zeros(vw, dtype=np.int32)
        for col in range(vw):
            rows = np.where(vehicle_mask[:, col] > 0)[0]
            if len(rows) > 0:
                bottom_profile[col] = rows[-1]
            else:
                bottom_profile[col] = -1

        # Lowest contact points (tyre contact y)
        valid_cols = np.where(bottom_profile > 0)[0]
        if len(valid_cols) == 0:
            return 1.0 - shadow_alpha

        lowest_y = int(np.max(bottom_profile[valid_cols]))
        ground_canvas_y = vy + lowest_y

        # 1. Tyre contact patches: find local maxima in bottom profile (tyres)
        tyre_patches = []
        step = max(5, vw // 30)
        for i in range(step, len(valid_cols) - step, step):
            col_idx = valid_cols[i]
            y_val = bottom_profile[col_idx]
            if y_val >= (lowest_y - int(vh * 0.06)):
                tyre_patches.append((col_idx, y_val))

        # Render tyre contact ellipses
        contact_layer = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        tyre_shadow_w = max(18, int(vw * 0.07))
        tyre_shadow_h = max(6, int(vh * 0.022))

        for (col_idx, y_val) in tyre_patches:
            cx = vx + col_idx
            cy = vy + y_val + 2
            if 0 <= cx < canvas_w and 0 <= cy < canvas_h:
                cv2.ellipse(
                    contact_layer,
                    (cx, cy),
                    (tyre_shadow_w, tyre_shadow_h),
                    0,
                    0,
                    360,
                    1.0,
                    -1,
                )

        contact_layer = cv2.GaussianBlur(contact_layer, (7, 7), 0)

        # 2. Underbody Chassis Ambient Occlusion
        # Extract lower 25% of vehicle silhouette for underbody shadow
        underbody_h = int(vh * 0.25)
        underbody_mask = np.zeros((vh, vw), dtype=np.uint8)
        underbody_mask[max(0, vh - underbody_h):, :] = vehicle_mask[max(0, vh - underbody_h):, :]

        # Squeeze vertically to ground plane
        underbody_shadow_h = max(12, int(vh * 0.07))
        squeezed_shadow = cv2.resize(
            underbody_mask.astype(np.float32) / 255.0,
            (vw, underbody_shadow_h),
            interpolation=cv2.INTER_LINEAR,
        )

        chassis_layer = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        chassis_y1 = ground_canvas_y - int(underbody_shadow_h * 0.4)
        chassis_y2 = chassis_y1 + underbody_shadow_h
        chassis_x1 = vx
        chassis_x2 = vx + vw

        # Clip bounds
        src_y1 = max(0, -chassis_y1)
        src_y2 = underbody_shadow_h - max(0, chassis_y2 - canvas_h)
        dst_y1 = max(0, chassis_y1)
        dst_y2 = min(canvas_h, chassis_y2)

        src_x1 = max(0, -chassis_x1)
        src_x2 = vw - max(0, chassis_x2 - canvas_w)
        dst_x1 = max(0, chassis_x1)
        dst_x2 = min(canvas_w, chassis_x2)

        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
            chassis_layer[dst_y1:dst_y2, dst_x1:dst_x2] = squeezed_shadow[src_y1:src_y2, src_x1:src_x2]

        chassis_layer = cv2.GaussianBlur(chassis_layer, (self.shadow_blur_radius, self.shadow_blur_radius), 0)

        # 3. Soft ambient cast pool
        pool_layer = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        pool_cx = vx + vw // 2
        pool_cy = ground_canvas_y + int(vh * 0.02)
        pool_rx = int(vw * 0.54)
        pool_ry = int(vh * 0.09)
        if 0 <= pool_cx < canvas_w and 0 <= pool_cy < canvas_h:
            cv2.ellipse(
                pool_layer,
                (pool_cx, pool_cy),
                (pool_rx, pool_ry),
                0,
                0,
                360,
                0.45,
                -1,
            )
        pool_layer = cv2.GaussianBlur(pool_layer, (31, 31), 0)

        # Combine shadow layers with weights
        combined_shadow = (
            contact_layer * 0.95
            + chassis_layer * 0.70
            + pool_layer * 0.40
        )
        combined_shadow = np.clip(combined_shadow * self.shadow_intensity, 0.0, 0.92)

        # Attenuation map: 1.0 means full light, (1.0 - shadow) reduces brightness
        return 1.0 - combined_shadow

    def create_floor_reflection(
        self,
        vehicle_rgba: np.ndarray,
        canvas_shape: Tuple[int, int],
        vehicle_offset: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize studio floor reflection:
        1. Vertical flip of the vehicle foreground.
        2. Vertical compression to match perspective floor angle.
        3. Progressive distance falloff (reflection fades rapidly away from tyres).
        4. Surface roughness blur (Gaussian blur increasing with distance).

        Args:
            vehicle_rgba: Isolated vehicle RGBA image.
            canvas_shape: (height, width) of studio background.
            vehicle_offset: (x, y) placement on canvas.

        Returns:
            Tuple of (reflection_rgb, reflection_alpha) both of canvas_shape.
        """
        canvas_h, canvas_w = canvas_shape[:2]
        vh, vw = vehicle_rgba.shape[:2]
        vx, vy = vehicle_offset

        reflection_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        reflection_alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        if self.reflection_opacity <= 0.0:
            return reflection_rgb, reflection_alpha

        # Extract vehicle RGB and Alpha
        v_rgb = vehicle_rgba[:, :, :3]
        v_alpha = vehicle_rgba[:, :, 3].astype(np.float32) / 255.0

        # Find ground line (lowest opaque pixel in vehicle)
        mask = vehicle_rgba[:, :, 3] > 30
        rows = np.where(mask)[0]
        if len(rows) == 0:
            return reflection_rgb, reflection_alpha

        ground_y_local = int(rows[-1])
        ground_canvas_y = vy + ground_y_local

        # 1. Vertically flip the vehicle image
        flipped_rgb = cv2.flip(v_rgb, 0)
        flipped_alpha = cv2.flip(v_alpha, 0)

        # The flipped ground line is at (vh - 1 - ground_y_local)
        flip_ground_y = vh - 1 - ground_y_local
        
        # Crop the portion that extends downward from the ground contact line
        refl_content_rgb = flipped_rgb[flip_ground_y:, :]
        refl_content_alpha = flipped_alpha[flip_ground_y:, :]
        rf_h = refl_content_rgb.shape[0]

        if rf_h <= 0:
            return reflection_rgb, reflection_alpha

        # 2. Vertical compression (perspective foreshortening)
        comp_factor = 0.70
        comp_h = max(4, int(rf_h * comp_factor))
        comp_rgb = cv2.resize(refl_content_rgb, (vw, comp_h), interpolation=cv2.INTER_LINEAR)
        comp_alpha = cv2.resize(refl_content_alpha, (vw, comp_h), interpolation=cv2.INTER_LINEAR)

        # 3. Progressive distance falloff gradient
        max_refl_h = max(
            1,
            int(vh * self.reflection_falloff_height)
        )

        comp_h = min(comp_h, max_refl_h)

        if comp_h <= 0:
            return reflection_rgb, reflection_alpha

        comp_rgb = comp_rgb[:comp_h, :]
        comp_alpha = comp_alpha[:comp_h, :]

        y_indices = np.arange(comp_h, dtype=np.float32)

        falloff = (
            1.0 - (y_indices / float(comp_h))
        ) ** 1.8

        comp_alpha = (
            comp_alpha
            * falloff[:, None]
            * self.reflection_opacity
        )

        # 4. Surface roughness blur
        k = self.reflection_blur_kernel
        comp_rgb = cv2.GaussianBlur(comp_rgb, (k, k), 0)

        # 5. Composite onto canvas coordinates
        dst_y1 = ground_canvas_y
        dst_y2 = min(canvas_h, dst_y1 + comp_h)
        actual_h = dst_y2 - dst_y1

        if actual_h > 0:
            dst_x1 = max(0, vx)
            dst_x2 = min(canvas_w, vx + vw)
            src_x1 = dst_x1 - vx
            src_x2 = src_x1 + (dst_x2 - dst_x1)

            reflection_rgb[dst_y1:dst_y2, dst_x1:dst_x2] = comp_rgb[:actual_h, src_x1:src_x2]
            reflection_alpha[dst_y1:dst_y2, dst_x1:dst_x2] = comp_alpha[:actual_h, src_x1:src_x2]

        return reflection_rgb, reflection_alpha

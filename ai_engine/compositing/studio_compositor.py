"""
Studio Compositor Module.
Composites isolated vehicles onto the approved fixed studio environment with:
- Proportional aspect-ratio-preserving scaling
- Ground plane positioning and tyre alignment
- Multi-layer compositing (background -> reflection -> shadow -> vehicle)
- Realistic ambient light integration
"""

import os
from typing import Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np

from ai_engine.common.types import ColorOrder
from ai_engine.compositing.shadow_reflection import ShadowReflectionGenerator


class StudioCompositor:
    """
    Composites isolated vehicle cutouts into the approved fixed studio background.
    """

    def __init__(
        self,
        approved_background_path: Optional[str] = None,
        vehicle_scale_pct: float = 0.72,
        ground_line_pct: float = 0.74,
        enable_shadows: bool = True,
        enable_reflections: bool = True,
        shadow_intensity: float = 0.70,
        reflection_opacity: float = 0.28,
    ):
        self.vehicle_scale_pct = vehicle_scale_pct
        self.ground_line_pct = ground_line_pct
        self.enable_shadows = enable_shadows
        self.enable_reflections = enable_reflections
        
        self.background_path = self._resolve_background_path(approved_background_path)
        self.base_background_bgr = self._load_background()
        
        self.shadow_reflection_gen = ShadowReflectionGenerator(
            shadow_intensity=shadow_intensity,
            reflection_opacity=reflection_opacity,
        )

    def _resolve_background_path(self, path: Optional[str]) -> str:
        if path and os.path.exists(path):
            return path
        candidates = [
            os.path.abspath("studio_assets/backgrounds/approved_studio.png"),
            os.path.abspath("../studio_assets/backgrounds/approved_studio.png"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "studio_assets/backgrounds/approved_studio.png"

    def _load_background(self) -> np.ndarray:
        if os.path.exists(self.background_path):
            bg = cv2.imread(self.background_path)
            if bg is not None:
                return bg
        
        # If not on disk, generate high-definition default studio (1920x1080)
        from scripts.generate_studio_background import generate_approved_studio
        return generate_approved_studio(output_path=self.background_path)

    def composite(
        self,
        foreground_rgba: np.ndarray,
        background: Optional[np.ndarray] = None,
        color_order: ColorOrder = ColorOrder.BGR,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Composite vehicle foreground onto studio background.

        Args:
            foreground_rgba: Isolated vehicle RGBA array (HxWx4).
            background: Optional custom background image. If None, uses approved studio asset.
            color_order: BGR (default OpenCV) or RGB.

        Returns:
            Tuple of (composited_image, metadata_dict).
        """
        # Select background canvas (BGR)
        if background is not None:
            if color_order == ColorOrder.RGB and background.ndim == 3:
                canvas_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
            else:
                canvas_bgr = background.copy()
        else:
            canvas_bgr = self.base_background_bgr.copy()

        canvas_h, canvas_w = canvas_bgr.shape[:2]

        # 1. Proportional vehicle scaling preserving aspect ratio
        scaled_rgba, scale_factor = self._scale_vehicle(foreground_rgba, canvas_w, canvas_h)
        vh, vw = scaled_rgba.shape[:2]

        # 2. Ground plane alignment: place tyres on the studio floor line
        v_alpha = scaled_rgba[:, :, 3]
        rows = np.where(v_alpha > 30)[0]
        if len(rows) > 0:
            tyre_contact_y = int(rows[-1])
        else:
            tyre_contact_y = vh - 1

        target_ground_y = int(canvas_h * self.ground_line_pct)
        offset_y = target_ground_y - tyre_contact_y
        offset_x = (canvas_w - vw) // 2  # Stage center alignment

        # Ensure vehicle fits vertically within canvas
        if (offset_y + vh) > canvas_h:
            offset_y = canvas_h - vh
        if offset_y < 0:
            offset_y = 0

        vehicle_offset = (offset_x, offset_y)

        # 3. Layer 1: Floor Reflection
        composite_bgr = canvas_bgr.copy().astype(np.float32)
        
        if self.enable_reflections:
            # Note: scaled_rgba is RGBA [R, G, B, A]
            # Convert scaled_rgba RGB channels to BGR for blending with canvas_bgr
            scaled_bgra = np.zeros_like(scaled_rgba)
            scaled_bgra[:, :, :3] = cv2.cvtColor(scaled_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            scaled_bgra[:, :, 3] = scaled_rgba[:, :, 3]

            refl_bgr, refl_alpha = self.shadow_reflection_gen.create_floor_reflection(
                vehicle_rgba=scaled_bgra,
                canvas_shape=(canvas_h, canvas_w),
                vehicle_offset=vehicle_offset,
            )
            alpha_3ch = refl_alpha[:, :, None]
            composite_bgr = composite_bgr * (1.0 - alpha_3ch) + refl_bgr.astype(np.float32) * alpha_3ch

        # 4. Layer 2: Contact & Ambient Shadows
        if self.enable_shadows:
            scaled_mask = (scaled_rgba[:, :, 3] > 30).astype(np.uint8) * 255
            shadow_map = self.shadow_reflection_gen.create_contact_shadow(
                vehicle_mask=scaled_mask,
                canvas_shape=(canvas_h, canvas_w),
                vehicle_offset=vehicle_offset,
            )
            # Multiply shadow attenuation onto background & reflection
            composite_bgr = composite_bgr * shadow_map[:, :, None]

        # 5. Layer 3: Vehicle Foreground
        v_bgr = cv2.cvtColor(scaled_rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
        v_alpha_f = (scaled_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

        # Calculate bounding coordinates on canvas
        vy1 = max(0, offset_y)
        vy2 = min(canvas_h, offset_y + vh)
        vx1 = max(0, offset_x)
        vx2 = min(canvas_w, offset_x + vw)

        sy1 = vy1 - offset_y
        sy2 = sy1 + (vy2 - vy1)
        sx1 = vx1 - offset_x
        sx2 = sx1 + (vx2 - vx1)

        target_region = composite_bgr[vy1:vy2, vx1:vx2]
        fg_crop = v_bgr[sy1:sy2, sx1:sx2]
        alpha_crop = v_alpha_f[sy1:sy2, sx1:sx2]

        # Alpha composite vehicle onto background
        composite_bgr[vy1:vy2, vx1:vx2] = (
            target_region * (1.0 - alpha_crop) + fg_crop * alpha_crop
        )

        final_composite_uint8 = np.clip(composite_bgr, 0, 255).astype(np.uint8)

        # Convert to target color order
        if color_order == ColorOrder.RGB:
            final_output = cv2.cvtColor(final_composite_uint8, cv2.COLOR_BGR2RGB)
        else:
            final_output = final_composite_uint8

        metadata = {
            "canvas_dimensions": [canvas_w, canvas_h],
            "vehicle_dimensions": [vw, vh],
            "scale_factor": round(float(scale_factor), 4),
            "vehicle_offset": [offset_x, offset_y],
            "ground_line_y": target_ground_y,
            "shadows_enabled": self.enable_shadows,
            "reflections_enabled": self.enable_reflections,
        }

        return final_output, metadata

    def _scale_vehicle(
        self, rgba: np.ndarray, canvas_w: int, canvas_h: int
    ) -> Tuple[np.ndarray, float]:
        """
        Scale vehicle to target width percentage while strictly preserving aspect ratio.
        """
        # Crop tight around vehicle alpha first to eliminate excess empty transparent border
        alpha = rgba[:, :, 3]
        rows = np.where(alpha > 10)[0]
        cols = np.where(alpha > 10)[1]

        if len(rows) == 0 or len(cols) == 0:
            return rgba, 1.0

        r_min, r_max = rows[0], rows[-1]
        c_min, c_max = cols[0], cols[-1]

        tight_rgba = rgba[r_min:r_max + 1, c_min:c_max + 1]
        th, tw = tight_rgba.shape[:2]

        target_w = int(canvas_w * self.vehicle_scale_pct)
        scale = target_w / float(tw)

        # Ensure height doesn't exceed 65% of canvas height
        if (th * scale) > (canvas_h * 0.65):
            scale = (canvas_h * 0.65) / float(th)

        new_w = max(10, int(round(tw * scale)))
        new_h = max(10, int(round(th * scale)))

        scaled = cv2.resize(tight_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return scaled, scale

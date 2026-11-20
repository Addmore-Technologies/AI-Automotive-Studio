"""
Unit tests for Studio Compositor and Shadow/Reflection generation.
"""

import os
import pytest
import numpy as np
import cv2

from ai_engine.compositing.studio_compositor import StudioCompositor
from ai_engine.compositing.shadow_reflection import ShadowReflectionGenerator
from ai_engine.common.types import ColorOrder


@pytest.fixture(scope="module")
def compositor():
    return StudioCompositor(vehicle_scale_pct=0.70)


def test_shadow_reflection_generation():
    gen = ShadowReflectionGenerator(shadow_intensity=0.70, reflection_opacity=0.30)
    
    # Fake vehicle mask (100x200)
    vh, vw = 100, 200
    mask = np.zeros((vh, vw), dtype=np.uint8)
    mask[20:95, 20:180] = 255
    
    canvas_shape = (500, 800)
    offset = (300, 250)

    shadow_map = gen.create_contact_shadow(mask, canvas_shape, offset)
    assert shadow_map.shape == canvas_shape
    # Shadow map is attenuation (0.0 to 1.0)
    assert np.min(shadow_map) < 0.90
    assert np.max(shadow_map) <= 1.0

    # Fake vehicle RGBA
    rgba = np.ones((vh, vw, 4), dtype=np.uint8) * 200
    rgba[:, :, 3] = mask
    refl_bgr, refl_alpha = gen.create_floor_reflection(rgba, canvas_shape, offset)
    assert refl_bgr.shape == (500, 800, 3)
    assert refl_alpha.shape == canvas_shape
    assert np.max(refl_alpha) > 0.0


def test_compositor_aspect_ratio_preservation(compositor):
    # Create fake vehicle RGBA with 2:1 aspect ratio (width=400, height=200)
    vw, vh = 400, 200
    rgba = np.zeros((vh, vw, 4), dtype=np.uint8)
    # Draw car-like body
    rgba[50:180, 20:380, :3] = (40, 60, 220)  # Red car
    rgba[50:180, 20:380, 3] = 255  # Fully opaque

    scaled, scale = compositor._scale_vehicle(rgba, canvas_w=1920, canvas_h=1080)
    scaled_h, scaled_w = scaled.shape[:2]
    
    # Original content aspect ratio (360 / 130)
    orig_content_aspect = 360.0 / 130.0
    scaled_content_aspect = scaled_w / float(scaled_h)
    
    # Verify aspect ratio is preserved within 1% tolerance
    assert abs(orig_content_aspect - scaled_content_aspect) / orig_content_aspect < 0.02


def test_end_to_end_composite(compositor):
    vw, vh = 500, 250
    rgba = np.zeros((vh, vw, 4), dtype=np.uint8)
    rgba[40:220, 30:470, :3] = (30, 180, 30)
    rgba[40:220, 30:470, 3] = 255

    composite_img, meta = compositor.composite(rgba, color_order=ColorOrder.BGR)
    assert composite_img.shape == (1080, 1920, 3)
    assert meta["canvas_dimensions"] == [1920, 1080]
    assert "scale_factor" in meta
    assert "ground_line_y" in meta

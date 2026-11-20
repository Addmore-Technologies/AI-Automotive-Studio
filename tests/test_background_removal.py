"""
Unit tests for Background Removal and Transparent Cutout.
"""

import os
import pytest
import numpy as np
from PIL import Image

from ai_engine.background_removal.remover import BackgroundRemover
from ai_engine.common.types import ColorOrder


@pytest.fixture
def remover():
    return BackgroundRemover(feather_radius=2, despill_edges=True)


def test_extract_foreground_dimensions_and_channels(remover):
    h, w = 200, 300
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    mask = np.zeros((h, w), dtype=np.uint8)
    # Circle in center
    mask[50:150, 80:220] = 255

    rgba = remover.extract_foreground(img, mask, color_order=ColorOrder.BGR)
    assert rgba.shape == (h, w, 4)
    assert rgba.dtype == np.uint8

    # Outside the circle must be completely transparent (alpha == 0)
    assert rgba[10, 10, 3] == 0
    # Inside the center must be fully opaque (alpha == 255)
    assert rgba[100, 150, 3] == 255


def test_save_and_load_cutout(remover, tmp_path):
    h, w = 150, 200
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:120, 40:160] = 255

    rgba = remover.extract_foreground(img, mask)
    out_file = str(tmp_path / "test_cutout.png")
    remover.save_cutout(rgba, out_file)

    assert os.path.exists(out_file)
    loaded = Image.open(out_file)
    assert loaded.mode == "RGBA"
    assert loaded.size == (w, h)

"""
Unit tests for Fictional Number Plate Replacement.
"""

import os
import pytest
import numpy as np
import cv2

from ai_engine.number_plate.plate_processor import PlateProcessor
from ai_engine.common.types import BoundingBox, ColorOrder

SAMPLE_DIR = os.path.abspath("sample_data/input")


@pytest.fixture
def plate_proc():
    return PlateProcessor(default_fictional_text="STUDIO 01")


def test_plate_processor_initialization(plate_proc):
    assert plate_proc.default_fictional_text == "STUDIO 01"


def test_fictional_plate_generation(plate_proc):
    plate_img = plate_proc._generate_fictional_plate(
        width=400,
        height=100,
        text="STUDIO 01",
        subtext="AI AUTOMOTIVE",
    )
    assert plate_img.shape == (100, 400, 3)
    assert plate_img.dtype == np.uint8
    # Ensure not uniform blank
    assert np.std(plate_img) > 10.0


def test_process_rear_plate(plate_proc):
    img_path = os.path.join(SAMPLE_DIR, "car_rear_plate.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_rear_plate.jpg not found")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    v_box = BoundingBox(x1=int(w * 0.1), y1=int(h * 0.1), x2=int(w * 0.9), y2=int(h * 0.95))

    out_img, res = plate_proc.process(img, vehicle_box=v_box, fictional_text="STUDIO 01")
    assert out_img.shape == img.shape
    # If detected, replaced_text should match
    if res.found:
        assert res.replaced_text == "STUDIO 01"
        assert res.box is not None
        assert res.quadrilateral is not None
        assert len(res.quadrilateral) == 4


def test_non_visible_plate_handling(plate_proc):
    # Pure blank image with no vehicle or plate
    blank = np.zeros((400, 600, 3), dtype=np.uint8)
    out_img, res = plate_proc.process(blank)
    assert res.found is False
    assert res.replaced_text == ""
    assert "No visible number plate" in res.status_note or "too small" in res.status_note
    # Input must be safely preserved
    np.testing.assert_array_equal(out_img, blank)

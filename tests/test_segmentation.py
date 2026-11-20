"""
Unit and integration tests for Vehicle Segmentation.
"""

import os
import pytest
import numpy as np
import cv2

from ai_engine.segmentation.vehicle_segmenter import VehicleSegmenter
from ai_engine.common.types import BoundingBox, ColorOrder

SAMPLE_DIR = os.path.abspath("sample_data/input")


@pytest.fixture(scope="module")
def segmenter():
    return VehicleSegmenter(confidence_threshold=0.30, refine_edges=True)


def test_segmenter_initialization(segmenter):
    assert segmenter is not None
    assert segmenter.model is not None


def test_segment_car_sedan(segmenter):
    img_path = os.path.join(SAMPLE_DIR, "car_sedan_front.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_sedan_front.jpg not found")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    res = segmenter.segment(img)
    assert res.mask is not None
    # Dimension matching input image
    assert res.mask.shape == (h, w)
    # Binary mask values 0 or 255
    unique_vals = np.unique(res.mask)
    assert set(unique_vals).issubset({0, 255})
    # Coverage check
    assert np.count_nonzero(res.mask) > 1000
    assert res.refined is True
    assert "status" in res.details


def test_segment_with_target_box(segmenter):
    img_path = os.path.join(SAMPLE_DIR, "car_sports_red.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_sports_red.jpg not found")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    box = BoundingBox(x1=int(w * 0.1), y1=int(h * 0.1), x2=int(w * 0.9), y2=int(h * 0.9))

    res = segmenter.segment(img, target_box=box)
    assert res.mask.shape == (h, w)
    assert np.count_nonzero(res.mask) > 0


def test_segment_blank_image(segmenter):
    blank = np.zeros((300, 400, 3), dtype=np.uint8)
    res = segmenter.segment(blank)
    assert res.mask.shape == (300, 400)
    assert np.count_nonzero(res.mask) == 0
    assert res.details.get("status") == "empty_mask"

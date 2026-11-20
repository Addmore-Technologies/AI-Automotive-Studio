"""
Unit and integration tests for Vehicle Detection.
"""

import os
import pytest
import numpy as np
import cv2

from ai_engine.detection.vehicle_detector import VehicleDetector
from ai_engine.common.types import MultiVehiclePolicy, ColorOrder, BoundingBox


SAMPLE_DIR = os.path.abspath("sample_data/input")


@pytest.fixture(scope="module")
def detector():
    return VehicleDetector(confidence_threshold=0.30)


def test_detector_initialization(detector):
    assert detector is not None
    assert detector.model is not None


def test_detect_sedan(detector):
    img_path = os.path.join(SAMPLE_DIR, "car_sedan_front.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_sedan_front.jpg not found")

    result = detector.detect(img_path)
    assert result.has_detection is True
    assert len(result.boxes) >= 1
    assert result.primary_box is not None
    assert result.primary_box.class_name in ["car", "truck", "vehicle"]
    assert result.primary_box.width > 100
    assert result.primary_box.height > 50


def test_detect_suv(detector):
    img_path = os.path.join(SAMPLE_DIR, "car_suv_side.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_suv_side.jpg not found")

    result = detector.detect(img_path)
    assert result.has_detection is True
    assert result.primary_box is not None
    assert result.primary_box.confidence > 0.40


def test_detect_sports_car(detector):
    img_path = os.path.join(SAMPLE_DIR, "car_sports_red.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_sports_red.jpg not found")

    result = detector.detect(img_path)
    assert result.has_detection is True
    assert result.primary_box is not None


def test_no_vehicle_detection(detector):
    # Pure noise / blank background
    blank = np.zeros((400, 600, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert result.has_detection is False
    assert len(result.boxes) == 0
    assert result.primary_box is None


def test_multi_vehicle_policies():
    boxes = [
        BoundingBox(x1=50, y1=50, x2=150, y2=150, confidence=0.85),   # area: 10000, center: (100, 100)
        BoundingBox(x1=300, y1=300, x2=550, y2=550, confidence=0.70), # area: 62500, center: (425, 425)
        BoundingBox(x1=200, y1=200, x2=320, y2=320, confidence=0.98), # area: 14400, center: (260, 260)
    ]
    det = VehicleDetector(multi_vehicle_policy=MultiVehiclePolicy.LARGEST)
    idx_largest = det._resolve_primary_vehicle(boxes, 600, 600)
    assert idx_largest == 1  # Largest area

    det_conf = VehicleDetector(multi_vehicle_policy=MultiVehiclePolicy.HIGHEST_CONF)
    idx_conf = det_conf._resolve_primary_vehicle(boxes, 600, 600)
    assert idx_conf == 2  # Highest conf (0.98)

    det_center = VehicleDetector(multi_vehicle_policy=MultiVehiclePolicy.CENTER)
    idx_center = det_center._resolve_primary_vehicle(boxes, 600, 600)
    # Center is at (300, 300); box 2 center is (260, 260) -> closest to center
    assert idx_center == 2


def test_rgb_color_order_support(detector):
    img_path = os.path.join(SAMPLE_DIR, "car_sedan_front.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image not found")

    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = detector.detect(rgb, color_order=ColorOrder.RGB)
    assert result.has_detection is True

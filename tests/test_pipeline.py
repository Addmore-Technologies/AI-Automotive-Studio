"""
Integration tests for End-to-End Pipeline and Video Frame processing.
"""

import os
import pytest
import numpy as np
import cv2

from ai_engine.pipeline import (
    AutomotiveStudioPipeline,
    process_image,
    process_frame,
)
from ai_engine.common.types import PipelineConfig

SAMPLE_DIR = os.path.abspath("sample_data/input")


@pytest.fixture(scope="module")
def pipeline():
    config = PipelineConfig(
        confidence_threshold=0.30,
        enable_plate_replacement=True,
        enable_shadows=True,
        enable_reflections=True,
    )
    return AutomotiveStudioPipeline(config)


def test_process_image_end_to_end(pipeline, tmp_path):
    img_path = os.path.join(SAMPLE_DIR, "car_sedan_front.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_sedan_front.jpg not found")

    out_path = str(tmp_path / "sedan_studio.png")
    res = pipeline.process_image(img_path, output_path=out_path)

    assert res["success"] is True
    assert res["status"] == "COMPLETED"
    assert os.path.exists(out_path)
    
    # Check output image dimensions
    out_img = cv2.imread(out_path)
    assert out_img is not None
    assert out_img.shape == (1080, 1920, 3)

    # Check cutout path and existence
    cutout_path = res.get("cutout_path")
    assert cutout_path is not None
    assert os.path.exists(cutout_path)

    # Check detection & segmentation metadata
    assert res["detection"]["has_detection"] is True
    assert res["segmentation"]["coverage_pct"] > 0
    assert "execution_time_ms" in res


def test_process_sports_red(pipeline, tmp_path):
    img_path = os.path.join(SAMPLE_DIR, "car_sports_red.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_sports_red.jpg not found")

    out_path = str(tmp_path / "sports_studio.png")
    res = pipeline.process_image(img_path, output_path=out_path)
    assert res["success"] is True
    assert os.path.exists(out_path)


def test_top_level_process_image_convenience(tmp_path):
    img_path = os.path.join(SAMPLE_DIR, "car_suv_side.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image car_suv_side.jpg not found")

    out_path = str(tmp_path / "suv_studio.png")
    res = process_image(img_path, output_path=out_path)
    assert res["success"] is True
    assert os.path.exists(out_path)


def test_process_video_frames_with_temporal_smoothing(pipeline):
    img_path = os.path.join(SAMPLE_DIR, "car_sedan_front.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Sample image not found")

    base_frame = cv2.imread(img_path)
    state = {}

    # Simulate 5 consecutive video frames with subtle camera vibration/noise
    frames_processed = []
    for f_idx in range(5):
        # Add slight jitter/noise to simulate video frame changes
        jittered_frame = base_frame.copy()
        if f_idx > 0:
            noise = np.random.randint(-3, 4, jittered_frame.shape, dtype=np.int16)
            jittered_frame = np.clip(jittered_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        comp_frame, meta = pipeline.process_frame(jittered_frame, state=state)
        assert comp_frame.shape == (1080, 1920, 3)
        assert meta["success"] is True
        frames_processed.append(comp_frame)

    assert len(frames_processed) == 5
    # State should have smoothed bounding box and mask
    assert "prev_box" in state
    assert "prev_mask" in state


def test_error_handling_invalid_file(pipeline):
    res = pipeline.process_image("non_existent_file_xyz_123.jpg")
    assert res["success"] is False
    assert res["status"] == "ERROR_FILE_NOT_FOUND"

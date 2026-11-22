"""
Tests for media upload validation: file formats, magic bytes, corrupted content,
size limits, and path traversal protection.
"""

import io
from fastapi import status
from PIL import Image

from backend.app.services.file_service import FileService, SecurityPathTraversalError


def test_valid_image_upload_passes_validation(sample_car_bytes):
    """Verify valid JPEG passes content inspection."""
    ext, w, h = FileService.validate_image_bytes(sample_car_bytes, "car.jpg")
    assert ext in ("jpg", "jpeg")
    assert w > 0 and h > 0


def test_reject_unsupported_image_extension(client):
    """Verify rejection of unsupported extensions (e.g. .txt, .pdf, .gif)."""
    response = client.post(
        "/api/image/process",
        files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert "UNSUPPORTED_FORMAT" in response.text


def test_reject_corrupted_image_magic_bytes(client):
    """Verify rejection of file with image extension but invalid header bytes."""
    corrupted_data = b"NOT_A_REAL_IMAGE_CONTENT"
    response = client.post(
        "/api/image/process",
        files={"file": ("fake.png", corrupted_data, "image/png")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "CORRUPTED_FILE" in response.text or "header" in response.text.lower()


def test_reject_oversized_image(client, monkeypatch):
    """Verify HTTP 413 when image exceeds configured max size limit."""
    from backend.app.core import config
    # Temporarily set limit to 100 bytes for test
    monkeypatch.setattr(config.settings, "MAX_IMAGE_SIZE_BYTES", 100)

    dummy_bytes = b"\xFF\xD8\xFF" + b"0" * 200
    response = client.post(
        "/api/image/process",
        files={"file": ("large.jpg", dummy_bytes, "image/jpeg")},
    )
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "FILE_TOO_LARGE" in response.text


def test_filename_sanitization_and_path_traversal():
    """Verify filename sanitization prevents directory traversal attacks."""
    evil_name = "../../etc/passwd"
    clean = FileService.sanitize_filename(evil_name)
    assert "/" not in clean
    assert ".." not in clean

    windows_evil = "..\\..\\boot.ini"
    clean_win = FileService.sanitize_filename(windows_evil)
    assert "\\" not in clean_win
    assert ".." not in clean_win


def test_reject_corrupted_video_file(client):
    """Verify rejection of video with corrupted or missing headers."""
    fake_video = b"NOT_A_REAL_MP4_STREAM"
    response = client.post(
        "/api/video/process",
        files={"file": ("corrupt.mp4", fake_video, "video/mp4")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "CORRUPTED_FILE" in response.text


def test_reject_unsupported_video_extension(client):
    """Verify rejection of unsupported video formats (e.g. .avi, .mkv)."""
    response = client.post(
        "/api/video/process",
        files={"file": ("video.avi", b"RIFF....AVI ", "video/x-msvideo")},
    )
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

"""
File handling, validation, sanitization, and temporary workspace management.
"""

import os
import shutil
import uuid
import re
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import io

from backend.app.core.config import settings


class FileValidationError(Exception):
    """Base exception for file validation failures."""
    pass


class UnsupportedFormatError(FileValidationError):
    """Raised when file extension or MIME type is not supported."""
    pass


class CorruptedFileError(FileValidationError):
    """Raised when file header, magic bytes, or decoding fails."""
    pass


class FileTooLargeError(FileValidationError):
    """Raised when file size exceeds maximum permitted bytes."""
    pass


class SecurityPathTraversalError(FileValidationError):
    """Raised when filename attempts path traversal."""
    pass


# Magic bytes signatures
JPEG_MAGIC = b"\xFF\xD8\xFF"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MP4_FTYP_SIGNATURE = b"ftyp"


class FileService:
    """Service for validating, storing, and safely managing uploaded media files."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize uploaded filename to prevent directory traversal or invalid characters.
        """
        if not filename:
            return "unnamed_media"
        # Disallow path separators, null bytes
        base_name = os.path.basename(filename.replace("\\", "/"))
        # Strip dangerous characters
        sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', base_name)
        # Avoid hidden files or empty names
        sanitized = sanitized.lstrip(".")
        if not sanitized:
            sanitized = f"media_{uuid.uuid4().hex[:8]}"
        return sanitized

    @staticmethod
    def create_job_workspace(job_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Create an isolated temporary directory for the job.
        Returns (job_id, job_temp_dir).
        """
        job_id = job_id or str(uuid.uuid4())
        job_dir = os.path.join(settings.TEMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_id, job_dir

    @classmethod
    def validate_image_bytes(cls, content: bytes, original_filename: str) -> Tuple[str, int, int]:
        """
        Validate image content: size, extension, magic bytes, and actual decode.
        Returns (extension, width, height).
        """
        # 1. Size check
        if len(content) == 0:
            raise CorruptedFileError("Uploaded image file is empty (0 bytes).")
        if len(content) > settings.MAX_IMAGE_SIZE_BYTES:
            max_mb = settings.MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
            raise FileTooLargeError(f"Image exceeds maximum allowed size of {max_mb} MB.")

        # 2. Extension check
        ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
        if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_IMAGE_EXTENSIONS)
            raise UnsupportedFormatError(
                f"Unsupported image format '.{ext}'. Accepted formats are: {allowed}."
            )

        # 3. Magic bytes check
        if ext in ("jpg", "jpeg"):
            if not content.startswith(JPEG_MAGIC):
                raise CorruptedFileError("File header does not match a valid JPEG image.")
        elif ext == "png":
            if not content.startswith(PNG_MAGIC):
                raise CorruptedFileError("File header does not match a valid PNG image.")

        # 4. Actual image decode check using PIL and OpenCV
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
                w, h = img.size
        except Exception as e:
            raise CorruptedFileError(f"Uploaded file cannot be decoded as a valid image: {e}")

        # Check OpenCV compatibility
        np_arr = np.frombuffer(content, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise CorruptedFileError("Image could not be parsed by computer vision engine.")

        h, w = decoded.shape[:2]
        if w < 10 or h < 10:
            raise CorruptedFileError(f"Image dimensions ({w}x{h}) are too small for processing.")

        return ext, w, h

    @classmethod
    def validate_video_file(cls, file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Validate saved video file: size, extension, magic bytes, and stream decode.
        Returns dictionary of video metadata.
        """
        if not os.path.exists(file_path):
            raise CorruptedFileError("Video file was not saved successfully.")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise CorruptedFileError("Uploaded video file is empty (0 bytes).")
        if file_size > settings.MAX_VIDEO_SIZE_BYTES:
            max_mb = settings.MAX_VIDEO_SIZE_BYTES // (1024 * 1024)
            raise FileTooLargeError(f"Video exceeds maximum allowed size of {max_mb} MB.")

        # Extension check
        ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
        if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_VIDEO_EXTENSIONS)
            raise UnsupportedFormatError(
                f"Unsupported video format '.{ext}'. Accepted formats are: {allowed}."
            )

        # Magic bytes check for ISO base media (MP4/MOV)
        with open(file_path, "rb") as f:
            header = f.read(32)

        # Most MP4/MOV files have 'ftyp' at offset 4 or 'moov'/'mdat'
        has_ftyp = MP4_FTYP_SIGNATURE in header[:16]
        has_qt_moov = b"moov" in header or b"mdat" in header or b"wide" in header
        if not (has_ftyp or has_qt_moov):
            raise CorruptedFileError("File header does not match a valid MP4 or QuickTime/MOV container.")

        # Decode verification using OpenCV VideoCapture
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise CorruptedFileError("Cannot open video stream. The file may be corrupt or encoded with an unsupported codec.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Test reading the first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.size == 0:
            raise CorruptedFileError("Video stream contains no readable visual frames.")

        if fps <= 0 or fps > 240:
            fps = 24.0  # Fallback default if fps metadata is missing/erroneous

        if total_frames <= 0:
            total_frames = 1

        duration_s = round(total_frames / fps, 2)

        return {
            "format": ext,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_s": duration_s,
            "file_size_bytes": file_size,
        }

    @staticmethod
    def cleanup_job_workspace(job_id: str) -> None:
        """
        Safely remove temporary workspace for a job.
        """
        job_dir = os.path.join(settings.TEMP_DIR, job_id)
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except OSError:
                pass

    @staticmethod
    def get_safe_result_path(filename: str) -> str:
        """
        Resolve and validate path within RESULTS_DIR against path traversal.
        """
        clean_name = os.path.basename(filename.replace("\\", "/"))
        full_path = os.path.abspath(os.path.join(settings.RESULTS_DIR, clean_name))
        results_dir_abs = os.path.abspath(settings.RESULTS_DIR)

        # Path traversal guard
        if os.path.commonpath([results_dir_abs, full_path]) != results_dir_abs:
            raise SecurityPathTraversalError(f"Access denied: invalid file path '{filename}'.")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Result file '{clean_name}' not found.")

        return full_path

"""
Centralized Configuration for AI Automotive Studio Backend.
"""

import os
from typing import List, Set


class Settings:
    """Application configuration loaded from environment or defaults."""

    # Application metadata
    APP_NAME: str = os.getenv("APP_NAME", "AI Automotive Studio API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,*",
        ).split(",")
        if origin.strip()
    ]

    # File upload limits
    MAX_IMAGE_SIZE_BYTES: int = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(25 * 1024 * 1024)))  # 25 MB
    MAX_VIDEO_SIZE_BYTES: int = int(os.getenv("MAX_VIDEO_SIZE_BYTES", str(100 * 1024 * 1024)))  # 100 MB

    # Allowed extensions and MIME types
    ALLOWED_IMAGE_EXTENSIONS: Set[str] = {"jpg", "jpeg", "png"}
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {"mp4", "mov"}

    ALLOWED_IMAGE_MIMES: Set[str] = {
        "image/jpeg",
        "image/png",
    }
    ALLOWED_VIDEO_MIMES: Set[str] = {
        "video/mp4",
        "video/quicktime",
    }

    # Project directories
    # Detect if running from repo root or backend directory
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/app/core
    _BACKEND_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))  # backend/
    _PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))  # repo root

    PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", _PROJECT_ROOT)
    BACKEND_DIR: str = os.getenv("BACKEND_DIR", _BACKEND_DIR)

    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(_PROJECT_ROOT, "uploads"))
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", os.path.join(_PROJECT_ROOT, "outputs"))
    TEMP_DIR: str = os.getenv("TEMP_DIR", os.path.join(_PROJECT_ROOT, "temp"))

    # Studio background and weights
    STUDIO_BACKGROUND_PATH: str = os.getenv(
        "STUDIO_BACKGROUND_PATH",
        os.path.join(_PROJECT_ROOT, "studio_assets", "backgrounds", "approved_studio.png"),
    )
    WEIGHTS_DIR: str = os.getenv(
        "WEIGHTS_DIR",
        os.path.join(_PROJECT_ROOT, "ai_engine", "weights"),
    )

    # Defaults for AI compositing
    DEFAULT_PLATE_TEXT: str = os.getenv("DEFAULT_PLATE_TEXT", "STUDIO 01")
    DEFAULT_PLATE_SUBTEXT: str = os.getenv("DEFAULT_PLATE_SUBTEXT", "AUTO-AI")
    DEFAULT_VEHICLE_SCALE: float = float(os.getenv("DEFAULT_VEHICLE_SCALE", "0.72"))

    def ensure_directories(self) -> None:
        """Ensure necessary runtime directories exist."""
        for path in [self.UPLOAD_DIR, self.RESULTS_DIR, self.TEMP_DIR]:
            os.makedirs(path, exist_ok=True)


settings = Settings()

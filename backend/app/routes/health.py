"""
Health check endpoints.
"""

import time
import os
from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.responses import HealthResponse
from backend.app.services.ai_pipeline_service import ai_pipeline_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Check backend and AI engine health")
@router.get("/api/health", response_model=HealthResponse, include_in_schema=False)
def get_health() -> HealthResponse:
    """
    Returns service health status, AI model availability, and approved studio asset status.
    """
    ready, weights = ai_pipeline_service.check_readiness()
    bg_exists = os.path.exists(settings.STUDIO_BACKGROUND_PATH)

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        app_name=settings.APP_NAME,
        ai_engine_ready=ready,
        approved_studio_exists=bg_exists,
        weights_available=weights,
        timestamp=time.time(),
    )

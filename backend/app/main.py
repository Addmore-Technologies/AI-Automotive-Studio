"""
Main FastAPI Application Entry Point for AI Automotive Studio.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.app.core.config import settings
from backend.app.routes import health, image, video, results
from backend.app.services.file_service import FileValidationError
from backend.app.services.ai_pipeline_service import (
    NoVehicleDetectedException,
    SegmentationFailedException,
    AiPipelineException,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to ensure required runtime directories exist."""
    settings.ensure_directories()
    yield


def create_app() -> FastAPI:
    """Application factory for FastAPI service."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="FastAPI service connecting Arya's React interface with the AI/ML vision pipeline.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 3. Custom exception handlers
    @app.exception_handler(NoVehicleDetectedException)
    async def no_vehicle_exception_handler(request: Request, exc: NoVehicleDetectedException):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "NO_VEHICLE_DETECTED",
                "error": str(exc),
                "detail": "No vehicle was detected in the provided media. Ensure the car is clearly visible.",
            },
        )

    @app.exception_handler(SegmentationFailedException)
    async def segmentation_exception_handler(request: Request, exc: SegmentationFailedException):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "SEGMENTATION_FAILED",
                "error": str(exc),
                "detail": "Failed to separate the vehicle from the background.",
            },
        )

    @app.exception_handler(FileValidationError)
    async def file_validation_exception_handler(request: Request, exc: FileValidationError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "FILE_VALIDATION_ERROR",
                "error": str(exc),
                "detail": str(exc),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "VALIDATION_ERROR",
                "error": "Request validation failed.",
                "detail": exc.errors(),
            },
        )

    # 4. Include Routers
    app.include_router(health.router)
    app.include_router(image.router)
    app.include_router(video.router)
    app.include_router(results.router)

    # Root welcome route
    @app.get("/", tags=["Root"])
    def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()

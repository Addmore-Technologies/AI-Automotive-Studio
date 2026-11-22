"""
Image processing orchestration service.
"""

import os
import uuid
from typing import Dict, Any, Optional

from backend.app.core.config import settings
from backend.app.schemas.requests import ImageProcessOptions
from backend.app.schemas.responses import ImageProcessResponse
from backend.app.services.file_service import FileService
from backend.app.services.ai_pipeline_service import ai_pipeline_service


class ImageService:
    """Service to orchestrate still vehicle image processing into studio renders."""

    @classmethod
    def process_image(
        cls,
        file_bytes: bytes,
        original_filename: str,
        options: Optional[ImageProcessOptions] = None,
    ) -> ImageProcessResponse:
        """
        Validate, isolate, plate-replace, and composite an uploaded car image.
        """
        options = options or ImageProcessOptions()

        # 1. Validate image content and dimensions
        ext, width, height = FileService.validate_image_bytes(file_bytes, original_filename)

        # 2. Setup job workspace
        job_id, job_dir = FileService.create_job_workspace()

        # Input and output file paths
        safe_name = FileService.sanitize_filename(original_filename)
        input_path = os.path.join(job_dir, f"input_{safe_name}")
        output_filename = f"studio_{job_id}.png"
        output_path = os.path.join(settings.RESULTS_DIR, output_filename)

        try:
            # 3. Write uploaded bytes to temp input file
            with open(input_path, "wb") as f:
                f.write(file_bytes)

            # 4. Execute AI/ML Vision Pipeline
            result = ai_pipeline_service.process_image(
                input_path=input_path,
                output_path=output_path,
                fictional_plate_text=options.fictional_plate_text,
                enable_shadows=options.enable_shadows,
                enable_reflections=options.enable_reflections,
                vehicle_scale_pct=options.vehicle_scale_pct,
            )

            # 5. Check if cutout was created
            cutout_filename = f"studio_{job_id}_cutout.png"
            cutout_path = os.path.join(settings.RESULTS_DIR, cutout_filename)
            has_cutout = os.path.exists(cutout_path)

            # 6. Construct result URLs
            result_url = f"/api/results/{output_filename}"
            download_url = f"/api/results/{output_filename}?download=true"
            cutout_url = f"/api/results/{cutout_filename}" if has_cutout else None

            exec_time_ms = result.get("execution_time_ms", 0.0)

            return ImageProcessResponse(
                job_id=job_id,
                status="completed",
                success=True,
                result_url=result_url,
                download_url=download_url,
                cutout_url=cutout_url,
                output_format="png",
                message="Studio image generated successfully.",
                metadata={
                    "original_filename": original_filename,
                    "input_dimensions": {"width": width, "height": height},
                    "output_dimensions": {"width": 1920, "height": 1080},
                    "detection": result.get("detection"),
                    "segmentation": result.get("segmentation"),
                    "number_plate": result.get("number_plate"),
                    "compositing": result.get("compositing"),
                },
                execution_time_ms=exec_time_ms,
            )

        finally:
            # 7. Safe workspace cleanup
            FileService.cleanup_job_workspace(job_id)


image_service = ImageService()

import logging

logger = logging.getLogger(__name__)

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from backend.app.schemas.requests import ImageProcessOptions
from backend.app.schemas.responses import ImageProcessResponse
from backend.app.services.image_service import image_service
from backend.app.services.file_service import (
    UnsupportedFormatError,
    CorruptedFileError,
    FileTooLargeError,
)
from backend.app.services.ai_pipeline_service import (
    NoVehicleDetectedException,
    SegmentationFailedException,
    AiPipelineException,
)

router = APIRouter(tags=["Image Processing"])


@router.post(
    "/api/image/process",
    response_model=ImageProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and process car image into studio render",
)
@router.post(
    "/process/image",
    response_model=ImageProcessResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def process_image_endpoint(
    file: UploadFile = File(..., description="Multipart vehicle image file (JPG, JPEG, PNG)"),
    fictional_plate_text: Optional[str] = Form(None, description="Custom text for fictional license plate"),
    enable_shadows: bool = Form(True, description="Enable ground contact shadow"),
    enable_reflections: bool = Form(True, description="Enable studio floor reflection"),
    vehicle_scale_pct: Optional[float] = Form(None, description="Vehicle scale percentage"),
) -> ImageProcessResponse:
    """
    Accept an uploaded vehicle image, run detection, segmentation, plate replacement,
    background removal, and composite onto the approved studio background.
    """
    filename = file.filename or "uploaded_image.png"

    try:
        content = await file.read()
        options = ImageProcessOptions(
            fictional_plate_text=fictional_plate_text,
            enable_shadows=enable_shadows,
            enable_reflections=enable_reflections,
            vehicle_scale_pct=vehicle_scale_pct,
        )

        return image_service.process_image(
            file_bytes=content,
            original_filename=filename,
            options=options,
        )

    except UnsupportedFormatError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"status": "UNSUPPORTED_FORMAT", "error": str(e)},
        )
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"status": "FILE_TOO_LARGE", "error": str(e)},
        )
    except CorruptedFileError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "CORRUPTED_FILE", "error": str(e)},
        )
    except NoVehicleDetectedException as e:
        # 422 Unprocessable Entity - matches Arya's frontend regex for 'no vehicle detected'
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "NO_VEHICLE_DETECTED", "error": str(e)},
        )
    except SegmentationFailedException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "SEGMENTATION_FAILED", "error": str(e)},
        )
    except AiPipelineException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "AI_PIPELINE_ERROR", "error": str(e)},
        )
    except Exception as e:
        logger.exception("Unexpected error during image processing")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "INTERNAL_ERROR",
                "error": f"An unexpected error occurred: {e}",
            },
        )
    finally:
        await file.close()

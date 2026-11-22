"""
Results and media delivery endpoints.
"""

import os
import mimetypes
from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import FileResponse

from backend.app.services.file_service import FileService, SecurityPathTraversalError

router = APIRouter(tags=["Results"])


@router.get(
    "/api/results/{filename}",
    summary="Retrieve or download generated studio result",
)
@router.get(
    "/results/{filename}",
    include_in_schema=False,
)
def get_result_file(
    filename: str,
    download: bool = Query(False, description="Whether to trigger browser download attachment"),
):
    """
    Serve generated studio image or video files with correct MIME types and download support.
    """
    try:
        file_path = FileService.get_safe_result_path(filename)
    except SecurityPathTraversalError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"status": "ACCESS_DENIED", "error": str(e)},
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"status": "FILE_NOT_FOUND", "error": str(e)},
        )

    # Determine media type
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "mp4": "video/mp4",
        "mov": "video/quicktime",
    }
    media_type = mime_map.get(ext, mimetypes.guess_type(file_path)[0] or "application/octet-stream")

    content_disposition_type = "attachment" if download else "inline"
    headers = {
        "Content-Disposition": f'{content_disposition_type}; filename="{os.path.basename(file_path)}"',
        "Accept-Ranges": "bytes",
    }

    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers=headers,
    )

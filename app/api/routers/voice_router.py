from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Header
from typing import Optional
import uuid
from app.api.services.upload_service import upload_audio_video_file
from app.api.responses.base import BaseResponse
from starlette import status
from loguru import logger as custom_logger


router = APIRouter()

@router.post("/speech-to-text")
async def speech_to_text_api(
    file: UploadFile = File(...),
    user_id: str = "default-user" 
):
    try:
        upload_result = await upload_audio_video_file(file, user_id)
        
        if upload_result is None:
            custom_logger.error("upload_audio_video_file returned None")
            return BaseResponse.error_response(
                "Upload service returned None - internal error",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        if not upload_result['success']:
            custom_logger.error(f"Upload failed: {upload_result['error']}")
            return BaseResponse.error_response(
                upload_result['error'],
                status.HTTP_400_BAD_REQUEST
            )
        
        return BaseResponse.success_response(
            "File uploaded and processed successfully",
            status.HTTP_200_OK,
            upload_result['data']
        )
        
    except Exception as e:
        custom_logger.error(f"Speech-to-text API error: {str(e)}", exc_info=True)
        return BaseResponse.error_response(
            f"Internal server error: {str(e)}",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )
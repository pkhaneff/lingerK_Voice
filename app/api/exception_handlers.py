from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
from typing import Union

from app.api.responses.base import BaseResponse

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTPException: {exc.detail} - Path: {request.url.path}")
    return BaseResponse.error_response(
        message=str(exc.detail),
        status_code=exc.status_code
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation Error: {exc.errors()} - Path: {request.url.path}")
    return BaseResponse.error_response(
        message=f"Validation Error: {exc.errors()}",
        status_code=422
    )

async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.error(f"Unhandled Exception: {str(exc)} - Path: {request.url.path}", exc_info=True)
    return BaseResponse.error_response(
        message="Internal Server Error: An unexpected error occurred.",
        status_code=500
    )

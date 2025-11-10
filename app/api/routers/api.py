"""API routes"""
from fastapi import APIRouter

from app.api.routers import upload_router

app = APIRouter()

app.include_router(
    upload_router.router,
    tags=["Voice Identification"],
    prefix="/v1",
)
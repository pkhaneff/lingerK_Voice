"""API routes"""
from fastapi import APIRouter

from app.api.routers import upload_router, semantic_router

app = APIRouter()

app.include_router(
    upload_router.router,
    tags=["Voice Identification"],
    prefix="/v1",
)

app.include_router(
    semantic_router.router,
    tags=["Semantic Analysis"],
    prefix="/v1",
)
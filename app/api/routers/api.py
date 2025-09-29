"""API routes"""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.routers import voice_router

app = APIRouter()

app.include_router(
    voice_router.router,
    tags=["Voice Identification"],
    prefix="/v1",
)
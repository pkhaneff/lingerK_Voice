from fastapi import APIRouter, Depends, HTTPException
from uuid import UUID

from app.api.responses.base import BaseResponse
from app.api.services.semantic.orchestrator import SemanticOrchestrator
from app.api.dependencies import get_semantic_orchestrator


router = APIRouter()


@router.get("/semantic/{audio_clean_id}")
async def get_semantic_document(
    audio_clean_id: UUID,
    orchestrator: SemanticOrchestrator = Depends(get_semantic_orchestrator)
):
    document = await orchestrator.get_document(audio_clean_id)
    
    if not document:
        raise HTTPException(404, "Semantic document not found")
    
    return BaseResponse.success_response(
        message="Semantic document retrieved successfully",
        data=document
    )

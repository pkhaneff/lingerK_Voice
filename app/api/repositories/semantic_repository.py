from typing import List, Dict
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.model.semantic_document import SemanticDocument


class SemanticRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, document_data: dict) -> UUID:
        doc = SemanticDocument(**document_data)
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc.doc_id

    async def get_by_audio_clean_id(self, audio_clean_id: UUID) -> SemanticDocument | None:
        result = await self.session.execute(
            select(SemanticDocument)
            .where(SemanticDocument.audio_clean_id == audio_clean_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, doc_id: UUID) -> SemanticDocument | None:
        result = await self.session.execute(
            select(SemanticDocument)
            .where(SemanticDocument.doc_id == doc_id)
        )
        return result.scalar_one_or_none()

    async def update_stage(self, doc_id: UUID, stage: str) -> bool:
        result = await self.session.execute(
            select(SemanticDocument)
            .where(SemanticDocument.doc_id == doc_id)
        )
        doc = result.scalar_one_or_none()
        
        if not doc:
            return False
        
        doc.processing_stage = stage
        await self.session.commit()
        return True

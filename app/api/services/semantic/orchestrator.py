from typing import List, Dict
from uuid import UUID
from loguru import logger

from app.api.services.semantic.text_preparation import TextPreparationService
from app.api.services.semantic.sentence_tokenizer import SentenceTokenizer
from app.api.services.semantic.text_normalizer import TextNormalizer
from app.api.services.semantic.document_type_detector import DocumentTypeDetector
from app.api.repositories.semantic_repository import SemanticRepository


class SemanticOrchestrator:
    def __init__(
        self,
        text_prep: TextPreparationService,
        repository: SemanticRepository
    ):
        self.text_prep = text_prep
        self.repository = repository

    async def process(self, audio_clean_id: UUID) -> dict:
        try:
            doc_data = await self.text_prep.prepare(audio_clean_id)
            doc_id = await self.repository.save(doc_data)
            
            logger.info(f"Semantic document created: {doc_id}")
            
            return {
                'success': True,
                'doc_id': doc_id,
                'sentence_count': doc_data['sentence_count']
            }
        except Exception as e:
            logger.error(f"Semantic processing failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    async def get_document(self, audio_clean_id: UUID) -> dict | None:
        doc = await self.repository.get_by_audio_clean_id(audio_clean_id)
        
        if not doc:
            return None
        
        return {
            'doc_id': str(doc.doc_id),
            'audio_clean_id': str(doc.audio_clean_id),
            'full_text': doc.full_text,
            'speaker_count': doc.speaker_count,
            'total_duration': doc.total_duration,
            'word_count': doc.word_count,
            'sentence_count': doc.sentence_count,
            'sentences': doc.sentences,
            'speaker_info': doc.speaker_info,
            'document_type': doc.document_type,
            'processing_stage': doc.processing_stage,
            'created_at': doc.created_at.isoformat() if doc.created_at else None
        }

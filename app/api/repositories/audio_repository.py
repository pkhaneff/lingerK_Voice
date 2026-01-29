from typing import Optional, Dict, List
import uuid
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.model.audio_model import AudioIngest
from app.api.model.audio_clean import AudioClean

class AudioRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_audio(
        self,
        file_name: str,
        storage_uri: str,
        user_id: str,
        duration: Optional[float] = None,
        codec: Optional[str] = None,
        is_video: bool = False
    ) -> AudioIngest:
        audio_record = AudioIngest(
            audio_id=uuid.uuid4(),
            file_name=file_name,
            storage_uri=storage_uri,
            duration=duration,
            codec=codec,
            user_id=uuid.UUID(user_id),
            status="uploaded",
            preprocessed=False,
            created_at=datetime.utcnow(),
            is_video=is_video
        )
        self.session.add(audio_record)
        return audio_record

    async def get_audio(self, audio_id: uuid.UUID) -> Optional[AudioIngest]:
        return await self.session.get(AudioIngest, audio_id)
        
    async def create_audio_clean(
        self,
        original_audio_id: uuid.UUID,
        storage_uri: str,
        processing_method: str = 'pyrnnoise'
    ) -> AudioClean:
        audio_clean_record = AudioClean(
            cleaned_audio_id=uuid.uuid4(),
            original_audio_id=original_audio_id,
            storage_uri=storage_uri,
            processing_method=processing_method,
            created_at=datetime.utcnow()
        )
        self.session.add(audio_clean_record)
        return audio_clean_record

    async def get_audio_clean(self, cleaned_audio_id: uuid.UUID) -> Optional[AudioClean]:
        return await self.session.get(AudioClean, cleaned_audio_id)

    async def update_processing_results(
        self,
        audio_id: uuid.UUID,
        combined_analysis: Dict,
        duration: Optional[float] = None
    ) -> None:
        stmt = (
            update(AudioIngest)
            .where(AudioIngest.audio_id == audio_id)
            .values(
                preprocessed=True,
                processed_time=datetime.utcnow(),
                status='completed',
                noise_analysis=combined_analysis,
                duration=duration if duration is not None else AudioIngest.duration
            )
        )
        await self.session.execute(stmt)

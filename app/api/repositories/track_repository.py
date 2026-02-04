from typing import List, Optional
import uuid
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.model.speaker_track_model import SpeakerTrack
from app.api.model.track_segment_model import TrackSegment
from app.api.repositories.interfaces import ITrackRepository

class TrackRepository(ITrackRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_track(
        self,
        audio_id: uuid.UUID,
        speaker_id: int,
        track_type: str,
        ranges: List,
        total_duration: float,
        coverage: float,
        transcript: Optional[str] = None,
        words: Optional[List] = None
    ) -> SpeakerTrack:
        track = SpeakerTrack(
            track_id=uuid.uuid4(),
            audio_id=audio_id,
            speaker_id=speaker_id,
            track_type=track_type,
            ranges=ranges,
            total_duration=total_duration,
            coverage=coverage,
            transcript=transcript,
            words=words,
            created_at=datetime.utcnow()
        )
        self.session.add(track)
        return track

    async def create_segment(
        self,
        track_id: uuid.UUID,
        segment_type: str,
        start_time: float,
        end_time: float,
        duration: float,
        confidence: Optional[float] = None,
        separation_method: Optional[str] = None
    ) -> TrackSegment:
        segment = TrackSegment(
            segment_id=uuid.uuid4(),
            track_id=track_id,
            segment_type=segment_type,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            confidence=confidence,
            separation_method=separation_method,
            created_at=datetime.utcnow()
        )
        self.session.add(segment)
        return segment

    async def get_tracks_by_audio_id(self, audio_id: uuid.UUID) -> List[SpeakerTrack]:
        result = await self.session.execute(
            select(SpeakerTrack).where(SpeakerTrack.audio_id == audio_id)
        )
        return result.scalars().all()

    async def get_segments_by_track_id(self, track_id: uuid.UUID) -> List[TrackSegment]:
        result = await self.session.execute(
            select(TrackSegment).where(TrackSegment.track_id == track_id)
        )
        return result.scalars().all()

    async def update_transcript(
        self,
        audio_id: uuid.UUID,
        speaker_id: int,
        transcript: str,
        words: List
    ) -> int:
        stmt = (
            update(SpeakerTrack)
            .where(
                SpeakerTrack.audio_id == audio_id,
                SpeakerTrack.speaker_id == speaker_id
            )
            .values(
                transcript=transcript,
                words=words
            )
        )
        result = await self.session.execute(stmt)
        return result.rowcount

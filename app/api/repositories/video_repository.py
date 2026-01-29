import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.model.video_model import VideoIngest

class VideoRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_video(
        self,
        audio_id: uuid.UUID,
        storage_uri: str
    ) -> VideoIngest:
        video_record = VideoIngest(
            video_id=uuid.uuid4(),
            audio_id=audio_id,
            storage_uri=storage_uri
        )
        self.session.add(video_record)
        return video_record

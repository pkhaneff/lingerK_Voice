import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.api.db.base import Base

class VideoIngest(Base):
    __tablename__ = "video_ingest"

    video_id   = sa.Column(UUID(as_uuid=True), primary_key=True,
                           server_default=sa.text("gen_random_uuid()"))
    audio_id   = sa.Column(UUID(as_uuid=True),
                           sa.ForeignKey("audio_ingest.audio_id", ondelete="CASCADE"),
                           nullable=False, unique=True)
    storage_uri = sa.Column(sa.Text, nullable=False)

    audio = relationship("AudioIngest", back_populates="video", lazy="select")

    __table_args__ = (
        sa.Index("idx_video_ingest_audio_id", "audio_id"),
    )

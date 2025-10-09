import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from app.api.db.base import Base

class AudioIngest(Base):
    __tablename__ = "audio_ingest"

    audio_id     = sa.Column(UUID(as_uuid=True), primary_key=True,
                             server_default=sa.text("gen_random_uuid()"))
    file_name    = sa.Column(sa.String(255), nullable=False)
    storage_uri  = sa.Column(sa.Text, nullable=False)            # S3 URI/path
    duration     = sa.Column(sa.Float, nullable=True)            # seconds
    codec        = sa.Column(sa.String(50), nullable=True)
    user_id      = sa.Column(UUID(as_uuid=True), nullable=False) # hoặc FK tới users
    status       = sa.Column(sa.String(20), nullable=False,
                             server_default=sa.text("'uploaded'"))
    preprocessed = sa.Column(sa.Boolean, nullable=False, server_default=sa.text("false"))
    created_at   = sa.Column(sa.DateTime(timezone=True), nullable=False,
                             server_default=sa.text("NOW()"))
    processed_time = sa.Column(sa.DateTime(timezone=True), nullable=True)
    is_video = sa.Column(sa.Boolean, nullable=False, server_default=sa.text("false"))
    noise_analysis = sa.Column(JSONB, nullable=True)

    video = relationship("VideoIngest", back_populates="audio", uselist=False, lazy="select")

    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('uploaded','processing','completed','failed')",
            name="audio_ingest_status_check"
        ),
        sa.Index("idx_audio_user_id", "user_id"),
        sa.Index("idx_audio_status", "status"),
        sa.Index("idx_audio_created_at", "created_at"),
        sa.Index("idx_audio_is_video", "is_video"),
    )

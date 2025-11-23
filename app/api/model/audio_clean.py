import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.api.db.base import Base


class AudioClean(Base):
    __tablename__ = "audio_clean"

    cleaned_audio_id = sa.Column(UUID(as_uuid=True), primary_key=True,
                                 server_default=sa.text("gen_random_uuid()"))
    original_audio_id = sa.Column(UUID(as_uuid=True),
                                 sa.ForeignKey("audio_ingest.audio_id", ondelete="CASCADE"),
                                 nullable=False)
    storage_uri = sa.Column(sa.Text, nullable=False)
    processing_method = sa.Column(sa.String(50), nullable=False,
                                 server_default=sa.text("'pyrnnoise'"))
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))

    original_audio = relationship("AudioIngest", lazy="select")
    notes = relationship("Note",back_populates="audio_clean", lazy="select")

    __table_args__ = (
        sa.Index("idx_audio_clean_original_id", "original_audio_id"),
        sa.Index("idx_audio_clean_created_at", "created_at"),
    )
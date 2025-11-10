import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.api.db.base import Base


class AudioSegment(Base):
    __tablename__ = "audio_segments"

    segment_id = sa.Column(UUID(as_uuid=True), primary_key=True,
                          server_default=sa.text("gen_random_uuid()"))
    audio_id = sa.Column(UUID(as_uuid=True),
                        sa.ForeignKey("audio_ingest.audio_id", ondelete="CASCADE"),
                        nullable=False)
    
    track_type = sa.Column(sa.String(20), nullable=False)  
    track_order = sa.Column(sa.Integer, nullable=False)    
    
    start_time = sa.Column(sa.Float, nullable=False)
    end_time = sa.Column(sa.Float, nullable=False)
    duration = sa.Column(sa.Float, nullable=False)
    
    coverage = sa.Column(sa.Float, nullable=True)          
    osd_confidence = sa.Column(sa.Float, nullable=True)    
    
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))
    
    audio = relationship("AudioIngest", back_populates="segments", lazy="select")

    __table_args__ = (
        sa.Index("idx_segments_audio_id", "audio_id"),
        sa.Index("idx_segments_track_type", "track_type"),
        sa.Index("idx_segments_track_order", "track_order"),
        sa.Index("idx_segments_start_time", "start_time"),
    )
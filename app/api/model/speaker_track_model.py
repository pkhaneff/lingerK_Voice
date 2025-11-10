import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.api.db.base import Base


class SpeakerTrack(Base):
    """High-level speaker track information."""
    __tablename__ = "speaker_tracks"

    track_id = sa.Column(UUID(as_uuid=True), primary_key=True,
                         server_default=sa.text("gen_random_uuid()"))
    audio_id = sa.Column(UUID(as_uuid=True),
                        sa.ForeignKey("audio_ingest.audio_id", ondelete="CASCADE"),
                        nullable=False)
    
    speaker_id = sa.Column(sa.Integer, nullable=False)
    track_type = sa.Column(sa.String(20), nullable=False)  
    
    ranges = sa.Column(JSONB, nullable=False)
    
    total_duration = sa.Column(sa.Float, nullable=False)
    coverage = sa.Column(sa.Float, nullable=False)  
    
    transcript = sa.Column(sa.Text, nullable=True)
    words = sa.Column(JSONB, nullable=True)  
    
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))
    
    audio = relationship("AudioIngest", back_populates="speaker_tracks", lazy="select")
    segments = relationship("TrackSegment", back_populates="track", 
                           cascade="all, delete-orphan", lazy="select")

    __table_args__ = (
        sa.CheckConstraint("track_type IN ('single','separated')", 
                          name='speaker_tracks_type_check'),
        sa.Index("idx_speaker_tracks_audio_id", "audio_id"),
        sa.Index("idx_speaker_tracks_speaker_id", "speaker_id"),
        sa.Index("idx_speaker_tracks_type", "track_type"),
        sa.Index("idx_speaker_tracks_duration", "total_duration"),
        sa.Index("idx_speaker_tracks_ranges", "ranges", postgresql_using="gin"),
    )
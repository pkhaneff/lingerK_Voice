import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.api.db.base import Base


class TrackSegment(Base):
    """Detailed segment information within a speaker track."""
    __tablename__ = "track_segments"

    segment_id = sa.Column(UUID(as_uuid=True), primary_key=True,
                          server_default=sa.text("gen_random_uuid()"))
    track_id = sa.Column(UUID(as_uuid=True),
                        sa.ForeignKey("speaker_tracks.track_id", ondelete="CASCADE"),
                        nullable=False)
    
    segment_type = sa.Column(sa.String(20), nullable=False)
    
    start_time = sa.Column(sa.Float, nullable=False)
    end_time = sa.Column(sa.Float, nullable=False)
    duration = sa.Column(sa.Float, nullable=False)
    
    confidence = sa.Column(sa.Float, nullable=True)
    separation_method = sa.Column(sa.String(50), nullable=True)
    
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))
    
    track = relationship("SpeakerTrack", back_populates="segments", lazy="select")

    __table_args__ = (
        sa.CheckConstraint("segment_type IN ('non-overlap','overlap')", 
                          name='track_segments_type_check'),
        sa.Index("idx_track_segments_track_id", "track_id"),
        sa.Index("idx_track_segments_type", "segment_type"),
        sa.Index("idx_track_segments_start_time", "start_time"),
        sa.Index("idx_track_segments_confidence", "confidence"),
    )
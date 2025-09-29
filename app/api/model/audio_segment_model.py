from sqlalchemy import Column, String, Float, ForeignKey, DateTime, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.api.db.base import Base

class AudioSegment(Base):
    __tablename__ = "audio_segments"
    
    segment_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    audio_id = Column(UUID(as_uuid=True), ForeignKey('audio_ingest.audio_id', ondelete='CASCADE'), nullable=False)
    start_time = Column(Float, nullable=False) 
    end_time = Column(Float, nullable=False)   
    duration = Column(Float, nullable=False)   
    confidence = Column(Float, nullable=True)   
    segment_type = Column(String(20), default='voice', nullable=False)  
    storage_uri = Column(String(500), nullable=True)  
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    
    audio = relationship("AudioIngest", back_populates="segments")
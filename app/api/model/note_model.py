import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.api.db.base import Base

class Note(Base):
    __tablename__ = "notes"

    note_id = sa.Column(UUID(as_uuid=True), primary_key=True,
                       server_default=sa.text("gen_random_uuid()"))
    audio_clean_id = sa.Column(UUID(as_uuid=True),
                              sa.ForeignKey("audio_clean.cleaned_audio_id", ondelete="CASCADE"),
                              nullable=False)
    
    title = sa.Column(sa.String(500), nullable=True)
    note_type = sa.Column(sa.String(50), nullable=True)
    summary = sa.Column(sa.Text, nullable=True)
    text = sa.Column(sa.Text, nullable=True) 
    
    highlights = sa.Column(JSONB, nullable=True)  
    tags = sa.Column(JSONB, nullable=True)        
    
    processing_status = sa.Column(sa.String(20), nullable=False,
                                 server_default=sa.text("'pending'"))
    processing_time = sa.Column(sa.Float, nullable=True)
    error_message = sa.Column(sa.Text, nullable=True)
    
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))
    updated_at = sa.Column(sa.DateTime(timezone=True), nullable=True)

    # SỬA: Thêm back_populates
    audio_clean = relationship("AudioClean", back_populates="notes", lazy="select")
    
    sections = relationship("NoteSection", back_populates="note", 
                           cascade="all, delete-orphan", 
                           order_by="NoteSection.section_order",
                           lazy="select")

    __table_args__ = (
        sa.CheckConstraint(
            "processing_status IN ('pending','processing','completed','failed')",
            name="notes_status_check"
        ),
        sa.Index("idx_notes_audio_clean_id", "audio_clean_id"),
        sa.Index("idx_notes_status", "processing_status"),
        sa.Index("idx_notes_created_at", "created_at"),
        sa.Index("idx_notes_note_type", "note_type"),
        sa.Index("idx_notes_highlights", "highlights", postgresql_using="gin"),
        sa.Index("idx_notes_tags", "tags", postgresql_using="gin"),
    )

class NoteSection(Base):
    __tablename__ = "note_sections"

    note_section_id = sa.Column(UUID(as_uuid=True), primary_key=True,
                               server_default=sa.text("gen_random_uuid()"))
    note_id = sa.Column(UUID(as_uuid=True),
                       sa.ForeignKey("notes.note_id", ondelete="CASCADE"),
                       nullable=False)
    
    heading = sa.Column(sa.String(500), nullable=False)
    bullets = sa.Column(JSONB, nullable=False) 
    
    section_order = sa.Column(sa.Integer, nullable=False, default=0)
    
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))

    note = relationship("Note", back_populates="sections", lazy="select")

    __table_args__ = (
        sa.Index("idx_note_sections_note_id", "note_id"),
        sa.Index("idx_note_sections_order", "section_order"),
        sa.Index("idx_note_sections_bullets", "bullets", postgresql_using="gin"),
    )
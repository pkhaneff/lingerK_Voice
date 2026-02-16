import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.api.db.base import Base


class SemanticDocument(Base):
    __tablename__ = "semantic_documents"

    doc_id = sa.Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()")
    )
    audio_clean_id = sa.Column(
        UUID(as_uuid=True),
        sa.ForeignKey("audio_clean.cleaned_audio_id", ondelete="CASCADE"),
        nullable=False
    )

    full_text = sa.Column(sa.Text, nullable=False)
    normalized_text = sa.Column(sa.Text)

    speaker_count = sa.Column(sa.Integer, nullable=False)
    total_duration = sa.Column(sa.Float, nullable=False)
    word_count = sa.Column(sa.Integer)
    sentence_count = sa.Column(sa.Integer)

    speaker_info = sa.Column(JSONB)
    sentences = sa.Column(JSONB)

    document_type = sa.Column(sa.String(50))
    processing_stage = sa.Column(
        sa.String(20),
        nullable=False,
        server_default=sa.text("'prepared'")
    )

    created_at = sa.Column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("NOW()")
    )
    updated_at = sa.Column(
        sa.DateTime(timezone=True),
        server_default=sa.text("NOW()")
    )

    __table_args__ = (
        sa.CheckConstraint(
            "processing_stage IN ('prepared','analyzing','chunked','indexed','failed')",
            name='semantic_docs_stage_check'
        ),
        sa.CheckConstraint(
            "document_type IN ('phone_call','online_class','meeting','other')",
            name='semantic_docs_type_check'
        ),
        sa.Index("idx_semantic_docs_audio_clean", "audio_clean_id"),
        sa.Index("idx_semantic_docs_stage", "processing_stage"),
        sa.Index("idx_semantic_docs_type", "document_type"),
        sa.Index("idx_semantic_docs_created", "created_at"),
    )

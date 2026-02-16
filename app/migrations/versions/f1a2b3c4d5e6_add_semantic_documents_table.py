"""add_semantic_documents_table

Revision ID: f1a2b3c4d5e6
Revises: 0d4a20ec27b6
Create Date: 2026-02-06 08:51:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'f1a2b3c4d5e6'
down_revision = '0d4a20ec27b6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'semantic_documents',
        sa.Column('doc_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('audio_clean_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('full_text', sa.Text(), nullable=False),
        sa.Column('normalized_text', sa.Text(), nullable=True),
        sa.Column('speaker_count', sa.Integer(), nullable=False),
        sa.Column('total_duration', sa.Float(), nullable=False),
        sa.Column('word_count', sa.Integer(), nullable=True),
        sa.Column('sentence_count', sa.Integer(), nullable=True),
        sa.Column('speaker_info', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('sentences', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('document_type', sa.String(length=50), nullable=True),
        sa.Column('processing_stage', sa.String(length=20), server_default=sa.text("'prepared'"), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.CheckConstraint("processing_stage IN ('prepared','analyzing','chunked','indexed','failed')", name='semantic_docs_stage_check'),
        sa.CheckConstraint("document_type IN ('phone_call','online_class','meeting','other')", name='semantic_docs_type_check'),
        sa.ForeignKeyConstraint(['audio_clean_id'], ['audio_clean.cleaned_audio_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('doc_id')
    )
    op.create_index('idx_semantic_docs_audio_clean', 'semantic_documents', ['audio_clean_id'], unique=False)
    op.create_index('idx_semantic_docs_created', 'semantic_documents', ['created_at'], unique=False)
    op.create_index('idx_semantic_docs_stage', 'semantic_documents', ['processing_stage'], unique=False)
    op.create_index('idx_semantic_docs_type', 'semantic_documents', ['document_type'], unique=False)


def downgrade():
    op.drop_index('idx_semantic_docs_type', table_name='semantic_documents')
    op.drop_index('idx_semantic_docs_stage', table_name='semantic_documents')
    op.drop_index('idx_semantic_docs_created', table_name='semantic_documents')
    op.drop_index('idx_semantic_docs_audio_clean', table_name='semantic_documents')
    op.drop_table('semantic_documents')

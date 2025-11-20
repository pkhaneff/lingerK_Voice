# revision identifiers, used by Alembic.
revision = '0d4a20ec27b6'
down_revision = '97049274243f'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

def upgrade() -> None:
    """Create note and note_section tables"""
    
    # Table 1: notes - chứa thông tin tổng quan
    op.create_table(
        'notes',
        sa.Column('note_id', UUID(as_uuid=True), 
                  server_default=sa.text('gen_random_uuid()'), 
                  nullable=False),
        sa.Column('audio_clean_id', UUID(as_uuid=True), nullable=False),
        
        # Content fields
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('note_type', sa.String(50), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),  # Full processed text
        
        # Structured data
        sa.Column('highlights', JSONB, nullable=True),  # Array of highlight strings
        sa.Column('tags', JSONB, nullable=True),        # Array of tag strings
        
        # Metadata
        sa.Column('processing_status', sa.String(20), nullable=False,
                  server_default=sa.text("'pending'")),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        
        # Foreign key to audio_clean
        sa.ForeignKeyConstraint(['audio_clean_id'], ['audio_clean.cleaned_audio_id'], 
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('note_id'),
        
        # Check constraint
        sa.CheckConstraint(
            "processing_status IN ('pending','processing','completed','failed')",
            name="notes_status_check"
        )
    )
    
    # Indexes for notes table
    op.create_index('idx_notes_audio_clean_id', 'notes', ['audio_clean_id'])
    op.create_index('idx_notes_status', 'notes', ['processing_status'])
    op.create_index('idx_notes_created_at', 'notes', ['created_at'])
    op.create_index('idx_notes_note_type', 'notes', ['note_type'])
    op.create_index('idx_notes_highlights', 'notes', ['highlights'], 
                    postgresql_using='gin')
    op.create_index('idx_notes_tags', 'notes', ['tags'], 
                    postgresql_using='gin')
    
    # Table 2: note_sections - chứa các sections chi tiết
    op.create_table(
        'note_sections',
        sa.Column('note_section_id', UUID(as_uuid=True), 
                  server_default=sa.text('gen_random_uuid()'), 
                  nullable=False),
        sa.Column('note_id', UUID(as_uuid=True), nullable=False),
        
        # Section content
        sa.Column('heading', sa.String(500), nullable=False),
        sa.Column('bullets', JSONB, nullable=False),  # Array of bullet point strings
        
        # Ordering
        sa.Column('section_order', sa.Integer(), nullable=False, default=0),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
        
        # Foreign key to notes
        sa.ForeignKeyConstraint(['note_id'], ['notes.note_id'], 
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('note_section_id')
    )
    
    # Indexes for note_sections table
    op.create_index('idx_note_sections_note_id', 'note_sections', ['note_id'])
    op.create_index('idx_note_sections_order', 'note_sections', ['section_order'])
    op.create_index('idx_note_sections_bullets', 'note_sections', ['bullets'], 
                    postgresql_using='gin')


def downgrade() -> None:
    """Drop note tables"""
    
    # Drop note_sections first (has foreign key)
    op.drop_index('idx_note_sections_bullets', table_name='note_sections')
    op.drop_index('idx_note_sections_order', table_name='note_sections')
    op.drop_index('idx_note_sections_note_id', table_name='note_sections')
    op.drop_table('note_sections')
    
    # Drop notes table
    op.drop_index('idx_notes_tags', table_name='notes')
    op.drop_index('idx_notes_highlights', table_name='notes')
    op.drop_index('idx_notes_note_type', table_name='notes')
    op.drop_index('idx_notes_created_at', table_name='notes')
    op.drop_index('idx_notes_status', table_name='notes')
    op.drop_index('idx_notes_audio_clean_id', table_name='notes')
    op.drop_table('notes')
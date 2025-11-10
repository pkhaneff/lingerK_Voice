# revision identifiers, used by Alembic.
revision = '8170fccd17da'
down_revision = 'a29e086a6bec'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

def upgrade() -> None:
    """Create audio_clean table."""
    op.create_table(
        'audio_clean',
        sa.Column('cleaned_audio_id', UUID(as_uuid=True), 
                  server_default=sa.text('gen_random_uuid()'), 
                  nullable=False),
        sa.Column('original_audio_id', UUID(as_uuid=True), nullable=False),
        sa.Column('storage_uri', sa.Text(), nullable=False),
        sa.Column('processing_method', sa.String(50), 
                  server_default=sa.text("'pyrnnoise'"), 
                  nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  server_default=sa.text('NOW()'), 
                  nullable=False),
        
        # Foreign key
        sa.ForeignKeyConstraint(['original_audio_id'], ['audio_ingest.audio_id'], 
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('cleaned_audio_id')
    )
    
    # Create indexes
    op.create_index('idx_audio_clean_original_id', 'audio_clean', ['original_audio_id'])
    op.create_index('idx_audio_clean_created_at', 'audio_clean', ['created_at'])

def downgrade() -> None:
    """Drop audio_clean table."""
    op.drop_index('idx_audio_clean_created_at', table_name='audio_clean')
    op.drop_index('idx_audio_clean_original_id', table_name='audio_clean')
    op.drop_table('audio_clean')

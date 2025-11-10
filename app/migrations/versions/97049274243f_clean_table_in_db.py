# revision identifiers, used by Alembic.
revision = '97049274243f'
down_revision = '8170fccd17da'
branch_labels = None
depends_on = None

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

def upgrade() -> None:
    """Drop unused tables to clean up database."""
    
    # Drop audio_segments table (deprecated - replaced by speaker_tracks + track_segments)
    # This table was created in e30b46a864bb but is no longer used
    # Current code uses save_hybrid_tracks() instead of save_tracks()
    
    print("Dropping deprecated audio_segments table...")
    
    # Drop indexes first
    op.drop_index('idx_segments_start_time', table_name='audio_segments', if_exists=True)
    op.drop_index('idx_segments_track_order', table_name='audio_segments', if_exists=True) 
    op.drop_index('idx_segments_track_type', table_name='audio_segments', if_exists=True)
    op.drop_index('idx_segments_audio_id', table_name='audio_segments', if_exists=True)
    
    # Drop table and all data
    op.drop_table('audio_segments', if_exists=True)
    
    # Drop refresh_tokens table (never used - leftover from old auth system)
    # This table was created in a29e086a6bec but has no corresponding model or usage
    
    print("Dropping unused refresh_tokens table...")
    
    # Drop indexes first  
    op.drop_index('ix_refresh_tokens_hashed_token', table_name='refresh_tokens', if_exists=True)
    
    # Drop table and all data
    op.drop_table('refresh_tokens', if_exists=True)
    
    print("Database cleanup completed successfully!")


def downgrade() -> None:
    """Recreate dropped tables (for rollback only - not recommended)."""
    
    # Recreate refresh_tokens table
    op.create_table('refresh_tokens',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('hashed_token', sa.String(length=512), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='NOW()', nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_refresh_tokens_hashed_token', 'refresh_tokens', ['hashed_token'], unique=True)
    
    # Recreate audio_segments table
    op.create_table('audio_segments',
        sa.Column('segment_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('audio_id', sa.UUID(), nullable=False),
        sa.Column('track_type', sa.String(20), nullable=False),
        sa.Column('track_order', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False), 
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('coverage', sa.Float(), nullable=True),
        sa.Column('osd_confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['audio_id'], ['audio_ingest.audio_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('segment_id')
    )
    
    # Recreate indexes
    op.create_index('idx_segments_audio_id', 'audio_segments', ['audio_id'])
    op.create_index('idx_segments_track_type', 'audio_segments', ['track_type'])
    op.create_index('idx_segments_track_order', 'audio_segments', ['track_order'])
    op.create_index('idx_segments_start_time', 'audio_segments', ['start_time'])
    
    print("Tables recreated for rollback (data will be empty)")

"""add audio_segment table

Revision ID: 03b543205cb2
Revises: a5eb0c98339d
Create Date: 2025-09-26 11:39:51.403461

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '03b543205cb2'
down_revision: Union[str, Sequence[str], None] = 'a5eb0c98339d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('audio_segments',
        sa.Column('segment_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('audio_id', sa.UUID(), nullable=False),
        
        # Track information
        sa.Column('track_type', sa.String(20), nullable=False),
        sa.Column('track_order', sa.Integer(), nullable=False),
        
        # Time information
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False), 
        sa.Column('duration', sa.Float(), nullable=False),
        
        # Metrics
        sa.Column('coverage', sa.Float(), nullable=True),
        sa.Column('osd_confidence', sa.Float(), nullable=True),
        
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['audio_id'], ['audio_ingest.audio_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('segment_id')
    )
    
    # Create indexes
    op.create_index('idx_segments_audio_id', 'audio_segments', ['audio_id'])
    op.create_index('idx_segments_track_type', 'audio_segments', ['track_type'])
    op.create_index('idx_segments_track_order', 'audio_segments', ['track_order'])
    op.create_index('idx_segments_start_time', 'audio_segments', ['start_time'])


def downgrade() -> None:
    op.drop_index('idx_segments_start_time', table_name='audio_segments')
    op.drop_index('idx_segments_track_order', table_name='audio_segments')
    op.drop_index('idx_segments_track_type', table_name='audio_segments')
    op.drop_index('idx_segments_audio_id', table_name='audio_segments')
    op.drop_table('audio_segments')
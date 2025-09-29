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
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False), 
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('segment_type', sa.String(20), server_default='voice', nullable=False),
        sa.Column('storage_uri', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.ForeignKeyConstraint(['audio_id'], ['audio_ingest.audio_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('segment_id')
    )
    op.create_index('idx_segments_audio_id', 'audio_segments', ['audio_id'])
    op.create_index('idx_segments_start_time', 'audio_segments', ['start_time'])
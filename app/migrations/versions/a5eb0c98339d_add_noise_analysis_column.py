"""add noise_analysis column

Revision ID: a5eb0c98339d
Revises: 31f9322caea1
Create Date: 2025-09-26 06:15:34.026574

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'a5eb0c98339d'
down_revision: Union[str, Sequence[str], None] = '31f9322caea1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add noise_analysis JSONB column to audio_ingest table."""
    op.add_column('audio_ingest', sa.Column('noise_analysis', JSONB, nullable=True))
    
    # Add index for better query performance on noise analysis
    op.create_index('idx_audio_noise_analysis', 'audio_ingest', ['noise_analysis'], unique=False, postgresql_using='gin')


def downgrade() -> None:
    """Remove noise_analysis column from audio_ingest table."""
    op.drop_index('idx_audio_noise_analysis', table_name='audio_ingest')
    op.drop_column('audio_ingest', 'noise_analysis')

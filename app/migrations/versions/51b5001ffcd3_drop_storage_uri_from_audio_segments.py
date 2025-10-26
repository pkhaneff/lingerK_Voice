# revision identifiers, used by Alembic.
revision = '51b5001ffcd3'
down_revision = '03b543205cb2'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa

def upgrade() -> None:
    """Drop storage_uri column from audio_segments"""
    op.execute('ALTER TABLE audio_segments DROP COLUMN IF EXISTS storage_uri')

def downgrade() -> None:
    """Add storage_uri column back"""
    op.add_column('audio_segments', 
                  sa.Column('storage_uri', sa.String(500), nullable=True))
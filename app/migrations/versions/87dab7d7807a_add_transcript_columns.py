# revision identifiers, used by Alembic.
revision = '87dab7d7807a'
down_revision = '2ea764c68b9c'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

def upgrade() -> None:
    """Add transcript and words columns to speaker_tracks table."""
    
    # Add transcript column (full text)
    op.add_column('speaker_tracks', 
                  sa.Column('transcript', sa.Text(), nullable=True))
    
    # Add words column (JSONB array of word objects)
    op.add_column('speaker_tracks',
                  sa.Column('words', JSONB, nullable=True))
    
    # Add index for words JSONB (kept)
    op.create_index('idx_speaker_tracks_words', 
                    'speaker_tracks', 
                    ['words'],
                    postgresql_using='gin')
    
    # Note: Full-text search index removed since Vietnamese config not available
    # If needed later, can add custom text search configuration


def downgrade() -> None:
    """Remove transcript and words columns."""
    
    op.drop_index('idx_speaker_tracks_words', table_name='speaker_tracks')
    op.drop_column('speaker_tracks', 'words')
    op.drop_column('speaker_tracks', 'transcript')

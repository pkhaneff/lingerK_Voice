from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = '2ea764c68b9c'
down_revision = '51b5001ffcd3'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa

def upgrade() -> None:
    # Table 1: speaker_tracks (high-level)
    op.create_table(
        'speaker_tracks',
        sa.Column('track_id', UUID(as_uuid=True), 
                  server_default=sa.text('gen_random_uuid()'), 
                  nullable=False),
        sa.Column('audio_id', UUID(as_uuid=True), nullable=False),
        
        # Speaker info
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('track_type', sa.String(20), nullable=False),
        
        # Summary ranges (JSONB array of [start, end])
        sa.Column('ranges', JSONB, nullable=False),
        
        # Metrics
        sa.Column('total_duration', sa.Float(), nullable=False),
        sa.Column('coverage', sa.Float(), nullable=False),
        
        # Timestamp
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  server_default=sa.text('NOW()'), 
                  nullable=False),
        
        # Foreign key
        sa.ForeignKeyConstraint(['audio_id'], ['audio_ingest.audio_id'], 
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('track_id'),
        
        # Check constraints
        sa.CheckConstraint("track_type IN ('single','separated')", 
                          name='speaker_tracks_type_check')
    )
    
    # Indexes for speaker_tracks
    op.create_index('idx_speaker_tracks_audio_id', 'speaker_tracks', ['audio_id'])
    op.create_index('idx_speaker_tracks_speaker_id', 'speaker_tracks', ['speaker_id'])
    op.create_index('idx_speaker_tracks_type', 'speaker_tracks', ['track_type'])
    op.create_index('idx_speaker_tracks_duration', 'speaker_tracks', ['total_duration'])
    op.create_index('idx_speaker_tracks_ranges', 'speaker_tracks', ['ranges'], 
                    postgresql_using='gin')
    
    # Table 2: track_segments (detail)
    op.create_table(
        'track_segments',
        sa.Column('segment_id', UUID(as_uuid=True), 
                  server_default=sa.text('gen_random_uuid()'), 
                  nullable=False),
        sa.Column('track_id', UUID(as_uuid=True), nullable=False),
        
        # Segment type
        sa.Column('segment_type', sa.String(20), nullable=False),
        
        # Time info
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('duration', sa.Float(), nullable=False),
        
        # Quality metrics
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('separation_method', sa.String(50), nullable=True),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  server_default=sa.text('NOW()'), 
                  nullable=False),
        
        # Foreign key
        sa.ForeignKeyConstraint(['track_id'], ['speaker_tracks.track_id'], 
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('segment_id'),
        
        # Check constraints
        sa.CheckConstraint("segment_type IN ('non-overlap','overlap')", 
                          name='track_segments_type_check')
    )
    
    # Indexes for track_segments
    op.create_index('idx_track_segments_track_id', 'track_segments', ['track_id'])
    op.create_index('idx_track_segments_type', 'track_segments', ['segment_type'])
    op.create_index('idx_track_segments_start_time', 'track_segments', ['start_time'])
    op.create_index('idx_track_segments_confidence', 'track_segments', ['confidence'])


def downgrade() -> None:
    op.drop_index('idx_track_segments_confidence', table_name='track_segments')
    op.drop_index('idx_track_segments_start_time', table_name='track_segments')
    op.drop_index('idx_track_segments_type', table_name='track_segments')
    op.drop_index('idx_track_segments_track_id', table_name='track_segments')
    op.drop_table('track_segments')
    
    # Drop speaker_tracks
    op.drop_index('idx_speaker_tracks_ranges', table_name='speaker_tracks')
    op.drop_index('idx_speaker_tracks_duration', table_name='speaker_tracks')
    op.drop_index('idx_speaker_tracks_type', table_name='speaker_tracks')
    op.drop_index('idx_speaker_tracks_speaker_id', table_name='speaker_tracks')
    op.drop_index('idx_speaker_tracks_audio_id', table_name='speaker_tracks')
    op.drop_table('speaker_tracks')

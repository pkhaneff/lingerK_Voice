import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger as custom_logger

from app.api.db.session import AsyncSessionLocal
from app.api.model.audio_model import AudioIngest
from app.api.model.video_model import VideoIngest
from app.api.model.audio_segment_model import AudioSegment
from app.api.model.speaker_track_model import SpeakerTrack
from app.api.model.track_segment_model import TrackSegment


class DBSaver:
    """Save and update metadata in database."""
    
    async def save_audio(
        self,
        file_name: str,
        storage_uri: str,
        user_id: str,
        duration: Optional[float] = None,
        codec: Optional[str] = None,
        is_video: bool = False
    ) -> Dict[str, Any]:
        """
        Save audio metadata.
        
        Args:
            file_name: Original filename
            storage_uri: S3 URI
            user_id: User ID
            duration: Audio duration in seconds
            codec: Audio codec
            is_video: Whether this audio is from video
            
        Returns:
            {'success': bool, 'data': {'audio_id'}, 'error': str}
        """
        try:
            custom_logger.info(f"Saving audio metadata: {file_name}, is_video={is_video}")
            
            audio_record = AudioIngest(
                audio_id=uuid.uuid4(),
                file_name=file_name,
                storage_uri=storage_uri,
                duration=duration,
                codec=codec,
                user_id=uuid.UUID(user_id),
                status="uploaded",
                preprocessed=False,
                created_at=datetime.utcnow(),
                is_video=is_video
            )
            
            async with AsyncSessionLocal() as session:
                session.add(audio_record)
                await session.commit()
                await session.refresh(audio_record)
                
                audio_id = str(audio_record.audio_id)
                custom_logger.info(f"Audio metadata saved: audio_id={audio_id}")
                
                return {
                    'success': True,
                    'data': {'audio_id': audio_id},
                    'error': None
                }
        
        except Exception as e:
            custom_logger.error(f"Failed to save audio: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def save_video(
        self,
        audio_id: str,
        video_storage_uri: str
    ) -> Dict[str, Any]:
        """
        Save video metadata.
        
        Args:
            audio_id: Audio ID (foreign key)
            video_storage_uri: Video S3 URI
            
        Returns:
            {'success': bool, 'data': {'video_id'}, 'error': str}
        """
        try:
            custom_logger.info(f"Saving video metadata: audio_id={audio_id}")
            
            video_record = VideoIngest(
                video_id=uuid.uuid4(),
                audio_id=uuid.UUID(audio_id),
                storage_uri=video_storage_uri
            )
            
            async with AsyncSessionLocal() as session:
                session.add(video_record)
                await session.commit()
                await session.refresh(video_record)
                
                video_id = str(video_record.video_id)
                custom_logger.info(f"Video metadata saved: video_id={video_id}")
                
                return {
                    'success': True,
                    'data': {'video_id': video_id},
                    'error': None
                }
        
        except Exception as e:
            custom_logger.error(f"Failed to save video: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def save_hybrid_tracks(
        self,
        audio_id: str,
        tracks: List[Dict]
    ) -> Dict[str, Any]:
        """
        Save tracks using hybrid approach (2 tables).
        
        NOW INCLUDES: transcript and words from Whisper
        """
        try:
            custom_logger.info(f"Saving {len(tracks)} hybrid tracks for audio_id={audio_id}")
            
            audio_uuid = uuid.UUID(audio_id)
            track_records = []
            segment_records = []
            
            for track in tracks:
                # Create SpeakerTrack record (parent)
                track_id = uuid.uuid4()
                
                speaker_track = SpeakerTrack(
                    track_id=track_id,
                    audio_id=audio_uuid,
                    speaker_id=track['speaker_id'],
                    track_type=track['type'],
                    ranges=track['ranges'],
                    total_duration=track['total_duration'],
                    coverage=track['coverage'],
                    transcript=track.get('transcript'),      # NEW: Add transcript
                    words=track.get('words'),                # NEW: Add words
                    created_at=datetime.utcnow()
                )
                track_records.append(speaker_track)
                
                # Create TrackSegment records for this track (children)
                segments = track.get('segments', [])
                custom_logger.debug(f"Track {track['speaker_id']} has {len(segments)} segments")
                
                for segment in segments:
                    track_segment = TrackSegment(
                        segment_id=uuid.uuid4(),
                        track_id=track_id,
                        segment_type=segment['segment_type'],
                        start_time=segment['start_time'],
                        end_time=segment['end_time'],
                        duration=segment['duration'],
                        confidence=segment.get('confidence'),
                        separation_method=segment.get('separation_method'),
                        created_at=datetime.utcnow()
                    )
                    segment_records.append(track_segment)
            
            # Save to database
            async with AsyncSessionLocal() as session:
                # Insert tracks first (parent records)
                session.add_all(track_records)
                await session.flush()
                
                # Insert segments (children records)
                session.add_all(segment_records)
                
                # Commit transaction
                await session.commit()
                
                custom_logger.info(
                    f"✅ Saved {len(track_records)} tracks "
                    f"and {len(segment_records)} segments successfully"
                )
                
                return {
                    'success': True,
                    'data': {
                        'tracks_saved': len(track_records),
                        'segments_saved': len(segment_records)
                    },
                    'error': None
                }
        
        except Exception as e:
            custom_logger.error(f"Failed to save hybrid tracks: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    
    async def save_tracks(
        self,
        audio_id: str,
        tracks: List[Dict]
    ) -> Dict[str, Any]:
        """
        Save tracks to database (OLD VERSION - for audio_segments table).
        
        DEPRECATED: Use save_hybrid_tracks() instead for new implementation.
        
        Args:
            audio_id: Audio ID
            tracks: List of track dictionaries
            
        Returns:
            {'success': bool, 'data': {'tracks_saved'}, 'error': str}
        """
        try:
            custom_logger.info(f"Saving {len(tracks)} tracks (old method) for audio_id={audio_id}")
            
            audio_uuid = uuid.UUID(audio_id)
            track_records = []
            
            for track in tracks:
                segment = AudioSegment(
                    segment_id=uuid.uuid4(),
                    audio_id=audio_uuid,
                    track_type=track['type'],
                    track_order=track['order'],
                    start_time=track['start_time'],
                    end_time=track['end_time'],
                    duration=track['duration'],
                    coverage=track['coverage'],
                    osd_confidence=track.get('osd_confidence'),
                    created_at=datetime.utcnow()
                )
                track_records.append(segment)
            
            async with AsyncSessionLocal() as session:
                session.add_all(track_records)
                await session.commit()
                
                custom_logger.info(f"Saved {len(track_records)} tracks successfully")
                
                return {
                    'success': True,
                    'data': {'tracks_saved': len(track_records)},
                    'error': None
                }
        
        except Exception as e:
            custom_logger.error(f"Failed to save tracks: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def update_processing_results(
        self,
        audio_id: str,
        noise_analysis: Dict,
        vad_analysis: Dict,
        osd_analysis: Dict
    ) -> Dict[str, Any]:
        """
        Update audio record with processing results.
        
        Args:
            audio_id: Audio ID
            noise_analysis: Noise reduction analysis
            vad_analysis: VAD analysis
            osd_analysis: OSD analysis
            
        Returns:
            {'success': bool, 'data': None, 'error': str}
        """
        try:
            async with AsyncSessionLocal() as session:
                audio_record = await session.get(AudioIngest, uuid.UUID(audio_id))
                
                if not audio_record:
                    custom_logger.error(f"Audio {audio_id} not found")
                    return {'success': False, 'data': None, 'error': 'Audio not found'}
                
                # Combine all analyses
                combined_analysis = {
                    'noise_segments': noise_analysis.get('noise_segments', []),
                    'vad_timeline': vad_analysis.get('vad_timeline', []),
                    'vad_statistics': vad_analysis.get('statistics', {}),
                    'osd_track_type': osd_analysis.get('track_type', 'unknown'),
                    'osd_tracks': osd_analysis.get('tracks', []),
                    'statistics': {
                        **noise_analysis.get('statistics', {}),
                        **vad_analysis.get('statistics', {}),
                        **osd_analysis.get('statistics', {})
                    },
                    'processed_at': datetime.utcnow().isoformat()
                }
                
                # Update record
                audio_record.preprocessed = True
                audio_record.processed_time = datetime.utcnow()
                audio_record.status = 'completed'
                audio_record.noise_analysis = combined_analysis
                
                # Update duration if available
                if 'total_duration' in combined_analysis['statistics']:
                    audio_record.duration = combined_analysis['statistics']['total_duration']
                
                await session.commit()
                custom_logger.info(f"Updated audio {audio_id} with processing results")
                
                return {'success': True, 'data': None, 'error': None}
                
        except Exception as e:
            custom_logger.error(f"Failed to update audio {audio_id}: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
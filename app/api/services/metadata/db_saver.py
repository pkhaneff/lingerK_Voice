import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger as custom_logger

from app.api.db.session import AsyncSessionLocal
from app.api.repositories.audio_repository import AudioRepository
from app.api.repositories.video_repository import VideoRepository
from app.api.repositories.track_repository import TrackRepository

class DBSaver:
    """
    Service to save and update metadata in database.
    Now acts as a Facade using Repositories.
    """
    
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
        """
        try:
            custom_logger.info(f"Saving audio metadata: {file_name}, is_video={is_video}")
            
            async with AsyncSessionLocal() as session:
                repo = AudioRepository(session)
                audio_record = await repo.create_audio(
                    file_name=file_name,
                    storage_uri=storage_uri,
                    user_id=user_id,
                    duration=duration,
                    codec=codec,
                    is_video=is_video
                )
                await session.commit()
                
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
        """
        try:
            custom_logger.info(f"Saving video metadata: audio_id={audio_id}")
            
            async with AsyncSessionLocal() as session:
                repo = VideoRepository(session)
                video_record = await repo.create_video(
                    audio_id=uuid.UUID(audio_id),
                    storage_uri=video_storage_uri
                )
                await session.commit()
                
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

    async def save_audio_clean(
        self,
        original_audio_id: str,
        storage_uri: str,
        processing_method: str = 'pyrnnoise'
    ) -> Dict[str, Any]:
        """
        Save cleaned audio metadata.
        """
        try:            
            custom_logger.info(f"Saving cleaned audio for original_id: {original_audio_id}")
            
            async with AsyncSessionLocal() as session:
                repo = AudioRepository(session)
                clean_record = await repo.create_audio_clean(
                    original_audio_id=uuid.UUID(original_audio_id),
                    storage_uri=storage_uri,
                    processing_method=processing_method
                )
                await session.commit()
                
                cleaned_audio_id = str(clean_record.cleaned_audio_id)
                custom_logger.info(f"Cleaned audio saved: cleaned_audio_id={cleaned_audio_id}")
                
                return {
                    'success': True, 
                    'data': {'cleaned_audio_id': cleaned_audio_id}, 
                    'error': None
                }
        except Exception as e:
            custom_logger.error(f"Failed to save audio clean: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    async def save_hybrid_tracks(
        self,
        audio_id: str,
        tracks: List[Dict]
    ) -> Dict[str, Any]:
        """
        Save tracks using hybrid approach (2 tables).
        """
        try:
            custom_logger.info(f"Saving {len(tracks)} hybrid tracks for audio_id={audio_id}")
            audio_uuid = uuid.UUID(audio_id)
            tracks_saved = 0
            segments_saved = 0

            async with AsyncSessionLocal() as session:
                repo = TrackRepository(session)
                
                for track in tracks:
                    track_record = await repo.create_track(
                        audio_id=audio_uuid,
                        speaker_id=track['speaker_id'],
                        track_type=track['type'],
                        ranges=track['ranges'],
                        total_duration=track['total_duration'],
                        coverage=track['coverage'],
                        transcript=track.get('transcript'),
                        words=track.get('words')
                    )
                    tracks_saved += 1
                    
                    for segment in track.get('segments', []):
                        await repo.create_segment(
                            track_id=track_record.track_id,
                            segment_type=segment['segment_type'],
                            start_time=segment['start_time'],
                            end_time=segment['end_time'],
                            duration=segment['duration'],
                            confidence=segment.get('confidence'),
                            separation_method=segment.get('separation_method')
                        )
                        segments_saved += 1
                
                await session.commit()
                
                custom_logger.info(f"Saved {tracks_saved} tracks and {segments_saved} segments")
                return {
                    'success': True,
                    'data': {
                        'tracks_saved': tracks_saved,
                        'segments_saved': segments_saved
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
        Deprecated but kept for compatibility if needed.
        """
        # For simplicity, keeping this one as is or we can remove it if sure it's unused.
        # But to be safe, let's just log a warning and return empty success or adapt it.
        # Given the previous code used it, let's just fail or implement if strictly needed.
        # The prompt implies moving to new structure, so let's skip deep implementation of legacy unless needed. 
        # Wait, I should probably implement it if it's called. The original code imported `AudioSegment`.
        
        from app.api.model.audio_segment_model import AudioSegment
        try:
            custom_logger.warning("save_tracks (legacy) called. Implementing via direct session for now.")
            # Simple implementation or logic to use legacy model
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
                return {'success': True, 'data': {'tracks_saved': len(track_records)}, 'error': None}
                
        except Exception as e:
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
        """
        try:
            async with AsyncSessionLocal() as session:
                repo = AudioRepository(session)
                
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
                
                duration = None
                if 'total_duration' in combined_analysis['statistics']:
                    duration = combined_analysis['statistics']['total_duration']
                    
                await repo.update_processing_results(
                    audio_id=uuid.UUID(audio_id),
                    combined_analysis=combined_analysis,
                    duration=duration
                )
                await session.commit()
                
                custom_logger.info(f"Updated audio {audio_id} with processing results")
                return {'success': True, 'data': None, 'error': None}
        except Exception as e:
            custom_logger.error(f"Failed to update audio {audio_id}: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    async def update_track_transcripts(
        self,
        audio_id: str,
        transcribed_tracks: List[Dict]
    ) -> Dict[str, Any]:
        """
        UPDATE existing speaker tracks with transcript and words data.
        """
        try:
            custom_logger.info(f"Updating transcripts for {len(transcribed_tracks)} tracks")
            audio_uuid = uuid.UUID(audio_id)
            updated_count = 0
            
            async with AsyncSessionLocal() as session:
                repo = TrackRepository(session)
                
                for track in transcribed_tracks:
                    count = await repo.update_transcript(
                        audio_id=audio_uuid,
                        speaker_id=track['speaker_id'],
                        transcript=track.get('transcript'),
                        words=track.get('words')
                    )
                    updated_count += count
                
                await session.commit()
                custom_logger.info(f"Updated {updated_count} tracks with transcripts")
                return {
                    'success': True, 
                    'data': {'tracks_updated': updated_count}, 
                    'error': None
                }
        except Exception as e:
            custom_logger.error(f"Failed to update track transcripts: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
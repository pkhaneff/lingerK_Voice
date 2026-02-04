import uuid
from typing import Dict, Any, List
from loguru import logger as custom_logger
from app.api.repositories.track_repository import TrackRepository

class TrackSaver:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def save_hybrid_tracks(
        self,
        audio_id: str,
        tracks: List[Dict]
    ) -> Dict[str, Any]:
        try:
            custom_logger.info(f"Saving {len(tracks)} hybrid tracks for audio_id={audio_id}")
            audio_uuid = uuid.UUID(audio_id)
            tracks_saved = 0
            segments_saved = 0

            async with self.session_factory() as session:
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
                return {'success': True, 'data': {'tracks_saved': tracks_saved, 'segments_saved': segments_saved}, 'error': None}
        except Exception as e:
            custom_logger.error(f"Failed to save hybrid tracks: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    async def update_track_transcripts(
        self,
        audio_id: str,
        transcribed_tracks: List[Dict]
    ) -> Dict[str, Any]:
        try:
            custom_logger.info(f"Updating transcripts for {len(transcribed_tracks)} tracks")
            audio_uuid = uuid.UUID(audio_id)
            updated_count = 0
            
            async with self.session_factory() as session:
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
                return {'success': True, 'data': {'tracks_updated': updated_count}, 'error': None}
        except Exception as e:
            custom_logger.error(f"Failed to update track transcripts: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

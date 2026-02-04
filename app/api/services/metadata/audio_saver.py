import uuid
from typing import Dict, Any, Optional
from loguru import logger as custom_logger
from app.api.repositories.audio_repository import AudioRepository
from app.api.repositories.video_repository import VideoRepository

class AudioSaver:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def save_audio(
        self,
        file_name: str,
        storage_uri: str,
        user_id: str,
        duration: Optional[float] = None,
        codec: Optional[str] = None,
        is_video: bool = False
    ) -> Dict[str, Any]:
        try:
            custom_logger.info(f"Saving audio metadata: {file_name}, is_video={is_video}")
            async with self.session_factory() as session:
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
                return {'success': True, 'data': {'audio_id': str(audio_record.audio_id)}, 'error': None}
        except Exception as e:
            custom_logger.error(f"Failed to save audio: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    async def save_video(self, audio_id: str, video_storage_uri: str) -> Dict[str, Any]:
        try:
            custom_logger.info(f"Saving video metadata: audio_id={audio_id}")
            async with self.session_factory() as session:
                repo = VideoRepository(session)
                video_record = await repo.create_video(
                    audio_id=uuid.UUID(audio_id),
                    storage_uri=video_storage_uri
                )
                await session.commit()
                return {'success': True, 'data': {'video_id': str(video_record.video_id)}, 'error': None}
        except Exception as e:
            custom_logger.error(f"Failed to save video: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    async def save_audio_clean(
        self,
        original_audio_id: str,
        storage_uri: str,
        processing_method: str = 'pyrnnoise'
    ) -> Dict[str, Any]:
        try:
            custom_logger.info(f"Saving cleaned audio for original_id: {original_audio_id}")
            async with self.session_factory() as session:
                repo = AudioRepository(session)
                clean_record = await repo.create_audio_clean(
                    original_audio_id=uuid.UUID(original_audio_id),
                    storage_uri=storage_uri,
                    processing_method=processing_method
                )
                await session.commit()
                return {'success': True, 'data': {'cleaned_audio_id': str(clean_record.cleaned_audio_id)}, 'error': None}
        except Exception as e:
            custom_logger.error(f"Failed to save audio clean: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    async def update_processing_results(
        self,
        audio_id: str,
        noise_analysis: Dict,
        vad_analysis: Dict,
        osd_analysis: Dict
    ) -> Dict[str, Any]:
        try:
            async with self.session_factory() as session:
                repo = AudioRepository(session)
                combined_analysis = {
                    'noise_segments': noise_analysis.get('noise_segments', []),
                    'vad_timeline': vad_analysis.get('vad_timeline', []),
                    'vad_statistics': vad_analysis.get('statistics', {}),
                    'osd_track_type': osd_analysis.get('track_type', 'unknown'),
                    'osd_tracks': osd_analysis.get('tracks', []),
                    'statistics': {**noise_analysis.get('statistics', {}), **vad_analysis.get('statistics', {}), **osd_analysis.get('statistics', {})},
                    'processed_at': 1 # Fixed placeholder or use datetime
                }
                # Fix datetime import
                from datetime import datetime
                combined_analysis['processed_at'] = datetime.utcnow().isoformat()
                
                duration = combined_analysis['statistics'].get('total_duration')
                await repo.update_processing_results(
                    audio_id=uuid.UUID(audio_id),
                    combined_analysis=combined_analysis,
                    duration=duration
                )
                await session.commit()
                return {'success': True, 'data': None, 'error': None}
        except Exception as e:
            custom_logger.error(f"Failed to update audio {audio_id}: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

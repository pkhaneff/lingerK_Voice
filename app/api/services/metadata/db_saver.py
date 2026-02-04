from typing import Dict, Any, Optional, List
from app.api.services.metadata.audio_saver import AudioSaver
from app.api.services.metadata.track_saver import TrackSaver
from loguru import logger as custom_logger

class DBSaver:
    """
    Facade for database saving operations.
    Delegates to specialized savers: AudioSaver and TrackSaver.
    """
    def __init__(self, session_factory):
        self.audio_saver = AudioSaver(session_factory)
        self.track_saver = TrackSaver(session_factory)

    async def save_audio(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.audio_saver.save_audio(*args, **kwargs)

    async def save_video(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.audio_saver.save_video(*args, **kwargs)

    async def save_audio_clean(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.audio_saver.save_audio_clean(*args, **kwargs)
    
    async def update_processing_results(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.audio_saver.update_processing_results(*args, **kwargs)

    async def save_hybrid_tracks(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.track_saver.save_hybrid_tracks(*args, **kwargs)

    async def update_track_transcripts(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.track_saver.update_track_transcripts(*args, **kwargs)

    async def save_tracks(self, *args, **kwargs) -> Dict[str, Any]:
        # Legacy support if needed, or redirect
        custom_logger.warning("Legacy save_tracks called")
        return {'success': False, 'error': 'Deprecated'}
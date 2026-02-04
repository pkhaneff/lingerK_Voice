from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import uuid

# Avoid importing models directly if possible to prevent circular imports during definition
# but we need them for type hinting. using 'if TYPE_CHECKING' is better but 
# for simplicity and explicit interfaces, we'll import or use forward refs.

class ITrackRepository(ABC):
    @abstractmethod
    async def create_track(
        self,
        audio_id: uuid.UUID,
        speaker_id: int,
        track_type: str,
        ranges: List,
        total_duration: float,
        coverage: float,
        transcript: Optional[str] = None,
        words: Optional[List] = None
    ) -> Any:
        pass

    @abstractmethod
    async def create_segment(
        self,
        track_id: uuid.UUID,
        segment_type: str,
        start_time: float,
        end_time: float,
        duration: float,
        confidence: Optional[float] = None,
        separation_method: Optional[str] = None
    ) -> Any:
        pass

    @abstractmethod
    async def update_transcript(
        self,
        audio_id: uuid.UUID,
        speaker_id: int,
        transcript: str,
        words: List
    ) -> int:
        pass

class IAudioRepository(ABC):
    @abstractmethod
    async def create_audio(
        self,
        file_name: str,
        storage_uri: str,
        user_id: str,
        duration: Optional[float] = None,
        codec: Optional[str] = None,
        is_video: bool = False
    ) -> Any:
        pass
    
    @abstractmethod
    async def create_audio_clean(
        self,
        original_audio_id: uuid.UUID,
        storage_uri: str,
        processing_method: str = 'pyrnnoise'
    ) -> Any:
        pass

    @abstractmethod
    async def update_processing_results(
        self,
        audio_id: uuid.UUID,
        combined_analysis: Dict,
        duration: Optional[float] = None
    ) -> None:
        pass

class IVideoRepository(ABC):
    @abstractmethod
    async def create_video(
        self,
        audio_id: uuid.UUID,
        storage_uri: str
    ) -> Any:
        pass

from fastapi import Depends
from app.api.services.upload.s3_uploader import S3Uploader
from app.api.services.processing.audio_extractor import AudioExtractor
from app.api.services.metadata.metadata_extractor import MetadataExtractor
from app.api.services.metadata.db_saver import DBSaver
from app.api.services.processing.noise_reducer import NoiseReducer
from app.api.services.processing.vad_analyzer import VADAnalyzer
from app.api.services.processing.osd_analyzer import OSDAnalyzer
from app.api.services.processing.audio_segment_processor import AudioSegmentProcessor
from app.api.services.processing.transcription_service import TranscriptionService
from app.api.services.workflow.ingestion_workflow import IngestionWorkflowService
from app.api.services.workflow.processing_workflow import ProcessingWorkflowService
from app.api.services.workflow.transcription_workflow import TranscriptionWorkflowService
from app.core.config import S3_PREFIX_AUDIO, S3_PREFIX_VIDEO
from app.api.repositories.audio_repository import AudioRepository
from app.api.repositories.track_repository import TrackRepository
from app.api.repositories.video_repository import VideoRepository
from app.api.db.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.services.metadata.audio_saver import AudioSaver
from app.api.services.metadata.track_saver import TrackSaver

# --- Low Level Services ---

def get_db_saver():
    return DBSaver()

def get_audio_uploader():
    return S3Uploader(S3_PREFIX_AUDIO)

def get_video_uploader():
    return S3Uploader(S3_PREFIX_VIDEO)

def get_metadata_extractor():
    return MetadataExtractor()

def get_audio_extractor():
    return AudioExtractor()

def get_noise_reducer():
    return NoiseReducer()

def get_vad_analyzer():
    return VADAnalyzer()

def get_osd_analyzer():
    return OSDAnalyzer()

def get_segment_processor():
    return AudioSegmentProcessor()

def get_transcription_service():
    return TranscriptionService()

# --- Repositories ---

async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session

def get_db_saver():
    return DBSaver(AsyncSessionLocal)

def get_audio_saver():
    return AudioSaver(AsyncSessionLocal)

def get_track_saver():
    return TrackSaver(AsyncSessionLocal)

def get_audio_repository(session: AsyncSession = Depends(get_db_session)):
    return AudioRepository(session)

def get_track_repository(session: AsyncSession = Depends(get_db_session)):
    return TrackRepository(session)
    
def get_video_repository(session: AsyncSession = Depends(get_db_session)):
    return VideoRepository(session)
        

# --- Workflow Services ---

def get_ingestion_workflow(
    db_saver: DBSaver = Depends(get_db_saver),
    metadata_extractor: MetadataExtractor = Depends(get_metadata_extractor),
    audio_uploader: S3Uploader = Depends(get_audio_uploader),
    video_uploader: S3Uploader = Depends(get_video_uploader),
    audio_extractor: AudioExtractor = Depends(get_audio_extractor)
) -> IngestionWorkflowService:
    return IngestionWorkflowService(
        db_saver=db_saver,
        metadata_extractor=metadata_extractor,
        audio_uploader=audio_uploader,
        video_uploader=video_uploader,
        audio_extractor=audio_extractor
    )

def get_processing_workflow(
    noise_reducer: NoiseReducer = Depends(get_noise_reducer),
    vad_analyzer: VADAnalyzer = Depends(get_vad_analyzer),
    osd_analyzer: OSDAnalyzer = Depends(get_osd_analyzer),
    segment_processor: AudioSegmentProcessor = Depends(get_segment_processor),
    db_saver: DBSaver = Depends(get_db_saver),
    uploader: S3Uploader = Depends(get_audio_uploader)
) -> ProcessingWorkflowService:
    return ProcessingWorkflowService(
        noise_reducer=noise_reducer,
        vad_analyzer=vad_analyzer,
        osd_analyzer=osd_analyzer,
        segment_processor=segment_processor,
        db_saver=db_saver,
        uploader=uploader
    )

def get_transcription_workflow(
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    db_saver: DBSaver = Depends(get_db_saver)
) -> TranscriptionWorkflowService:
    return TranscriptionWorkflowService(
        transcription_service=transcription_service,
        db_saver=db_saver
    )

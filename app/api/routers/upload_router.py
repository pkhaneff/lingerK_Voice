from typing import Tuple
from datetime import datetime
import uuid
import bcrypt
import base64
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from loguru import logger as custom_logger
from sqlalchemy import select

from app.api.responses.base import BaseResponse
from app.core.config import (
    AUDIO_EXTS, VIDEO_EXTS, AUDIO_FILE_SEIZE, VIDEO_FILE_SEIZE,
    S3_PREFIX_AUDIO, S3_PREFIX_VIDEO, GEMINI_API_KEY
)
from app.api.db.session import AsyncSessionLocal
from app.api.model.api_key import ApiKey
from app.api.services.workflow.ingestion_workflow import IngestionWorkflowService
from app.api.services.workflow.processing_workflow import ProcessingWorkflowService
from app.api.services.workflow.transcription_workflow import TranscriptionWorkflowService
from app.api.repositories.audio_repository import AudioRepository
from app.api.repositories.track_repository import TrackRepository
from app.api.dependencies import (
    get_ingestion_workflow,
    get_processing_workflow,
    get_transcription_workflow,
    get_audio_repository,
    get_track_repository
)
from app.api.utils.s3_key_utils import generate_s3_key

router = APIRouter()

# --- Helpers ---

async def validate_api_key(api_key: str) -> str:
    """Validate API key and return user_id."""
    if not api_key:
        raise HTTPException(400, "API key is required")
    
    try:
        custom_logger.info(f"Validating API key: {api_key[:20]}...")
        
        if not api_key.startswith("sk_live_"):
            custom_logger.error(f"Invalid API key format")
            raise HTTPException(401, "Invalid API key format")
        
        encoded_part = api_key[len("sk_live_"):]
        
        padding = 4 - (len(encoded_part) % 4)
        if padding != 4:
            encoded_part += "=" * padding
        
        try:
            decoded = base64.urlsafe_b64decode(encoded_part).decode('utf-8')
            key_id_str, secret = decoded.split(":", 1)  
            key_id = uuid.UUID(key_id_str)
            
            custom_logger.info(f"Key ID: {key_id}")
            
        except Exception as decode_error:
            custom_logger.error(f"Decode failed: {decode_error}")
            raise HTTPException(401, "Invalid API key encoding")
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ApiKey.user_id, ApiKey.hashed_key).where(
                    ApiKey.id == key_id, 
                    ApiKey.is_active == True
                )
            )
            api_key_record = result.first()
            
            if not api_key_record:
                custom_logger.error("API key not found or inactive")
                raise HTTPException(401, "Invalid API key")
            
            # custom_logger.info("Verifying secret hash...")
            hash_check = bcrypt.checkpw(secret.encode('utf-8'), api_key_record.hashed_key.encode('utf-8')) 
            # custom_logger.info(f"Hash verification: {hash_check}")
            
            if hash_check:
                user_id = str(api_key_record.user_id)
                custom_logger.info(f"API key validated for user: {user_id}")
                return user_id
            else:
                custom_logger.error("Secret verification failed")
                raise HTTPException(401, "Invalid API key")
            
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error")

# --- Endpoints ---

@router.post("/upload")
async def upload_file_api(
    api_key: str = Form(...),
    file: UploadFile = File(...),
    workflow: IngestionWorkflowService = Depends(get_ingestion_workflow)
):
    """Upload audio/video file and return audio_id."""
    try:
        user_id = await validate_api_key(api_key)
        
        # S3 key generation could be inside workflow, but generating here to pass uniform ID is fine too.
        s3_key = generate_s3_key(file.filename)
        
        result = await workflow.handle_upload(file=file, s3_key=s3_key, user_id=user_id)
        
        return BaseResponse.success_response(
            message="File uploaded successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.post("/process")
async def process_audio_api(
    api_key: str = Form(...),
    audio_id: str = Form(...),
    workflow: ProcessingWorkflowService = Depends(get_processing_workflow),
    audio_repo: AudioRepository = Depends(get_audio_repository)
):
    """Process audio: noise reduction, VAD, OSD, separation."""
    try:
        custom_logger.info(f"Starting processing: {audio_id}")
        
        user_id = await validate_api_key(api_key)
        
        audio_record = await audio_repo.get_audio(uuid.UUID(audio_id))
        
        if not audio_record:
            raise HTTPException(404, "Audio not found")
        
        if str(audio_record.user_id) != user_id:
            raise HTTPException(403, "Access denied")
        
        if audio_record.preprocessed:
            raise HTTPException(400, "Audio already processed")
        
        # Determine S3 Key
        if audio_record.storage_uri.startswith('s3://'):
            s3_uri_parts = audio_record.storage_uri.replace('s3://', '').split('/', 1)
            audio_s3_key = s3_uri_parts[1] if len(s3_uri_parts) > 1 else s3_uri_parts[0]
        else:
            storage_uri_parts = audio_record.storage_uri.split('/')
            audio_s3_key = '/'.join(storage_uri_parts[-2:])
            
        total_duration = audio_record.duration if audio_record.duration else 0
    
        # Delegate to Workflow Service
        result = await workflow.process_audio(
            audio_id=audio_id,
            audio_s3_key=audio_s3_key,
            total_duration=total_duration
        )
        
        if not result['success']:
            raise HTTPException(500, result['error'])
            
        return BaseResponse.success_response(
            message="Processing completed successfully",
            data=result['data']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.post("/transcribe")
async def transcribe_audio_api(
    api_key: str = Form(...),
    audio_id_clean: str = Form(...),
    workflow: TranscriptionWorkflowService = Depends(get_transcription_workflow),
    audio_repo: AudioRepository = Depends(get_audio_repository),
    track_repo: TrackRepository = Depends(get_track_repository)
):
    """Transcribe processed audio tracks."""
    try:
        custom_logger.info(f"Starting transcription: {audio_id_clean}")
        
        user_id = await validate_api_key(api_key)
        
        tracks = []
        separated_regions = []
        
        audio_clean_record = await audio_repo.get_audio_clean(uuid.UUID(audio_id_clean))
        if not audio_clean_record:
            raise HTTPException(404, "Cleaned audio not found")
        
        audio_record = await audio_repo.get_audio(audio_clean_record.original_audio_id)
        if not audio_record:
            raise HTTPException(404, "Original audio not found")
        
        if str(audio_record.user_id) != user_id:
            raise HTTPException(403, "Access denied")
        
        if not audio_record.preprocessed:
            raise HTTPException(400, "Audio not processed yet. Call /process first")
        
        # Determine Cleaned S3 Key
        if audio_clean_record.storage_uri.startswith('s3://'):
            s3_uri_parts = audio_clean_record.storage_uri.replace('s3://', '').split('/', 1)
            cleaned_s3_key = s3_uri_parts[1] if len(s3_uri_parts) > 1 else s3_uri_parts[0]
        else:
            storage_uri_parts = audio_clean_record.storage_uri.split('/')
            cleaned_s3_key = '/'.join(storage_uri_parts[-2:])
        
        # Fetch Tracks using Repository
        speaker_tracks_records = await track_repo.get_tracks_by_audio_id(audio_clean_record.original_audio_id)
        
        if not speaker_tracks_records:
            raise HTTPException(400, "No tracks found. Call /process first")
        
        if any(track.transcript for track in speaker_tracks_records):
             return BaseResponse.success_response(
                message="Audio already transcribed",
                data={'status': 'already_transcribed'}
             )

        # Build tracks object
        for track_record in speaker_tracks_records:
            track_segments = await track_repo.get_segments_by_track_id(track_record.track_id)
            
            segments = []
            for seg in track_segments:
                segments.append({
                    'segment_type': seg.segment_type,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration,
                    'confidence': seg.confidence,
                    'separation_method': seg.separation_method
                })
            
            tracks.append({
                'speaker_id': track_record.speaker_id,
                'type': track_record.track_type,
                'ranges': track_record.ranges,
                'total_duration': track_record.total_duration,
                'coverage': track_record.coverage,
                'segments': segments 
            })

        # Delegate to Workflow
        result = await workflow.transcribe_audio(
            audio_id=str(audio_record.audio_id),
            audio_id_clean=audio_id_clean,
            cleaned_s3_key=cleaned_s3_key,
            speaker_tracks=tracks,
            separated_regions=separated_regions
        )
        
        if not result['success']:
            raise HTTPException(500, result['error'])
            
        return BaseResponse.success_response(
            message="Transcription completed successfully",
            data=result['data']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")
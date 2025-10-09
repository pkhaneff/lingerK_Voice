from typing import Tuple
from datetime import datetime
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger as custom_logger

from app.api.services.upload.s3_uploader import S3Uploader
from app.api.services.processing.audio_extractor import AudioExtractor
from app.api.services.processing.noise_reducer import NoiseReducer
from app.api.services.processing.vad_analyzer import VADAnalyzer
from app.api.services.metadata.metadata_extractor import MetadataExtractor
from app.api.services.metadata.db_saver import DBSaver
from app.api.responses.base import BaseResponse
from app.core.config import (
    AUDIO_EXTS, VIDEO_EXTS, AUDIO_FILE_SEIZE, VIDEO_FILE_SEIZE,
    S3_PREFIX_AUDIO, S3_PREFIX_VIDEO
)

router = APIRouter()


def validate_file_extension(filename: str) -> Tuple[str, str]:
    """Validate file extension and return file type."""
    if not filename:
        raise HTTPException(400, "Filename is required")
    
    ext = filename.split(".")[-1].lower()
    
    if ext in AUDIO_EXTS:
        return 'audio', ext
    elif ext in VIDEO_EXTS:
        return 'video', ext
    else:
        raise HTTPException(400, f"Unsupported format. Audio: {AUDIO_EXTS} | Video: {VIDEO_EXTS}")


def validate_file_size(file_size: int, file_type: str) -> None:
    """Validate file size."""
    max_size = AUDIO_FILE_SEIZE if file_type == 'audio' else VIDEO_FILE_SEIZE
    
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        raise HTTPException(400, f"{file_type.title()} too large. Max: {max_mb:.0f}MB")


def generate_s3_key(filename: str) -> str:
    """Generate unique S3 key."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}_{filename}"


@router.post("/speech-to-text")
async def speech_to_text_api(
    file: UploadFile = File(...),
    user_id: str = "default-user"
):
    """
    Upload audio/video and process.
    
    Flow:
    - Audio: Upload -> DB -> Noise -> VAD -> Update DB
    - Video: Upload video -> Extract audio -> Upload audio -> DB -> Noise -> VAD -> Update DB
    """
    try:
        custom_logger.info(f"Starting upload: {file.filename}")
        
        # Validate
        file_type, ext = validate_file_extension(file.filename)
        validate_file_size(file.size, file_type)
        s3_key = generate_s3_key(file.filename)
        
        audio_id = None
        audio_s3_key = None
        
        # Initialize services
        metadata_extractor = MetadataExtractor()
        db_saver = DBSaver()
        
        if file_type == 'audio':
            custom_logger.info("Processing AUDIO")
            
            # Upload audio to S3
            audio_uploader = S3Uploader(S3_PREFIX_AUDIO)
            upload_result = await audio_uploader.upload_file(file, s3_key)
            
            if not upload_result['success']:
                raise HTTPException(500, f"Upload failed: {upload_result['error']}")
            
            audio_s3_key = upload_result['data']['s3_key']
            audio_s3_url = upload_result['data']['s3_url']
            
            # Extract metadata
            metadata_result = await metadata_extractor.extract_from_s3(audio_s3_key)
            if not metadata_result['success']:
                custom_logger.warning(f"Metadata extraction failed: {metadata_result['error']}")
                duration, codec = None, None
            else:
                duration = metadata_result['data']['duration']
                codec = metadata_result['data']['codec']
            
            # Save to DB
            save_result = await db_saver.save_audio(
                file_name=file.filename,
                storage_uri=audio_s3_url,
                user_id=user_id,
                duration=duration,
                codec=codec,
                is_video=False
            )
            
            if not save_result['success']:
                raise HTTPException(500, f"DB save failed: {save_result['error']}")
            
            audio_id = save_result['data']['audio_id']
            
        else:  # video
            custom_logger.info("Processing VIDEO")
            
            # Upload video to S3
            video_uploader = S3Uploader(S3_PREFIX_VIDEO)
            video_upload_result = await video_uploader.upload_file(file, s3_key)
            
            if not video_upload_result['success']:
                raise HTTPException(500, f"Video upload failed: {video_upload_result['error']}")
            
            video_s3_key = video_upload_result['data']['s3_key']
            video_s3_url = video_upload_result['data']['s3_url']
            
            # Extract audio from video
            audio_extractor = AudioExtractor()
            extraction_result = await audio_extractor.extract_from_s3(video_s3_key)
            
            if not extraction_result['success']:
                raise HTTPException(500, f"Audio extraction failed: {extraction_result['error']}")
            
            audio_content = extraction_result['data']['audio_content']
            duration = extraction_result['data']['duration']
            codec = extraction_result['data']['codec']
            
            # Upload extracted audio to S3 (reuse audio uploader)
            audio_filename = s3_key.rsplit('.', 1)[0] + '.mp3'
            audio_uploader = S3Uploader(S3_PREFIX_AUDIO)
            audio_upload_result = await audio_uploader.upload_bytes(
                content=audio_content,
                s3_key=audio_filename,
                content_type='audio/mpeg'
            )
            
            if not audio_upload_result['success']:
                raise HTTPException(500, f"Audio upload failed: {audio_upload_result['error']}")
            
            audio_s3_key = audio_upload_result['data']['s3_key']
            audio_s3_url = audio_upload_result['data']['s3_url']
            
            # Save audio to DB
            audio_save_result = await db_saver.save_audio(
                file_name=audio_filename,
                storage_uri=audio_s3_url,
                user_id=user_id,
                duration=duration,
                codec=codec,
                is_video=True
            )
            
            if not audio_save_result['success']:
                raise HTTPException(500, f"Audio DB save failed: {audio_save_result['error']}")
            
            audio_id = audio_save_result['data']['audio_id']
            
            # Save video to DB
            video_save_result = await db_saver.save_video(
                audio_id=audio_id,
                video_storage_uri=video_s3_url
            )
            
            if not video_save_result['success']:
                custom_logger.warning(f"Video DB save failed: {video_save_result['error']}")
        
        # Process audio: Noise reduction
        custom_logger.info("Starting noise reduction")
        noise_reducer = NoiseReducer()
        noise_result = await noise_reducer.reduce_noise_from_s3(audio_s3_key)
        
        if not noise_result['success']:
            custom_logger.warning(f"Noise reduction failed: {noise_result['error']}")
            return BaseResponse.success_response(
                message="Upload successful but processing failed",
                data={
                    'audio_id': audio_id,
                    'file_name': file.filename,
                    's3_key': audio_s3_key,
                    'processing': 'failed'
                }
            )
        
        cleaned_audio_path = noise_result['data']['cleaned_path']
        noise_analysis = noise_result['data']['noise_analysis']
        
        # Process audio: VAD
        custom_logger.info("Starting VAD analysis")
        vad_analyzer = VADAnalyzer()
        vad_result = await vad_analyzer.analyze_voice_activity(cleaned_audio_path)
        
        if not vad_result['success']:
            custom_logger.warning(f"VAD failed: {vad_result['error']}")
            vad_analysis = {'vad_segments': [], 'statistics': {}}
        else:
            vad_analysis = vad_result['data']
        
        # Update DB with processing results
        update_result = await db_saver.update_processing_results(
            audio_id=audio_id,
            noise_analysis=noise_analysis,
            vad_analysis=vad_analysis
        )
        
        if not update_result['success']:
            custom_logger.warning(f"DB update failed: {update_result['error']}")
        
        # Cleanup temp file
        from pathlib import Path
        if cleaned_audio_path and Path(cleaned_audio_path).exists():
            try:
                Path(cleaned_audio_path).unlink()
                custom_logger.debug(f"Cleaned up temp file: {cleaned_audio_path}")
            except Exception as e:
                custom_logger.warning(f"Cleanup failed: {str(e)}")
        
        # Return success response
        return BaseResponse.success_response(
            message="Upload and processing completed successfully",
            data={
                'audio_id': audio_id,
                'file_name': file.filename,
                'file_type': file_type,
                's3_key': audio_s3_key,
                'processing': {
                    'status': 'completed',
                    'noise_reduction': {
                        'noise_ratio': noise_analysis.get('statistics', {}).get('noise_ratio', 0),
                        'segments_count': noise_analysis.get('statistics', {}).get('segments_count', 0)
                    },
                    'vad_analysis': {
                        'voice_segments_count': len(vad_analysis.get('vad_segments', [])),
                        'voice_activity_ratio': vad_analysis.get('statistics', {}).get('voice_activity_ratio', 0)
                    }
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")
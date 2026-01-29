from typing import Tuple, Dict, Any, Optional
from fastapi import UploadFile, HTTPException
import uuid
from loguru import logger as custom_logger

from app.api.services.upload.s3_uploader import S3Uploader
from app.api.services.processing.audio_extractor import AudioExtractor
from app.api.services.metadata.metadata_extractor import MetadataExtractor
from app.api.services.metadata.db_saver import DBSaver
from app.core.config import (
    AUDIO_EXTS, VIDEO_EXTS, AUDIO_FILE_SEIZE, VIDEO_FILE_SEIZE,
    S3_PREFIX_AUDIO, S3_PREFIX_VIDEO
)

class IngestionWorkflowService:
    def __init__(
        self,
        db_saver: DBSaver,
        metadata_extractor: MetadataExtractor,
        audio_uploader: S3Uploader,
        video_uploader: S3Uploader,
        audio_extractor: AudioExtractor
    ):
        self.db_saver = db_saver
        self.metadata_extractor = metadata_extractor
        self.audio_uploader = audio_uploader
        self.video_uploader = video_uploader
        self.audio_extractor = audio_extractor

    async def _validate_file(self, filename: str, file_size: int) -> Tuple[str, str]:
        if not filename:
            raise HTTPException(400, "Filename is required")
        
        ext = filename.split(".")[-1].lower()
        
        if ext in AUDIO_EXTS:
            file_type = 'audio'
        elif ext in VIDEO_EXTS:
            file_type = 'video'
        else:
            raise HTTPException(400, f"Unsupported format. Audio: {AUDIO_EXTS} | Video: {VIDEO_EXTS}")
            
        max_size = AUDIO_FILE_SEIZE if file_type == 'audio' else VIDEO_FILE_SEIZE
        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            raise HTTPException(400, f"{file_type.title()} too large. Max: {max_mb:.0f}MB")
            
        return file_type, ext

    async def handle_upload(self, file: UploadFile, s3_key: str, user_id: str) -> Dict[str, Any]:
        custom_logger.info(f"Starting ingestion workflow for: {file.filename}")
        
        file_type, ext = await self._validate_file(file.filename, file.size)
        
        audio_id = None
        final_s3_key = None
        
        if file_type == 'audio':
            custom_logger.info("Processing AUDIO upload")
            upload_result = await self.audio_uploader.upload_file(file, s3_key)
            
            if not upload_result['success']:
                raise HTTPException(500, f"Upload failed: {upload_result['error']}")
            
            final_s3_key = upload_result['data']['s3_key']
            audio_s3_url = upload_result['data']['s3_url']
            
            metadata_result = await self.metadata_extractor.extract_from_s3(final_s3_key)
            if not metadata_result['success']:
                custom_logger.warning(f"Metadata extraction failed")
                duration, codec = None, None
            else:
                duration = metadata_result['data']['duration']
                codec = metadata_result['data']['codec']
            
            save_result = await self.db_saver.save_audio(
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
            
        else:
            custom_logger.info("Processing VIDEO upload")
            video_upload_result = await self.video_uploader.upload_file(file, s3_key)
            
            if not video_upload_result['success']:
                raise HTTPException(500, f"Video upload failed")
            
            video_s3_key = video_upload_result['data']['s3_key']
            video_s3_url = video_upload_result['data']['s3_url']
            
            extraction_result = await self.audio_extractor.extract_from_s3(video_s3_key)
            if not extraction_result['success']:
                raise HTTPException(500, f"Audio extraction failed")
            
            audio_content = extraction_result['data']['audio_content']
            duration = extraction_result['data']['duration']
            codec = extraction_result['data']['codec']
            
            audio_filename = s3_key.rsplit('.', 1)[0] + '.mp3'
            
            # Using audio_uploader for the extracted audio
            audio_upload_result = await self.audio_uploader.upload_bytes(
                content=audio_content,
                s3_key=audio_filename,
                content_type='audio/mpeg'
            )
            
            if not audio_upload_result['success']:
                raise HTTPException(500, f"Extracted audio upload failed")
            
            final_s3_key = audio_upload_result['data']['s3_key']
            audio_s3_url = audio_upload_result['data']['s3_url']
            
            audio_save_result = await self.db_saver.save_audio(
                file_name=audio_filename,
                storage_uri=audio_s3_url,
                user_id=user_id,
                duration=duration,
                codec=codec,
                is_video=True
            )
            
            if not audio_save_result['success']:
                raise HTTPException(500, f"Audio DB save failed")
            
            audio_id = audio_save_result['data']['audio_id']
            
            await self.db_saver.save_video(
                audio_id=audio_id,
                video_storage_uri=video_s3_url
            )

        return {
            'audio_id': audio_id,
            'file_name': file.filename,
            'file_type': file_type,
            's3_key': final_s3_key,
            'status': 'uploaded'
        }

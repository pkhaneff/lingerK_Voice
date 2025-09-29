import io
from typing import Dict, Any
from fastapi import UploadFile
from loguru import logger as custom_logger

from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws.s3.repository.object import put_object
from app.api.infra.aws.s3.entity.object import S3Object
from app.core.config import S3_PREFIX_AUDIO

class AudioUploadService:
    def __init__(self):
        self.bucket_name = s3_bucket
        self.s3_prefix = S3_PREFIX_AUDIO
    
    async def upload(self, file: UploadFile, s3_key: str, file_content: bytes) -> Dict[str, Any]:
        try:
            
            full_s3_key = f"{self.s3_prefix}/{s3_key}"
            
            content_type = self._get_content_type(file.filename)
            
            # Create S3Object
            s3_object = S3Object(
                body=file_content,
                content_length=len(file_content),
                content_type=content_type,
                key=full_s3_key,
                last_modified=None
            )
            
            # Upload to S3
            put_result = put_object(s3_object, self.bucket_name)
            
            s3_url = f"s3://{self.bucket_name}/{full_s3_key}"
            custom_logger.info(f"Audio uploaded successfully: {s3_url}")
            
            return {
                'success': True,
                's3_key': full_s3_key,
                's3_url': s3_url,
                'bucket_name': self.bucket_name
            }
        except Exception as e:
            custom_logger.error(f"AudioUploadService upload failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    async def upload_raw_content(self, s3_key: str, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """
        Upload raw audio content to S3 (for extracted audio from video)
        
        Args:
            s3_key: S3 key for the file
            file_content: Raw audio bytes
            content_type: MIME type (e.g., 'audio/mpeg')
            
        Returns:
            Upload result dictionary
        """
        try:
            full_s3_key = f"{self.s3_prefix}/{s3_key}"
            
            s3_object = S3Object(
                body=io.BytesIO(file_content),
                content_length=len(file_content),
                content_type=content_type,
                key=full_s3_key,
                last_modified=None
            )
            
            put_result = put_object(s3_object, self.bucket_name)
            
            s3_url = f"s3://{self.bucket_name}/{full_s3_key}"
            custom_logger.info(f"Raw audio content uploaded successfully: {s3_url}")
            
            return {
                'success': True,
                's3_key': full_s3_key,
                's3_url': s3_url,
                'bucket_name': self.bucket_name
            }
        except Exception as e:
            custom_logger.error(f"AudioUploadService upload_raw_content failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _get_content_type(self, filename: str) -> str:
        """Safely detect content type from filename"""
        try:
            custom_logger.debug(f"_get_content_type called with: {repr(filename)} (type: {type(filename)})")
            
            if not filename or not isinstance(filename, str):
                custom_logger.warning(f"Invalid filename, using default audio/ogg: {repr(filename)}")
                return 'audio/ogg'
            
            filename_lower = filename.lower()
            custom_logger.debug(f"Filename lowercased: {repr(filename_lower)}")
            
            if filename_lower.endswith('.m4a'):
                content_type = 'audio/mp4'
            elif filename_lower.endswith('.ogg'):
                content_type = 'audio/ogg'
            elif filename_lower.endswith('.mp3'):
                content_type = 'audio/mpeg'
            elif filename_lower.endswith('.wav'):
                content_type = 'audio/wav'
            else:
                content_type = 'audio/ogg'
            
            custom_logger.debug(f"Detected content type: {content_type}")
            return content_type
            
        except Exception as e:
            custom_logger.error(f"Error in _get_content_type: {str(e)} | filename: {repr(filename)}")
            return 'audio/ogg'
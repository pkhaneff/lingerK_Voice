import io
from typing import Dict, Any
from fastapi import UploadFile
from loguru import logger as custom_logger

from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws import session
from app.core.config import S3_PREFIX_VIDEO, VIDEO_FILE_SEIZE, VIDEO_MULTIPART_THRESHOLD, VIDEO_CHUNK_SIZE_SMALL, VIDEO_CHUNK_SIZE_LARGE, MAX_REQUEST_SIZE

class VideoUploadService:
    def __init__(self):
        self.bucket_name = s3_bucket
        self.s3_prefix = S3_PREFIX_VIDEO
        self.s3_client = session.client("s3")
        
        # Optimized settings for large files
        self.multipart_threshold = VIDEO_MULTIPART_THRESHOLD
        self.chunk_size_small = VIDEO_CHUNK_SIZE_SMALL
        self.chunk_size_large = VIDEO_CHUNK_SIZE_LARGE
        self.max_file_size = MAX_REQUEST_SIZE
    
    async def upload(self, file: UploadFile, s3_key: str, file_size: int) -> Dict[str, Any]:
        try:
            custom_logger.info(f"Starting video upload: {s3_key}, size: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
            
            # Validate file size
            if file_size > self.max_file_size:
                return {
                    'success': False, 
                    'error': f'File too large. Max size: {self.max_file_size/1024/1024:.0f}MB'
                }
            
            full_s3_key = f"{self.s3_prefix}/{s3_key}"
            
            # Choose upload method based on file size
            if file_size > self.multipart_threshold:
                custom_logger.info(f"Using multipart upload for large file: {file_size:,} bytes")
                return await self._multipart_upload(file, full_s3_key, file_size)
            else:
                custom_logger.info(f"Using direct upload for smaller file: {file_size:,} bytes")
                return await self._direct_upload(file, full_s3_key, file_size)
                
        except Exception as e:
            custom_logger.error(f"Video upload failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    async def _direct_upload(self, file: UploadFile, s3_key: str, file_size: int):
        """Direct upload for files < 100MB"""
        from app.api.infra.aws.s3.repository.object import put_object
        from app.api.infra.aws.s3.entity.object import S3Object
        
        try:
            await file.seek(0)
            file_content = await file.read()
            
            s3_object = S3Object(
                body=io.BytesIO(file_content),
                content_length=file_size,
                content_type=self._get_video_content_type(s3_key),
                key=s3_key,
                last_modified=None
            )
            
            put_object(s3_object, self.bucket_name)
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            
            custom_logger.info(f"Direct upload completed: {s3_url}")
            return {
                'success': True, 
                's3_key': s3_key, 
                's3_url': s3_url, 
                'bucket_name': self.bucket_name
            }
            
        except Exception as e:
            custom_logger.error(f"Direct upload failed: {str(e)}")
            raise e
    
    async def _multipart_upload(self, file: UploadFile, s3_key: str, file_size: int):
        """Multipart upload for files >= 100MB"""
        upload_id = None
        
        try:
            # Choose optimal chunk size based on file size
            chunk_size = self._get_optimal_chunk_size(file_size)
            custom_logger.info(f"Using chunk size: {chunk_size/1024/1024:.1f}MB")
            
            # Initiate multipart upload
            response = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name, 
                Key=s3_key, 
                ContentType=self._get_video_content_type(s3_key)
            )
            upload_id = response['UploadId']
            custom_logger.info(f"Multipart upload initiated: {upload_id}")
            
            # Upload parts in chunks
            await file.seek(0)
            parts = []
            part_number = 1
            bytes_uploaded = 0
            
            while bytes_uploaded < file_size:
                # Calculate current chunk size
                current_chunk_size = min(chunk_size, file_size - bytes_uploaded)
                
                # Read chunk without loading entire file into memory
                chunk_data = await file.read(current_chunk_size)
                
                if not chunk_data:
                    break
                
                # Upload part
                custom_logger.debug(f"Uploading part {part_number}: {len(chunk_data):,} bytes")
                part_response = self.s3_client.upload_part(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=io.BytesIO(chunk_data)
                )
                
                parts.append({
                    'PartNumber': part_number, 
                    'ETag': part_response['ETag']
                })
                
                bytes_uploaded += len(chunk_data)
                part_number += 1
                
                # Log progress
                progress = (bytes_uploaded / file_size) * 100
                custom_logger.info(f"Upload progress: {progress:.1f}% ({bytes_uploaded:,}/{file_size:,} bytes)")
            
            # Complete multipart upload
            custom_logger.info(f"Completing multipart upload with {len(parts)} parts")
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            custom_logger.info(f"Multipart upload completed successfully: {s3_url}")
            
            return {
                'success': True, 
                's3_key': s3_key, 
                's3_url': s3_url, 
                'bucket_name': self.bucket_name,
                'parts_count': len(parts),
                'total_size': bytes_uploaded
            }
            
        except Exception as e:
            custom_logger.error(f"Multipart upload failed: {str(e)}")
            
            # Abort multipart upload to avoid charges
            if upload_id:
                try:
                    custom_logger.info(f"Aborting multipart upload: {upload_id}")
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        UploadId=upload_id
                    )
                except Exception as abort_error:
                    custom_logger.error(f"Failed to abort multipart upload: {str(abort_error)}")
            
            raise e
    
    def _get_optimal_chunk_size(self, file_size: int) -> int:
        """Calculate optimal chunk size based on file size"""
        if file_size < 200 * 1024 * 1024:  # < 200MB
            return VIDEO_CHUNK_SIZE_SMALL  # 10MB chunks
        elif file_size < 500 * 1024 * 1024:  # 200MB - 500MB
            return 25 * 1024 * 1024  # 25MB chunks
        else:  # > 500MB
            return VIDEO_CHUNK_SIZE_LARGE  # 50MB chunks
    
    def _get_video_content_type(self, s3_key: str) -> str:
        """Return MP4 content type (only MP4 files are supported)"""
        return 'video/mp4'
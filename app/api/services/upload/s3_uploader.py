"""Async S3 uploader with streaming - Memory efficient"""
import io
import aioboto3
from typing import Dict, Any
from fastapi import UploadFile
from loguru import logger as custom_logger

from app.api.infra.aws.s3 import s3_bucket
from app.core.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION


class S3Uploader:
    """Async S3 uploader with streaming for memory efficiency"""
    
    def __init__(self, s3_prefix: str):
        self.bucket_name = s3_bucket
        self.s3_prefix = s3_prefix
        self.session = aioboto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
    
    async def upload_file(self, file: UploadFile, s3_key: str) -> Dict[str, Any]:
        """
        Upload file to S3 with streaming (memory efficient).
        
        Memory usage: ~10MB buffer instead of full file size
        """
        try:
            full_s3_key = f"{self.s3_prefix}/{s3_key}"
            file_size = file.size
            
            custom_logger.info(f"📤 Uploading to S3: {full_s3_key}, size: {file_size:,} bytes")
            
            # Use multipart for files > 100MB
            if file_size > 100 * 1024 * 1024:
                result = await self._multipart_upload(file, full_s3_key, file_size)
            else:
                result = await self._direct_upload(file, full_s3_key, file_size)
            
            return result
            
        except Exception as e:
            custom_logger.error(f"❌ Upload failed: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def _direct_upload(
        self,
        file: UploadFile,
        s3_key: str,
        file_size: int
    ) -> Dict[str, Any]:
        """Direct streaming upload for files < 100MB"""
        try:
            await file.seek(0)
            content_type = self._detect_content_type(file.filename)
            
            async with self.session.client('s3') as s3:
                # ✅ Stream upload directly - NO memory loading
                await s3.upload_fileobj(
                    file.file,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': content_type}
                )
            
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            custom_logger.info(f"✅ Upload completed: {s3_url}")
            
            return {
                'success': True,
                'data': {
                    's3_key': s3_key,
                    's3_url': s3_url,
                    'bucket_name': self.bucket_name
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Direct upload failed: {str(e)}")
            raise
    
    async def _multipart_upload(
        self,
        file: UploadFile,
        s3_key: str,
        file_size: int
    ) -> Dict[str, Any]:
        """Async multipart upload for files >= 100MB"""
        upload_id = None
        
        try:
            chunk_size = self._calculate_chunk_size(file_size)
            content_type = self._detect_content_type(file.filename)
            
            async with self.session.client('s3') as s3:
                # Create multipart upload
                response = await s3.create_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    ContentType=content_type
                )
                upload_id = response['UploadId']
                custom_logger.info(f"🔄 Multipart upload started: {upload_id}")
                
                await file.seek(0)
                parts = []
                part_number = 1
                bytes_uploaded = 0
                
                # Upload parts in chunks
                while bytes_uploaded < file_size:
                    current_chunk_size = min(chunk_size, file_size - bytes_uploaded)
                    chunk_data = await file.read(current_chunk_size)
                    
                    if not chunk_data:
                        break
                    
                    # Upload part (async, non-blocking)
                    part_response = await s3.upload_part(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk_data
                    )
                    
                    parts.append({
                        'PartNumber': part_number,
                        'ETag': part_response['ETag']
                    })
                    
                    bytes_uploaded += len(chunk_data)
                    part_number += 1
                    
                    progress = (bytes_uploaded / file_size) * 100
                    custom_logger.info(f"📊 Progress: {progress:.1f}%")
                
                # Complete multipart upload
                await s3.complete_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': parts}
                )
                
                s3_url = f"s3://{self.bucket_name}/{s3_key}"
                custom_logger.info(f"✅ Multipart upload completed: {s3_url}")
                
                return {
                    'success': True,
                    'data': {
                        's3_key': s3_key,
                        's3_url': s3_url,
                        'bucket_name': self.bucket_name,
                        'parts_count': len(parts)
                    },
                    'error': None
                }
        
        except Exception as e:
            custom_logger.error(f"❌ Multipart upload failed: {str(e)}")
            
            # Cleanup on failure
            if upload_id:
                try:
                    async with self.session.client('s3') as s3:
                        await s3.abort_multipart_upload(
                            Bucket=self.bucket_name,
                            Key=s3_key,
                            UploadId=upload_id
                        )
                    custom_logger.info(f"🗑️ Aborted upload: {upload_id}")
                except:
                    pass
            
            raise
    
    async def upload_bytes(
        self,
        content: bytes,
        s3_key: str,
        content_type: str
    ) -> Dict[str, Any]:
        """Upload raw bytes to S3"""
        try:
            full_s3_key = f"{self.s3_prefix}/{s3_key}"
            
            async with self.session.client('s3') as s3:
                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=full_s3_key,
                    Body=content,
                    ContentType=content_type
                )
            
            s3_url = f"s3://{self.bucket_name}/{full_s3_key}"
            custom_logger.info(f"✅ Bytes uploaded: {s3_url}")
            
            return {
                'success': True,
                'data': {
                    's3_key': full_s3_key,
                    's3_url': s3_url,
                    'bucket_name': self.bucket_name
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Bytes upload failed: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _calculate_chunk_size(self, file_size: int) -> int:
        """Calculate optimal chunk size"""
        if file_size < 200 * 1024 * 1024:  # < 200MB
            return 10 * 1024 * 1024  # 10MB
        elif file_size < 500 * 1024 * 1024:  # < 500MB
            return 25 * 1024 * 1024  # 25MB
        else:  # >= 500MB
            return 50 * 1024 * 1024  # 50MB
    
    def _detect_content_type(self, filename: str) -> str:
        """Detect content type from filename"""
        if not filename:
            return 'application/octet-stream'
        
        ext = filename.lower().split('.')[-1]
        
        content_types = {
            'm4a': 'audio/mp4',
            'ogg': 'audio/ogg',
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'mp4': 'video/mp4'
        }
        
        return content_types.get(ext, 'application/octet-stream')
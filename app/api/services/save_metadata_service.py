from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from loguru import logger as custom_logger

from app.api.db.session import AsyncSessionLocal
from app.api.model.audio_model import AudioIngest
from app.api.model.video_model import VideoIngest
from app.api.services.audio_metadata_extractor import AudioMetadataExtractor

class MetadataService:
    """Service to save file metadata to database"""
    
    def __init__(self):
        self.metadata_extractor = AudioMetadataExtractor()
    
    async def save_audio_metadata(
        self, 
        file_name: str,
        storage_uri: str,
        user_id: str,
        is_video: bool = False
    ) -> Dict[str, Any]:
        """
        Save audio metadata to database
        
        Args:
            file_name: Original filename
            storage_uri: S3 URI (s3://bucket/key)
            user_id: User UUID
            is_video: True if this audio was extracted from video
            
        Returns:
            Dict with save result and audio_id
        """
        try:
            custom_logger.info(f"Saving audio metadata: {file_name}, is_video={is_video}")
            
            # Extract S3 key from URI
            s3_key = storage_uri.replace(f"s3://", "").split("/", 1)[1]
            
            # Extract metadata from audio file
            metadata = await self.metadata_extractor.extract_metadata(s3_key)
            
            # Create audio record
            audio_record = AudioIngest(
                audio_id=uuid.uuid4(),
                file_name=file_name,
                storage_uri=storage_uri,
                duration=metadata.get('duration'),
                codec=metadata.get('codec'),
                user_id=uuid.UUID(user_id),
                status="uploaded",
                preprocessed=False,
                created_at=datetime.utcnow(),
                is_video=is_video
            )
            
            # Save to database
            async with AsyncSessionLocal() as session:
                session.add(audio_record)
                await session.commit()
                await session.refresh(audio_record)
                
                audio_id = str(audio_record.audio_id)
                custom_logger.info(f"Audio metadata saved successfully: audio_id={audio_id}")
                
                return {
                    'success': True,
                    'audio_id': audio_id,
                    'duration': metadata.get('duration'),
                    'codec': metadata.get('codec')
                }
        
        except Exception as e:
            custom_logger.error(f"Failed to save audio metadata: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def save_video_metadata(
        self,
        audio_id: str,
        storage_uri: str
    ) -> Dict[str, Any]:
        """
        Save video metadata to database
        
        Args:
            audio_id: UUID of related audio record
            storage_uri: S3 URI of video file
            
        Returns:
            Dict with save result and video_id
        """
        try:
            custom_logger.info(f"Saving video metadata: audio_id={audio_id}")
            
            # Create video record
            video_record = VideoIngest(
                video_id=uuid.uuid4(),
                audio_id=uuid.UUID(audio_id),
                storage_uri=storage_uri
            )
            
            # Save to database
            async with AsyncSessionLocal() as session:
                session.add(video_record)
                await session.commit()
                await session.refresh(video_record)
                
                video_id = str(video_record.video_id)
                custom_logger.info(f"Video metadata saved successfully: video_id={video_id}")
                
                return {
                    'success': True,
                    'video_id': video_id
                }
        
        except Exception as e:
            custom_logger.error(f"Failed to save video metadata: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def save_upload_metadata(
        self,
        file_name: str,
        file_type: str,  # 'audio' or 'video'
        audio_s3_uri: str,
        video_s3_uri: Optional[str],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Main function to save upload metadata based on file type
        
        Args:
            file_name: Original filename
            file_type: 'audio' or 'video'
            audio_s3_uri: S3 URI of audio file
            video_s3_uri: S3 URI of video file (only for video uploads)
            user_id: User UUID
            
        Returns:
            Dict with save result
        """
        try:
            custom_logger.info(f"Saving upload metadata: {file_name}, type={file_type}")
            
            if file_type == 'audio':
                # Save audio metadata
                result = await self.save_audio_metadata(
                    file_name=file_name,
                    storage_uri=audio_s3_uri,
                    user_id=user_id,
                    is_video=False
                )
                
                return {
                    'success': result['success'],
                    'audio_id': result.get('audio_id'),
                    'error': result.get('error')
                }
            
            elif file_type == 'video':
                # Save audio metadata (extracted from video)
                audio_result = await self.save_audio_metadata(
                    file_name=file_name,
                    storage_uri=audio_s3_uri,
                    user_id=user_id,
                    is_video=True
                )
                
                if not audio_result['success']:
                    return audio_result
                
                # Save video metadata
                video_result = await self.save_video_metadata(
                    audio_id=audio_result['audio_id'],
                    storage_uri=video_s3_uri
                )
                
                if not video_result['success']:
                    return video_result
                
                return {
                    'success': True,
                    'audio_id': audio_result['audio_id'],
                    'video_id': video_result['video_id']
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Invalid file type: {file_type}'
                }
        
        except Exception as e:
            custom_logger.error(f"Failed to save upload metadata: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
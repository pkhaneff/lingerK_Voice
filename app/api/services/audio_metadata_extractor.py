import tempfile
from pathlib import Path
from typing import Dict, Optional
from loguru import logger as custom_logger
import asyncio
import functools

from moviepy import AudioFileClip
from app.api.infra.aws.s3.repository.object import get_object
from app.api.infra.aws.s3 import s3_bucket


class AudioMetadataExtractor:
    """Extract metadata from audio files stored in S3"""
    
    def __init__(self):
        self.bucket_name = s3_bucket
    
    async def extract_metadata(self, s3_key: str) -> Dict[str, Optional[str]]:
        """
        Extract duration and codec from audio file in S3
        
        Args:
            s3_key: S3 key of the audio file
            
        Returns:
            Dict with duration and codec info
        """
        temp_file_path = None
        
        try:
            custom_logger.info(f"Extracting metadata from audio: {s3_key}")
            
            # Download audio from S3 to temp file
            with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Get audio object from S3
            audio_object = get_object(s3_key, self.bucket_name)
            
            # Write to temp file
            with open(temp_file_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            custom_logger.debug(f"Audio downloaded to temp file: {temp_file_path}")
            
            # Extract metadata using MoviePy in thread
            loop = asyncio.get_event_loop()
            metadata = await loop.run_in_executor(
                None, 
                functools.partial(self._extract_metadata_sync, temp_file_path)
            )
            
            custom_logger.info(f"Metadata extracted: duration={metadata.get('duration')}s, codec={metadata.get('codec')}")
            return metadata
            
        except Exception as e:
            custom_logger.error(f"Failed to extract metadata from {s3_key}: {str(e)}")
            return {'duration': None, 'codec': None, 'error': str(e)}
        
        finally:
            # Cleanup temp file
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    custom_logger.debug(f"Cleaned up temp file: {temp_file_path}")
                except Exception as e:
                    custom_logger.warning(f"Failed to cleanup temp file: {str(e)}")
    
    def _extract_metadata_sync(self, file_path: str) -> Dict[str, Optional[str]]:
        """Synchronously extract metadata using MoviePy"""
        audio_clip = None
        
        try:
            custom_logger.debug(f"Loading audio file for metadata: {file_path}")
            
            # Load audio file
            audio_clip = AudioFileClip(file_path)
            
            # Extract duration
            duration = audio_clip.duration  # in seconds
            
            # Try to get codec info from filename extension
            file_ext = Path(file_path).suffix.lower()
            codec_map = {
                '.mp3': 'mp3',
                '.m4a': 'aac',
                '.ogg': 'ogg',
                '.wav': 'pcm',
                '.aac': 'aac'
            }
            codec = codec_map.get(file_ext, 'unknown')
            
            return {
                'duration': round(duration, 2) if duration else None,
                'codec': codec
            }
            
        except Exception as e:
            custom_logger.error(f"Error extracting metadata: {str(e)}")
            return {'duration': None, 'codec': None, 'error': str(e)}
        
        finally:
            if audio_clip:
                try:
                    audio_clip.close()
                except:
                    pass
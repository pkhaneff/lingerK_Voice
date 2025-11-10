import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import functools
from loguru import logger as custom_logger
from moviepy import AudioFileClip

from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws.s3.repository.object import get_object


class MetadataExtractor:
    """Extract metadata from audio files."""
    
    def __init__(self):
        self.bucket_name = s3_bucket
    
    async def extract_from_s3(self, audio_s3_key: str) -> Dict[str, Any]:
        """
        Download audio from S3, extract metadata.
        
        Args:
            audio_s3_key: Full S3 key of audio
            
        Returns:
            {'success': bool, 'data': {'duration', 'codec'}, 'error': str}
        """
        temp_file_path = None
        
        try:
            custom_logger.info(f"Extracting metadata from: {audio_s3_key}")
            
            with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as f:
                temp_file_path = f.name
            
            audio_object = get_object(audio_s3_key, self.bucket_name)
            with open(temp_file_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            custom_logger.debug(f"Audio downloaded to: {temp_file_path}")
            
            loop = asyncio.get_event_loop()
            metadata = await loop.run_in_executor(
                None,
                functools.partial(self._extract_sync, temp_file_path)
            )
            
            custom_logger.info(f"Metadata extracted: duration={metadata['data']['duration']}s, codec={metadata['data']['codec']}")
            return metadata
            
        except Exception as e:
            custom_logger.error(f"Metadata extraction failed: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
        
        finally:
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    custom_logger.debug(f"Cleaned up: {temp_file_path}")
                except Exception as e:
                    custom_logger.warning(f"Cleanup failed: {str(e)}")
    
    def _extract_sync(self, file_path: str) -> Dict[str, Any]:
        """Sync extraction"""
        audio_clip = None
        
        try:
            audio_clip = AudioFileClip(file_path)
            duration = audio_clip.duration
            
            file_ext = Path(file_path).suffix.lower()
            codec_map = {
                '.mp3': 'mp3',
                '.m4a': 'aac',
                '.ogg': 'ogg',
                '.wav': 'pcm',
            }
            codec = codec_map.get(file_ext, 'unknown')
            
            return {
                'success': True,
                'data': {
                    'duration': round(duration, 2) if duration else None,
                    'codec': codec
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Sync extraction failed: {str(e)}")
            return {'success': False, 'data': None, 'error': str(e)}
        
        finally:
            if audio_clip:
                try:
                    audio_clip.close()
                except:
                    pass
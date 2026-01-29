import tempfile
from pathlib import Path
from typing import Dict, Any
import asyncio
import functools
from loguru import logger as custom_logger
from moviepy import VideoFileClip

from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws.s3.repository.object import get_object


class AudioExtractor:
    """Extract audio from video files."""
    
    def __init__(self):
        self.bucket_name = s3_bucket
    
    async def extract_from_s3(self, video_s3_key: str) -> Dict[str, Any]:
        """
        Download video from S3, extract audio to temp file.
        
        Args:
            video_s3_key: Full S3 key of video
            
        Returns:
            {'success': bool, 'data': {'audio_path', 'duration', 'codec'}, 'error': str}
        """
        temp_video_path = None
        temp_audio_path = None
        
        try:
            custom_logger.info(f"Extracting audio from: {video_s3_key}")
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                temp_video_path = f.name
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_audio_path = f.name
            
            video_object = await get_object(video_s3_key, self.bucket_name)
            
            def _write_file():
                with open(temp_video_path, 'wb') as f:
                    f.write(video_object.body.read())
                    
            await loop.run_in_executor(None, _write_file)
            
            custom_logger.info(f"Video downloaded to: {temp_video_path}")
            
            result = await self._extract_audio(temp_video_path, temp_audio_path)
            
            if not result['success']:
                self._cleanup_files(temp_video_path, temp_audio_path)
                return result
            
            self._cleanup_files(temp_video_path)
            
            with open(temp_audio_path, 'rb') as f:
                audio_content = f.read()
            
            return {
                'success': True,
                'data': {
                    'audio_content': audio_content,
                    'audio_path': temp_audio_path,
                    'duration': result['data']['duration'],
                    'codec': 'mp3'
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Audio extraction failed: {str(e)}", exc_info=True)
            self._cleanup_files(temp_video_path, temp_audio_path)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def _extract_audio(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """Extract audio using MoviePy in thread"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(self._extract_sync, video_path, audio_path)
            )
            return result
            
        except Exception as e:
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _extract_sync(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """Sync extraction in thread"""
        video_clip = None
        audio_clip = None
        
        try:
            video_clip = VideoFileClip(video_path)
            
            if video_clip.audio is None:
                return {'success': False, 'data': None, 'error': 'Video has no audio track'}
            
            audio_clip = video_clip.audio
            
            try:
                audio_clip.write_audiofile(audio_path, codec='mp3', bitrate='128k')
            except TypeError:
                try:
                    audio_clip.write_audiofile(audio_path, codec='mp3')
                except TypeError:
                    audio_clip.write_audiofile(audio_path)
            
            if not Path(audio_path).exists():
                return {'success': False, 'data': None, 'error': 'Audio file not created'}
            
            audio_size = Path(audio_path).stat().st_size
            duration = audio_clip.duration
            
            custom_logger.info(f"Audio extracted: {audio_size:,} bytes, {duration:.2f}s")
            
            return {
                'success': True,
                'data': {
                    'size': audio_size,
                    'duration': duration
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Extraction sync failed: {str(e)}")
            return {'success': False, 'data': None, 'error': str(e)}
        
        finally:
            if audio_clip:
                audio_clip.close()
            if video_clip:
                video_clip.close()
    
    def _cleanup_files(self, *file_paths):
        """Cleanup temp files"""
        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    custom_logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    custom_logger.warning(f"Cleanup failed {file_path}: {str(e)}")
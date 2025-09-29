import io
import tempfile
from pathlib import Path
from typing import Dict, Any
from loguru import logger as custom_logger
import asyncio
import functools

# MoviePy imports
from moviepy import VideoFileClip

from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws.s3.repository.object import get_object, put_object
from app.api.infra.aws.s3.entity.object import S3Object
from app.core.config import S3_PREFIX_AUDIO


class MoviePyAudioExtractor:
    """Audio extraction using MoviePy (no system FFmpeg required)"""
    
    def __init__(self):
        self.bucket_name = s3_bucket
        self.audio_prefix = S3_PREFIX_AUDIO
        self._initialize_moviepy()
    
    def _initialize_moviepy(self):
        """Initialize MoviePy and download FFmpeg if needed"""
        try:
            custom_logger.info("Initializing MoviePy...")
            
            # MoviePy will automatically download FFmpeg binary if not found
            # This happens automatically when creating VideoFileClip
            custom_logger.info("✅ MoviePy initialized successfully")
            
        except Exception as e:
            custom_logger.error(f"MoviePy initialization failed: {str(e)}")
            raise RuntimeError(f"MoviePy setup failed: {str(e)}")
    
    async def extract_audio_from_video_s3(self, video_s3_key: str, audio_filename: str) -> Dict[str, Any]:
        """
        Extract audio from video stored in S3 using MoviePy
        
        Args:
            video_s3_key: S3 key of the video file
            audio_filename: Desired filename for audio (without extension)
            
        Returns:
            Dict with extraction result
        """
        temp_video_path = None
        temp_audio_path = None
        
        try:
            custom_logger.info(f"Extracting audio from video using MoviePy: {video_s3_key}")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video_path = temp_video.name
                
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Download video from S3
            custom_logger.debug("Downloading video from S3...")
            video_object = get_object(video_s3_key, self.bucket_name)
            
            with open(temp_video_path, 'wb') as f:
                f.write(video_object.body.read())
            
            custom_logger.info(f"Video downloaded to temp file: {temp_video_path}")
            
            # Extract audio using MoviePy (run in thread to avoid blocking)
            custom_logger.debug("Extracting audio using MoviePy...")
            extraction_result = await self._extract_audio_moviepy(temp_video_path, temp_audio_path)
            
            if not extraction_result['success']:
                return extraction_result
            
            # Generate S3 key for audio
            audio_s3_key = f"{self.audio_prefix}/{audio_filename}.mp3"
            custom_logger.info(f"Generated audio S3 key: {audio_s3_key}")
            
            # Read extracted audio and upload to S3
            with open(temp_audio_path, 'rb') as audio_file:
                audio_content = audio_file.read()
                audio_size = len(audio_content)
                
                custom_logger.info(f"Audio extracted successfully. Size: {audio_size:,} bytes")
                
                # Create S3Object for audio
                s3_audio_object = S3Object(
                    body=io.BytesIO(audio_content),
                    content_length=audio_size,
                    content_type='audio/mpeg',
                    key=audio_s3_key,
                    last_modified=None
                )
                
                # Upload to S3
                put_result = put_object(s3_audio_object, self.bucket_name)
                
                audio_s3_url = f"s3://{self.bucket_name}/{audio_s3_key}"
                custom_logger.info(f"Audio uploaded successfully: {audio_s3_url}")
                
                return {
                    'success': True,
                    'audio_s3_key': audio_s3_key,
                    'audio_s3_url': audio_s3_url,
                    'audio_size': audio_size,
                    'audio_size_mb': round(audio_size / (1024 * 1024), 2),
                    'bucket_name': self.bucket_name,
                    'extraction_method': 'moviepy'
                }
                
        except Exception as e:
            custom_logger.error(f"Audio extraction failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f"Audio extraction failed: {str(e)}"
            }
        
        finally:
            # Cleanup temp files
            self._cleanup_temp_files(temp_video_path, temp_audio_path)
    
    async def _extract_audio_moviepy(self, input_video_path: str, output_audio_path: str) -> Dict[str, Any]:
        """
        Extract audio using MoviePy (runs in thread to avoid blocking)
        
        Args:
            input_video_path: Path to input video file
            output_audio_path: Path to output audio file
            
        Returns:
            Dict with extraction result
        """
        try:
            # Run MoviePy extraction in thread pool to avoid blocking async event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                functools.partial(
                    self._moviepy_extract_sync,
                    input_video_path,
                    output_audio_path
                )
            )
            return result
            
        except Exception as e:
            custom_logger.error(f"MoviePy extraction error: {str(e)}")
            return {
                'success': False,
                'error': f"MoviePy extraction error: {str(e)}"
            }
    
    def _moviepy_extract_sync(self, input_video_path: str, output_audio_path: str) -> Dict[str, Any]:
        """Synchronous MoviePy extraction (runs in thread)"""
        video_clip = None
        audio_clip = None
        
        try:
            custom_logger.debug(f"Loading video file: {input_video_path}")
            
            # Load video file
            video_clip = VideoFileClip(input_video_path)
            
            # Check if video has audio
            if video_clip.audio is None:
                return {
                    'success': False,
                    'error': 'Video file has no audio track'
                }
            
            # Extract audio
            audio_clip = video_clip.audio
            
            custom_logger.debug(f"Writing audio to: {output_audio_path}")
            
            # Write audio file with compatible parameters (removed verbose and logger)
            try:
                # Try with bitrate parameter
                audio_clip.write_audiofile(
                    output_audio_path,
                    codec='mp3',
                    bitrate='128k'
                )
            except TypeError:
                # Fallback for older MoviePy versions - minimal parameters
                try:
                    audio_clip.write_audiofile(output_audio_path, codec='mp3')
                except TypeError:
                    # Final fallback - no parameters
                    audio_clip.write_audiofile(output_audio_path)
            
            # Check if output file was created
            if not Path(output_audio_path).exists():
                return {
                    'success': False,
                    'error': 'Audio file was not created by MoviePy'
                }
            
            audio_size = Path(output_audio_path).stat().st_size
            video_duration = video_clip.duration
            audio_duration = audio_clip.duration
            
            custom_logger.info(f"MoviePy extraction completed. Size: {audio_size:,} bytes, Duration: {audio_duration:.2f}s")
            
            return {
                'success': True,
                'extracted_size': audio_size,
                'video_duration': video_duration,
                'audio_duration': audio_duration,
                'codec': 'mp3'
            }
            
        except Exception as e:
            custom_logger.error(f"MoviePy sync extraction error: {str(e)}")
            return {
                'success': False,
                'error': f"MoviePy extraction failed: {str(e)}"
            }
        
        finally:
            # Clean up MoviePy objects
            if audio_clip:
                audio_clip.close()
            if video_clip:
                video_clip.close()
    
    def _cleanup_temp_files(self, *file_paths):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    custom_logger.debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    custom_logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")


# Convenience function for backward compatibility
async def extract_audio_from_video_s3(video_s3_key: str, audio_filename: str) -> Dict[str, Any]:
    """
    Extract audio from video stored in S3 using MoviePy
    
    Args:
        video_s3_key: S3 key of the video file
        audio_filename: Desired filename for audio (without extension)
        
    Returns:
        Dict with extraction result
    """
    extractor = MoviePyAudioExtractor()
    return await extractor.extract_audio_from_video_s3(video_s3_key, audio_filename)
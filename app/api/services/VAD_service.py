import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger as custom_logger
import uuid
from app.core.config import HF_TOKEN

# Try importing pyannote - graceful fallback if not available
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError as e:
    custom_logger.warning(f"pyannote.audio not available: {e}")
    PYANNOTE_AVAILABLE = False

from app.api.infra.aws.s3.repository.object import get_object
from app.api.infra.aws.s3 import s3_bucket

class VoiceActivityDetectionService:
    def __init__(self):
        self.bucket_name = s3_bucket
        self.pipeline = None
        self.model_available = False
        
        if PYANNOTE_AVAILABLE:
            self._load_model()
        else:
            custom_logger.error("Pyannote not available - VAD service disabled")
    
    def _load_model(self):
        """Load pyannote VAD model following HuggingFace documentation"""
        try:
            custom_logger.info("Loading pyannote/voice-activity-detection model...")
            
            # Check for HuggingFace token in environment
            hf_token = HF_TOKEN
            
            if not hf_token:
                custom_logger.error("🔐 HuggingFace token REQUIRED for pyannote/voice-activity-detection model!")
                custom_logger.error("Steps to setup:")
                custom_logger.error("1. Visit https://hf.co/pyannote/voice-activity-detection")
                custom_logger.error("2. Login and accept user conditions")
                custom_logger.error("3. Visit https://hf.co/settings/tokens to create access token")
                custom_logger.error("4. Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable")
                custom_logger.error("5. Restart the service")
                self.pipeline = None
                self.model_available = False
                return
                
            try:
                # Load pipeline following HuggingFace docs pattern
                custom_logger.info(f"Pipeline symbol = {Pipeline} (module: {getattr(Pipeline, '__module__', None)})")
                self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=hf_token)
                custom_logger.info(f"Loaded pipeline type = {type(self.pipeline)}")
                
                # Verify pipeline is loaded and callable
                if self.pipeline is None:
                    raise RuntimeError("Pipeline loaded but is None")
                
                custom_logger.info("Testing pipeline callable...")
                # Test that pipeline is callable (without actual inference)
                if not callable(self.pipeline):
                    raise RuntimeError("Pipeline is not callable")
                
                self.model_available = True
                custom_logger.info("✅ VAD model loaded successfully with authentication")
                
            except Exception as auth_error:
                custom_logger.error(f"Failed to load VAD model: {str(auth_error)}")
                
                # Provide specific error guidance
                error_msg = str(auth_error).lower()
                if "private" in error_msg or "gated" in error_msg or "access" in error_msg:
                    custom_logger.error("🔐 Model access issue:")
                    custom_logger.error("1. Ensure you logged into HuggingFace and accepted conditions at:")
                    custom_logger.error("   https://hf.co/pyannote/voice-activity-detection") 
                    custom_logger.error("2. Verify your token has correct permissions")
                    custom_logger.error("3. Try regenerating your token")
                elif "connection" in error_msg or "network" in error_msg:
                    custom_logger.error("🌐 Network issue - check internet connectivity")
                elif "version" in error_msg or "compatibility" in error_msg:
                    custom_logger.error("📦 Version issue - ensure pyannote.audio 2.1+ is installed:")
                    custom_logger.error("   pip install pyannote.audio>=2.1")
                else:
                    custom_logger.error(f"🔧 Other error: {str(auth_error)}")
                
                self.pipeline = None
                self.model_available = False
                
        except Exception as e:
            custom_logger.error(f"VAD model initialization failed: {str(e)}")
            self.pipeline = None
            self.model_available = False
    
    async def detect_voice_activity(self, audio_s3_key: str) -> Dict[str, Any]:
        """
        Detect voice activity using proper pyannote API pattern from HuggingFace docs
        """
        # Check if model is available
        if not self.model_available or self.pipeline is None:
            custom_logger.error("VAD model not available - cannot perform analysis")
            return {
                'success': False,
                'error': 'VAD model not loaded. Please setup HuggingFace authentication first.'
            }
        
        temp_audio_path = None
        
        try:
            custom_logger.info(f"Starting VAD analysis for: {audio_s3_key}")
            
            # Download audio from S3
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                audio_object = get_object(audio_s3_key, self.bucket_name)
                with open(temp_audio_path, 'wb') as f:
                    f.write(audio_object.body.read())
                custom_logger.info(f"Audio downloaded to: {temp_audio_path}")
            except Exception as download_error:
                custom_logger.error(f"Failed to download audio from S3: {str(download_error)}")
                return {
                    'success': False,
                    'error': f'Failed to download audio: {str(download_error)}'
                }
            
            # Verify file exists and has content
            if not Path(temp_audio_path).exists() or Path(temp_audio_path).stat().st_size == 0:
                custom_logger.error("Downloaded audio file is empty or does not exist")
                return {
                    'success': False,
                    'error': 'Downloaded audio file is empty'
                }
            
            # Run VAD analysis using HuggingFace docs pattern
            custom_logger.info("Running VAD analysis using official API pattern...")
            try:
                # Use the pattern from HuggingFace docs
                vad_output = self.pipeline(temp_audio_path)
                
                # Extract timeline using official API: output.get_timeline().support()
                if hasattr(vad_output, 'get_timeline'):
                    speech_timeline = vad_output.get_timeline().support()
                else:
                    # Fallback to direct timeline access
                    speech_timeline = vad_output
                
                custom_logger.info(f"VAD analysis completed, processing {len(speech_timeline)} speech segments")
                
            except Exception as vad_error:
                custom_logger.error(f"VAD pipeline execution failed: {str(vad_error)}")
                return {
                    'success': False,
                    'error': f'VAD analysis failed: {str(vad_error)}'
                }
            
            voice_segments = []
            total_voice_duration = 0.0
            
            try:
                for idx, speech_segment in enumerate(speech_timeline):
                    start_time = float(speech_segment.start)
                    end_time = float(speech_segment.end)
                    duration = end_time - start_time
                    
                    base_key = audio_s3_key.replace('uploads/', '').rsplit('.', 1)[0]
                    segment_storage_uri = f"s3://{self.bucket_name}/segments/{base_key}_segment_{idx:03d}.wav"
                    
                    confidence_score = min(0.95, 0.5 + (duration / 10.0)) 
                    
                    voice_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'confidence': confidence_score, 
                        'segment_type': 'voice',
                        'storage_uri': segment_storage_uri
                    })
                    total_voice_duration += duration
                
            except Exception as segment_error:
                custom_logger.error(f"Error processing VAD segments: {str(segment_error)}")
                return {
                    'success': False,
                    'error': f'Error processing VAD results: {str(segment_error)}'
                }
            
            try:
                file_size = Path(temp_audio_path).stat().st_size
                estimated_duration = max(total_voice_duration, file_size / (16000 * 2))
                voice_ratio = total_voice_duration / estimated_duration if estimated_duration > 0 else 0.0
            except Exception:
                voice_ratio = 0.0
                estimated_duration = total_voice_duration
            
            custom_logger.info(f"VAD completed: {len(voice_segments)} voice segments, {voice_ratio:.2%} voice activity")
            
            return {
                'success': True,
                'segments': voice_segments,
                'total_segments': len(voice_segments),
                'total_voice_duration': total_voice_duration,
                'voice_activity_ratio': voice_ratio,
                'model_used': 'pyannote/voice-activity-detection',
                'api_version': 'official_huggingface_pattern'
            }
            
        except Exception as e:
            custom_logger.error(f"VAD analysis failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f"VAD analysis failed: {str(e)}"
            }
        
        finally:
            # Cleanup temp file
            if temp_audio_path and Path(temp_audio_path).exists():
                try:
                    Path(temp_audio_path).unlink()
                    custom_logger.debug(f"Cleaned up temp file: {temp_audio_path}")
                except Exception as e:
                    custom_logger.warning(f"Failed to cleanup temp file: {str(e)}")

# Factory function with better error handling
def create_vad_service() -> Optional[VoiceActivityDetectionService]:
    """Create VAD service with comprehensive error handling"""
    try:
        return VoiceActivityDetectionService()
    except Exception as e:
        custom_logger.error(f"Failed to create VAD service: {str(e)}")
        return None
    
async def save_audio_segments(audio_id: uuid.UUID, segments: List[Dict[str, Any]]) -> bool:
    """Save VAD segments to database"""
    from app.api.model.audio_segment_model import AudioSegment
    from app.api.db.session import AsyncSessionLocal
    
    try:
        async with AsyncSessionLocal() as session:
            custom_logger.info(f"Saving {len(segments)} segments for audio_id: {audio_id}")
            
            segment_objects = []
            for segment in segments:
                segment_obj = AudioSegment(
                    audio_id=audio_id,
                    start_time=float(segment['start_time']),
                    end_time=float(segment['end_time']),
                    duration=float(segment['duration']),
                    confidence=float(segment['confidence']) if segment.get('confidence') is not None else None,
                    segment_type=segment.get('segment_type', 'voice'),
                    storage_uri=segment.get('storage_uri')
                )
                segment_objects.append(segment_obj)
            
            session.add_all(segment_objects)
            await session.commit()
            
            custom_logger.info(f"Successfully saved {len(segment_objects)} segments to database")
            return True
            
    except Exception as e:
        custom_logger.error(f"Failed to save segments for audio {audio_id}: {str(e)}", exc_info=True)
        return False
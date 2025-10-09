from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger as custom_logger

from app.core.config import HF_TOKEN, DEVICE
from app.data_model.model_registry import ModelRegistry
from app.data_model.model_config import MODEL_CONFIGS
from app.data_model.implementations.vad_model import VADModel


class VADAnalyzer:
    """Analyze voice activity from cleaned audio."""
    
    def __init__(self):
        self.vad_model: Optional[VADModel] = None
        self._model_loading = False
    
    async def analyze_voice_activity(self, cleaned_audio_path: str) -> Dict[str, Any]:
        """
        Run VAD on cleaned audio file.
        
        Args:
            cleaned_audio_path: Path to cleaned audio file
            
        Returns:
            {'success': bool, 'data': {'vad_segments', 'statistics'}, 'error': str}
        """
        # Ensure model loaded
        if not await self._ensure_model_loaded():
            return {'success': False, 'data': None, 'error': 'VAD model not loaded'}
        
        try:
            custom_logger.info(f"Starting VAD analysis: {cleaned_audio_path}")
            
            # Verify file
            if not Path(cleaned_audio_path).exists():
                return {'success': False, 'data': None, 'error': f'File not found: {cleaned_audio_path}'}
            
            # Run prediction
            vad_output = await self.vad_model.predict(cleaned_audio_path)
            
            if not vad_output:
                return {'success': False, 'data': None, 'error': 'VAD prediction failed'}
            
            # Extract segments
            vad_segments = []
            total_voice_duration = 0.0
            
            for idx, speech_segment in enumerate(vad_output):
                start_time = float(speech_segment.start)
                end_time = float(speech_segment.end)
                duration = end_time - start_time
                
                vad_segments.append({
                    'segment_index': idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })
                total_voice_duration += duration
            
            # Calculate statistics
            file_size = Path(cleaned_audio_path).stat().st_size
            estimated_duration = max(total_voice_duration, file_size / (16000 * 2))
            voice_ratio = total_voice_duration / estimated_duration if estimated_duration > 0 else 0.0
            
            statistics = {
                'total_segments': len(vad_segments),
                'total_voice_duration': float(total_voice_duration),
                'voice_activity_ratio': float(voice_ratio),
                'estimated_total_duration': float(estimated_duration)
            }
            
            custom_logger.info(f"VAD completed: {len(vad_segments)} segments, {voice_ratio:.1%} voice ratio")
            
            return {
                'success': True,
                'data': {
                    'vad_segments': vad_segments,
                    'statistics': statistics
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"VAD analysis failed: {e}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def _ensure_model_loaded(self) -> bool:
        """Lazy load model"""
        if self.vad_model and self.vad_model.is_loaded:
            return True
        
        if self._model_loading:
            custom_logger.warning("Model is already being loaded")
            return False
        
        try:
            self._model_loading = True
            
            # Get from registry
            registry = ModelRegistry.get_instance()
            self.vad_model = registry.get('vad')
            
            if not self.vad_model:
                # Create new model
                config = {
                    'model_name': MODEL_CONFIGS['vad'].model_name,
                    'hf_token': HF_TOKEN,
                    'device': DEVICE,
                    'cache_dir': MODEL_CONFIGS['vad'].cache_dir
                }
                
                custom_logger.info("Creating new VAD model...")
                self.vad_model = VADModel(config=config)
                
                # Load model
                custom_logger.info("Loading VAD model...")
                if not await self.vad_model.load_model():
                    custom_logger.error("Failed to load VAD model")
                    self.vad_model = None
                    return False
                
                # Register
                registry.register('vad', self.vad_model)
                custom_logger.info("VAD model loaded and registered")
            
            return self.vad_model.is_loaded
            
        except Exception as e:
            custom_logger.error(f"Model loading error: {e}", exc_info=True)
            self.vad_model = None
            return False
        finally:
            self._model_loading = False
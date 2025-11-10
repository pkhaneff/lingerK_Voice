from pathlib import Path
from typing import Dict, Any
from loguru import logger as custom_logger
from huggingface_hub import login

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError as e:
    custom_logger.warning(f"pyannote.audio not available: {e}")
    PYANNOTE_AVAILABLE = False

from app.data_model.base_model import BaseModel


class VADModel(BaseModel):
    """VAD Model implementation using pyannote"""
    
    async def _download_model(self, dest_path: Path) -> bool:
        """
        Pyannote auto-downloads from HuggingFace
        No manual download needed
        """
        try:
            custom_logger.info("Pyannote will auto-download from HuggingFace")
            hf_token = self.config.get('hf_token')
            if hf_token:
                login(token=hf_token, add_to_git_credential=False)
                custom_logger.info(" Logged in to HuggingFace")
            else:
                custom_logger.warning("No HF token provided")
            return True
        except Exception as e:
            custom_logger.error(f"HuggingFace login failed: {e}")
            return False
    
    async def load_model(self) -> bool:
        """Load VAD model from HuggingFace"""
        if not PYANNOTE_AVAILABLE:
            custom_logger.error("pyannote.audio not available")
            return False
        
        try:
            custom_logger.info("Loading pyannote VAD model...")
            
            hf_token = self.config.get('hf_token')
            if hf_token:
                try:
                    login(token=hf_token, add_to_git_credential=False)
                    custom_logger.info(" HuggingFace authentication success")
                except Exception as e:
                    custom_logger.warning(f"HF login warning: {e}")
            
            model_name = self.config.get('model_name', 'pyannote/voice-activity-detection')
            cache_dir = self.config.get('cache_dir')
            
            custom_logger.info(f"Loading model: {model_name}")
            custom_logger.info(f"Cache dir: {cache_dir}")
            
            self.model = Pipeline.from_pretrained(
                model_name,
                cache_dir=str(cache_dir) if cache_dir else None
            )
            
            device = self.config.get('device', 'cpu')
            try:
                self.model.to(device)
                custom_logger.info(f" VAD model loaded on device: {device}")
            except Exception as e:
                custom_logger.warning(f"Could not move to {device}: {e}, using CPU")
                self.model.to('cpu')
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            custom_logger.error(f"VAD model load failed: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    async def predict(self, audio_path: str):
        """Run VAD prediction"""
        if not self.is_loaded:
            custom_logger.error("Model not loaded")
            return None
        
        try:
            custom_logger.info(f"Running VAD on: {audio_path}")
            
            # Run VAD
            vad_output = self.model(audio_path)
            
            # Extract timeline
            if hasattr(vad_output, 'get_timeline'):
                speech_timeline = vad_output.get_timeline().support()
            else:
                speech_timeline = vad_output
            
            custom_logger.info(f" VAD completed: {len(speech_timeline)} segments")
            return speech_timeline
            
        except Exception as e:
            custom_logger.error(f"VAD prediction failed: {e}", exc_info=True)
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            try:
                del self.model
                self.model = None
                self.is_loaded = False
                custom_logger.info("VAD model cleaned up")
            except Exception as e:
                custom_logger.warning(f"Cleanup error: {e}")
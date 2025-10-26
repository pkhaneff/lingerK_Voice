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


class OSDModel(BaseModel):
    """Overlapped Speech Detection Model using pyannote"""
    
    async def _download_model(self, dest_path: Path) -> bool:
        """Pyannote auto-downloads from HuggingFace"""
        try:
            custom_logger.info("Pyannote OSD will auto-download from HuggingFace")
            hf_token = self.config.get('hf_token')
            if hf_token:
                login(token=hf_token, add_to_git_credential=False)
                custom_logger.info("✅ Logged in to HuggingFace for OSD")
            else:
                custom_logger.warning("No HF token provided for OSD")
            return True
        except Exception as e:
            custom_logger.error(f"HuggingFace login failed for OSD: {e}")
            return False
    
    async def load_model(self) -> bool:
        """Load OSD model from HuggingFace"""
        if not PYANNOTE_AVAILABLE:
            custom_logger.error("pyannote.audio not available")
            return False
        
        try:
            custom_logger.info("Loading pyannote OSD model...")
            
            hf_token = self.config.get('hf_token')
            if hf_token:
                try:
                    login(token=hf_token, add_to_git_credential=False)
                    custom_logger.info("✅ HuggingFace authentication success for OSD")
                except Exception as e:
                    custom_logger.warning(f"HF login warning: {e}")
            
            model_name = self.config.get('model_name', 'pyannote/overlapped-speech-detection')
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
                custom_logger.info(f"✅ OSD model loaded on device: {device}")
            except Exception as e:
                custom_logger.warning(f"Could not move to {device}: {e}, using CPU")
                self.model.to('cpu')
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            custom_logger.error(f"OSD model load failed: {e}", exc_info=True)
            self.is_loaded = False
            return False
    
    async def predict(self, audio_path: str):
        """Run OSD prediction with posterior probabilities"""
        if not self.is_loaded:
            custom_logger.error("OSD model not loaded")
            return None
        
        try:
            custom_logger.info(f"Running OSD on: {audio_path}")
            from pyannote.audio import Inference
            
            # Run OSD with scores
            osd_output = self.model(audio_path)
            
            # Extract both timeline and scores
            if hasattr(osd_output, 'get_timeline'):
                overlap_timeline = osd_output.get_timeline().support()
            else:
                overlap_timeline = osd_output
            
            try:
                if hasattr(self.model, '_segmentation'):
                    inference = self.model._segmentation
                    scores = inference(audio_path)
                else:
                    custom_logger.warning("Cannot access internal scores")
                    scores = None
            except Exception as e:
                custom_logger.warning(f"Cannot extract scores: {e}")
                scores = None
            custom_logger.info(f"✅ OSD completed: {len(overlap_timeline)} overlap segments")
            
            # Return both timeline and raw output for confidence extraction
            return {
                'timeline': overlap_timeline,
                'scores': scores 
            }
            
        except Exception as e:
            custom_logger.error(f"OSD prediction failed: {e}", exc_info=True)
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            try:
                del self.model
                self.model = None
                self.is_loaded = False
                custom_logger.info("OSD model cleaned up")
            except Exception as e:
                custom_logger.warning(f"Cleanup error: {e}")
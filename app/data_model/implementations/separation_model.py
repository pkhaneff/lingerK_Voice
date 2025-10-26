from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import functools
import numpy as np
import torch
import librosa
from loguru import logger as custom_logger

try:
    from asteroid.models import BaseModel as AsteroidBaseModel
    ASTEROID_AVAILABLE = True
except ImportError:
    ASTEROID_AVAILABLE = False
    custom_logger.warning("asteroid not available")

from app.data_model.base_model import BaseModel
from app.core.config import DEVICE


class ConvTasNetSeparator(BaseModel):
    """Source separation using ConvTasNet model from Asteroid."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = str(DEVICE) if DEVICE else "cpu"
        
        if not ASTEROID_AVAILABLE:
            raise RuntimeError("asteroid is required but not installed")
    
    async def _download_model(self, dest_path: Path) -> bool:
        """Asteroid auto-downloads from HuggingFace."""
        try:
            custom_logger.info("Asteroid will auto-download model from HuggingFace")
            custom_logger.info(f"Model will be saved to: {self.config['cache_dir']}")
            
            self.config['cache_dir'].mkdir(parents=True, exist_ok=True)
            return True
            
        except Exception as e:
            custom_logger.error(f"Model setup failed: {e}")
            return False
    
    async def load_model(self) -> bool:
        """Load ConvTasNet model from HuggingFace."""
        if self.is_loaded:
            return True
        
        if not await self.ensure_model_downloaded():
            custom_logger.error("Model download check failed")
            return False
        
        try:
            custom_logger.info("Loading ConvTasNet model...")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                functools.partial(self._load_sync)
            )
            
            self.is_loaded = True
            custom_logger.info(f"✅ ConvTasNet loaded on {self.device}")
            return True
            
        except Exception as e:
            custom_logger.error(f"Model load failed: {e}", exc_info=True)
            return False
    
    def _load_sync(self):
        """Sync model loading."""
        model_name = self.config.get('model_name', 'JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k')
        cache_dir = self.config.get('cache_dir')
        
        custom_logger.info(f"Loading from: {model_name}")
        custom_logger.info(f"Cache dir: {cache_dir}")
        
        # ✅ Load Asteroid model from HuggingFace
        self.model = AsteroidBaseModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        custom_logger.info("Model loaded successfully")
    
    async def separate_overlap_regions(
        self,
        audio_path: str,
        overlap_regions: List[Dict],
        sr: int = 16000
    ) -> Dict[str, Any]:
        """Separate sources from overlap regions."""
        if not self.is_loaded:
            if not await self.load_model():
                return {'success': False, 'data': None, 'error': 'Model not loaded'}
        
        try:
            custom_logger.info(f"Separating {len(overlap_regions)} overlap regions")
            
            # Load full audio
            audio_data, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
            
            separated_regions = []
            
            for idx, region in enumerate(overlap_regions):
                start_time = region['start_time']
                end_time = region['end_time']
                
                # Extract region
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                region_audio = audio_data[start_sample:end_sample]
                
                # Separate
                loop = asyncio.get_event_loop()
                sources = await loop.run_in_executor(
                    None,
                    functools.partial(self._separate_sync, region_audio, sr)
                )
                
                if sources is None:
                    custom_logger.warning(f"Separation failed for region {idx}")
                    continue
                
                separated_regions.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'source_0': sources[0],
                    'source_1': sources[1],
                    'confidence': region.get('confidence', 0.75)
                })
                
                custom_logger.info(f"Separated region {idx+1}/{len(overlap_regions)}")
            
            return {
                'success': True,
                'data': {
                    'separated_regions': separated_regions,
                    'sample_rate': sr
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Separation failed: {e}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _separate_sync(self, audio: np.ndarray, sr: int) -> Optional[List[np.ndarray]]:
        """Sync separation."""
        try:
            # Ensure (samples,) shape
            if audio.ndim != 1:
                audio = audio.flatten()
            
            # Convert to tensor (1, samples)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            audio_tensor = audio_tensor.to(self.device)
            
            # Separate
            with torch.no_grad():
                # Asteroid model returns (batch, n_sources, samples)
                est_sources = self.model(audio_tensor)
            
            # Convert back to numpy
            est_sources = est_sources.cpu().numpy()
            
            # Extract 2 sources
            if est_sources.shape[1] < 2:
                custom_logger.error(f"Expected 2 sources, got {est_sources.shape[1]}")
                return None
            
            source_0 = est_sources[0, 0, :]  # Speaker A
            source_1 = est_sources[0, 1, :]  # Speaker B
            
            return [source_0, source_1]
            
        except Exception as e:
            custom_logger.error(f"Sync separation failed: {e}", exc_info=True)
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        if self.model:
            try:
                del self.model
                self.model = None
                self.is_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                custom_logger.info("Separator cleaned up")
            except Exception as e:
                custom_logger.warning(f"Cleanup error: {e}")
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
from .model_downloader import ModelDownloader
from loguru import logger as custom_logger

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_loaded = False
        self.downloader = ModelDownloader()
    
    async def ensure_model_downloaded(self) -> bool:
        """Đảm bảo model đã được download"""
        model_path = self.config['cache_dir'] / self.config['model_file']
        
        if not self.downloader.model_exists(model_path):
            custom_logger.info(f"Model not found, downloading to {model_path}")
            return await self._download_model(model_path)
        
        custom_logger.info(f"Model exists: {model_path}")
        return True
    
    @abstractmethod
    async def _download_model(self, dest_path: Path) -> bool:
        """Implement download logic for specific model"""
        pass
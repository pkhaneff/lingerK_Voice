from pathlib import Path
from typing import Optional
import hashlib
import json
from loguru import logger

from app.data_model.model_config import MODEL_STORAGE_BASE

class ModelDownloader:
    CHECKSUMS_FILE = MODEL_STORAGE_BASE / "checksums.json"
    
    @classmethod
    def download_model(cls, url: str, dest_path: Path, 
                       expected_hash: Optional[str] = None) -> bool:
        """Download model từ URL về local storage"""
        try:
            # Download logic
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Verify checksum
            if expected_hash and not cls._verify_checksum(dest_path, expected_hash):
                logger.error(f"Checksum mismatch for {dest_path}")
                return False
            
            # Save checksum
            cls._save_checksum(dest_path, expected_hash)
            
            logger.info(f"Model downloaded: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    @staticmethod
    def _verify_checksum(file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest() == expected_hash
    
    @classmethod
    def _save_checksum(cls, file_path: Path, checksum: str):
        """Save checksum to registry"""
        checksums = {}
        if cls.CHECKSUMS_FILE.exists():
            with open(cls.CHECKSUMS_FILE) as f:
                checksums = json.load(f)
        
        checksums[str(file_path)] = checksum
        
        with open(cls.CHECKSUMS_FILE, 'w') as f:
            json.dump(checksums, f, indent=2)
    
    @classmethod
    def model_exists(cls, model_path: Path) -> bool:
        """Check if model exists locally"""
        return model_path.exists() and model_path.stat().st_size > 0
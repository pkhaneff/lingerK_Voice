from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from app.core.config import DEVICE

MODEL_STORAGE_BASE = Path("app/data_model/storage")
MODEL_STORAGE_BASE.mkdir(parents=True, exist_ok=True)

@dataclass
class VADConfig:
    model_name: str = "pyannote/voice-activity-detection"
    hf_token: Optional[str] = None
    cache_dir: Path = MODEL_STORAGE_BASE / "vad"
    model_file: str = "pytorch_model.bin"
    device: str = DEVICE

@dataclass  
class NoiseReductionConfig:
    sample_rate: int = 48000
    cache_dir: Path = MODEL_STORAGE_BASE / "noise_reduction"
    model_file: str = "rnnoise_model.pt"

MODEL_CONFIGS = {
    "vad": VADConfig(),
    "noise_reduction": NoiseReductionConfig()
}

# Tạo folders khi import
for config_name, config in MODEL_CONFIGS.items():
    if hasattr(config, 'cache_dir'):
        config.cache_dir.mkdir(parents=True, exist_ok=True)
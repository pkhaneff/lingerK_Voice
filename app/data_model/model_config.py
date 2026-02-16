from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from app.core.config import DEVICE

# Use absolute path to ensure cache is found regardless of working directory
MODEL_STORAGE_BASE = Path(__file__).parent / "storage"
MODEL_STORAGE_BASE.mkdir(parents=True, exist_ok=True)

@dataclass
class VADConfig:
    model_name: str = "pyannote/voice-activity-detection"
    hf_token: Optional[str] = None
    cache_dir: Path = MODEL_STORAGE_BASE / "vad"
    model_file: str = "pytorch_model.bin"
    device: str = DEVICE

@dataclass
class OSDConfig:
    model_name: str = "pyannote/overlapped-speech-detection"
    hf_token: Optional[str] = None
    cache_dir: Path = MODEL_STORAGE_BASE / "osd"
    model_file: str = "pytorch_model.bin"
    device: str = DEVICE

@dataclass
class SeparationConfig:
    model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    cache_dir: Path = MODEL_STORAGE_BASE / "separation"
    model_file: str = "pytorch_model.bin"
    device: str = DEVICE

MODEL_CONFIGS = {
    "vad": VADConfig(),
    "osd": OSDConfig(),
    "separation": SeparationConfig()
}

for config_name, config in MODEL_CONFIGS.items():
    if hasattr(config, 'cache_dir'):
        config.cache_dir.mkdir(parents=True, exist_ok=True)
from __future__ import annotations
import logging, sys, os
import torch
from loguru import logger as custom_logger
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

config = Config(".env")

API_PREFIX: str = "/api"
VERSION: str = "0.0.0"
PROJECT_NAME: str = config("PROJECT_NAME", default="BAP Voice Identification")
DEBUG: bool = config("DEBUG", cast=bool, default=False)
ALLOWED_HOSTS: list[str] = config(
    "ALLOWED_HOSTS",
    cast=CommaSeparatedStrings,
    default="",
)
def _probe_cuda() -> torch.device:
    wanted = config("DEVICE", default="cuda" if torch.cuda.is_available() else "cpu")
    if wanted != "cuda":
        return torch.device("cpu")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() = False")
        _ = torch.cuda.device_count()           
        _ = torch.cuda.current_device()          
        _ = torch.cuda.get_device_name(0)        
        custom_logger.info("CUDA probe OK")
        return torch.device("cuda")
    except Exception as e:
        custom_logger.warning(f"CUDA probe failed -> fallback CPU: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")

DEVICE = _probe_cuda()

AWS_ACCESS_KEY: str = config("AWS_ACCESS_KEY")
AWS_SECRET_KEY: str = config("AWS_SECRET_KEY")
AWS_REGION: str = config("AWS_REGION", default="ap-northeast-1")
HF_TOKEN: str = config("HF_TOKEN", default="")

AUDIO_FILE_SEIZE: int = config("AUDIO_FILE_SEIZE", cast=int, default=52428800)  # 10MB
VIDEO_FILE_SEIZE: int = config("VIDEO_FILE_SEIZE", cast=int, default=1073741824) 
MAX_REQUEST_SIZE: int = config("MAX_REQUEST_SIZE", cast=int, default=1073741824) 
VIDEO_MULTIPART_THRESHOLD: int = config("VIDEO_MULTIPART_THRESHOLD", cast=int, default=104857600)
VIDEO_CHUNK_SIZE_SMALL: int = config("VIDEO_CHUNK_SIZE_SMALL", cast=int, default=10485760) 
VIDEO_CHUNK_SIZE_LARGE: int = config("VIDEO_CHUNK_SIZE_LARGE", cast=int, default=52428800)  

S3_PREFIX_AUDIO: str = config("S3_PREFIX_AUDIO", default="uploads/audio")
S3_PREFIX_VIDEO: str = config("S3_PREFIX_VIDEO", default="uploads/video")

AUDIO_EXTS: list[str] = config("AUDIO_EXTS", cast=CommaSeparatedStrings, default="m4a,ogg")
VIDEO_EXTS: list[str] = config("VIDEO_EXTS", cast=CommaSeparatedStrings, default="mp4")

# Timeouts (seconds)
UPLOAD_TIMEOUT: int = config("UPLOAD_TIMEOUT", cast=int, default=300)      # 5 minutes
REQUEST_TIMEOUT: int = config("REQUEST_TIMEOUT", cast=int, default=300)

# Database
DB_HOST: str = config("DB_HOST", default="localhost")
DB_PORT: int = config("DB_PORT", cast=int, default=5432)
DB_USER_NAME: str = config("DB_USER_NAME", default="postgres")
DB_PASSWORD: str = config("DB_PASSWORD", default="")
DB_DATABASE: str = config("DB_DATABASE", default="BAP_Voice_Identification")
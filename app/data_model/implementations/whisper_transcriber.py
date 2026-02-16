from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import functools
import re
import torch
import librosa
import numpy as np
from loguru import logger as custom_logger
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperTranscriber:
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.pipe = None  # ASR pipeline for automatic chunking
        self.device = None
        self.is_loaded = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if transformers is available."""
        try:
            self.available = True
            custom_logger.info("Transformers available")
        except ImportError:
            self.available = False
            custom_logger.warning("Transformers not installed. Install: pip install transformers")
    
    @property
    def is_available(self) -> bool:
        return self.available
    
    async def load_model(self) -> bool:
        """Load Whisper model with optimizations."""
        if not self.is_available:
            custom_logger.error("Transformers not available")
            return False
        
        if self.is_loaded:
            custom_logger.info("Model already loaded")
            return True
        
        try:
            custom_logger.info("Loading PhoWhisper-medium (VinAI's Vietnamese ASR model)...")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_sync)
            
            if self.model is None or self.processor is None:
                custom_logger.error("Model or processor is None after loading")
                return False
            
            self._cleanup_unused_files()
            
            self.is_loaded = True
            custom_logger.info("PhoWhisper loaded successfully")
            return True
            
        except Exception as e:
            custom_logger.error(f"Model load failed: {e}", exc_info=True)
            return False
    
    def _load_sync(self):
        """Sync load - optimized version with GPU support and pipeline for chunking."""

        model_id = "vinai/PhoWhisper-medium"
        # Use absolute path to ensure cache is found regardless of working directory
        cache_dir = Path(__file__).parent.parent / "storage" / "phowhisper"
        cache_dir.mkdir(parents=True, exist_ok=True)

        custom_logger.info(f"Loading from: {model_id}")
        custom_logger.info(f"Cache dir: {cache_dir}")

        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16
            custom_logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            custom_logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            custom_logger.info("Using CPU")

        self.device = device

        # Create pipeline with automatic chunking support
        custom_logger.info("Creating ASR pipeline with chunking support...")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs={"cache_dir": str(cache_dir)},
            chunk_length_s=30,  # Process in 30-second chunks
            stride_length_s=5,   # 5-second overlap between chunks for smooth transitions
        )

        # Configure pipeline for Vietnamese
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
            language="vi",
            task="transcribe"
        )

        custom_logger.info(f"Pipeline configured on {device} with 30s chunking")

        # Also load processor for compatibility (if needed elsewhere)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(cache_dir)
        )
        self.model = self.pipe.model  # Keep reference for cleanup

        custom_logger.info(f"Configuration completed on {device}")
    
    def _cleanup_unused_files(self):
        """Remove unnecessary files to save disk space."""
        cache_path = Path(__file__).parent.parent / "storage" / "phowhisper"
        
        if not cache_path.exists():
            return
        
        unused_patterns = [
            "**/pytorch_model.bin",
            "**/flax_model.msgpack",
            "**/model.fp32*.safetensors",
            "**/pytorch_model.fp32*.bin",
            "**/normalizer.json",  
        ]
        
        deleted_size = 0
        deleted_count = 0
        
        for pattern in unused_patterns:
            for file in cache_path.glob(pattern):
                if file.is_file():
                    try:
                        size = file.stat().st_size
                        file.unlink()
                        deleted_size += size
                        deleted_count += 1
                        custom_logger.debug(f"Deleted: {file.name} ({size / 1e6:.1f} MB)")
                    except Exception as e:
                        custom_logger.warning(f"Failed to delete {file.name}: {e}")
        
        if deleted_count > 0:
            custom_logger.info(
                f"Cleaned up {deleted_count} files, "
                f"freed {deleted_size / 1e9:.2f} GB"
            )
    
    async def transcribe_track(self, audio_path: str, language: str = 'vi') -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_path: Path to cleaned audio file (after noise reduction)
            language: Language code (default: 'vi' for Vietnamese)
            
        Returns:
            {
                'success': bool,
                'data': {
                    'text': str,          # Full transcript
                    'segments': [],       # Empty (not used)
                    'words': [            # Word-level timestamps
                        {
                            'word': str,
                            'start': float,
                            'end': float,
                            'confidence': float
                        }
                    ]
                },
                'error': str
            }
        """
        if not self.is_available:
            return {
                'success': False,
                'data': None,
                'error': 'Transformers not available'
            }
        
        if not self.is_loaded:
            custom_logger.info("Model not loaded, loading now...")
            if not await self.load_model():
                return {
                    'success': False,
                    'data': None,
                    'error': 'Failed to load model'
                }
        
        try:
            if not Path(audio_path).exists():
                return {
                    'success': False,
                    'data': None,
                    'error': f'Audio file not found: {audio_path}'
                }
            
            custom_logger.info(f"Transcribing: {audio_path}")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(self._transcribe_sync, audio_path, language)
            )
            
            return result
            
        except Exception as e:
            custom_logger.error(f"Transcription failed: {e}", exc_info=True)
            return {
                'success': False,
                'data': None,
                'error': f'Transcription error: {str(e)}'
            }
    
    def _transcribe_sync(self, audio_path: str, language: str) -> Dict[str, Any]:
        """Sync transcribe using pipeline with automatic chunking for long audio."""
        try:
            if self.pipe is None:
                return {'success': False, 'data': None, 'error': 'Pipeline is None'}

            # Load and analyze audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            # Voice analysis (simplified)
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

            mean_rms = np.mean(rms)
            threshold = max(0.001, mean_rms - 0.5 * np.std(rms))
            voice_frames = np.sum(rms > threshold)
            voice_percentage = (voice_frames * hop_length / len(audio)) * 100

            custom_logger.info(
                f"Audio: {duration:.1f}s, voice: {voice_percentage:.1f}%, "
                f"rms: {np.sqrt(np.mean(audio**2)):.4f}"
            )

            # Calculate expected chunks
            num_chunks = int(np.ceil(duration / 30))
            custom_logger.info(f"Pipeline will process in ~{num_chunks} chunks (30s each with 5s overlap)")

            # Use pipeline with chunking - it automatically handles long audio
            try:
                result = self.pipe(
                    audio,
                    return_timestamps="word",  # Get word-level timestamps
                    generate_kwargs={
                        "language": language,
                        "task": "transcribe",
                    }
                )

                if not result:
                    return {'success': False, 'data': None, 'error': 'Pipeline returned empty result'}

                # Extract text and chunks
                full_text = result.get("text", "")
                chunks = result.get("chunks", [])

                custom_logger.info(f"✅ Pipeline completed: {len(chunks)} chunks processed")

                if not full_text:
                    return {'success': False, 'data': None, 'error': 'Empty transcription'}

                # Convert chunks to word-level timestamps
                words = []
                for chunk in chunks:
                    word_text = chunk.get("text", "").strip()
                    if not word_text:
                        continue

                    timestamp = chunk.get("timestamp", (0.0, 0.0))
                    start_time = timestamp[0] if timestamp[0] is not None else 0.0
                    end_time = timestamp[1] if timestamp[1] is not None else start_time + 1.0

                    # Clean word
                    clean_word = self._clean_word(word_text)
                    if clean_word:
                        words.append({
                            'word': clean_word,
                            'start': float(start_time),
                            'end': float(end_time),
                            'confidence': 1.0
                        })

                custom_logger.info(f"✅ Transcribed: {len(full_text)} chars, {len(words)} words")
                if full_text:
                    preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
                    custom_logger.info(f"Preview: {preview}")

                return {
                    'success': True,
                    'data': {
                        'text': full_text.strip(),
                        'segments': [],
                        'words': words
                    },
                    'error': None
                }

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return {'success': False, 'data': None, 'error': f'GPU OOM: audio too long ({duration:.1f}s)'}
            except Exception as pipe_error:
                custom_logger.error(f"Pipeline error: {pipe_error}", exc_info=True)
                return {'success': False, 'data': None, 'error': f'Pipeline failed: {str(pipe_error)}'}

        except Exception as e:
            custom_logger.error(f"Transcription failed: {e}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _clean_word(self, word: str) -> str:
        """
        Clean word by removing special tokens and punctuation.
        
        Keep Vietnamese characters and basic punctuation.
        """
        word = re.sub(r'<\|[^|]+\|>', '', word)
        
        word = word.strip()
        
        if re.match(r'^[^\w\s]+$', word):
            return ''
        
        return word

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.model:
                del self.model
                self.model = None
                custom_logger.debug("Model deleted")
            
            if self.processor:
                del self.processor
                self.processor = None
                custom_logger.debug("Processor deleted")
                
            if self.pipe:
                del self.pipe
                self.pipe = None
                custom_logger.debug("Pipeline deleted")
            
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                custom_logger.debug("CUDA cache cleared")
            
            custom_logger.info("Whisper transcriber cleaned up")
            
        except Exception as e:
            custom_logger.warning(f"Cleanup error: {e}")
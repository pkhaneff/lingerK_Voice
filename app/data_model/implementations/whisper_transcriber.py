from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import functools
import re
import torch
from loguru import logger as custom_logger


class WhisperTranscriber:
    """Transcribe audio using Whisper large-v3 - Optimized for Vietnamese."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None  # Will be set during load
        self.is_loaded = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if transformers is available."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
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
            custom_logger.info("Loading Whisper large-v3 (optimized for Vietnamese)...")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_sync)
            
            if self.model is None or self.processor is None:
                custom_logger.error("Model or processor is None after loading")
                return False
            
            # Cleanup unused files after first load
            self._cleanup_unused_files()
            
            self.is_loaded = True
            custom_logger.info("âœ… Whisper loaded successfully")
            return True
            
        except Exception as e:
            custom_logger.error(f"Model load failed: {e}", exc_info=True)
            return False
    
    def _load_sync(self):
        """Sync load - optimized version with GPU support."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        model_id = "openai/whisper-medium"
        cache_dir = "app/data_model/storage/whisper"
        
        custom_logger.info(f"Loading from: {model_id}")
        custom_logger.info(f"Cache dir: {cache_dir}")
        
        # âœ… Auto-detect device (GPU/CPU)
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16  # FP16 for GPU
            custom_logger.info(f"ðŸš€ Using GPU: {torch.cuda.get_device_name(0)}")
            custom_logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = "cpu"
            torch_dtype = torch.float32  # FP32 for CPU
            custom_logger.info("ðŸ’» Using CPU")
        
        # âœ… Load model with device-specific settings
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,        # FP16 for GPU, FP32 for CPU
            # low_cpu_mem_usage=True,         # Reduce RAM during load
            use_safetensors=True,           # Use .safetensors format (faster)
            cache_dir=cache_dir,
            attn_implementation="sdpa"      # Use scaled dot-product attention
        )
        
        # Move model to device
        self.model.to(device)
        self.device = device
        
        custom_logger.info(f"Model loaded on {device}, loading processor...")
        
        # âœ… Load processor (tokenizer + feature extractor)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        
        custom_logger.info("Processor loaded, configuring for Vietnamese...")
        
        # âœ… Configure for Vietnamese transcription
        self.model.generation_config.language = "vi"
        self.model.generation_config.task = "transcribe"
        
        # âœ… Disable internal VAD logic (we have our own VAD)
        self.model.generation_config.condition_on_prev_tokens = False
        self.model.generation_config.no_speech_threshold = 1.0  # Assume all speech
        
        custom_logger.info(f"âœ… Configuration completed on {device}")
    
    def _cleanup_unused_files(self):
        """Remove unnecessary files to save disk space."""
        cache_path = Path("app/data_model/storage/whisper")
        
        if not cache_path.exists():
            return
        
        # Files khÃ´ng cáº§n thiáº¿t
        unused_patterns = [
            "**/pytorch_model.bin",      # Duplicate cá»§a safetensors
            "**/flax_model.msgpack",     # Flax version (khÃ´ng dÃ¹ng)
            "**/model.fp32*.safetensors", # FP32 version (náº¿u Ä‘Ã£ cÃ³ FP16)
            "**/pytorch_model.fp32*.bin",
            "**/normalizer.json",        # English normalizer (khÃ´ng cáº§n cho VI)
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
        """
        Sync transcribe - optimized version with GPU support.
        
        This runs in executor thread to avoid blocking.
        """
        try:
            if self.model is None or self.processor is None:
                return {
                    'success': False,
                    'data': None,
                    'error': 'Model or processor is None'
                }
            
            # Load audio (librosa automatically resamples to 16kHz)
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # ===== LOG AUDIO INPUT STATS =====
            duration = len(audio) / sr
            non_zero = np.count_nonzero(np.abs(audio) > 0.001)
            voice_samples = non_zero
            silence_samples = len(audio) - non_zero
            
            custom_logger.info(f"📥 Whisper Input Audio Stats:")
            custom_logger.info(f"   Duration: {duration:.2f}s ({len(audio)} samples)")
            custom_logger.info(f"   Voice samples: {voice_samples} ({voice_samples/len(audio)*100:.1f}%)")
            custom_logger.info(f"   Silence samples: {silence_samples} ({silence_samples/len(audio)*100:.1f}%)")
            # =================================
            
            # âœ… Convert audio to mel-spectrogram (using preprocessor_config.json)
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # âœ… Move inputs to device (GPU/CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            custom_logger.debug(f"Audio preprocessed to mel-spectrogram on {self.device}")
            
            # âœ… Generate transcription with word timestamps
            gen_kwargs = {
                "language": language,
                "task": "transcribe",
                
                # ===== DISABLE VAD LOGIC (we use our own VAD) =====
                "condition_on_prev_tokens": False,
                "no_speech_threshold": 1.0,       # Assume all is speech
                "logprob_threshold": -1.0,        # Don't filter by log prob
                "compression_ratio_threshold": 10.0,  # Don't filter by compression
                
                # ===== ENABLE WORD TIMESTAMPS =====
                "return_timestamps": True,
                
                # ===== GENERATION SETTINGS =====
                "num_beams": 1,                   # Greedy search (faster)
                "max_new_tokens": 448,
            }
            
            custom_logger.info(f"Generating transcription on {self.device}...")
            
            # Generate token IDs
            pred_ids = self.model.generate(**inputs, **gen_kwargs)
            
            custom_logger.debug(f"Generated {pred_ids.shape[1]} tokens")
            
            # âœ… Decode with timestamps
            # skip_special_tokens=False to keep timestamp tokens
            decoded = self.processor.batch_decode(
                pred_ids,
                skip_special_tokens=False
            )
            
            raw_text = decoded[0]
            custom_logger.debug(f"Raw decoded text length: {len(raw_text)}")
            
            # âœ… Parse word timestamps from decoded text
            words = self._parse_word_timestamps(raw_text)
            
            # âœ… Extract clean text (remove timestamp tokens)
            clean_text = self._remove_timestamp_tokens(raw_text)
            
            # ===== LOG TRANSCRIPT OUTPUT =====
            custom_logger.info(f"\n📝 Whisper Output:")
            custom_logger.info(f"   Text length: {len(clean_text)} chars")
            custom_logger.info(f"   Words count: {len(words)}")
            custom_logger.info(f"   Preview: {clean_text[:200]}")
            if len(clean_text) > 200:
                custom_logger.info(f"   ... (truncated)")
            # ================================
            
            custom_logger.info(
                f"âœ… Transcription completed: {len(clean_text)} chars, "
                f"{len(words)} words"
            )
            
            return {
                'success': True,
                'data': {
                    'text': clean_text,
                    'segments': [],  # Not used (we have word-level instead)
                    'words': words
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Sync transcribe failed: {e}", exc_info=True)
            return {
                'success': False,
                'data': None,
                'error': str(e)
            }
    
    def _parse_word_timestamps(self, raw_text: str) -> List[Dict]:
        """
        Parse word timestamps from Whisper decoded output.
        
        Whisper output format: <|0.00|>word1 word2<|2.50|>word3<|5.00|>
        
        Returns:
            [
                {'word': 'word1', 'start': 0.00, 'end': 2.50, 'confidence': 1.0},
                {'word': 'word2', 'start': 0.00, 'end': 2.50, 'confidence': 1.0},
                {'word': 'word3', 'start': 2.50, 'end': 5.00, 'confidence': 1.0},
            ]
        """
        try:
            words = []
            
            # Pattern to match timestamp tokens: <|12.34|>
            timestamp_pattern = r'<\|(\d+\.\d+)\|>'
            
            # Split by timestamps
            parts = re.split(timestamp_pattern, raw_text)
            
            # parts = ['', '0.00', 'word1 word2', '2.50', 'word3', '5.00', '']
            # Odd indices are timestamps, even indices are text
            
            timestamps = []
            texts = []
            
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Timestamp
                    try:
                        timestamps.append(float(part))
                    except ValueError:
                        custom_logger.warning(f"Invalid timestamp: {part}")
                elif i % 2 == 0 and part.strip():  # Text
                    texts.append(part.strip())
            
            # Match texts with timestamps
            for i, text in enumerate(texts):
                start_time = timestamps[i] if i < len(timestamps) else 0.0
                end_time = timestamps[i + 1] if i + 1 < len(timestamps) else start_time + 1.0
                
                # Split text into individual words
                word_tokens = text.split()
                
                if not word_tokens:
                    continue
                
                # Distribute time evenly across words in this segment
                segment_duration = end_time - start_time
                time_per_word = segment_duration / len(word_tokens) if len(word_tokens) > 0 else 0
                
                for j, word in enumerate(word_tokens):
                    # Clean word (remove special tokens)
                    clean_word = self._clean_word(word)
                    
                    if not clean_word:
                        continue
                    
                    word_start = start_time + (j * time_per_word)
                    word_end = start_time + ((j + 1) * time_per_word)
                    
                    words.append({
                        'word': clean_word,
                        'start': float(word_start),
                        'end': float(word_end),
                        'confidence': 1.0  # Whisper doesn't provide word-level confidence
                    })
            
            custom_logger.debug(f"Parsed {len(words)} words with timestamps")
            return words
            
        except Exception as e:
            custom_logger.error(f"Word timestamp parsing failed: {e}", exc_info=True)
            return []
    
    def _clean_word(self, word: str) -> str:
        """
        Clean word by removing special tokens and punctuation.
        
        Keep Vietnamese characters and basic punctuation.
        """
        # Remove Whisper special tokens
        word = re.sub(r'<\|[^|]+\|>', '', word)
        
        # Remove leading/trailing whitespace
        word = word.strip()
        
        # Remove standalone punctuation-only tokens
        if re.match(r'^[^\w\s]+$', word):
            return ''
        
        return word
    
    def _remove_timestamp_tokens(self, raw_text: str) -> str:
        """
        Remove timestamp tokens from text.
        
        Example: '<|0.00|>Hello world<|2.50|>' â†’ 'Hello world'
        """
        # Remove all timestamp tokens
        clean_text = re.sub(r'<\|[\d.]+\|>', '', raw_text)
        
        # Remove other special tokens
        clean_text = re.sub(r'<\|[^|]+\|>', '', clean_text)
        
        # Clean up extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text.strip()
    
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
            
            self.is_loaded = False
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                custom_logger.debug("CUDA cache cleared")
            
            custom_logger.info("Whisper transcriber cleaned up")
            
        except Exception as e:
            custom_logger.warning(f"Cleanup error: {e}")
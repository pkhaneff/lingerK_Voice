from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import functools
import re
import torch
import librosa
import numpy as np
from loguru import logger as custom_logger
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class WhisperTranscriber:
    
    def __init__(self):
        self.model = None
        self.processor = None
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
        """Sync load - optimized version with GPU support."""
        
        model_id = "vinai/PhoWhisper-medium"
        cache_dir = "app/data_model/storage/phowhisper"
        
        custom_logger.info(f"Loading from: {model_id}")
        custom_logger.info(f"Cache dir: {cache_dir}")
        
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16 
            custom_logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            custom_logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = "cpu"
            torch_dtype = torch.float32  
            custom_logger.info(" Using CPU")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,        
            low_cpu_mem_usage=True,      
            cache_dir=cache_dir,
        )
        
        self.model.to(device)
        self.device = device
        
        custom_logger.info(f"Model loaded on {device}, loading processor...")
        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        
        custom_logger.info("Processor loaded, configuring for Vietnamese...")
        
        self.model.generation_config.language = "vi"
        self.model.generation_config.task = "transcribe"
        
        self.model.generation_config.condition_on_prev_tokens = False
        self.model.generation_config.no_speech_threshold = 1.0 
        
        self.model.generation_config.logprob_threshold = None
        self.model.generation_config.compression_ratio_threshold = None
        
        custom_logger.info(f"Configuration completed on {device}")
    
    def _cleanup_unused_files(self):
        """Remove unnecessary files to save disk space."""
        cache_path = Path("app/data_model/storage/phowhisper")
        
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
            
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            custom_logger.info("Enhanced Transcription v2.0 - Debug fixes enabled!")
            
            duration = len(audio) / sr
            frame_length = int(0.025 * sr)  
            hop_length = int(0.010 * sr) 
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            mean_rms = np.mean(rms)
            std_rms = np.std(rms)
            dynamic_threshold = max(0.001, mean_rms - 0.5 * std_rms)  
            
            voice_frames = np.sum(rms > dynamic_threshold)
            silence_frames = len(rms) - voice_frames
            
            voice_samples_estimate = int(voice_frames * hop_length)
            silence_samples_estimate = len(audio) - voice_samples_estimate
            
            voice_percentage = voice_samples_estimate / len(audio) * 100
            silence_percentage = silence_samples_estimate / len(audio) * 100
            
            max_amplitude = np.max(np.abs(audio))
            audio_rms = np.sqrt(np.mean(audio**2))
            zero_crossings = np.sum(np.diff(np.signbit(audio)))
            
            custom_logger.info(f"Whisper Input Audio Stats:")
            custom_logger.info(f"Duration: {duration:.2f}s ({len(audio)} samples)")
            custom_logger.info(f"Voice estimate: {voice_samples_estimate} samples ({voice_percentage:.1f}%)")
            custom_logger.info(f"Silence estimate: {silence_samples_estimate} samples ({silence_percentage:.1f}%)")
            custom_logger.info(f"Quality: max_amp={max_amplitude:.4f}, rms={audio_rms:.4f}, zcr={zero_crossings}")
            custom_logger.info(f"RMS threshold: {dynamic_threshold:.4f} (adaptive)")
            
            if max_amplitude < 0.01:
                custom_logger.warning("Very low audio amplitude - may affect transcription quality")
            if voice_percentage < 50:
                custom_logger.warning("Low voice content detected - check audio preprocessing")
            if audio_rms < 0.005:
                custom_logger.warning("Very quiet audio - consider audio normalization")

            
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device).half() for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            custom_logger.debug(f"Audio preprocessed to mel-spectrogram on {self.device}")
            
            duration_minutes = duration / 60.0
            dynamic_max_tokens = max(1500, int(duration_minutes * 400 * 1.3))  
            dynamic_max_tokens = min(dynamic_max_tokens, 4096)  # Cap at 4096
            
            custom_logger.info(f"Dynamic max_tokens: {dynamic_max_tokens} (duration: {duration:.1f}s, {duration_minutes:.1f}min)")
            
            if duration > 300:
                custom_logger.warning(f"Very long audio ({duration:.1f}s). Risk of GPU OOM. Consider segmentation.")
            elif duration > 180:
                custom_logger.info(f"Long audio ({duration:.1f}s). Monitoring GPU memory usage.")
            
            gen_kwargs = {
                "language": language,
                "task": "transcribe",
                "return_timestamps": True,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": dynamic_max_tokens,
                
                "condition_on_prev_tokens": False,
                "no_speech_threshold": None,           
                "logprob_threshold": None,               
                "compression_ratio_threshold": None,   
                
                "use_cache": True,
                "forced_decoder_ids": None,
                "suppress_tokens": None,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                
                "min_length": max(200, int(duration_minutes * 150)),  
                "length_penalty": 1.0,                            
                "repetition_penalty": 1.0,                        
                "no_repeat_ngram_size": 0,                         
            }
            
            custom_logger.info(f"Generating transcription on {self.device}...")
            custom_logger.info(f"Expected tokens needed: ~{int(duration_minutes * 300)} (based on duration)")
            
            original_config = {}
            if hasattr(self.model, 'generation_config'):
                config = self.model.generation_config
                original_config = {
                    'no_speech_threshold': getattr(config, 'no_speech_threshold', None),
                    'logprob_threshold': getattr(config, 'logprob_threshold', None),
                    'compression_ratio_threshold': getattr(config, 'compression_ratio_threshold', None),
                }
                
                config.no_speech_threshold = None
                config.logprob_threshold = None  
                config.compression_ratio_threshold = None
                custom_logger.info("Model generation config overridden")
            
            custom_logger.info(f"Early stopping disabled: no_speech_threshold={gen_kwargs['no_speech_threshold']}")
            custom_logger.info(f"Early stopping disabled: logprob_threshold={gen_kwargs['logprob_threshold']}")
            custom_logger.info(f"Early stopping disabled: compression_ratio_threshold={gen_kwargs['compression_ratio_threshold']}")
            custom_logger.info(f"Min generation length: {gen_kwargs['min_length']} tokens")
            
            custom_logger.info("Starting token generation with enhanced debugging...")
            try:
                pred_ids = self.model.generate(**inputs, **gen_kwargs)
                
                try:
                    from .whisper_generation_debug import debug_whisper_generation
                    debug_result = debug_whisper_generation(
                        self.processor, self.model, pred_ids, duration, dynamic_max_tokens
                    )
                    
                    if debug_result:
                        tokens_info = debug_result['tokens']
                        eos_info = debug_result['eos'] 
                        quality_info = debug_result['quality']
                        
                        custom_logger.info(f"Generation Analysis:")
                        custom_logger.info(f"Content tokens: {tokens_info['content']}")
                        custom_logger.info(f"Quality status: {quality_info['token_count_status']}")
                        custom_logger.info(f"Quality score: {quality_info['quality_score']}/100")
                        
                        if eos_info['found']:
                            custom_logger.info(f"EOS found at position: {eos_info['first_position']}")
                            if eos_info['early_termination']:
                                custom_logger.error("EARLY EOS TERMINATION DETECTED!")
                                custom_logger.error("This explains the incomplete transcription!")
                            else:
                                custom_logger.info("EOS at end - normal termination")
                        else:
                            custom_logger.info("No EOS found - may have hit max_tokens")
                        
                        # Quality alerts
                        if quality_info['early_stop_risk']:
                            custom_logger.warning("Early stopping risk detected!")
                        
                        if quality_info['recommendations']:
                            custom_logger.info("Recommendations:")
                            for rec in quality_info['recommendations'][:3]:
                                custom_logger.info(f"{rec}")
                    else:
                        custom_logger.warning("Generation debugging returned no results")
                        
                except ImportError as e:
                    custom_logger.warning(f"Generation debugging not available: {e}")
                    generated_tokens = pred_ids.shape[1]
                    expected_tokens = int(duration_minutes * 300)
                    coverage_ratio = generated_tokens / expected_tokens if expected_tokens > 0 else 0
                    
                    custom_logger.info(f"Basic Analysis (fallback):")
                    custom_logger.info(f"Generated tokens: {generated_tokens}")
                    custom_logger.info(f"Expected tokens: ~{expected_tokens}")
                    custom_logger.info(f"Coverage ratio: {coverage_ratio:.2f} ({coverage_ratio*100:.1f}%)")
                    
                    if coverage_ratio < 0.5:
                        custom_logger.error("SEVERELY LOW TOKEN COUNT - likely early termination!")
                    elif coverage_ratio < 0.8:
                        custom_logger.warning("Low token count - may be incomplete transcription")
                        
                except Exception as e:
                    custom_logger.error(f"Generation debugging failed: {e}", exc_info=True)
                
                if pred_ids is None or pred_ids.size(1) == 0:
                    custom_logger.error("Generation failed: Empty pred_ids")
                    return {
                        'success': False,
                        'data': None,
                        'error': 'Generation failed: No tokens generated'
                    }
                
                generated_tokens = pred_ids.shape[1]
                custom_logger.info(f"Generated {generated_tokens} tokens (max allowed: {dynamic_max_tokens})")
                
                if generated_tokens < 5: 
                    custom_logger.error(f"Generation failed: Only {generated_tokens} tokens generated")
                    return {
                        'success': False,
                        'data': None,
                        'error': f'Generation failed: Insufficient tokens ({generated_tokens})'
                    }
                    
            except torch.cuda.OutOfMemoryError as e:
                custom_logger.error(f"CUDA OOM during generation: {e}")
                torch.cuda.empty_cache()
                return {
                    'success': False,
                    'data': None,
                    'error': f'GPU memory exceeded. Audio too long ({duration:.1f}s). Try shorter audio segments.'
                }
            except Exception as e:
                custom_logger.error(f"Generation failed: {e}", exc_info=True)
                return {
                    'success': False,
                    'data': None,
                    'error': f'Generation error: {str(e)}'
                }
            
            if hasattr(self.model, 'generation_config') and original_config:
                config = self.model.generation_config
                config.no_speech_threshold = original_config['no_speech_threshold']
                config.logprob_threshold = original_config['logprob_threshold'] 
                config.compression_ratio_threshold = original_config['compression_ratio_threshold']
                custom_logger.debug("Model generation config restored")
            
            decoded = self.processor.batch_decode(
                pred_ids,
                skip_special_tokens=False
            )
            
            if not decoded or len(decoded) == 0:
                custom_logger.error("Decoding failed: Empty decoded results")
                return {
                    'success': False,
                    'data': None,
                    'error': 'Decoding failed: No text generated'
                }
            
            raw_text = decoded[0]
            custom_logger.debug(f"Raw decoded text length: {len(raw_text)}")
            
            words = self._parse_word_timestamps(raw_text)
            
            clean_text = self._remove_timestamp_tokens(raw_text)
            
            custom_logger.info(f"\nWhisper Output:")
            custom_logger.info(f"Text length: {len(clean_text)} chars")
            custom_logger.info(f"Words count: {len(words)}")
            custom_logger.info(f"Preview: {clean_text[:200]}")
            if len(clean_text) > 200:
                custom_logger.info(f"(truncated)")
            
            custom_logger.info(
                f"Transcription completed: {len(clean_text)} chars, "
                f"{len(words)} words"
            )
            
            return {
                'success': True,
                'data': {
                    'text': clean_text,
                    'segments': [], 
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
            
            timestamp_pattern = r'<\|(\d+\.\d+)\|>'
            
            parts = re.split(timestamp_pattern, raw_text)
            
            timestamps = []
            texts = []
            
            for i, part in enumerate(parts):
                if i % 2 == 1:  
                    try:
                        timestamps.append(float(part))
                    except ValueError:
                        custom_logger.warning(f"Invalid timestamp: {part}")
                elif i % 2 == 0 and part.strip():
                    texts.append(part.strip())
            
            for i, text in enumerate(texts):
                start_time = timestamps[i] if i < len(timestamps) else 0.0
                end_time = timestamps[i + 1] if i + 1 < len(timestamps) else start_time + 1.0
                
                word_tokens = text.split()
                
                if not word_tokens:
                    continue
                
                segment_duration = end_time - start_time
                time_per_word = segment_duration / len(word_tokens) if len(word_tokens) > 0 else 0
                
                for j, word in enumerate(word_tokens):
                    clean_word = self._clean_word(word)
                    
                    if not clean_word:
                        continue
                    
                    word_start = start_time + (j * time_per_word)
                    word_end = start_time + ((j + 1) * time_per_word)
                    
                    words.append({
                        'word': clean_word,
                        'start': float(word_start),
                        'end': float(word_end),
                        'confidence': 1.0 
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
        word = re.sub(r'<\|[^|]+\|>', '', word)
        
        word = word.strip()
        
        if re.match(r'^[^\w\s]+$', word):
            return ''
        
        return word
    
    def _remove_timestamp_tokens(self, raw_text: str) -> str:
        """
        Remove timestamp tokens from text.
        """
        clean_text = re.sub(r'<\|[\d.]+\|>', '', raw_text)
        
        clean_text = re.sub(r'<\|[^|]+\|>', '', clean_text)
        
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                custom_logger.debug("CUDA cache cleared")
            
            custom_logger.info("Whisper transcriber cleaned up")
            
        except Exception as e:
            custom_logger.warning(f"Cleanup error: {e}")
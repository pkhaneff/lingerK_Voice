import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List
import asyncio
import functools
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
from loguru import logger as custom_logger

try:
    import pyrnnoise
    PYRNNOISE_AVAILABLE = True
except ImportError:
    PYRNNOISE_AVAILABLE = False
    custom_logger.warning("pyrnnoise not available")

from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws.s3.repository.object import get_object


class NoiseReducer:
    """Reduce noise from audio files."""
    
    def __init__(self):
        self.bucket_name = s3_bucket
        if not PYRNNOISE_AVAILABLE:
            raise RuntimeError("pyrnnoise is required but not installed")
    
    async def analyze_noise_only_from_s3(self, audio_s3_key: str) -> Dict[str, Any]:
        """
        Analyze noise in audio without applying reduction.
        
        Args:
            audio_s3_key: Full S3 key of audio
            
        Returns:
            {'success': bool, 'data': {'noise_segments': [...], 'statistics': {...}}, 'error': str}
        """
        temp_audio_path = None
        
        try:
            custom_logger.info(f"Starting noise analysis: {audio_s3_key}")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_audio_path = f.name
            
            audio_object = await get_object(audio_s3_key, self.bucket_name)
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            result = await self._analyze_noise_only(temp_audio_path)
            
            self._cleanup_files(temp_audio_path)
            
            return result
            
        except Exception as e:
            custom_logger.error(f"Noise analysis failed: {str(e)}", exc_info=True)
            self._cleanup_files(temp_audio_path)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def _analyze_noise_only(self, audio_path: str) -> Dict[str, Any]:
        """Analyze noise in thread"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(self._analyze_noise_sync, audio_path)
            )
            return result
            
        except Exception as e:
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _analyze_noise_sync(self, audio_path: str) -> Dict[str, Any]:
        """Sync noise analysis"""
        try:
            audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
            original_duration = len(audio_data) / sr
            custom_logger.info(f"Audio loaded for noise analysis: {original_duration:.2f}s, sr={sr}Hz")
            
            noise_segments = self._detect_noise_segments(audio_data, sr)
            
            total_noise_duration = sum(seg['duration'] for seg in noise_segments)
            noise_ratio = total_noise_duration / original_duration if original_duration > 0 else 0
            
            noise_analysis = {
                'noise_segments': noise_segments,
                'statistics': {
                    'total_duration': float(original_duration),
                    'noise_duration': float(total_noise_duration),
                    'clean_duration': float(original_duration - total_noise_duration),
                    'noise_ratio': float(noise_ratio),
                    'segments_count': len(noise_segments),
                    'sample_rate': int(sr),
                    'analysis_method': 'spectral_energy'
                },
                'processed_at': datetime.utcnow().isoformat()
            }
            
            custom_logger.info(f"Noise analysis completed: {len(noise_segments)} segments, {noise_ratio:.1%} ratio")
            
            return {
                'success': True,
                'data': noise_analysis,
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Sync noise analysis failed: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _detect_noise_segments(
        self, 
        audio_data: np.ndarray, 
        sr: int,
        noise_threshold: float = 0.02,
        min_segment_length: float = 0.5
    ) -> List[Dict]:
        """
        Detect noise segments using spectral energy analysis.
        
        Args:
            audio_data: Audio data
            sr: Sample rate
            noise_threshold: Energy threshold for noise detection
            min_segment_length: Minimum segment length in seconds
            
        Returns:
            List of noise segments with start/end times
        """
        try:
            frame_length = int(0.025 * sr)  
            hop_length = int(0.010 * sr)   
            
            rms_energy = librosa.feature.rms(
                y=audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=sr,
                hop_length=hop_length
            )[0]
            
            rms_norm = rms_energy / (np.max(rms_energy) + 1e-8)
            centroid_norm = spectral_centroid / (np.max(spectral_centroid) + 1e-8)
            
            energy_score = 1.0 - rms_norm  
            spectral_score = np.abs(centroid_norm - np.median(centroid_norm)) 
            
            noise_score = (energy_score * 0.7) + (spectral_score * 0.3)
            
            is_noise_frame = noise_score > noise_threshold
            
            noise_segments = []
            in_noise_segment = False
            segment_start = 0
            
            for i, is_noise in enumerate(is_noise_frame):
                sample_idx = i * hop_length
                
                if is_noise and not in_noise_segment:
                    in_noise_segment = True
                    segment_start = sample_idx
                    
                elif not is_noise and in_noise_segment:
                    segment_end = min(sample_idx, len(audio_data))
                    segment_duration = (segment_end - segment_start) / sr
                    
                    if segment_duration >= min_segment_length:
                        start_frame = max(0, segment_start // hop_length)
                        end_frame = min(len(noise_score), segment_end // hop_length)
                        avg_noise_score = np.mean(noise_score[start_frame:end_frame]) if end_frame > start_frame else 0.0
                        
                        noise_segments.append({
                            'start_time': float(segment_start / sr),
                            'end_time': float(segment_end / sr),
                            'duration': float(segment_duration),
                            'noise_score': float(avg_noise_score)
                        })
                    
                    in_noise_segment = False
            
            if in_noise_segment:
                segment_end = len(audio_data)
                segment_duration = (segment_end - segment_start) / sr
                
                if segment_duration >= min_segment_length:
                    start_frame = max(0, segment_start // hop_length)
                    end_frame = min(len(noise_score), segment_end // hop_length)
                    avg_noise_score = np.mean(noise_score[start_frame:end_frame]) if end_frame > start_frame else 0.0
                    
                    noise_segments.append({
                        'start_time': float(segment_start / sr),
                        'end_time': float(segment_end / sr),
                        'duration': float(segment_duration),
                        'noise_score': float(avg_noise_score)
                    })
            
            return noise_segments
            
        except Exception as e:
            custom_logger.error(f"Noise detection failed: {str(e)}")
            return []

    async def reduce_noise_from_s3(self, audio_s3_key: str) -> Dict[str, Any]:
        """
        Download audio from S3, reduce noise, return cleaned audio path.
        
        Args:
            audio_s3_key: Full S3 key of audio
            
        Returns:
            {'success': bool, 'data': {'cleaned_path', 'noise_analysis'}, 'error': str}
        """
        temp_original_path = None
        temp_cleaned_path = None
        
        try:
            custom_logger.info(f"Starting noise reduction: {audio_s3_key}")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_original_path = f.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_cleaned_path = f.name
            
            audio_object = await get_object(audio_s3_key, self.bucket_name)
            with open(temp_original_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            result = await self._process_noise_reduction(temp_original_path, temp_cleaned_path)
            
            if not result['success']:
                self._cleanup_files(temp_original_path, temp_cleaned_path)
                return result
            
            self._cleanup_files(temp_original_path)
            
            return {
                'success': True,
                'data': {
                    'cleaned_path': temp_cleaned_path,
                    'noise_analysis': result['data']['noise_analysis']
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Noise reduction failed: {str(e)}", exc_info=True)
            self._cleanup_files(temp_original_path, temp_cleaned_path)
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def _process_noise_reduction(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Process in thread"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(self._process_sync, input_path, output_path)
            )
            return result
            
        except Exception as e:
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _process_sync(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Sync processing"""
        try:
            audio_data, sr = librosa.load(input_path, sr=None, mono=True)
            original_duration = len(audio_data) / sr
            custom_logger.info(f"Audio loaded: {original_duration:.2f}s, sr={sr}Hz")
            
            audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
            
            cleaned_audio = self._apply_pyrnnoise(audio_data, sr)
            method = "pyrnnoise"
            
            if cleaned_audio is None:
                custom_logger.warning("PyRNNoise failed, using fallback")
                cleaned_audio = self._fallback_spectral_subtraction(audio_data, sr)
                method = "spectral_subtraction"
            
            cleaned_audio = self._ensure_same_length(audio_data, cleaned_audio)
            cleaned_audio = np.clip(cleaned_audio, -1.0, 1.0).astype(np.float32)
            
            noise_segments = self._extract_noise_segments(audio_data, cleaned_audio, sr)
            
            sf.write(output_path, cleaned_audio, sr)
            
            total_noise_duration = sum(seg['duration'] for seg in noise_segments)
            noise_ratio = total_noise_duration / original_duration if original_duration > 0 else 0
            
            noise_analysis = {
                'noise_segments': noise_segments,
                'statistics': {
                    'total_duration': float(original_duration),
                    'noise_duration': float(total_noise_duration),
                    'clean_duration': float(original_duration - total_noise_duration),
                    'noise_ratio': float(noise_ratio),
                    'segments_count': len(noise_segments),
                    'sample_rate': int(sr),
                    'reduction_method': method
                },
                'processed_at': datetime.utcnow().isoformat()
            }
            
            custom_logger.info(f"Noise reduction completed: {len(noise_segments)} segments, {noise_ratio:.1%} ratio")
            
            return {
                'success': True,
                'data': {'noise_analysis': noise_analysis},
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Sync processing failed: {str(e)}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _apply_pyrnnoise(self, audio_data: np.ndarray, sr: int) -> np.ndarray | None:
        """Apply PyRNNoise"""
        if not PYRNNOISE_AVAILABLE:
            return None
        
        try:
            target_sr = 48000
            
            if sr != target_sr:
                x = librosa.resample(audio_data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            else:
                x = audio_data.astype(np.float32)
            
            x = np.clip(x, -1.0, 1.0).astype(np.float32)
            
            x_i16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
            
            if x_i16.ndim == 1:
                x_i16_2d = x_i16[np.newaxis, :]
            else:
                x_i16_2d = x_i16
            
            rnn = pyrnnoise.RNNoise(sample_rate=target_sr)
            
            processed_blocks = []
            for vad_prob, den_frame in rnn.denoise_chunk(x_i16_2d, partial=True):
                den = np.asarray(den_frame)
                if den.ndim == 2:
                    den = den[0]
                if den.dtype != np.int16:
                    den = den.astype(np.int16)
                processed_blocks.append(den)
            
            if not processed_blocks:
                raise RuntimeError("RNNoise yielded no frames")
            
            processed_i16 = np.concatenate(processed_blocks, axis=0)
            
            try:
                rnn.reset()
            except:
                pass
            del rnn
            
            y = (processed_i16.astype(np.float32) / 32767.0).astype(np.float32)
            
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=target_sr, target_sr=sr).astype(np.float32)
            
            if len(y) != len(audio_data):
                if len(y) > len(audio_data):
                    y = y[:len(audio_data)]
                else:
                    y = np.pad(y, (0, len(audio_data) - len(y)), mode='constant')
            
            custom_logger.info(f"RNNoise completed: {len(y)} samples")
            return y
            
        except Exception as e:
            custom_logger.error(f"PyRNNoise failed: {e}", exc_info=True)
            return None
    
    def _fallback_spectral_subtraction(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Fallback method"""
        try:
            custom_logger.info("Using spectral subtraction fallback")
            
            noise_duration = min(0.5, len(audio_data) / sr)
            noise_samples = int(noise_duration * sr)
            noise_segment = audio_data[:noise_samples]
            
            stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_stft = librosa.stft(noise_segment, n_fft=2048, hop_length=512)
            noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
            
            alpha = 2.0
            beta = 0.01
            
            enhanced_magnitude = magnitude - alpha * noise_magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return cleaned_audio.astype(np.float32)
            
        except Exception as e:
            custom_logger.error(f"Fallback failed: {str(e)}")
            return audio_data.astype(np.float32)
    
    def _ensure_same_length(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Ensure same length"""
        ref_len = len(reference)
        target_len = len(target)
        
        if target_len == ref_len:
            return target
        elif target_len > ref_len:
            return target[:ref_len]
        else:
            padding = ref_len - target_len
            return np.pad(target, (0, padding), mode='constant', constant_values=0)
    
    def _extract_noise_segments(
        self, original: np.ndarray, cleaned: np.ndarray, sr: int,
        threshold: float = 0.1, min_segment_length: float = 0.5
    ) -> List[Dict]:
        """Extract noise segments metadata"""
        try:
            cleaned = self._ensure_same_length(original, cleaned)
            noise_diff = original.astype(np.float32) - cleaned.astype(np.float32)
            
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            
            noise_rms = librosa.feature.rms(
                y=noise_diff,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            noise_frames = noise_rms > threshold
            
            noise_segments = []
            in_noise_segment = False
            segment_start = 0
            
            for i, is_noise in enumerate(noise_frames):
                sample_idx = i * hop_length
                
                if is_noise and not in_noise_segment:
                    in_noise_segment = True
                    segment_start = sample_idx
                    
                elif not is_noise and in_noise_segment:
                    segment_end = min(sample_idx, len(original))
                    segment_duration = (segment_end - segment_start) / sr
                    
                    if segment_duration >= min_segment_length:
                        segment_audio = noise_diff[segment_start:segment_end]
                        segment_rms = np.sqrt(np.mean(segment_audio**2)) if len(segment_audio) > 0 else 0.0
                        
                        noise_segments.append({
                            'start_time': float(segment_start / sr),
                            'end_time': float(segment_end / sr),
                            'duration': float(segment_duration),
                            'rms_level': float(segment_rms)
                        })
                    
                    in_noise_segment = False
            
            if in_noise_segment:
                segment_end = len(original)
                segment_duration = (segment_end - segment_start) / sr
                
                if segment_duration >= min_segment_length:
                    segment_audio = noise_diff[segment_start:segment_end]
                    segment_rms = np.sqrt(np.mean(segment_audio**2)) if len(segment_audio) > 0 else 0.0
                    
                    noise_segments.append({
                        'start_time': float(segment_start / sr),
                        'end_time': float(segment_end / sr),
                        'duration': float(segment_duration),
                        'rms_level': float(segment_rms)
                    })
            
            return noise_segments
            
        except Exception as e:
            custom_logger.error(f"Noise segment extraction failed: {str(e)}")
            return []
    
    def _cleanup_files(self, *file_paths):
        """Cleanup temp files"""
        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    custom_logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    custom_logger.warning(f"Cleanup failed {file_path}: {str(e)}")
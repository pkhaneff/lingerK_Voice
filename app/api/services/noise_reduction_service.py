import io
import json
import tempfile
import asyncio
import functools
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from uuid import UUID

import numpy as np
import librosa
import soundfile as sf
from loguru import logger as custom_logger

try:
    import pyrnnoise
except ImportError:
    custom_logger.error("pyrnnoise not installed. Install with: pip install pyrnnoise")
    pyrnnoise = None

from app.api.infra.aws.s3 import s3_bucket
from app.api.infra.aws.s3.repository.object import get_object, put_object
from app.api.infra.aws.s3.entity.object import S3Object


class NoiseReductionService:
    """Service for noise reduction using PyRNNoise with segment preservation"""
    
    def __init__(self):
        self.bucket_name = s3_bucket
        
        if pyrnnoise is None:
            raise RuntimeError("pyrnnoise is required but not installed")
    
    async def process_audio_noise_reduction(self, audio_id: UUID, audio_s3_key: str) -> Dict[str, Any]:
        """
        Process audio with noise reduction and extract noise segments for evaluation
        
        Args:
            audio_id: UUID of audio record
            audio_s3_key: S3 key of original audio file
            
        Returns:
            Dict with processing results
        """
        temp_original_path = None
        temp_cleaned_path = None
        temp_noise_path = None
        
        try:
            custom_logger.info(f"Starting noise reduction for audio_id: {audio_id}")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_orig:
                temp_original_path = temp_orig.name
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_clean:
                temp_cleaned_path = temp_clean.name  
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_noise:
                temp_noise_path = temp_noise.name
            
            custom_logger.debug(f"Downloading audio from S3: {audio_s3_key}")
            audio_object = get_object(audio_s3_key, self.bucket_name)
            
            with open(temp_original_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            processing_result = await self._process_noise_reduction_async(
                temp_original_path, temp_cleaned_path, temp_noise_path
            )
            
            if not processing_result['success']:
                return processing_result
            
            upload_result = await self._upload_processed_files(
                audio_id, audio_s3_key, temp_cleaned_path, temp_noise_path, 
                processing_result['noise_analysis']
            )
            
            if not upload_result['success']:
                return upload_result
            
            result = {
                'success': True,
                'audio_id': str(audio_id),
                'cleaned_audio_key': upload_result['cleaned_key'],
                'noise_analysis': processing_result['noise_analysis'],
                'storage_keys': upload_result['storage_keys'],
                'processing_stats': processing_result['stats']
            }
            
            db_update_success = await self._update_database_with_noise_analysis(audio_id, result)
            result['db_updated'] = db_update_success
            
            custom_logger.info(f"Noise reduction completed for audio_id: {audio_id}")
            return result
            
        except Exception as e:
            custom_logger.error(f"Noise reduction failed for audio_id {audio_id}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f"Noise reduction failed: {str(e)}"
            }
        
        finally:
            self._cleanup_temp_files(temp_original_path, temp_cleaned_path, temp_noise_path)
    
    async def _process_noise_reduction_async(
        self, input_path: str, cleaned_path: str, noise_path: str
    ) -> Dict[str, Any]:
        """Process noise reduction in thread to avoid blocking event loop"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    self._process_noise_reduction_sync,
                    input_path, cleaned_path, noise_path
                )
            )
            return result
            
        except Exception as e:
            custom_logger.error(f"Async noise reduction failed: {str(e)}")
            return {
                'success': False,
                'error': f"Async processing error: {str(e)}"
            }
    
    def _process_noise_reduction_sync(self, input_path: str, cleaned_path: str, noise_path: str) -> Dict[str, Any]:
        try:
            custom_logger.debug(f"Loading audio file: {input_path}")
            audio_data, sr = librosa.load(input_path, sr=None, mono=True)
            original_duration = len(audio_data) / sr
            custom_logger.info(f"Audio loaded: duration={original_duration:.2f}s, sr={sr}Hz, samples={len(audio_data)}")

            audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)

            custom_logger.debug("Applying PyRNNoise...")
            used_method = "unknown"

            cleaned_audio = self._apply_pyrnnoise(audio_data, sr)
            if cleaned_audio is not None:
                used_method = "pyrnnoise_48k"
                out_sr = sr
            else:
                custom_logger.warning("PyRNNoise failed, falling back to spectral subtraction")
                cleaned_audio = self._fallback_spectral_subtraction(audio_data, sr)
                used_method = "spectral_subtraction_fallback"
                out_sr = sr

            cleaned_audio = self._ensure_same_length(audio_data, cleaned_audio)
            cleaned_audio = np.clip(cleaned_audio, -1.0, 1.0).astype(np.float32)

            noise_segments, noise_combined = self._extract_noise_segments(audio_data, cleaned_audio, sr)

            sf.write(cleaned_path, cleaned_audio, out_sr)

            if len(noise_combined) > 0:
                sf.write(noise_path, noise_combined.astype(np.float32), sr)
            else:
                sf.write(noise_path, np.array([0.0], dtype=np.float32), sr)

            total_noise_duration = sum(seg['duration'] for seg in noise_segments)
            noise_ratio = total_noise_duration / original_duration if original_duration > 0 else 0

            noise_analysis = {
                'total_duration': float(original_duration),
                'noise_duration': float(total_noise_duration),
                'clean_duration': float(original_duration - total_noise_duration),
                'noise_ratio': float(noise_ratio),
                'segments_count': len(noise_segments),
                'segments': noise_segments,
                'processed_at': datetime.utcnow().isoformat(),
                'sample_rate': int(out_sr),
                'reduction_method': used_method
            }

            stats = {
                'original_samples': len(audio_data),
                'cleaned_samples': len(cleaned_audio),
                'noise_samples': len(noise_combined),
                'resampled': False  
            }

            custom_logger.info(
                f"Noise reduction completed: {len(noise_segments)} segments, "
                f"{noise_ratio:.1%} noise ratio, {total_noise_duration:.2f}s noise removed"
            )

            return {'success': True, 'noise_analysis': noise_analysis, 'stats': stats}

        except Exception as e:
            custom_logger.error(f"Sync noise reduction failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': f"Processing error: {str(e)}"}

    
    def _apply_pyrnnoise(self, audio_data: np.ndarray, sr: int) -> np.ndarray | None:
        if pyrnnoise is None:
            custom_logger.error("pyrnnoise not installed")
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
            elif x_i16.ndim == 2 and x_i16.shape[0] == 1:
                x_i16_2d = x_i16
            else:
                x_i16_2d = np.mean(x_i16, axis=0, dtype=np.int16, keepdims=True)

            rnn = pyrnnoise.RNNoise(sample_rate=target_sr)
            custom_logger.info("RNNoise initialized; using denoise_chunk(partial=True)")

            processed_blocks = []
            total_frames = 0
            for vad_prob, den_frame in rnn.denoise_chunk(x_i16_2d, partial=True):
                den = np.asarray(den_frame)
                if den.ndim == 2:
                    if den.shape[0] != 1:
                        den = den[0]
                    else:
                        den = den[0]
                if den.ndim != 1:
                    raise ValueError(f"Unexpected denoised frame shape: {den.shape}")
                if den.dtype != np.int16:
                    den = den.astype(np.int16)
                processed_blocks.append(den)
                total_frames += 1

            if not processed_blocks:
                raise RuntimeError("RNNoise yielded no frames")

            processed_i16 = np.concatenate(processed_blocks, axis=0)

            try:
                rnn.reset()
            except Exception as e:
                custom_logger.warning(f"RNNoise reset failed: {e}")
            del rnn

            y = (processed_i16.astype(np.float32) / 32767.0).astype(np.float32)

            if sr != target_sr:
                y = librosa.resample(y, orig_sr=target_sr, target_sr=sr).astype(np.float32)

            if len(y) != len(audio_data):
                if len(y) > len(audio_data):
                    y = y[:len(audio_data)]
                else:
                    y = np.pad(y, (0, len(audio_data) - len(y)), mode='constant')

            custom_logger.info(f"RNNoise done: frames={total_frames}, samples_out={len(y)}")
            return y

        except Exception as e:
            custom_logger.error(f"PyRNNoise failed: {e}", exc_info=True)
            return None
    
    def _ensure_same_length(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Ensure target array has same length as reference"""
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
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Extract noise segments by analyzing difference between original and cleaned audio
        """
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
            combined_noise_parts = []
            
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
                        confidence = min(segment_rms / threshold, 1.0) if threshold > 0 else 0.0
                        
                        noise_segment = {
                            'start_time': float(segment_start / sr),
                            'end_time': float(segment_end / sr),
                            'duration': float(segment_duration),
                            'confidence_score': float(confidence),
                            'rms_level': float(segment_rms),
                            'sample_start': int(segment_start),
                            'sample_end': int(segment_end)
                        }
                        
                        noise_segments.append(noise_segment)
                        combined_noise_parts.append(segment_audio)
                    
                    in_noise_segment = False
            
            if in_noise_segment:
                segment_end = len(original)
                segment_duration = (segment_end - segment_start) / sr
                
                if segment_duration >= min_segment_length:
                    segment_audio = noise_diff[segment_start:segment_end]
                    segment_rms = np.sqrt(np.mean(segment_audio**2)) if len(segment_audio) > 0 else 0.0
                    confidence = min(segment_rms / threshold, 1.0) if threshold > 0 else 0.0
                    
                    noise_segment = {
                        'start_time': float(segment_start / sr),
                        'end_time': float(segment_end / sr),
                        'duration': float(segment_duration),
                        'confidence_score': float(confidence),
                        'rms_level': float(segment_rms),
                        'sample_start': int(segment_start),
                        'sample_end': int(segment_end)
                    }
                    
                    noise_segments.append(noise_segment)
                    combined_noise_parts.append(segment_audio)
            
            if combined_noise_parts:
                combined_noise = np.concatenate(combined_noise_parts)
            else:
                combined_noise = np.array([], dtype=np.float32)
            
            custom_logger.debug(
                f"Extracted {len(noise_segments)} noise segments, "
                f"total noise samples: {len(combined_noise)}"
            )
            
            return noise_segments, combined_noise.astype(np.float32)
            
        except Exception as e:
            custom_logger.error(f"Noise segment extraction failed: {str(e)}", exc_info=True)
            return [], np.array([], dtype=np.float32)
    
    async def _upload_processed_files(
        self, audio_id: UUID, original_s3_key: str, cleaned_path: str, 
        noise_path: str, noise_analysis: Dict
    ) -> Dict[str, Any]:
        """Upload processed audio files to S3"""
        try:
            base_key = original_s3_key.rsplit('.', 1)[0]  
            cleaned_key = f"{base_key}_cleaned.wav"
            noise_key = f"{base_key}_noise.wav"
            metadata_key = f"{base_key}_noise_analysis.json"
            
            uploaded_keys = {}
            
            with open(cleaned_path, 'rb') as f:
                cleaned_content = f.read()
                
            s3_cleaned = S3Object(
                body=io.BytesIO(cleaned_content),
                content_length=len(cleaned_content),
                content_type='audio/wav',
                key=cleaned_key,
                last_modified=None
            )
            
            put_object(s3_cleaned, self.bucket_name)
            uploaded_keys['cleaned_audio'] = cleaned_key
            custom_logger.debug(f"Uploaded cleaned audio: {cleaned_key}")
            
            noise_file_size = Path(noise_path).stat().st_size
            if noise_file_size > 100: 
                with open(noise_path, 'rb') as f:
                    noise_content = f.read()
                    
                s3_noise = S3Object(
                    body=io.BytesIO(noise_content),
                    content_length=len(noise_content),
                    content_type='audio/wav',
                    key=noise_key,
                    last_modified=None
                )
                
                put_object(s3_noise, self.bucket_name)
                uploaded_keys['noise_audio'] = noise_key
                custom_logger.debug(f"Uploaded noise audio: {noise_key}")
            
            metadata_json = json.dumps(noise_analysis, indent=2).encode('utf-8')
            s3_metadata = S3Object(
                body=io.BytesIO(metadata_json),
                content_length=len(metadata_json),
                content_type='application/json',
                key=metadata_key,
                last_modified=None
            )
            
            put_object(s3_metadata, self.bucket_name)
            uploaded_keys['noise_metadata'] = metadata_key
            custom_logger.debug(f"Uploaded noise metadata: {metadata_key}")
            
            return {
                'success': True,
                'cleaned_key': cleaned_key,
                'storage_keys': uploaded_keys
            }
            
        except Exception as e:
            custom_logger.error(f"Upload processed files failed: {str(e)}")
            return {
                'success': False,
                'error': f"Upload failed: {str(e)}"
            }
    
    def _cleanup_temp_files(self, *file_paths):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    custom_logger.debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    custom_logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")
    
    def _fallback_spectral_subtraction(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Fallback noise reduction using basic spectral subtraction with librosa
        """
        try:
            custom_logger.info("Using fallback spectral subtraction method")
            
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
            
            custom_logger.info("Fallback spectral subtraction completed")
            return cleaned_audio.astype(np.float32)
            
        except Exception as e:
            custom_logger.error(f"Fallback spectral subtraction failed: {str(e)}")
            custom_logger.warning("Returning original audio without noise reduction")
            return audio_data.astype(np.float32)
    
    async def _update_database_with_noise_analysis(self, audio_id: UUID, noise_result: Dict[str, Any]) -> bool:
        """
        Update audio record with noise analysis results
        TODO: Implement actual database update using SQLAlchemy/MetadataService
        """
        try:
            update_data = {
                'preprocessed': True,
                'noise_analysis': noise_result['noise_analysis'],
                'processed_time': datetime.utcnow(),
                'status': 'completed',
                'storage_uri': f"s3://{self.bucket_name}/{noise_result['cleaned_audio_key']}"  
            }
            
            custom_logger.info(f"[PLACEHOLDER] Would update audio {audio_id} with noise analysis in database")
            custom_logger.debug(f"Update data: {update_data}")
            
            return True  
            
        except Exception as e:
            custom_logger.error(f"Failed to update audio {audio_id} with noise analysis: {str(e)}")
            return False


async def process_audio_noise_reduction(audio_id: UUID, audio_s3_key: str) -> Dict[str, Any]:
    """
    Process audio with noise reduction
    
    Args:
        audio_id: UUID of audio record in database
        audio_s3_key: S3 key of the audio file
        
    Returns:
        Dict with processing results and noise analysis
    """
    service = NoiseReductionService()
    return await service.process_audio_noise_reduction(audio_id, audio_s3_key)
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
import soundfile as sf
from loguru import logger as custom_logger


class AudioSegmentProcessor:
    """Process audio segments: calculate removal ranges and splice audio."""
    
    def __init__(self):
        self.temp_files = []
    
    def calculate_removal_segments(
        self, 
        noise_timeline, 
        voice_timeline, 
        total_duration, 
        min_segment_duration = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate segments to remove: noise AND no voice.
        
        Args:
            noise_timeline: List of noise segments with start/end times
            voice_timeline: List of (start, end) voice segments
            total_duration: Total audio duration
            min_segment_duration: Minimum duration to consider for removal
            
        Returns:
            {
                'segments_to_remove': [(start, end), ...],
                'segments_to_keep': [(start, end), ...],
                'statistics': {...}
            }
        """
        try:
            custom_logger.info("Calculating audio segments to remove...")
            
            noise_ranges = [
                (seg['start_time'], seg['end_time']) 
                for seg in noise_timeline
            ]
            
            custom_logger.info(f"Noise segments: {len(noise_ranges)}")
            custom_logger.info(f"Voice segments: {len(voice_timeline)}")
            
            segments_to_remove = []
            
            for noise_start, noise_end in noise_ranges:
                has_voice_overlap = False
                
                for voice_start, voice_end in voice_timeline:
                    if not (noise_end <= voice_start or noise_start >= voice_end):
                        has_voice_overlap = True
                        break
                
                if not has_voice_overlap:
                    duration = noise_end - noise_start
                    if duration >= min_segment_duration:
                        segments_to_remove.append((noise_start, noise_end))
                        custom_logger.debug(f"Remove segment: {noise_start:.2f}-{noise_end:.2f}s")
            
            segments_to_keep = voice_timeline.copy()
            segments_to_remove = self._calculate_silence_gaps(voice_timeline, total_duration)
            
            total_remove_duration = sum(end - start for start, end in segments_to_remove)
            total_keep_duration = sum(end - start for start, end in segments_to_keep)
            
            statistics = {
                'total_duration': float(total_duration),
                'remove_segments_count': len(segments_to_remove),
                'remove_duration': float(total_remove_duration),
                'keep_segments_count': len(segments_to_keep),
                'keep_duration': float(total_keep_duration),
                'removal_ratio': float(total_remove_duration / total_duration) if total_duration > 0 else 0.0
            }
            
            custom_logger.info(
                f"Segments analysis: Remove {len(segments_to_remove)} segments "
                f"({total_remove_duration:.1f}s, {statistics['removal_ratio']:.1%})"
            )
            
            return {
                'segments_to_remove': segments_to_remove,
                'segments_to_keep': segments_to_keep,
                'statistics': statistics
            }
            
        except Exception as e:
            custom_logger.error(f"Segment calculation failed: {e}", exc_info=True)
            return {
                'segments_to_remove': [],
                'segments_to_keep': [(0, total_duration)],
                'statistics': {'error': str(e)}
            }
    
    def _calculate_silence_gaps(self, voice_timeline, total_duration):
        """Calculate silence gaps between voice segments."""
        if not voice_timeline:
            return [(0, total_duration)]
        
        sorted_voice = sorted(voice_timeline)
        silence_gaps = []
        
        if sorted_voice[0][0] > 0:
            silence_gaps.append((0, sorted_voice[0][0]))
        
        for i in range(len(sorted_voice) - 1):
            current_end = sorted_voice[i][1]
            next_start = sorted_voice[i + 1][0]
            if current_end < next_start:
                silence_gaps.append((current_end, next_start))
        
        last_end = sorted_voice[-1][1]
        if last_end < total_duration:
            silence_gaps.append((last_end, total_duration))
        
        return silence_gaps
    
    async def splice_audio(
        self,
        audio_s3_key: str,
        segments_to_keep: List[Tuple[float, float]],
        sr: int = 16000
    ) -> Dict[str, Any]:
        """
        Splice audio by keeping only specified segments.
        
        Args:
            audio_s3_key: S3 key of original audio
            segments_to_keep: List of (start, end) segments to keep
            sr: Sample rate
            
        Returns:
            {
                'success': bool,
                'data': {'spliced_audio_path': str, 'final_duration': float},
                'error': str
            }
        """
        temp_original_path = None
        temp_spliced_path = None
        
        try:
            custom_logger.info(f"Splicing audio: {len(segments_to_keep)} segments to keep")
            
            from app.api.infra.aws.s3.repository.object import get_object
            from app.api.infra.aws.s3 import s3_bucket
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_original_path = f.name
            
            audio_object = await get_object(audio_s3_key, s3_bucket)
            with open(temp_original_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            audio_data, loaded_sr = librosa.load(temp_original_path, sr=sr, mono=True)
            custom_logger.info(f"Original audio loaded: {len(audio_data)} samples, {loaded_sr}Hz")
            
            spliced_segments = []
            
            for segment_start, segment_end in segments_to_keep:
                start_sample = int(segment_start * sr)
                end_sample = int(segment_end * sr)
                
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                if start_sample < end_sample:
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    segment_audio = self._clean_segment_boundaries(segment_audio)
                    
                    if len(segment_audio) > 0:  
                        spliced_segments.append(segment_audio)
                        custom_logger.debug(
                            f"Extracted segment: {segment_start:.2f}-{segment_end:.2f}s "
                            f"({len(segment_audio)} samples, cleaned)"
                        )
            
            if spliced_segments:
                spliced_audio = np.concatenate(spliced_segments)
            else:
                custom_logger.warning("No segments to keep, creating empty audio")
                spliced_audio = np.zeros(int(0.1 * sr))  
            
            final_duration = len(spliced_audio) / sr
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_spliced_path = f.name
            
            sf.write(temp_spliced_path, spliced_audio, sr)
            self.temp_files.append(temp_spliced_path)
            
            custom_logger.info(
                f"Audio spliced successfully: "
                f"{len(segments_to_keep)} segments -> {final_duration:.2f}s"
            )
            
            self._cleanup_files(temp_original_path)
            
            return {
                'success': True,
                'data': {
                    'spliced_audio_path': temp_spliced_path,
                    'final_duration': float(final_duration),
                    'segments_count': len(segments_to_keep)
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Audio splicing failed: {e}", exc_info=True)
            self._cleanup_files(temp_original_path, temp_spliced_path)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _clean_segment_boundaries(
        self, 
        segment_audio: np.ndarray, 
        threshold: float = 0.001
    ) -> np.ndarray:
        """
        Remove silence from beginning and end of audio segment.
        This prevents gaps when concatenating segments.
        """
        if len(segment_audio) == 0:
            return segment_audio
        
        non_silent_indices = np.where(np.abs(segment_audio) > threshold)[0]
        
        if len(non_silent_indices) == 0:
            return segment_audio[:160] if len(segment_audio) >= 160 else segment_audio
        
        start_idx = non_silent_indices[0]
        end_idx = non_silent_indices[-1] + 1
        
        return segment_audio[start_idx:end_idx]
    
    def _cleanup_files(self, *file_paths):
        """Cleanup temp files."""
        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    custom_logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    custom_logger.warning(f"Cleanup failed {file_path}: {e}")
    
    def map_timeline_to_spliced_audio(
        self,
        original_timeline: List[Tuple[float, float]],
        segments_to_keep: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Map timeline from original audio to spliced audio coordinates.
        
        Args:
            original_timeline: Timeline segments from original audio
            segments_to_keep: Segments that were kept in spliced audio
            
        Returns:
            Mapped timeline for spliced audio
        """
        try:
            custom_logger.info(f"Mapping timeline: {len(original_timeline)} segments")
            
            if not original_timeline or not segments_to_keep:
                return []
            
            sorted_keep_segments = sorted(segments_to_keep)
            mapped_timeline = []
            
            spliced_offset = 0.0
            
            for original_start, original_end in original_timeline:
                mapped_segments = []
                
                for keep_start, keep_end in sorted_keep_segments:
                    if not (original_end <= keep_start or original_start >= keep_end):
                        overlap_start = max(original_start, keep_start)
                        overlap_end = min(original_end, keep_end)
                        
                        if overlap_start < overlap_end:
                            segment_offset = self._calculate_offset_for_segment(
                                keep_start, sorted_keep_segments
                            )
                            
                            mapped_start = segment_offset + (overlap_start - keep_start)
                            mapped_end = segment_offset + (overlap_end - keep_start)
                            
                            mapped_segments.append((mapped_start, mapped_end))
                
                mapped_timeline.extend(mapped_segments)
            
            merged_timeline = self._merge_adjacent_segments(mapped_timeline)
            
            custom_logger.info(f"Timeline mapped: {len(mapped_timeline)} â†’ {len(merged_timeline)} segments")
            
            return merged_timeline
            
        except Exception as e:
            custom_logger.error(f"Timeline mapping failed: {e}", exc_info=True)
            return []
    
    def _calculate_offset_for_segment(
        self, 
        segment_start: float, 
        sorted_keep_segments: List[Tuple[float, float]]
    ) -> float:
        """Calculate cumulative offset for a segment in spliced audio."""
        offset = 0.0
        
        for keep_start, keep_end in sorted_keep_segments:
            if keep_start < segment_start:
                offset += (keep_end - keep_start)
            elif keep_start == segment_start:
                break
            else:
                break
        
        return offset
    
    def _merge_adjacent_segments(
        self, 
        segments: List[Tuple[float, float]],
        gap_threshold: float = 0.1
    ) -> List[Tuple[float, float]]:
        """Merge adjacent or closely spaced segments."""
        if not segments:
            return []
        
        sorted_segments = sorted(segments)
        merged = []
        
        current_start, current_end = sorted_segments[0]
        
        for start, end in sorted_segments[1:]:
            if start <= current_end + gap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        
        return merged

    def cleanup(self):
        """Cleanup all temp files."""
        self._cleanup_files(*self.temp_files)
        self.temp_files.clear()
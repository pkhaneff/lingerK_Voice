import tempfile
from pathlib import Path
from typing import Dict, List
import numpy as np
import librosa
import soundfile as sf
from loguru import logger as custom_logger


class TrackAudioGenerator:
    """Generate complete audio files for each speaker track."""
    
    def __init__(self):
        self.temp_files = []
    
    def generate_tracks(
        self,
        cleaned_audio_path: str,
        speaker_tracks: List[Dict],
        separated_regions: List[Dict],
        sr: int = 16000
    ) -> Dict[int, str]:
        try:
            custom_logger.info(f"Generating track audio files from: {cleaned_audio_path}")
            
            custom_logger.info(f"TrackAudioGenerator received {len(speaker_tracks)} tracks")
            for i, track in enumerate(speaker_tracks):
                custom_logger.info(f"Input Track {i}: speaker_id={track.get('speaker_id')}, segments={len(track.get('segments', []))}")
                if track.get('segments'):
                    for j, seg in enumerate(track['segments'][:2]): 
                        custom_logger.debug(f"Segment {j+1}: {seg.get('segment_type')} {seg.get('start_time'):.2f}s-{seg.get('end_time'):.2f}s")
            
            audio_data, loaded_sr = librosa.load(cleaned_audio_path, sr=sr, mono=True)
            custom_logger.info(f"Loaded audio: {len(audio_data)} samples, sr={loaded_sr}Hz")
            
            track_audios = {}
            for track in speaker_tracks:
                speaker_id = track['speaker_id']
                track_audios[speaker_id] = np.zeros_like(audio_data)
            
            for track in speaker_tracks:
                speaker_id = track['speaker_id']
                
                custom_logger.info(f"\n{'='*50}")
                custom_logger.info(f"Processing Speaker {speaker_id} Track:")
                custom_logger.info(f"Type: {track.get('type', 'unknown')}")
                custom_logger.info(f"Total duration: {track.get('total_duration', 0):.2f}s")
                custom_logger.info(f"Segments count: {len(track['segments'])}")
                
                total_voice_samples = 0
                
                for seg_idx, segment in enumerate(track['segments']):
                    start_sample = int(segment['start_time'] * sr)
                    end_sample = int(segment['end_time'] * sr)
                    
                    start_sample = max(0, start_sample)
                    end_sample = min(len(audio_data), end_sample)
                    
                    if start_sample >= end_sample:
                        custom_logger.warning(f"Invalid segment: {start_sample} >= {end_sample}")
                        continue
                    
                    segment_samples = end_sample - start_sample
                    total_voice_samples += segment_samples
                    
                    if seg_idx < 3:
                        custom_logger.info(
                            f"   Filling segment {seg_idx+1}: {segment['start_time']:.2f}s â†’ {segment['end_time']:.2f}s "
                            f"({segment['segment_type']}, {segment_samples} samples)"
                        )
                    
                    if segment['segment_type'] == 'non-overlap':
                        track_audios[speaker_id][start_sample:end_sample] = \
                            audio_data[start_sample:end_sample]
                        
                    else: 
                        sep_audio = self._get_separated_audio(
                            separated_regions,
                            speaker_id,
                            segment['start_time'],
                            segment['end_time']
                        )
                        
                        if sep_audio is not None:
                            expected_len = end_sample - start_sample
                            if len(sep_audio) != expected_len:
                                if len(sep_audio) > expected_len:
                                    sep_audio = sep_audio[:expected_len]
                                else:
                                    sep_audio = np.pad(
                                        sep_audio,
                                        (0, expected_len - len(sep_audio)),
                                        mode='constant'
                                    )
                            
                            track_audios[speaker_id][start_sample:end_sample] = sep_audio
                
                track_audio = track_audios[speaker_id]
                total_samples = len(track_audio)
                non_zero_samples = np.count_nonzero(np.abs(track_audio) > 0.0001)
                zero_samples = total_samples - non_zero_samples
                # zero_samples = 0.0
                total_duration = total_samples / sr
                voice_duration = non_zero_samples / sr
                silence_duration = zero_samples / sr
                
                custom_logger.info(f"\nTrack Audio Stats for Speaker {speaker_id}:")
                custom_logger.info(f"   Total: {total_duration:.2f}s ({total_samples} samples)")
                custom_logger.info(f"   Voice: {voice_duration:.2f}s ({non_zero_samples} samples, {non_zero_samples/total_samples*100:.1f}%)")
                custom_logger.info(f"   Silence: {silence_duration:.2f}s ({zero_samples} samples, {zero_samples/total_samples*100:.1f}%)")
                custom_logger.info(f"{'='*50}\n")
            
            output_paths = {}
            for speaker_id, audio in track_audios.items():
                audio = self._remove_all_silence(audio)
                
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=f'_speaker_{speaker_id}.wav',
                    delete=False
                )
                temp_path = temp_file.name
                temp_file.close()
                
                sf.write(temp_path, audio, sr)
                output_paths[speaker_id] = temp_path
                self.temp_files.append(temp_path)
                
                custom_logger.info(f"Generated track for speaker {speaker_id}: {temp_path}")
            
            return output_paths
            
        except Exception as e:
            custom_logger.error(f"Track generation failed: {e}", exc_info=True)
            self.cleanup()
            raise
    
    def _get_separated_audio(
        self,
        separated_regions: List[Dict],
        speaker_id: int,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """Extract separated audio for specific time range."""
        for region in separated_regions:
            region_start = region['start_time']
            region_end = region['end_time']
            
            # Check overlap
            if not (end_time <= region_start or start_time >= region_end):
                source_key = f'source_{speaker_id}'
                
                if source_key not in region:
                    custom_logger.warning(f"Source {source_key} not found in region")
                    return None
                
                sep_audio = region[source_key]
                
                sr = 16000
                region_duration = region_end - region_start
                
                rel_start = max(0, start_time - region_start)
                rel_end = min(region_duration, end_time - region_start)
                
                start_sample = int(rel_start * sr)
                end_sample = int(rel_end * sr)
                
                start_sample = max(0, start_sample)
                end_sample = min(len(sep_audio), end_sample)
                
                return sep_audio[start_sample:end_sample]
        
        custom_logger.warning(f"No separated region found for {start_time}-{end_time}")
        return None
    
    def _remove_all_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.0001
    ) -> np.ndarray:
        """
        Remove ALL silence segments, not just leading/trailing.
        This prevents Whisper early stopping on silence gaps.
        """
        if len(audio) == 0:
            return audio
        
        non_silent_indices = np.where(np.abs(audio) > threshold)[0]
        
        if len(non_silent_indices) == 0:
            return audio[:160]
        
        return audio[non_silent_indices]
    
    def _remove_leading_trailing_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01
    ) -> np.ndarray:
        """Remove silence at start and end."""
        if len(audio) == 0:
            return audio
        
        non_silent = np.where(np.abs(audio) > threshold)[0]
        
        if len(non_silent) == 0:
            return audio
        
        start_idx = non_silent[0]
        end_idx = non_silent[-1] + 1
        
        return audio[start_idx:end_idx]
    
    def cleanup(self, specific_paths: Dict[int, str] = None):
        """Cleanup temp files."""
        paths_to_clean = specific_paths.values() if specific_paths else self.temp_files
        
        for path in paths_to_clean:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                    custom_logger.debug(f"Cleaned up: {path}")
                except Exception as e:
                    custom_logger.warning(f"Cleanup failed {path}: {e}")
        
        if not specific_paths:
            self.temp_files.clear()
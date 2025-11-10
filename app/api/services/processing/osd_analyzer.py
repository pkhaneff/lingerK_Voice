from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger as custom_logger
import numpy as np
import librosa

from app.core.config import HF_TOKEN, DEVICE
from app.data_model.model_registry import ModelRegistry
from app.data_model.implementations.osd_model import OSDModel
from app.data_model.implementations.separation_model import ConvTasNetSeparator
from app.api.services.processing.speaker_assigner import SpeakerAssigner


class OSDAnalyzer:
    """
    Analyze overlapped speech and create tracks.
    
    Logic:
    1. Run OSD to detect overlap regions
    2. Check criteria for single vs multi
    3. Single: return 1 track
    4. Multi: separate sources + assign speakers + stitch tracks
    """
    
    def __init__(self):
        self.osd_model: Optional[OSDModel] = None
        self.separator: Optional[ConvTasNetSeparator] = None
        self.assigner = SpeakerAssigner(
            merge_gap_threshold=2.0,
            min_range_len=10.0,
            min_track_total=20.0
        )
        self._model_loading = False
    
    async def analyze_overlap(
        self,
        cleaned_audio_path: str,
        total_audio_duration: float
    ) -> Dict[str, Any]:
        """
        Main entry point.
        
        Args:
            cleaned_audio_path: Path to cleaned audio (100% voice content)
            total_audio_duration: Duration of cleaned audio
            
        Returns:
            {
                'success': bool,
                'data': {
                    'track_type': 'single' or 'multi',
                    'tracks': [...],
                    'statistics': {...}
                },
                'error': str
            }
        """
        if not await self._ensure_osd_loaded():
            return {'success': False, 'data': None, 'error': 'OSD model not loaded'}
        
        try:
            custom_logger.info(f"Starting OSD analysis: {cleaned_audio_path}")
            
            if not Path(cleaned_audio_path).exists():
                return {'success': False, 'data': None, 'error': f'File not found'}
            
            osd_output = await self.osd_model.predict(cleaned_audio_path)
            
            if not osd_output:
                return {'success': False, 'data': None, 'error': 'OSD prediction failed'}
            
            overlap_timeline = osd_output['timeline']
            raw_output = osd_output.get('raw_output')
            
            overlap_segments = self._extract_overlap_segments(
                overlap_timeline,
                raw_output
            )
            
            custom_logger.info(f"OSD found {len(overlap_segments)} overlap segments")
            
            decision = self._check_criteria(
                overlap_segments,
                total_audio_duration 
            )
            
            track_type = decision['track_type']
            custom_logger.info(f"Decision: {track_type}")
            
            if track_type == 'single':
                tracks_result = self._create_single_track(
                    total_audio_duration
                )
            else:  
                tracks_result = await self._create_multi_tracks(
                    cleaned_audio_path,
                    decision['significant_overlaps'],
                    total_audio_duration
                )
            
            if not tracks_result['success']:
                return tracks_result
            
            tracks = tracks_result['data']['tracks']
            statistics = self._calculate_statistics(
                tracks,
                overlap_segments,
                total_audio_duration
            )
            
            return {
                'success': True,
                'data': {
                    'track_type': track_type,
                    'tracks': tracks,
                    'statistics': statistics
                },
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"OSD analysis failed: {e}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _extract_overlap_segments(
        self,
        overlap_timeline,
        raw_output
    ) -> List[Dict]:
        """Extract overlap segments with confidence."""
        segments = []
        
        for idx, segment in enumerate(overlap_timeline):
            start_time = float(segment.start)
            end_time = float(segment.end)
            duration = end_time - start_time
            
            confidence = self._calculate_confidence(
                raw_output.get('scores') if isinstance(raw_output, dict) else None,
                start_time,
                end_time
            )
            
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'confidence': confidence
            })
        
        return segments
    
    def _calculate_confidence(self, scores, start_time: float, end_time: float) -> float:
        """Calculate average confidence from scores."""
        try:
            if scores is None:
                return 0.75
            
            if not (hasattr(scores, 'sliding_window') and hasattr(scores, 'data')):
                custom_logger.warning(f"Scores type invalid: {type(scores)}")
                return 0.75
            
            sliding_window = scores.sliding_window
            start_frame = int(start_time / sliding_window.step)
            end_frame = int(end_time / sliding_window.step)
            
            start_frame = max(0, start_frame)
            end_frame = min(len(scores.data), end_frame)
            
            if start_frame >= end_frame:
                return 0.75
            
            segment_scores = scores.data[start_frame:end_frame]
            
            if segment_scores.ndim == 2 and segment_scores.shape[1] >= 2:
                overlap_probs = segment_scores[:, 1]
                confidence = float(np.mean(overlap_probs))
            elif segment_scores.ndim == 1:
                confidence = float(np.mean(segment_scores))
            else:
                confidence = float(np.mean(segment_scores))
            
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            custom_logger.warning(f"Could not calculate confidence: {e}")
            return 0.75
    
    def _check_criteria(
        self,
        overlap_segments: List[Dict],
        cleaned_audio_duration: float
    ) -> Dict[str, Any]:
        if not overlap_segments:
            return {
                'track_type': 'single',
                'significant_overlaps': []
            }
        
        total_voice_duration = cleaned_audio_duration
        
        merged_overlaps = self._merge_overlaps(overlap_segments, gap_threshold=2.0)
        
        significant = [
            seg for seg in merged_overlaps
            if seg['duration'] >= 5.0
        ]
        
        total_overlap_duration = sum(seg['duration'] for seg in merged_overlaps)
        overlap_ratio = total_overlap_duration / total_voice_duration if total_voice_duration > 0 else 0
        
        has_long_overlap = len(significant) > 0
        has_high_ratio = overlap_ratio >= 0.05
        has_multiple_regions = len(significant) >= 2
        
        custom_logger.info(f"Overlap criteria (cleaned audio logic):")
        custom_logger.info(f"  - Voice duration: {total_voice_duration:.2f}s (100% of cleaned)")
        custom_logger.info(f"  - Long overlaps (>=5s): {len(significant)}")
        custom_logger.info(f"  - Overlap ratio: {overlap_ratio:.1%}")
        custom_logger.info(f"  - Multiple regions: {has_multiple_regions}")
        
        if (has_long_overlap or has_high_ratio) and has_multiple_regions:
            return {
                'track_type': 'multi',
                'significant_overlaps': significant
            }
        else:
            return {
                'track_type': 'single',
                'significant_overlaps': []
            }
    
    def _merge_overlaps(
        self,
        overlaps: List[Dict],
        gap_threshold: float
    ) -> List[Dict]:
        """Merge overlaps cÃƒÂ¡ch nhau < threshold."""
        if not overlaps:
            return []
        
        sorted_overlaps = sorted(overlaps, key=lambda x: x['start_time'])
        merged = []
        
        current = sorted_overlaps[0].copy()
        
        for overlap in sorted_overlaps[1:]:
            gap = overlap['start_time'] - current['end_time']
            
            if gap <= gap_threshold:
                # Merge
                current['end_time'] = max(current['end_time'], overlap['end_time'])
                current['duration'] = current['end_time'] - current['start_time']
                current['confidence'] = max(current['confidence'], overlap['confidence'])
            else:
                merged.append(current)
                current = overlap.copy()
        
        merged.append(current)
        return merged
    
    def _create_single_track(
        self,
        cleaned_audio_duration: float
    ) -> Dict[str, Any]:
        """Create 1 single track covering entire cleaned audio."""
        try:
            if cleaned_audio_duration <= 0:
                return {'success': False, 'data': None, 'error': 'Invalid audio duration'}
            
            custom_logger.info(f"Creating single track for cleaned audio:")
            custom_logger.info(f"Duration: {cleaned_audio_duration:.2f}s (100% voice content)")
            start_time = 0.0
            end_time = cleaned_audio_duration
            duration = cleaned_audio_duration
            coverage = 100.0 

            custom_logger.info(
                f"Creating SINGLE segment: {start_time:.2f}s → {end_time:.2f}s\n"
                f"Total: {duration:.2f}s | Voice: {duration:.2f}s | Silence: 0.0s"
            )
            
            segments = [{
                'segment_type': 'non-overlap',
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(duration),
                'confidence': 0.95,
                'separation_method': None
            }]
            
            tracks = [{
                'type': 'single',
                'speaker_id': 0,
                'ranges': [(start_time, end_time)],
                'total_duration': float(duration),
                'coverage': float(coverage),
                'segments': segments
            }]
            
            custom_logger.info(f"Created single track: {duration:.1f}s ({coverage:.1f}%)")
            
            return {
                'success': True,
                'data': {'tracks': tracks},
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Single track creation failed: {e}")
            return {'success': False, 'data': None, 'error': str(e)}
    
    async def _create_multi_tracks(
        self,
        audio_path: str,
        significant_overlaps: List[Dict],
        total_duration: float
    ) -> Dict[str, Any]:
        """Create multiple tracks via separation."""
        try:
            custom_logger.info("Creating multi tracks...")
            
            if not self.separator:
                from app.data_model.model_config import MODEL_CONFIGS
        
                config = {
                    'model_name': MODEL_CONFIGS['separation'].model_name,
                    'cache_dir': MODEL_CONFIGS['separation'].cache_dir,
                    'device': DEVICE
                }
                self.separator = ConvTasNetSeparator(config=config)
                if not await self.separator.load_model():
                    return {'success': False, 'data': None, 'error': 'Separator not loaded'}
            
            sep_result = await self.separator.separate_overlap_regions(
                audio_path,
                significant_overlaps,
                sr=16000
            )
            
            if not sep_result['success']:
                return sep_result
            
            separated_regions = sep_result['data']['separated_regions']
            
            cleaned_vad_timeline = [(0.0, total_duration)]
            
            tracks_result = self.assigner.create_tracks(
                audio_path,
                cleaned_vad_timeline,
                separated_regions,
                total_duration,
                sr=16000
            )
            
            if not tracks_result['success']:
                return tracks_result
            
            tracks = tracks_result['data']['tracks']
            
            for track in tracks:
                track['type'] = 'separated'
            
            custom_logger.info(f"Created {len(tracks)} multi tracks")
            
            return {
                'success': True,
                'data': {'tracks': tracks},
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Multi track creation failed: {e}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}
    
    def _calculate_statistics(
        self,
        tracks: List[Dict],
        overlap_segments: List[Dict],
        total_duration: float
    ) -> Dict:
        """Calculate statistics."""
        total_overlap_duration = sum(seg['duration'] for seg in overlap_segments)
        
        return {
            'total_tracks': len(tracks),
            'total_duration': float(total_duration),
            'overlap_segments_count': len(overlap_segments),
            'overlap_duration': float(total_overlap_duration),
            'overlap_ratio': float(total_overlap_duration / total_duration) if total_duration > 0 else 0.0,
            'tracks_detail': [
                {
                    'speaker_id': t['speaker_id'],
                    'duration': t['total_duration'],
                    'coverage': t['coverage']
                }
                for t in tracks
            ]
        }
    
    async def _ensure_osd_loaded(self) -> bool:
        """Lazy load OSD model."""
        if self.osd_model and self.osd_model.is_loaded:
            return True
        
        if self._model_loading:
            return False
        
        try:
            self._model_loading = True
            
            registry = ModelRegistry.get_instance()
            self.osd_model = registry.get('osd')
            
            if not self.osd_model:
                from app.data_model.model_config import MODEL_CONFIGS
                
                config = {
                    'model_name': 'pyannote/overlapped-speech-detection',
                    'hf_token': HF_TOKEN,
                    'device': DEVICE,
                    'cache_dir': Path("app/data_model/storage/osd")
                }
                
                self.osd_model = OSDModel(config=config)
                
                if not await self.osd_model.load_model():
                    self.osd_model = None
                    return False
                
                registry.register('osd', self.osd_model)
            
            return self.osd_model.is_loaded
            
        except Exception as e:
            custom_logger.error(f"OSD loading error: {e}", exc_info=True)
            self.osd_model = None
            return False
        finally:
            self._model_loading = False
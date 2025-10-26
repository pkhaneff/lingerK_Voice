from typing import Dict, Any, List, Tuple
import numpy as np
import librosa
from loguru import logger as custom_logger


class SpeakerAssigner:
    """Assign non-overlap regions to speakers and stitch tracks."""
    
    def __init__(
        self,
        merge_gap_threshold: float = 2.0,
        min_range_len: float = 10.0,
        min_track_total: float = 20.0
    ):
        """
        Args:
            merge_gap_threshold: Merge gaps < this (seconds)
            min_range_len: Min length for continuous segment (seconds)
            min_track_total: Min total length for track (seconds)
        """
        self.merge_gap_threshold = merge_gap_threshold
        self.min_range_len = min_range_len
        self.min_track_total = min_track_total
    
    def create_tracks(
        self,
        audio_path: str,
        vad_timeline: List[Tuple[float, float]],
        separated_regions: List[Dict],
        total_duration: float,
        sr: int = 16000
    ) -> Dict[str, Any]:
        """
        Tạo N tracks ổn định từ VAD và separated regions.
        
        Args:
            audio_path: Path to cleaned audio
            vad_timeline: List of (start, end) tuples from VAD
            separated_regions: Output from separation
            total_duration: Total audio duration
            sr: Sample rate
            
        Returns:
            {
                'success': bool,
                'data': {
                    'tracks': [
                        {
                            'speaker_id': int,
                            'ranges': [(start, end), ...],
                            'total_duration': float,
                            'coverage': float,
                            'segments': [  # NEW: detailed segments
                                {
                                    'segment_type': 'overlap' or 'non-overlap',
                                    'start_time': float,
                                    'end_time': float,
                                    'duration': float,
                                    'confidence': float,
                                    'separation_method': str or None
                                }
                            ]
                        }
                    ]
                },
                'error': str
            }
        """
        try:
            custom_logger.info("Creating speaker tracks...")
            
            # Load full audio
            audio_data, _ = librosa.load(audio_path, sr=sr, mono=True)
            
            # Step 1: Identify overlap and non-overlap regions
            overlap_regions = [
                (r['start_time'], r['end_time']) 
                for r in separated_regions
            ]
            
            non_overlap_regions = self._get_non_overlap_regions(
                vad_timeline,
                overlap_regions
            )
            
            custom_logger.info(f"Non-overlap regions: {len(non_overlap_regions)}")
            custom_logger.info(f"Overlap regions: {len(overlap_regions)}")
            
            # Step 2: Assign non-overlap regions to speakers
            speaker_assignments = self._assign_non_overlap(
                audio_data,
                non_overlap_regions,
                separated_regions,
                sr
            )
            
            # Step 3: Build raw tracks (speaker_id -> list of ranges)
            raw_tracks = {0: [], 1: []}
            
            # Add non-overlap regions
            for region, speaker_id in zip(non_overlap_regions, speaker_assignments):
                start, end = region
                segment = {
                    'segment_type': 'non-overlap',
                    'start_time': float(start),
                    'end_time': float(end),
                    'duration': float(end - start),
                    'confidence': 0.95,
                    'separation_method': None  # ✅ Non-overlap không cần separation
                }
                raw_tracks[speaker_id].append(segment)
            
            # Add separated overlap regions
            for sep_region in separated_regions:
                start = sep_region['start_time']
                end = sep_region['end_time']
                duration = end - start

                segment_a = {
                    'segment_type': 'overlap',
                    'start_time': float(start),
                    'end_time': float(end),
                    'duration': float(duration),
                    'confidence': sep_region.get('confidence', 0.75),
                    'separation_method': 'ConvTasNet'  
                }
                raw_tracks[0].append(segment_a)
                
                segment_b = {
                    'segment_type': 'overlap',
                    'start_time': float(start),
                    'end_time': float(end),
                    'duration': float(duration),
                    'confidence': sep_region.get('confidence', 0.75),
                    'separation_method': 'ConvTasNet'  
                }
                raw_tracks[1].append(segment_b)
            
            # Step 4: Stitch tracks
            stitched_tracks = []
            
            for speaker_id, segment in raw_tracks.items():
                if not segment:
                    continue
                
                # Sort and merge
                sorted_segment = sorted(segment, key=lambda x: x['start_time'])
                ranges = [(s['start_time'], s['end_time']) for s in sorted_segment]
                merged_ranges = self._merge_ranges(
                    ranges,
                    gap_threshold=self.merge_gap_threshold
                )
                merged_segments = self._merge_segments_by_ranges(
                    sorted_segment,
                    merged_ranges
                )
                
                # Filter by min_range_len
                filtered_segments = [
                    s for s in merged_segments
                    if s['duration'] >= self.min_range_len
                ]
                
                # Calculate total duration
                total_dur = sum(s['duration'] for s in filtered_segments)
                
                # Filter by min_track_total
                if total_dur < self.min_track_total:
                    custom_logger.info(
                        f"Speaker {speaker_id} filtered out: "
                        f"total {total_dur:.1f}s < {self.min_track_total}s"
                    )
                    continue
                
                coverage = (total_dur / total_duration) * 100 if total_duration > 0 else 0

                final_ranges = [(s['start_time'], s['end_time']) for s in filtered_segments]
                
                stitched_tracks.append({
                    'speaker_id': speaker_id,
                    'ranges': final_ranges,
                    'total_duration': float(total_dur),
                    'coverage': float(coverage),
                    'segments': filtered_segments
                })
            
            custom_logger.info(f"Created {len(stitched_tracks)} final tracks")
            
            return {
                'success': True,
                'data': {'tracks': stitched_tracks},
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Track creation failed: {e}", exc_info=True)
            return {'success': False, 'data': None, 'error': str(e)}

    def _merge_segments_by_ranges(
        self,
        segments: List[Dict],
        merged_ranges: List[Tuple[float, float]]
    ) -> List[Dict]:
        merged_segments = []
        
        for range_start, range_end in merged_ranges:
            segments_in_range = [
                s for s in segments
                if not (s['end_time'] <= range_start or s['start_time'] >= range_end)
            ]
            
            if not segments_in_range:
                continue
            
            has_overlap = any(s['segment_type'] == 'overlap' for s in segments_in_range)
            segment_type = 'overlap' if has_overlap else 'non-overlap'
            
            separation_method = None
            if has_overlap:
                overlap_segments = [s for s in segments_in_range if s['segment_type'] == 'overlap']
                if overlap_segments:
                    separation_method = overlap_segments[0]['separation_method']
            
            avg_confidence = np.mean([s['confidence'] for s in segments_in_range])
            
            merged_segment = {
                'segment_type': segment_type,
                'start_time': float(range_start),
                'end_time': float(range_end),
                'duration': float(range_end - range_start),
                'confidence': float(avg_confidence),
                'separation_method': separation_method 
            }
            
            merged_segments.append(merged_segment)
        
        return merged_segments

    def _get_non_overlap_regions(
        self,
        vad_timeline: List[Tuple[float, float]],
        overlap_regions: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Tìm vùng VAD không overlap."""
        non_overlap = []
        
        for vad_start, vad_end in vad_timeline:
            overlaps = False
            
            for ovlp_start, ovlp_end in overlap_regions:
                if not (vad_end <= ovlp_start or vad_start >= ovlp_end):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlap.append((vad_start, vad_end))
        
        return non_overlap
    
    def _assign_non_overlap(
        self,
        audio_data: np.ndarray,
        non_overlap_regions: List[Tuple[float, float]],
        separated_regions: List[Dict],
        sr: int
    ) -> List[int]:
        """
        Assign non-overlap regions to speakers.
        
        Strategy: So sánh spectral similarity với separated sources
        """
        if not separated_regions:
            return [0] * len(non_overlap_regions)
        
        assignments = []
        
        ref_embeddings = self._extract_reference_embeddings(
            separated_regions,
            sr
        )
        
        for region_start, region_end in non_overlap_regions:
            start_sample = int(region_start * sr)
            end_sample = int(region_end * sr)
            region_audio = audio_data[start_sample:end_sample]
            
            # Extract embedding
            region_embedding = self._extract_embedding(region_audio, sr)
            
            # Compare with references
            similarities = [
                self._cosine_similarity(region_embedding, ref_emb)
                for ref_emb in ref_embeddings
            ]
            
            # Assign to most similar speaker
            speaker_id = int(np.argmax(similarities))
            assignments.append(speaker_id)
        
        return assignments
    
    def _extract_reference_embeddings(
        self,
        separated_regions: List[Dict],
        sr: int
    ) -> List[np.ndarray]:
        """Extract reference embeddings for 2 speakers."""
        source_0_chunks = []
        source_1_chunks = []
        
        for region in separated_regions:
            source_0_chunks.append(region['source_0'])
            source_1_chunks.append(region['source_1'])
        
        # Concatenate and extract embeddings
        source_0_concat = np.concatenate(source_0_chunks)
        source_1_concat = np.concatenate(source_1_chunks)
        
        emb_0 = self._extract_embedding(source_0_concat, sr)
        emb_1 = self._extract_embedding(source_1_concat, sr)
        
        return [emb_0, emb_1]
    
    def _extract_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral embedding (MFCC mean)."""
        if len(audio) < 512:
            return np.zeros(20)
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        embedding = np.mean(mfcc, axis=1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    
    def _merge_ranges(
        self,
        ranges: List[Tuple[float, float]],
        gap_threshold: float
    ) -> List[Tuple[float, float]]:
        """Merge ranges với gap < threshold."""
        if not ranges:
            return []
        
        merged = []
        current_start, current_end = ranges[0]
        
        for start, end in ranges[1:]:
            gap = start - current_end
            
            if gap <= gap_threshold:
                # Merge
                current_end = max(current_end, end)
            else:
                # Save current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add last range
        merged.append((current_start, current_end))
        
        return merged
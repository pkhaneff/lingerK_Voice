import uuid
import tempfile
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from loguru import logger as custom_logger
from fastapi import HTTPException

from app.api.services.processing.noise_reducer import NoiseReducer
from app.api.services.processing.vad_analyzer import VADAnalyzer
from app.api.services.processing.osd_analyzer import OSDAnalyzer
from app.api.services.processing.audio_segment_processor import AudioSegmentProcessor
from app.api.services.metadata.db_saver import DBSaver
from app.api.services.upload.s3_uploader import S3Uploader
from app.api.infra.aws.s3.repository.object import get_object
from app.api.infra.aws.s3 import s3_bucket
from app.api.utils.s3_key_utils import generate_cleaned_s3_key
from app.core.config import S3_PREFIX_AUDIO

class ProcessingWorkflowService:
    """
    Orchestrates the audio processing workflow:
    Noise Reduction -> VAD -> Splice -> OSD -> Save Results
    """
    
    def __init__(
        self,
        noise_reducer: NoiseReducer,
        vad_analyzer: VADAnalyzer,
        osd_analyzer: OSDAnalyzer,
        segment_processor: AudioSegmentProcessor,
        db_saver: DBSaver,
        uploader: S3Uploader
    ):
        self.noise_reducer = noise_reducer
        self.vad_analyzer = vad_analyzer
        self.osd_analyzer = osd_analyzer
        self.segment_processor = segment_processor
        self.db_saver = db_saver
        self.uploader = uploader

    async def process_audio(self, audio_id: str, audio_s3_key: str, total_duration: float) -> Dict[str, Any]:
        """
        Execute the full processing pipeline for a given audio file.
        """
        cleaned_audio_path = None
        cleaned_s3_key = None
        cleaned_audio_id = None
        cleaned_s3_url = None
        
        try:
            # Step 1: Noise Analysis
            custom_logger.info("Step 1/5: Noise Analysis")
            noise_result = await self.noise_reducer.analyze_noise_only_from_s3(audio_s3_key)
            
            if not noise_result['success']:
                custom_logger.error(f"Noise analysis failed: {noise_result['error']}")
                noise_timeline = []
                noise_analysis = {'statistics': {'total_duration': 0}}
            else:
                noise_timeline = noise_result['data']['noise_segments']
                noise_analysis = noise_result['data']
            
            # Step 2: VAD Analysis (Requires Async File I/O fix from Phase 1)
            custom_logger.info("Step 2/5: VAD Analysis")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_original_path = f.name
            
            try:
                # Async S3 Get and Async Write
                audio_object = await get_object(audio_s3_key, s3_bucket)
                
                loop = asyncio.get_event_loop()
                def _write_temp():
                    with open(temp_original_path, 'wb') as f:
                        f.write(audio_object.body.read())
                await loop.run_in_executor(None, _write_temp)
                
                vad_result = await self.vad_analyzer.analyze_voice_activity(temp_original_path)
                
                if not vad_result['success']:
                    custom_logger.error(f"VAD failed: {vad_result['error']}")
                    vad_timeline = []
                    vad_analysis = {}
                else:
                    vad_timeline = vad_result['data']['vad_timeline']
                    vad_analysis = vad_result['data']
                    
            finally:
                if Path(temp_original_path).exists():
                    try:
                        Path(temp_original_path).unlink()
                    except:
                        pass

            # Step 3: Segment Calculation
            custom_logger.info("Step 3/5: Calculate Audio Segments")
            # Use provided duration or fallback to noise analysis duration
            duration_to_use = total_duration if total_duration > 0 else noise_analysis.get('statistics', {}).get('total_duration', 0)
            
            segments_result = self.segment_processor.calculate_removal_segments(
                noise_timeline=noise_timeline,
                voice_timeline=vad_timeline,
                total_duration=duration_to_use
            )
            
            segments_to_keep = segments_result['segments_to_keep']
            segments_to_remove = segments_result['segments_to_remove']
            segment_statistics = segments_result['statistics']
            
            # Step 4: Splicing
            custom_logger.info("Step 4/5: Audio Splicing")
            splice_result = await self.segment_processor.splice_audio(
                audio_s3_key=audio_s3_key,
                segments_to_keep=segments_to_keep
            )
            
            if not splice_result['success']:
                return {
                    'success': False,
                    'error': f"Audio splicing failed: {splice_result['error']}"
                }
            
            cleaned_audio_path = splice_result['data']['spliced_audio_path']
            final_duration = splice_result['data']['final_duration']
            
            # Step 5: OSD Analysis
            custom_logger.info("Step 5/5: OSD Analysis")
            osd_result = await self.osd_analyzer.analyze_overlap(
                cleaned_audio_path=cleaned_audio_path,
                total_audio_duration=final_duration
            )
            
            tracks = []
            track_type = 'unknown'
            osd_statistics = {}
            
            if not osd_result['success']:
                custom_logger.error(f"OSD failed: {osd_result.get('error')}")
                # Fallback logic
                if final_duration > 0:
                     tracks = [{
                        'type': 'single',
                        'speaker_id': 0,
                        'ranges': [(0, final_duration)],
                        'total_duration': float(final_duration),
                        'coverage': 100.0,
                        'segments': [{
                            'segment_type': 'non-overlap',
                            'start_time': 0.0,
                            'end_time': float(final_duration),
                            'duration': float(final_duration),
                            'confidence': 0.5,
                            'separation_method': None
                        }]
                    }]
                     track_type = 'single_fallback'
            else:
                tracks = osd_result['data']['tracks']
                track_type = osd_result['data']['track_type']
                osd_statistics = osd_result['data']['statistics']
            
            # Save Tracks to DB
            if tracks:
                await self.db_saver.save_hybrid_tracks(audio_id=audio_id, tracks=tracks)

            # Save Analysis Results
            combined_analysis = {
                'noise_segments': noise_timeline,
                'noise_statistics': noise_analysis.get('statistics', {}),
                'vad_timeline': vad_timeline,
                'vad_statistics': vad_analysis,
                'segments_removed': segments_to_remove,
                'segments_kept': segments_to_keep,
                'segment_statistics': segment_statistics,
                'osd_track_type': track_type,
                'osd_tracks': tracks,
                'osd_statistics': osd_statistics,
                'final_duration': final_duration,
                'processing_method': 'splice_based_cleaning',
            }
            
            await self.db_saver.update_processing_results(
                audio_id=audio_id,
                noise_analysis=combined_analysis,
                vad_analysis=vad_analysis,
                osd_analysis={
                    'track_type': track_type,
                    'tracks': tracks,
                    'statistics': osd_statistics
                }
            )
            
            # Upload Cleaned Audio
            if cleaned_audio_path and Path(cleaned_audio_path).exists():
                try:
                    cleaned_s3_key = generate_cleaned_s3_key(audio_s3_key)
                    upload_key = cleaned_s3_key.replace(f"{S3_PREFIX_AUDIO}/", "")
                    
                    # Async read
                    loop = asyncio.get_event_loop()
                    def _read_cleaned():
                        with open(cleaned_audio_path, 'rb') as f:
                            return f.read()
                    cleaned_audio_content = await loop.run_in_executor(None, _read_cleaned)
                    
                    upload_result = await self.uploader.upload_bytes(
                        content=cleaned_audio_content,
                        s3_key=upload_key,
                        content_type='audio/wav'
                    )
                    
                    if upload_result['success']:
                        cleaned_s3_url = upload_result['data']['s3_url']
                        save_result = await self.db_saver.save_audio_clean(
                            original_audio_id=audio_id,
                            storage_uri=cleaned_s3_url,
                            processing_method="splice_based_cleaning"
                        )
                        if save_result['success']:
                            cleaned_audio_id = save_result['data']['cleaned_audio_id']
                
                finally:
                    # Async cleanup
                    def _cleanup():
                        if Path(cleaned_audio_path).exists():
                            Path(cleaned_audio_path).unlink()
                    await loop.run_in_executor(None, _cleanup)

            # Final Response Data
            return {
                'success': True,
                'data': {
                    'audio_id': audio_id,
                    'status': 'processed',
                    'track_type': track_type,
                    'tracks_count': len(tracks),
                    'cleaned_audio_id': cleaned_audio_id,
                    'cleaned_s3_key': cleaned_s3_key,
                    'processing': {
                        'method': 'splice_based_cleaning',
                        'original_duration': duration_to_use,
                        'final_duration': final_duration,
                        'duration_reduction': duration_to_use - final_duration,
                        'noise_analysis': {
                            'noise_segments_count': len(noise_timeline),
                        },
                        'vad_analysis': {
                            'original_voice_segments': len(vad_timeline),
                        },
                        'tracks': {
                            'total_tracks': len(tracks)
                        }
                    }
                }
            }

        except Exception as e:
            custom_logger.error(f"Workflow processing failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
        finally:
             self.segment_processor.cleanup()

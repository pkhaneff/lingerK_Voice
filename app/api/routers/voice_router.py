from typing import Tuple
from datetime import datetime
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger as custom_logger

from app.api.services.upload.s3_uploader import S3Uploader
from app.api.services.processing.audio_extractor import AudioExtractor
from app.api.services.processing.noise_reducer import NoiseReducer
from app.api.services.processing.vad_analyzer import VADAnalyzer
from app.api.services.processing.osd_analyzer import OSDAnalyzer
from app.api.services.processing.transcription_service import TranscriptionService  # NEW
from app.api.services.metadata.metadata_extractor import MetadataExtractor
from app.api.services.metadata.db_saver import DBSaver
from app.api.responses.base import BaseResponse
from app.data_model.implementations.separation_model import ConvTasNetSeparator  # NEW
from app.core.config import (
    AUDIO_EXTS, VIDEO_EXTS, AUDIO_FILE_SEIZE, VIDEO_FILE_SEIZE,
    S3_PREFIX_AUDIO, S3_PREFIX_VIDEO
)

router = APIRouter()


def validate_file_extension(filename: str) -> Tuple[str, str]:
    """Validate file extension and return file type."""
    if not filename:
        raise HTTPException(400, "Filename is required")
    
    ext = filename.split(".")[-1].lower()
    
    if ext in AUDIO_EXTS:
        return 'audio', ext
    elif ext in VIDEO_EXTS:
        return 'video', ext
    else:
        raise HTTPException(400, f"Unsupported format. Audio: {AUDIO_EXTS} | Video: {VIDEO_EXTS}")


def validate_file_size(file_size: int, file_type: str) -> None:
    """Validate file size."""
    max_size = AUDIO_FILE_SEIZE if file_type == 'audio' else VIDEO_FILE_SEIZE
    
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        raise HTTPException(400, f"{file_type.title()} too large. Max: {max_mb:.0f}MB")


def generate_s3_key(filename: str) -> str:
    """Generate unique S3 key."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}_{filename}"


@router.post("/speech-to-text")
async def speech_to_text_api(
    file: UploadFile = File(...),
    user_id: str = "default-user"
):
    cleaned_audio_path = None
    
    try:
        custom_logger.info(f"Starting upload: {file.filename}")
        
        # Validate
        file_type, ext = validate_file_extension(file.filename)
        validate_file_size(file.size, file_type)
        s3_key = generate_s3_key(file.filename)
        
        audio_id = None
        audio_s3_key = None
        
        # Initialize services
        metadata_extractor = MetadataExtractor()
        db_saver = DBSaver()
        
        # ========== UPLOAD PHASE ==========
        if file_type == 'audio':
            custom_logger.info("Processing AUDIO")
            
            audio_uploader = S3Uploader(S3_PREFIX_AUDIO)
            upload_result = await audio_uploader.upload_file(file, s3_key)
            
            if not upload_result['success']:
                raise HTTPException(500, f"Upload failed: {upload_result['error']}")
            
            audio_s3_key = upload_result['data']['s3_key']
            audio_s3_url = upload_result['data']['s3_url']
            
            metadata_result = await metadata_extractor.extract_from_s3(audio_s3_key)
            if not metadata_result['success']:
                custom_logger.warning(f"Metadata extraction failed")
                duration, codec = None, None
            else:
                duration = metadata_result['data']['duration']
                codec = metadata_result['data']['codec']
            
            save_result = await db_saver.save_audio(
                file_name=file.filename,
                storage_uri=audio_s3_url,
                user_id=user_id,
                duration=duration,
                codec=codec,
                is_video=False
            )
            
            if not save_result['success']:
                raise HTTPException(500, f"DB save failed: {save_result['error']}")
            
            audio_id = save_result['data']['audio_id']
            
        else:  # video
            custom_logger.info("Processing VIDEO")
            
            video_uploader = S3Uploader(S3_PREFIX_VIDEO)
            video_upload_result = await video_uploader.upload_file(file, s3_key)
            
            if not video_upload_result['success']:
                raise HTTPException(500, f"Video upload failed")
            
            video_s3_key = video_upload_result['data']['s3_key']
            video_s3_url = video_upload_result['data']['s3_url']
            
            audio_extractor = AudioExtractor()
            extraction_result = await audio_extractor.extract_from_s3(video_s3_key)
            
            if not extraction_result['success']:
                raise HTTPException(500, f"Audio extraction failed")
            
            audio_content = extraction_result['data']['audio_content']
            duration = extraction_result['data']['duration']
            codec = extraction_result['data']['codec']
            
            audio_filename = s3_key.rsplit('.', 1)[0] + '.mp3'
            audio_uploader = S3Uploader(S3_PREFIX_AUDIO)
            audio_upload_result = await audio_uploader.upload_bytes(
                content=audio_content,
                s3_key=audio_filename,
                content_type='audio/mpeg'
            )
            
            if not audio_upload_result['success']:
                raise HTTPException(500, f"Audio upload failed")
            
            audio_s3_key = audio_upload_result['data']['s3_key']
            audio_s3_url = audio_upload_result['data']['s3_url']
            
            audio_save_result = await db_saver.save_audio(
                file_name=audio_filename,
                storage_uri=audio_s3_url,
                user_id=user_id,
                duration=duration,
                codec=codec,
                is_video=True
            )
            
            if not audio_save_result['success']:
                raise HTTPException(500, f"Audio DB save failed")
            
            audio_id = audio_save_result['data']['audio_id']
            
            video_save_result = await db_saver.save_video(
                audio_id=audio_id,
                video_storage_uri=video_s3_url
            )
            
            if not video_save_result['success']:
                custom_logger.warning(f"Video DB save failed")
        
        # ========== PROCESSING PHASE ==========
        
        # Step 1: Noise Reduction
        custom_logger.info("Step 1/4: Noise Reduction")
        noise_reducer = NoiseReducer()
        noise_result = await noise_reducer.reduce_noise_from_s3(audio_s3_key)
        
        if not noise_result['success']:
            custom_logger.error(f"Noise reduction failed: {noise_result['error']}")
            return BaseResponse.success_response(
                message="Upload successful but processing failed at noise reduction",
                data={
                    'audio_id': audio_id,
                    'file_name': file.filename,
                    's3_key': audio_s3_key,
                    'processing': 'failed_at_noise_reduction'
                }
            )
        
        cleaned_audio_path = noise_result['data']['cleaned_path']
        noise_analysis = noise_result['data']['noise_analysis']
        total_audio_duration = noise_analysis.get('statistics', {}).get('total_duration', 0)
        
        # Step 2: VAD (timeline only)
        custom_logger.info("Step 2/4: VAD Analysis")
        vad_analyzer = VADAnalyzer()
        vad_result = await vad_analyzer.analyze_voice_activity(cleaned_audio_path)
        
        if not vad_result['success']:
            custom_logger.error(f"VAD failed: {vad_result['error']}")
            vad_timeline = []
        else:
            vad_timeline = vad_result['data']['vad_timeline']
        
        # Step 3: OSD with separation
        custom_logger.info("Step 3/4: OSD + Separation Analysis")
        osd_analyzer = OSDAnalyzer()
        osd_result = await osd_analyzer.analyze_overlap(
            cleaned_audio_path=cleaned_audio_path,
            vad_timeline=vad_timeline,
            total_audio_duration=total_audio_duration
        )
        
        if not osd_result['success']:
            custom_logger.error(f"OSD failed: {osd_result['error']}")
            tracks = []
            track_type = 'unknown'
            osd_statistics = {}
        else:
            tracks = osd_result['data']['tracks']
            track_type = osd_result['data']['track_type']
            osd_statistics = osd_result['data']['statistics']
        
        # ========== NEW: STEP 4 - TRANSCRIPTION ==========
        custom_logger.info("Step 4/4: Transcription")
        
        if tracks and cleaned_audio_path:
            # Get separated regions if multi-track
            separated_regions = []
            
            if track_type == 'multi':
                custom_logger.info("Multi-track detected, extracting separated regions...")
                
                # Extract overlap segments
                overlap_segments = []
                for track in tracks:
                    for segment in track.get('segments', []):
                        if segment.get('segment_type') == 'overlap':
                            overlap_segments.append({
                                'start_time': segment['start_time'],
                                'end_time': segment['end_time'],
                                'duration': segment['duration']
                            })
                
                # Deduplicate overlaps
                unique_overlaps = []
                seen = set()
                for seg in overlap_segments:
                    key = (seg['start_time'], seg['end_time'])
                    if key not in seen:
                        seen.add(key)
                        unique_overlaps.append(seg)
                
                # Run separation to get audio data
                if unique_overlaps:
                    custom_logger.info(f"Running separation for {len(unique_overlaps)} overlap regions")
                    separator = ConvTasNetSeparator()
                    await separator.load_model()
                    
                    sep_result = await separator.separate_overlap_regions(
                        cleaned_audio_path,
                        unique_overlaps,
                        sr=16000
                    )
                    
                    if sep_result['success']:
                        separated_regions = sep_result['data']['separated_regions']
                        custom_logger.info(f"✅ Separated {len(separated_regions)} regions")
                    else:
                        custom_logger.warning(f"Separation failed: {sep_result['error']}")
            
            # Run transcription
            transcription_service = TranscriptionService()
            transcription_result = await transcription_service.transcribe_all_tracks(
                cleaned_audio_path,
                tracks,
                separated_regions
            )
            
            if transcription_result['success']:
                tracks = transcription_result['data']['tracks']
                custom_logger.info(f"✅ Transcription completed for {len(tracks)} tracks")
                
                # Log preview of transcripts
                for track in tracks:
                    speaker_id = track.get('speaker_id', 0)
                    transcript = track.get('transcript', '')
                    words_count = len(track.get('words', []))
                    custom_logger.info(
                        f"Speaker {speaker_id}: {words_count} words, "
                        f"text preview: {transcript[:50]}..."
                    )
            else:
                custom_logger.warning(f"Transcription failed: {transcription_result['error']}")
                # Continue without transcription
        else:
            custom_logger.warning("Skipping transcription: no tracks or cleaned audio")
        
        # ========== SAVE TO DATABASE ==========
        
        # Save tracks to database using hybrid approach (now with transcripts)
        if tracks:
            custom_logger.info(f"Saving {len(tracks)} tracks to database")
            save_tracks_result = await db_saver.save_hybrid_tracks(
                audio_id=audio_id,
                tracks=tracks
            )
            
            if not save_tracks_result['success']:
                custom_logger.warning(f"Track save failed: {save_tracks_result['error']}")
            else:
                custom_logger.info(
                    f"✅ Saved {save_tracks_result['data']['tracks_saved']} tracks "
                    f"and {save_tracks_result['data']['segments_saved']} segments"
                )
        
        # Update DB with results
        update_result = await db_saver.update_processing_results(
            audio_id=audio_id,
            noise_analysis=noise_analysis,
            vad_analysis=vad_result['data'] if vad_result['success'] else {},
            osd_analysis={
                'track_type': track_type,
                'tracks': tracks,
                'statistics': osd_statistics
            }
        )
        
        if not update_result['success']:
            custom_logger.warning(f"DB update failed: {update_result['error']}")
        
        # Cleanup
        if cleaned_audio_path and Path(cleaned_audio_path).exists():
            try:
                Path(cleaned_audio_path).unlink()
                custom_logger.debug(f"Cleaned up temp file: {cleaned_audio_path}")
            except Exception as e:
                custom_logger.warning(f"Cleanup failed: {str(e)}")
        
        # ========== RETURN RESPONSE WITH TRANSCRIPTS ==========
        return BaseResponse.success_response(
            message="Upload and processing completed successfully",
            data={
                'audio_id': audio_id,
                'file_name': file.filename,
                'file_type': file_type,
                's3_key': audio_s3_key,
                'processing': {
                    'status': 'completed',
                    'track_type': track_type,
                    'noise_reduction': {
                        'noise_ratio': noise_analysis.get('statistics', {}).get('noise_ratio', 0),
                        'segments_count': noise_analysis.get('statistics', {}).get('segments_count', 0)
                    },
                    'vad_analysis': {
                        'voice_segments_count': len(vad_timeline),
                        'voice_activity_ratio': vad_result['data'].get('voice_activity_ratio', 0) if vad_result['success'] else 0
                    },
                    'tracks': {
                        'total_tracks': len(tracks),
                        'tracks_detail': [
                            {
                                'speaker_id': t.get('speaker_id', 0),
                                'type': t.get('type', 'unknown'),
                                'duration': t.get('total_duration', 0),
                                'coverage': t.get('coverage', 0),
                                'ranges_count': len(t.get('ranges', [])),
                                # NEW: Transcription data
                                'transcript': t.get('transcript', ''),
                                'words_count': len(t.get('words', [])),
                                'transcript_preview': t.get('transcript', '')[:100] if t.get('transcript') else ''
                            }
                            for t in tracks
                        ]
                    }
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Upload failed: {str(e)}", exc_info=True)
        
        # Cleanup on error
        if cleaned_audio_path and Path(cleaned_audio_path).exists():
            try:
                Path(cleaned_audio_path).unlink()
            except:
                pass
        
        raise HTTPException(500, f"Internal server error: {str(e)}")
from typing import Tuple
from datetime import datetime
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from loguru import logger as custom_logger
import bcrypt
import base64
from sqlalchemy import select

from app.api.services.upload.s3_uploader import S3Uploader
from app.api.services.processing.audio_extractor import AudioExtractor
from app.api.services.processing.noise_reducer import NoiseReducer
from app.api.services.processing.vad_analyzer import VADAnalyzer
from app.api.services.processing.osd_analyzer import OSDAnalyzer
from app.api.services.processing.transcription_service import TranscriptionService
from app.api.services.processing.audio_segment_processor import AudioSegmentProcessor
from app.api.services.metadata.metadata_extractor import MetadataExtractor
from app.api.services.metadata.db_saver import DBSaver
from app.api.responses.base import BaseResponse
from app.data_model.implementations.separation_model import ConvTasNetSeparator
from app.api.services.processing.normalize_service import GeminiTextNormalizer
from app.api.model.note_model import Note, NoteSection 
from app.core.config import (
    AUDIO_EXTS, VIDEO_EXTS, AUDIO_FILE_SEIZE, VIDEO_FILE_SEIZE,
    S3_PREFIX_AUDIO, S3_PREFIX_VIDEO, GEMINI_API_KEY
)
from app.api.db.session import AsyncSessionLocal
from app.api.model.api_key import ApiKey
from app.api.model.user import User
from app.api.model.audio_model import AudioIngest
from app.api.model.audio_clean import AudioClean

router = APIRouter()


async def validate_api_key(api_key: str) -> str:
    """Validate API key and return user_id."""
    if not api_key:
        raise HTTPException(400, "API key is required")
    
    try:
        custom_logger.info(f"Validating API key: {api_key[:20]}...")
        
        if not api_key.startswith("sk_live_"):
            custom_logger.error(f"Invalid API key format")
            raise HTTPException(401, "Invalid API key format")
        
        encoded_part = api_key[len("sk_live_"):]
        
        padding = 4 - (len(encoded_part) % 4)
        if padding != 4:
            encoded_part += "=" * padding
        
        try:
            decoded = base64.urlsafe_b64decode(encoded_part).decode('utf-8')
            key_id_str, secret = decoded.split(":", 1)  
            key_id = uuid.UUID(key_id_str)
            
            custom_logger.info(f"Key ID: {key_id}")
            custom_logger.info(f"Secret: {secret[:10]}...")
            
        except Exception as decode_error:
            custom_logger.error(f"Decode failed: {decode_error}")
            raise HTTPException(401, "Invalid API key encoding")
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ApiKey.user_id, ApiKey.hashed_key).where(
                    ApiKey.id == key_id, 
                    ApiKey.is_active == True
                )
            )
            api_key_record = result.first()
            
            if not api_key_record:
                custom_logger.error("API key not found or inactive")
                raise HTTPException(401, "Invalid API key")
            
            custom_logger.info("Verifying secret hash...")
            hash_check = bcrypt.checkpw(secret.encode('utf-8'), api_key_record.hashed_key.encode('utf-8')) 
            custom_logger.info(f"Hash verification: {hash_check}")
            
            if hash_check:
                user_id = str(api_key_record.user_id)
                custom_logger.info(f"API key validated for user: {user_id}")
                return user_id
            else:
                custom_logger.error("Secret verification failed")
                raise HTTPException(401, "Invalid API key")
            
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error")


def generate_cleaned_s3_key(original_s3_key: str) -> str:
    """Generate S3 key for cleaned audio."""
    filename = original_s3_key.split('/')[-1]
    name_without_ext = filename.rsplit('.', 1)[0]
    return original_s3_key.replace(filename, f"{name_without_ext}_cleaned.wav")


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


@router.post("/upload")
async def upload_file_api(
    api_key: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload audio/video file and return audio_id."""
    try:
        custom_logger.info(f"Starting upload: {file.filename}")
        
        user_id = await validate_api_key(api_key)
        
        file_type, ext = validate_file_extension(file.filename)
        validate_file_size(file.size, file_type)
        s3_key = generate_s3_key(file.filename)
        
        audio_id = None
        audio_s3_key = None
        
        metadata_extractor = MetadataExtractor()
        db_saver = DBSaver()
        
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
            
        else:  
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
        
        return BaseResponse.success_response(
            message="File uploaded successfully",
            data={
                'audio_id': audio_id,
                'file_name': file.filename,
                'file_type': file_type,
                's3_key': audio_s3_key,
                'status': 'uploaded'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.post("/process")
async def process_audio_api(
    api_key: str = Form(...),
    audio_id: str = Form(...)
):
    """Process audio: noise reduction, VAD, OSD, separation."""
    cleaned_audio_path = None
    
    try:
        custom_logger.info(f"Starting processing: {audio_id}")
        
        user_id = await validate_api_key(api_key)
        
        async with AsyncSessionLocal() as session:
            audio_record = await session.get(AudioIngest, uuid.UUID(audio_id))
            if not audio_record:
                raise HTTPException(404, "Audio not found")
            
            if str(audio_record.user_id) != user_id:
                raise HTTPException(403, "Access denied")
            
            if audio_record.preprocessed:
                raise HTTPException(400, "Audio already processed")
            
            if audio_record.storage_uri.startswith('s3://'):
                s3_uri_parts = audio_record.storage_uri.replace('s3://', '').split('/', 1)
                audio_s3_key = s3_uri_parts[1] if len(s3_uri_parts) > 1 else s3_uri_parts[0]
            else:
                storage_uri_parts = audio_record.storage_uri.split('/')
                audio_s3_key = '/'.join(storage_uri_parts[-2:])
        
        db_saver = DBSaver()
        
        
        custom_logger.info("Step 1/5: Noise Analysis")
        noise_reducer = NoiseReducer()
        noise_result = await noise_reducer.analyze_noise_only_from_s3(audio_s3_key)
        
        if not noise_result['success']:
            custom_logger.error(f"Noise analysis failed: {noise_result['error']}")
            noise_timeline = []
            noise_analysis = {'statistics': {'total_duration': 0}}
        else:
            noise_timeline = noise_result['data']['noise_segments']
            noise_analysis = noise_result['data']
        
        total_audio_duration = noise_analysis.get('statistics', {}).get('total_duration', 0)
        custom_logger.info(f"Noise analysis: {len(noise_timeline)} segments detected")
        
        custom_logger.info("Step 2/5: VAD Analysis") 
        vad_analyzer = VADAnalyzer()
        import tempfile
        from app.api.infra.aws.s3.repository.object import get_object
        from app.api.infra.aws.s3 import s3_bucket
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_original_path = f.name
        
        try:
            audio_object = get_object(audio_s3_key, s3_bucket)
            with open(temp_original_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            vad_result = await vad_analyzer.analyze_voice_activity(temp_original_path)
            
            if not vad_result['success']:
                custom_logger.error(f"VAD failed: {vad_result['error']}")
                vad_timeline = []
                vad_analysis = {}
            else:
                vad_timeline = vad_result['data']['vad_timeline']
                vad_analysis = vad_result['data']
            
            custom_logger.info(f"VAD analysis: {len(vad_timeline)} voice segments detected")
            
        finally:
            try:
                Path(temp_original_path).unlink()
            except:
                pass
        
        custom_logger.info("Step 3/5: Calculate Audio Segments")
        
        segment_processor = AudioSegmentProcessor()
        segments_result = segment_processor.calculate_removal_segments(
            noise_timeline=noise_timeline,
            voice_timeline=vad_timeline,
            total_duration=total_audio_duration
        )
        
        segments_to_keep = segments_result['segments_to_keep']
        segments_to_remove = segments_result['segments_to_remove']
        segment_statistics = segments_result['statistics']
        
        custom_logger.info(
            f"Segments calculation: Remove {len(segments_to_remove)} segments, "
            f"Keep {len(segments_to_keep)} segments"
        )
        
        custom_logger.info("Step 4/5: Audio Splicing")
        splice_result = await segment_processor.splice_audio(
            audio_s3_key=audio_s3_key,
            segments_to_keep=segments_to_keep
        )
        
        if not splice_result['success']:
            custom_logger.error(f"Audio splicing failed: {splice_result['error']}")
            return BaseResponse.success_response(
                message="Processing failed at audio splicing",
                data={
                    'audio_id': audio_id,
                    'status': 'failed_at_splicing'
                }
            )
        
        cleaned_audio_path = splice_result['data']['spliced_audio_path']
        final_duration = splice_result['data']['final_duration']
        custom_logger.info(f"Audio spliced: {final_duration:.2f}s clean audio created")
        
        custom_logger.info("Step 5/5: OSD Analysis")
        
        custom_logger.info("Audio clean contains voice-only content, no VAD mapping needed")
        
        osd_analyzer = OSDAnalyzer()
        osd_result = await osd_analyzer.analyze_overlap(
            cleaned_audio_path=cleaned_audio_path,
            total_audio_duration=final_duration
        )
        
        if not osd_result['success']:
            custom_logger.error(f"OSD failed: {osd_result['error']}")
            
            if final_duration > 0:
                custom_logger.info("Creating fallback single track for transcription")
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
                tracks = []
                track_type = 'unknown'
            osd_statistics = {}
        else:
            tracks = osd_result['data']['tracks']
            track_type = osd_result['data']['track_type']
            osd_statistics = osd_result['data']['statistics']
            
            custom_logger.info(f"OSD returned {len(tracks)} tracks")
            for i, track in enumerate(tracks):
                custom_logger.info(f"OSD Track {i}: speaker_id={track.get('speaker_id')}, segments={len(track.get('segments', []))}")
                if track.get('segments'):
                    for j, seg in enumerate(track['segments'][:2]): 
                        custom_logger.debug(f"   Segment {j+1}: {seg.get('segment_type')} {seg.get('start_time'):.2f}s-{seg.get('end_time'):.2f}s")
        
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
                    f"Saved {save_tracks_result['data']['tracks_saved']} tracks "
                    f"and {save_tracks_result['data']['segments_saved']} segments"
                )
        
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
        
        update_result = await db_saver.update_processing_results(
            audio_id=audio_id,
            noise_analysis=combined_analysis,
            vad_analysis=vad_analysis,
            osd_analysis={
                'track_type': track_type,
                'tracks': tracks,
                'statistics': osd_statistics
            }
        )
        
        if not update_result['success']:
            custom_logger.warning(f"DB update failed: {update_result['error']}")
        
        cleaned_s3_url = None
        cleaned_audio_id = None
        cleaned_s3_key = None

        if cleaned_audio_id:
            try:
                async with AsyncSessionLocal() as session:
                    clean_record = await session.get(AudioClean, uuid.UUID(cleaned_audio_id))
                    if clean_record and clean_record.storage_uri:
                        if clean_record.storage_uri.startswith('s3://'):
                            s3_uri_parts = clean_record.storage_uri.replace('s3://', '').split('/', 1)
                            cleaned_s3_key = s3_uri_parts[1] if len(s3_uri_parts) > 1 else s3_uri_parts[0]
                        else:
                            storage_uri_parts = clean_record.storage_uri.split('/')
                            cleaned_s3_key = '/'.join(storage_uri_parts[-2:])
                        
                        custom_logger.info(f"Cleaned audio S3 key: {cleaned_s3_key}")
            except Exception as e:
                custom_logger.warning(f"Failed to get cleaned S3 key: {e}")
                    
        
        if cleaned_audio_path and Path(cleaned_audio_path).exists():
            try:
                custom_logger.info("Uploading cleaned audio to S3...")
                
                cleaned_s3_key = generate_cleaned_s3_key(audio_s3_key)
                
                with open(cleaned_audio_path, 'rb') as f:
                    cleaned_audio_content = f.read()
                
                uploader = S3Uploader(S3_PREFIX_AUDIO)
                upload_key = cleaned_s3_key.replace(f"{S3_PREFIX_AUDIO}/", "")
                
                upload_result = await uploader.upload_bytes(
                    content=cleaned_audio_content,
                    s3_key=upload_key,
                    content_type='audio/wav'
                )
                
                if upload_result['success']:
                    cleaned_s3_url = upload_result['data']['s3_url']
                    custom_logger.info(f"Cleaned audio uploaded: {cleaned_s3_url}")
                    
                    save_cleaned_result = await db_saver.save_cleaned_audio(
                        original_audio_id=audio_id,
                        cleaned_storage_uri=cleaned_s3_url,
                        processing_method="splice_based_cleaning"
                    )
                    
                    if save_cleaned_result['success']:
                        cleaned_audio_id = save_cleaned_result['data']['cleaned_audio_id']
                        custom_logger.info(f"Cleaned audio record saved: {cleaned_audio_id}")
                    else:
                        custom_logger.error(f"Save cleaned audio failed: {save_cleaned_result['error']}")
                else:
                    custom_logger.error(f"Upload failed: {upload_result['error']}")
                    
            except Exception as e:
                custom_logger.error(f"Upload cleaned audio failed: {str(e)}")
            
            finally:
                try:
                    Path(cleaned_audio_path).unlink()
                    custom_logger.debug(f"Cleaned up temp file: {cleaned_audio_path}")
                except Exception as e:
                    custom_logger.warning(f"Cleanup failed: {str(e)}")
        
        segment_processor.cleanup()
        
        return BaseResponse.success_response(
            message="Processing completed successfully",
            data={
                'audio_id': audio_id,
                'status': 'processed',
                'track_type': track_type,
                'tracks_count': len(tracks),
                'cleaned_audio_id': cleaned_audio_id,
                'cleaned_s3_key': cleaned_s3_key,
                'processing': {
                    'method': 'splice_based_cleaning',
                    'original_duration': total_audio_duration,
                    'final_duration': final_duration,
                    'duration_reduction': total_audio_duration - final_duration,
                    'noise_analysis': {
                        'noise_segments_count': len(noise_timeline),
                        'noise_ratio': noise_analysis.get('statistics', {}).get('noise_ratio', 0)
                    },
                    'vad_analysis': {
                        'original_voice_segments': len(vad_timeline),
                        'voice_activity_ratio': vad_analysis.get('voice_activity_ratio', 0),
                        'note': 'Cleaned audio contains voice-only content'
                    },
                    'segment_processing': {
                        'segments_removed': len(segments_to_remove),
                        'segments_kept': len(segments_to_keep),
                        'removal_ratio': segment_statistics.get('removal_ratio', 0)
                    },
                    'tracks': {
                        'total_tracks': len(tracks),
                        'tracks_detail': [
                            {
                                'speaker_id': t.get('speaker_id', 0),
                                'type': t.get('type', 'unknown'),
                                'duration': t.get('total_duration', 0),
                                'coverage': t.get('coverage', 0),
                                'ranges_count': len(t.get('ranges', []))
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
        custom_logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.post("/transcribe")
async def transcribe_audio_api(
    api_key: str = Form(...),
    audio_id_clean: str = Form(...)
):
    """Transcribe processed audio tracks."""
    try:
        custom_logger.info(f"Starting transcription: {audio_id_clean}")
        
        user_id = await validate_api_key(api_key)
        
        async with AsyncSessionLocal() as session:
            
            audio_clean_record = await session.get(AudioClean, uuid.UUID(audio_id_clean))
            if not audio_clean_record:
                raise HTTPException(404, "Cleaned audio not found")
            
            audio_record = await session.get(AudioIngest, audio_clean_record.original_audio_id)
            if not audio_record:
                raise HTTPException(404, "Original audio not found")
            
            if str(audio_record.user_id) != user_id:
                raise HTTPException(403, "Access denied")
            
            if not audio_record.preprocessed:
                raise HTTPException(400, "Audio not processed yet. Call /process first")
            
            from sqlalchemy import select
            from app.api.model.speaker_track_model import SpeakerTrack
            
            result = await session.execute(
                select(SpeakerTrack).where(SpeakerTrack.audio_id == audio_clean_record.original_audio_id)
            )
            speaker_tracks_records = result.scalars().all()
            
            if not speaker_tracks_records:
                raise HTTPException(400, "No tracks found. Call /process first")
            
            if any(track.transcript for track in speaker_tracks_records):
                return BaseResponse.success_response(
                    message="Audio already transcribed",
                    data={
                        'audio_id_clean': audio_id_clean,
                        'status': 'already_transcribed',
                        'tracks': [
                            {
                                'speaker_id': track.speaker_id,
                                'transcript': track.transcript,
                                'words_count': len(track.words) if track.words else 0
                            }
                            for track in speaker_tracks_records
                        ]
                    }
                )
            
            if audio_clean_record.storage_uri.startswith('s3://'):
                s3_uri_parts = audio_clean_record.storage_uri.replace('s3://', '').split('/', 1)
                cleaned_s3_key = s3_uri_parts[1] if len(s3_uri_parts) > 1 else s3_uri_parts[0]
            else:
                storage_uri_parts = audio_clean_record.storage_uri.split('/')
                cleaned_s3_key = '/'.join(storage_uri_parts[-2:])
        
        tracks = []
        for track_record in speaker_tracks_records:
            from app.api.model.track_segment_model import TrackSegment
            
            custom_logger.info(f"Loading segments for track_id: {track_record.track_id}")
            
            segments_result = await session.execute(
                select(TrackSegment).where(TrackSegment.track_id == track_record.track_id)
            )
            track_segments = segments_result.scalars().all()
            
            custom_logger.info(f"Found {len(track_segments)} segments in DB for track {track_record.speaker_id}")
            
            segments = []
            for seg in track_segments:
                segment_dict = {
                    'segment_type': seg.segment_type,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration,
                    'confidence': seg.confidence,
                    'separation_method': seg.separation_method
                }
                segments.append(segment_dict)
                custom_logger.debug(f"   Segment: {seg.segment_type} {seg.start_time:.2f}s-{seg.end_time:.2f}s")
            
            track_dict = {
                'speaker_id': track_record.speaker_id,
                'type': track_record.track_type,
                'ranges': track_record.ranges,
                'total_duration': track_record.total_duration,
                'coverage': track_record.coverage,
                'segments': segments 
            }
            tracks.append(track_dict)
            
            custom_logger.info(f"Loaded track {track_record.speaker_id}: {len(segments)} segments")
            custom_logger.info(f"Track dict segments: {len(track_dict['segments'])}")
        
        try:
            from app.api.infra.aws.s3.repository.object import get_object
            from app.api.infra.aws.s3 import s3_bucket
            import tempfile
            
            custom_logger.info(f"Downloading cleaned audio from S3: {cleaned_s3_key}")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                cleaned_audio_path = f.name
            
            audio_object = get_object(cleaned_s3_key, s3_bucket)
            with open(cleaned_audio_path, 'wb') as f:
                f.write(audio_object.body.read())
            
            custom_logger.info(f"Cleaned audio downloaded to: {cleaned_audio_path}")
            
        except Exception as e:
            raise HTTPException(500, f"Failed to download cleaned audio: {str(e)}")
        
        separated_regions = []
        
        transcription_service = TranscriptionService()
        transcription_result = await transcription_service.transcribe_all_tracks(
            cleaned_audio_path,
            tracks,
            separated_regions
        )
        
        if not transcription_result['success']:
            raise HTTPException(500, f"Transcription failed: {transcription_result['error']}")
        
        transcribed_tracks = transcription_result['data']['tracks']
        
        db_saver = DBSaver()
        update_result = await db_saver.update_track_transcripts(
            audio_id=str(audio_clean_record.original_audio_id),
            transcribed_tracks=transcribed_tracks
        )
        
        if not update_result['success']:
            custom_logger.warning(f"Failed to update transcripts: {update_result['error']}")
        
        if cleaned_audio_path and Path(cleaned_audio_path).exists():
            try:
                Path(cleaned_audio_path).unlink()
                custom_logger.debug(f"Cleaned up temp file: {cleaned_audio_path}")
            except Exception as e:
                custom_logger.warning(f"Cleanup failed: {str(e)}")
        
        return BaseResponse.success_response(
            message="Transcription completed successfully",
            data={
                'audio_id_clean': audio_id_clean,
                'original_audio_id': str(audio_clean_record.original_audio_id),
                'status': 'transcribed',
                'summary': {
                    'total_tracks': len(transcribed_tracks),
                    'total_words': sum(len(t.get('words', [])) for t in transcribed_tracks),
                    'total_characters': sum(len(t.get('transcript', '')) for t in transcribed_tracks)
                },
                'tracks': [
                    {
                        'speaker_id': t.get('speaker_id', 0),
                        'transcript': t.get('transcript', ''),
                        'words_count': len(t.get('words', [])),
                        'words': t.get('words', []),
                        'char_count': len(t.get('transcript', ''))
                    }
                    for t in transcribed_tracks
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")
    
@router.post("/get_full_transcript")
async def get_full_transcript_api(
    api_key: str = Form(...),
    audio_id_clean: str = Form(...)
):
    """Get full transcript with word-level timings from database."""
    try:
        custom_logger.info(f"Getting full transcript: {audio_id_clean}")
        
        user_id = await validate_api_key(api_key)
        
        async with AsyncSessionLocal() as session:
            from app.api.model.speaker_track_model import SpeakerTrack
            from app.api.model.track_segment_model import TrackSegment
            from sqlalchemy import select
            
            audio_clean_record = await session.get(AudioClean, uuid.UUID(audio_id_clean))
            if not audio_clean_record:
                raise HTTPException(404, "Cleaned audio not found")
            
            audio_record = await session.get(AudioIngest, audio_clean_record.original_audio_id)
            if not audio_record:
                raise HTTPException(404, "Original audio not found")
            
            if str(audio_record.user_id) != user_id:
                raise HTTPException(403, "Access denied")
            
            result = await session.execute(
                select(SpeakerTrack).where(
                    SpeakerTrack.audio_id == audio_clean_record.original_audio_id
                ).order_by(SpeakerTrack.speaker_id)
            )
            speaker_tracks = result.scalars().all()
            
            if not speaker_tracks:
                raise HTTPException(404, "No tracks found")
            
            if not any(track.transcript for track in speaker_tracks):
                raise HTTPException(400, "Audio not transcribed yet. Call /transcribe first")
            
            tracks_data = []
            
            for track in speaker_tracks:
                segments_result = await session.execute(
                    select(TrackSegment).where(
                        TrackSegment.track_id == track.track_id
                    ).order_by(TrackSegment.start_time)
                )
                segments = segments_result.scalars().all()
                
                track_data = {
                    'track_id': str(track.track_id),
                    'speaker_id': track.speaker_id,
                    'track_type': track.track_type,
                    'total_duration': track.total_duration,
                    'coverage': track.coverage,
                    'ranges': track.ranges,
                    'transcript': {
                        'full_text': track.transcript or '',
                        'word_count': len(track.words) if track.words else 0,
                        'words': track.words or [],
                        'char_count': len(track.transcript) if track.transcript else 0
                    },
                    'segments': [
                        {
                            'segment_id': str(seg.segment_id),
                            'segment_type': seg.segment_type,
                            'start_time': seg.start_time,
                            'end_time': seg.end_time,
                            'duration': seg.duration,
                            'confidence': seg.confidence
                        }
                        for seg in segments
                    ]
                }
                tracks_data.append(track_data)
            
            total_words = sum(len(track.words) if track.words else 0 for track in speaker_tracks)
            total_chars = sum(len(track.transcript) if track.transcript else 0 for track in speaker_tracks)
            full_transcript = '\n\n'.join([
                f"Speaker {track.speaker_id}: {track.transcript}" 
                for track in speaker_tracks if track.transcript
            ])
            
            return BaseResponse.success_response(
                message="Full transcript retrieved successfully",
                data={
                    'audio_id_clean': audio_id_clean,
                    'original_audio_id': str(audio_clean_record.original_audio_id),
                    'status': 'transcribed',
                    'summary': {
                        'total_tracks': len(tracks_data),
                        'total_words': total_words,
                        'total_characters': total_chars,
                        'transcribed_tracks': len([t for t in speaker_tracks if t.transcript])
                    },
                    'full_transcript': full_transcript,
                    'tracks': tracks_data 
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        custom_logger.error(f"Get transcript failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")
@router.post("/normalize_text")
async def normalize_text_api(
    api_key: str = Form(...),
    text: str = Form(...)
):
    """
    Normalize Vietnamese text from speech-to-text
    
    Simple endpoint: input text, output normalized text
    
    Args:
        api_key: User API key
        text: Raw text from STT (no punctuation, fillers, etc)
        
    Returns:
        {
            'success': bool,
            'data': {
                'original_text': str,
                'normalized_text': str,
                'statistics': {
                    'original_length': int,
                    'normalized_length': int,
                    'reduction_ratio': float,
                    'processing_time': float,
                    'cost_estimate': float
                }
            }
        }
    """
    try:
        custom_logger.info(f"Normalizing text: {len(text)} chars")
        
        # Validate API key
        user_id = await validate_api_key(api_key)
        
        # Validate input
        if not text or len(text.strip()) < 10:
            raise HTTPException(400, "Text must be at least 10 characters")
        
        # Check Gemini API key
        if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
            raise HTTPException(500, "Gemini API key not configured")
        
        # Normalize with Gemini (single pass)
        normalizer = GeminiTextNormalizer(api_key=GEMINI_API_KEY)
        result = await normalizer.normalize_text(text)
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Normalization failed')
            custom_logger.error(f"Normalization error: {error_msg}")
            raise HTTPException(500, error_msg)
        
        # Return clean response
        return BaseResponse.success_response(
            message="Text normalized successfully",
            data={
                'original_text': text,
                'normalized_text': result['data']['normalized_text'],
                'statistics': result['data']['statistics']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        custom_logger.error(f"Normalization failed: {e}")
        custom_logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(500, f"Internal server error: {str(e)}")
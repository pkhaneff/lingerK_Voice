import uuid
import re
import unicodedata
from datetime import datetime
from typing import Dict, Any
from fastapi import UploadFile
from loguru import logger as custom_logger

from app.api.db.session import AsyncSessionLocal
from app.api.model.audio_model import AudioIngest
from app.api.services.audio_upload_service import AudioUploadService
from app.api.services.video_upload_service import VideoUploadService
from app.api.services.extract_audio import extract_audio_from_video_s3
from app.api.services.noise_reduction_service import process_audio_noise_reduction
from app.api.services.VAD_service import VoiceActivityDetectionService
from app.api.services.VAD_service import save_audio_segments
from app.core.config import AUDIO_EXTS, VIDEO_EXTS, AUDIO_FILE_SEIZE, VIDEO_FILE_SEIZE

class MP4Validator:
    """Simple MP4 validator using magic bytes"""
    
    @classmethod
    async def validate_mp4_file(cls, file) -> tuple[bool, str]:
        """Validate if file is a valid MP4"""
        try:
            # Read first 32 bytes for validation
            await file.seek(0)
            header_bytes = await file.read(32)
            
            # Reset to beginning after reading
            await file.seek(0)
            
            if len(header_bytes) < 8:
                return False, "File too small to validate MP4 format"
            
            # Look for 'ftyp' box which should be near the beginning
            for i in range(0, min(20, len(header_bytes) - 4)):
                chunk = header_bytes[i:i+4]
                if chunk == b'ftyp':
                    custom_logger.debug("Found MP4 ftyp signature")
                    return True, "Valid MP4 file detected"
            
            return False, "File does not appear to be a valid MP4 format"
            
        except Exception as e:
            custom_logger.error(f"Error validating MP4 file: {str(e)}")
            return False, f"MP4 validation error: {str(e)}"

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing Vietnamese accents and special characters
    Convert to ASCII-safe filename for S3 upload
    """
    custom_logger.info(f"clean_filename called with: {repr(filename)} (type: {type(filename)})")
    
    if not filename or not isinstance(filename, str):
        custom_logger.warning(f"Invalid filename input: {repr(filename)}, returning 'file'")
        return "file"
    
    try:
        custom_logger.debug(f"Original filename: {filename}")
        
        # Step 1: Normalize Unicode characters (Vietnamese accents -> base characters)
        normalized = unicodedata.normalize('NFD', filename)
        custom_logger.debug(f"After normalization: {repr(normalized)}")
        
        # Step 2: Remove accent marks (keep only ASCII characters)
        ascii_filename = normalized.encode('ascii', 'ignore').decode('ascii')
        custom_logger.debug(f"After ASCII conversion: {repr(ascii_filename)}")
        
        # Step 3: Replace remaining special characters with underscore
        clean_filename = re.sub(r'[^\w\.\-]', '_', ascii_filename)
        custom_logger.debug(f"After regex substitution: {repr(clean_filename)}")
        
        # Step 4: Remove multiple underscores
        clean_filename = re.sub(r'_{2,}', '_', clean_filename)
        custom_logger.debug(f"After removing multiple underscores: {repr(clean_filename)}")
        
        # Step 5: Remove leading/trailing underscores
        clean_filename = clean_filename.strip('_')
        custom_logger.debug(f"After stripping underscores: {repr(clean_filename)}")
        
        # Step 6: Ensure we still have a valid filename
        if not clean_filename or clean_filename == '.':
            custom_logger.warning(f"Empty filename after cleaning, returning 'file'")
            return "file"
        
        custom_logger.info(f"Cleaned filename: {repr(clean_filename)}")
        return clean_filename
        
    except Exception as e:
        custom_logger.error(f"Error in clean_filename: {str(e)} | Input: {repr(filename)}")
        try:
            safe_filename = re.sub(r'[^\w\.\-]', '_', str(filename))
            custom_logger.info(f"Fallback filename: {repr(safe_filename)}")
            return safe_filename or "file"
        except Exception as fallback_error:
            custom_logger.error(f"Even fallback failed: {str(fallback_error)}")
            return "file"

async def save_audio_metadata_to_db(audio_data: Dict[str, Any], user_id: str) -> uuid.UUID:
    from app.api.model.audio_model import AudioIngest
    from app.api.db.session import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        codec = detect_codec_from_filename(audio_data['file_name'])
        
        audio_record = AudioIngest(
            file_name=audio_data['file_name'],
            storage_uri=audio_data['storage_uri'],
            user_id=uuid.UUID(user_id),
            is_video=audio_data['is_video'],
            status='uploaded',
            codec=codec
        )
        
        session.add(audio_record)
        await session.commit()
        await session.refresh(audio_record)
        
        return audio_record.audio_id

def detect_codec_from_filename(filename: str) -> str:
    """Detect codec from file extension"""
    if filename.lower().endswith('.ogg'):
        return 'ogg'
    elif filename.lower().endswith('.mp3'):
        return 'mp3'
    elif filename.lower().endswith('.wav'):
        return 'wav'
    elif filename.lower().endswith('.m4a'):
        return 'aac'
    return 'unknown'

async def update_audio_with_noise_analysis(audio_id: uuid.UUID, noise_result: Dict[str, Any]) -> bool:
    """Update audio record with noise analysis results"""
    from app.api.model.audio_model import AudioIngest
    from app.api.db.session import AsyncSessionLocal
    
    custom_logger.info(f"Starting update for audio_id: {audio_id}")
    custom_logger.info(f"noise_result keys: {list(noise_result.keys())}")
    custom_logger.info(f"noise_result data: {noise_result}")
    
    try:
        async with AsyncSessionLocal() as session:
            # Tìm record
            custom_logger.info(f"Looking for audio record: {audio_id}")
            audio_record = await session.get(AudioIngest, audio_id)
            
            if not audio_record:
                custom_logger.error(f"Audio record {audio_id} NOT FOUND!")
                return False
            
            custom_logger.info(f"Found audio record: {audio_record.audio_id}")
            
            # Update basic fields
            audio_record.preprocessed = True
            audio_record.processed_time = datetime.utcnow()
            audio_record.status = 'completed' if noise_result['success'] else 'failed'
            custom_logger.info(f"Updated basic fields")
            
            # Update duration
            if 'noise_analysis' in noise_result and 'total_duration' in noise_result['noise_analysis']:
                old_duration = audio_record.duration
                audio_record.duration = noise_result['noise_analysis']['total_duration']
                custom_logger.info(f"Duration: {old_duration} -> {audio_record.duration}")
            
            if 'noise_analysis' in noise_result:
                old_noise = audio_record.noise_analysis
                audio_record.noise_analysis = noise_result['noise_analysis']
                custom_logger.info(f"noise_analysis: {old_noise} -> {audio_record.noise_analysis}")
            else:
                custom_logger.error(f"NO 'noise_analysis' key in noise_result!")
            
            # Commit changes
            custom_logger.info(f"Committing changes...")
            await session.commit()
            custom_logger.info(f"Successfully committed changes for {audio_id}")
            
            # Verify the update
            await session.refresh(audio_record)
            custom_logger.info(f"Verified - noise_analysis in DB: {audio_record.noise_analysis}")
            custom_logger.info(f"Verified - duration in DB: {audio_record.duration}")
            
            return True
            
    except Exception as e:
        custom_logger.error(f"Exception in update_audio_with_noise_analysis: {str(e)}", exc_info=True)
        return False

async def upload_audio_video_file(file: UploadFile, user_id: str) -> Dict[str, Any]:
    """
    Main upload function with automatic audio extraction for video files
    and noise reduction processing for all audio
    - Audio: mp3, wav, ogg, m4a, aac (max 50MB) 
    - Video: mp4 only (max 1GB) with audio extraction to S3
    """
    try:
        custom_logger.info(f"Upload started for file: {file.filename}")
        
        if not file.filename:
            custom_logger.error("Filename is None or empty")
            return {'success': False, 'error': 'Filename is required'}
        
        custom_logger.debug(f"Processing filename: {repr(file.filename)}")
        
        # Extract file extension
        try:
            filename_parts = file.filename.split(".")
            custom_logger.debug(f"Filename parts: {filename_parts}")
            file_extension = filename_parts[-1].lower()
            custom_logger.debug(f"File extension: {repr(file_extension)}")
        except Exception as e:
            custom_logger.error(f"Error splitting filename: {str(e)} | Filename: {repr(file.filename)}")
            return {'success': False, 'error': f'Invalid filename format: {str(e)}'}
        
        # Determine file type with MP4-specific validation
        if file_extension in AUDIO_EXTS:
            file_type = 'audio'
            max_size = AUDIO_FILE_SEIZE
            custom_logger.info(f"Detected audio file with extension: {file_extension}")
            
        elif file_extension in VIDEO_EXTS:  # Only 'mp4'
            file_type = 'video'
            max_size = VIDEO_FILE_SEIZE
            custom_logger.info(f"Detected video file with extension: {file_extension}")
            
        else:
            # Specific error messages for better UX
            if file_extension in ['avi', 'mov', 'wmv', 'flv', 'mkv', 'webm']:
                custom_logger.warning(f"Unsupported video format: {file_extension}")
                return {
                    'success': False, 
                    'error': 'Only MP4 format is supported for video files. Please convert your video to MP4.'
                }
            else:
                supported_audio = ", ".join(AUDIO_EXTS)
                custom_logger.warning(f"Unsupported file extension: {file_extension}")
                return {
                    'success': False, 
                    'error': f'Unsupported file format. Supported formats: Audio: {supported_audio} | Video: MP4 only'
                }
        
        # For video files: validate MP4 format using magic bytes
        validation_message = 'N/A'
        if file_type == 'video':
            custom_logger.debug("Validating MP4 format...")
            is_valid_mp4, validation_message = await MP4Validator.validate_mp4_file(file)
            
            if not is_valid_mp4:
                custom_logger.warning(f"MP4 validation failed: {validation_message}")
                return {
                    'success': False, 
                    'error': f'Invalid MP4 file: {validation_message}. Please ensure your file is a valid MP4 format.'
                }
            
            custom_logger.info(f"MP4 validation passed: {validation_message}")
        
        # Get file size efficiently
        custom_logger.debug("Getting file size...")
        file_size = file.size
        custom_logger.info(f"File size: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
        
        # Validate file size
        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            custom_logger.error(f"{file_type.title()} file too large: {file_size:,} > {max_size:,}")
            return {
                'success': False, 
                'error': f"{file_type.title()} file too large. Maximum size: {max_mb:.0f}MB"
            }
        
        # Generate S3 key
        custom_logger.debug("Generating S3 key...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            custom_logger.debug(f"Timestamp: {timestamp}, Unique ID: {unique_id}")
            
            raw_filename = clean_filename(file.filename)
            custom_logger.debug(f"Raw filename after cleaning: {repr(raw_filename)}")
            
            s3_key = f"{timestamp}_{unique_id}_{raw_filename}"
            custom_logger.info(f"Generated S3 key: {repr(s3_key)}")
            
        except Exception as e:
            custom_logger.error(f"Error generating S3 key: {str(e)} | Original filename: {repr(file.filename)}")
            return {'success': False, 'error': f'Error generating S3 key: {str(e)}'}
        
        # Upload based on file type
        custom_logger.info(f"Starting {file_type} upload with key: {s3_key}")
        try:
            if file_type == 'audio':
                # For audio: read content into memory (smaller files)
                await file.seek(0)
                file_content = await file.read()
                custom_logger.debug(f"Read audio content: {len(file_content):,} bytes")
                
                result = await AudioUploadService().upload(file, s3_key, file_content)
                
            else:  # video
                # For video: stream upload without loading full content into memory
                await file.seek(0)
                custom_logger.debug(f"Starting video stream upload: {file_size:,} bytes")
                
                result = await VideoUploadService().upload(file, s3_key, file_size)
            
            custom_logger.info(f"Upload service result: {result}")
            
        except Exception as e:
            custom_logger.error(f"Upload service error: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'Upload service error: {str(e)}'}
        
        # Process result
        if not result.get('success', False):
            custom_logger.error(f"Upload failed: {result.get('error', 'Unknown error')}")
            return {'success': False, 'error': result.get('error', 'Upload failed with unknown error')}
        
        custom_logger.info("Upload successful")
        
        # Determine upload method
        upload_method = "direct"
        if file_type == 'video' and result.get('parts_count', 1) > 1:
            upload_method = "multipart"
        
        # Prepare base response data
        response_data = {
            'file_name': file.filename,
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'file_type': file_type,
            's3_key': result['s3_key'],
            's3_url': result['s3_url'],
            'bucket_name': result['bucket_name'],
            'upload_method': upload_method,
            'parts_count': result.get('parts_count', 1),
            'validation': validation_message
        }
        
        # Save audio metadata to database first
        audio_s3_key = result['s3_key']  # For audio files, this is the direct key
        audio_metadata = {
            'file_name': file.filename,
            'storage_uri': result['s3_url'],
            'file_size': file_size,
            'file_type': file_type,
            'is_video': file_type == 'video'
        }
        
        try:
            # Save to database
            audio_id = await save_audio_metadata_to_db(audio_metadata, user_id)
            response_data['audio_id'] = str(audio_id)
            custom_logger.info(f"Saved audio metadata to DB with ID: {audio_id}")
        except Exception as e:
            custom_logger.error(f"Failed to save to database: {str(e)}", exc_info=True)
            # Continue processing but note the DB error
            response_data['audio_id'] = None
            response_data['db_error'] = f"Database save failed: {str(e)}"
        
        # For video files: extract audio immediately after upload
        if file_type == 'video':
            custom_logger.info("Starting audio extraction from uploaded video...")
            try:
                # Generate audio filename (same as video but without extension)
                audio_filename = f"{timestamp}_{unique_id}_{raw_filename.rsplit('.', 1)[0]}"
                
                # Extract audio
                extraction_result = await extract_audio_from_video_s3(
                    video_s3_key=result['s3_key'],
                    audio_filename=audio_filename
                )
                
                if extraction_result.get('success', False):
                    custom_logger.info(f"Audio extraction successful: {extraction_result['audio_s3_key']}")
                    
                    # Update audio S3 key for noise reduction
                    audio_s3_key = extraction_result['audio_s3_key']
                    
                    # Add audio info to response
                    response_data.update({
                        'audio_extraction': 'success',
                        'audio_s3_key': extraction_result['audio_s3_key'],
                        'audio_s3_url': extraction_result['audio_s3_url'],
                        'audio_size': extraction_result.get('audio_size', 0),
                        'audio_size_mb': extraction_result.get('audio_size_mb', 0.0)
                    })
                    
                else:
                    custom_logger.error(f"Audio extraction failed: {extraction_result.get('error', 'Unknown extraction error')}")
                    
                    # Still return video upload success, but note extraction failure
                    response_data.update({
                        'audio_extraction': 'failed',
                        'audio_extraction_error': extraction_result.get('error', 'Unknown extraction error'),
                        'audio_s3_key': None,
                        'audio_s3_url': None
                    })
                    
                    # Skip further processing and return early
                    return {
                        'success': True,
                        'data': response_data
                    }
                    
            except Exception as e:
                custom_logger.error(f"Audio extraction exception: {str(e)}", exc_info=True)
                response_data.update({
                    'audio_extraction': 'failed',
                    'audio_extraction_error': f"Extraction exception: {str(e)}",
                    'audio_s3_key': None,
                    'audio_s3_url': None
                })
                
                # Skip further processing and return early
                return {
                    'success': True,
                    'data': response_data
                }
        
        # Apply noise reduction to audio (for both audio files and extracted audio from video)
        # Only proceed if we have valid audio_id and audio_s3_key
        if response_data.get('audio_id') and audio_s3_key:
            custom_logger.info(f"Starting noise reduction for audio: {audio_s3_key}")
            try:
                noise_result = await process_audio_noise_reduction(response_data['audio_id'], audio_s3_key)
                
                if noise_result and noise_result.get('success', False):
                    custom_logger.info("Noise reduction completed successfully")
                    
                    # Update database with noise analysis
                    try:
                        db_update_success = await update_audio_with_noise_analysis(uuid.UUID(response_data['audio_id']), noise_result)
                    except Exception as db_e:
                        custom_logger.error(f"Database update failed: {str(db_e)}")
                        db_update_success = False
                    
                    # Add noise reduction info to response
                    response_data.update({
                        'noise_reduction': 'success',
                        'cleaned_audio_key': noise_result.get('cleaned_audio_key'),
                        'noise_analysis': {
                            'noise_ratio': noise_result.get('noise_analysis', {}).get('noise_ratio', 0.0),
                            'noise_duration': noise_result.get('noise_analysis', {}).get('noise_duration', 0.0),
                            'segments_count': noise_result.get('noise_analysis', {}).get('segments_count', 0),
                            'total_duration': noise_result.get('noise_analysis', {}).get('total_duration', 0.0)
                        },
                        'db_updated': db_update_success
                    })
                    
                    # Run VAD analysis on cleaned audio if noise reduction was successful
                    cleaned_audio_key = noise_result.get('cleaned_audio_key')
                    if cleaned_audio_key:
                        custom_logger.info(f"Starting VAD analysis for cleaned audio: {cleaned_audio_key}")
                        try:
                            vad_service = VoiceActivityDetectionService()
                            vad_result = await vad_service.detect_voice_activity(cleaned_audio_key)
                            
                            if vad_result and vad_result.get('success', False):
                                custom_logger.info(f"VAD analysis completed: {len(vad_result.get('segments', []))} voice segments")
                                response_data.update({
                                    'vad_analysis': 'completed',
                                    'voice_segments': len(vad_result.get('segments', [])),
                                    'voice_activity_ratio': vad_result.get('voice_activity_ratio', 0.0),
                                    'total_voice_duration': vad_result.get('total_voice_duration', 0.0)
                                })
                                
                                # Store segments in database (implement repository layer)
                                # await save_audio_segments(audio_id, vad_result['segments'])
                            else:
                                custom_logger.error(f"VAD analysis failed: {vad_result.get('error', 'Unknown VAD error') if vad_result else 'VAD returned None'}")
                                response_data.update({
                                    'vad_analysis': 'failed',
                                    'vad_error': vad_result.get('error', 'Unknown VAD error') if vad_result else 'VAD service returned None'
                                })
                        except Exception as e:
                            custom_logger.error(f"VAD processing error: {str(e)}", exc_info=True)
                            response_data.update({
                                'vad_analysis': 'failed',
                                'vad_error': str(e)
                            })
                    
                else:
                    custom_logger.error(f"Noise reduction failed: {noise_result.get('error', 'Unknown noise reduction error') if noise_result else 'Noise reduction returned None'}")
                    response_data.update({
                        'noise_reduction': 'failed',
                        'noise_reduction_error': noise_result.get('error', 'Unknown noise reduction error') if noise_result else 'Noise reduction service returned None'
                    })
                    
            except Exception as e:
                custom_logger.error(f"Noise reduction exception: {str(e)}", exc_info=True)
                response_data.update({
                    'noise_reduction': 'failed',
                    'noise_reduction_error': f"Noise reduction exception: {str(e)}"
                })
        else:
            custom_logger.warning(f"Skipping noise reduction - audio_id: {response_data.get('audio_id')}, audio_s3_key: {audio_s3_key}")
            response_data.update({
                'noise_reduction': 'skipped',
                'noise_reduction_reason': 'Missing audio_id or audio_s3_key'
            })
        
        # ✅ ALWAYS RETURN SUCCESS RESPONSE
        return {
            'success': True,
            'data': response_data
        }
        
    except Exception as e:
        # ✅ ALWAYS RETURN ERROR RESPONSE 
        custom_logger.error(f"Upload failed with exception: {str(e)}", exc_info=True)
        return {
            'success': False, 
            'error': f"Upload failed: {str(e)}"
        }
async def upload_audio_video_file(file: UploadFile, user_id: str) -> Dict[str, Any]:
    """
    Main upload function with automatic audio extraction for video files
    and noise reduction processing for all audio
    - Audio: mp3, wav, ogg, m4a, aac (max 50MB) 
    - Video: mp4 only (max 1GB) with audio extraction to S3
    """
    try:
        custom_logger.info(f"Upload started for file: {file.filename}")
        
        if not file.filename:
            custom_logger.error("Filename is None or empty")
            return {'success': False, 'error': 'Filename is required'}
        
        custom_logger.debug(f"Processing filename: {repr(file.filename)}")
        
        # Extract file extension
        try:
            filename_parts = file.filename.split(".")
            custom_logger.debug(f"Filename parts: {filename_parts}")
            file_extension = filename_parts[-1].lower()
            custom_logger.debug(f"File extension: {repr(file_extension)}")
        except Exception as e:
            custom_logger.error(f"Error splitting filename: {str(e)} | Filename: {repr(file.filename)}")
            return {'success': False, 'error': f'Invalid filename format: {str(e)}'}
        
        # Determine file type with MP4-specific validation
        if file_extension in AUDIO_EXTS:
            file_type = 'audio'
            max_size = AUDIO_FILE_SEIZE
            custom_logger.info(f"Detected audio file with extension: {file_extension}")
            
        elif file_extension in VIDEO_EXTS:  # Only 'mp4'
            file_type = 'video'
            max_size = VIDEO_FILE_SEIZE
            custom_logger.info(f"Detected video file with extension: {file_extension}")
            
        else:
            # Specific error messages for better UX
            if file_extension in ['avi', 'mov', 'wmv', 'flv', 'mkv', 'webm']:
                custom_logger.warning(f"Unsupported video format: {file_extension}")
                return {
                    'success': False, 
                    'error': 'Only MP4 format is supported for video files. Please convert your video to MP4.'
                }
            else:
                supported_audio = ", ".join(AUDIO_EXTS)
                custom_logger.warning(f"Unsupported file extension: {file_extension}")
                return {
                    'success': False, 
                    'error': f'Unsupported file format. Supported formats: Audio: {supported_audio} | Video: MP4 only'
                }
        
        # For video files: validate MP4 format using magic bytes
        validation_message = 'N/A'
        if file_type == 'video':
            custom_logger.debug("Validating MP4 format...")
            is_valid_mp4, validation_message = await MP4Validator.validate_mp4_file(file)
            
            if not is_valid_mp4:
                custom_logger.warning(f"MP4 validation failed: {validation_message}")
                return {
                    'success': False, 
                    'error': f'Invalid MP4 file: {validation_message}. Please ensure your file is a valid MP4 format.'
                }
            
            custom_logger.info(f"MP4 validation passed: {validation_message}")
        
        # Get file size efficiently
        custom_logger.debug("Getting file size...")
        file_size = file.size
        custom_logger.info(f"File size: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
        
        # Validate file size
        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            custom_logger.error(f"{file_type.title()} file too large: {file_size:,} > {max_size:,}")
            return {
                'success': False, 
                'error': f"{file_type.title()} file too large. Maximum size: {max_mb:.0f}MB"
            }
        
        # Generate S3 key
        custom_logger.debug("Generating S3 key...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            custom_logger.debug(f"Timestamp: {timestamp}, Unique ID: {unique_id}")
            
            raw_filename = clean_filename(file.filename)
            custom_logger.debug(f"Raw filename after cleaning: {repr(raw_filename)}")
            
            s3_key = f"{timestamp}_{unique_id}_{raw_filename}"
            custom_logger.info(f"Generated S3 key: {repr(s3_key)}")
            
        except Exception as e:
            custom_logger.error(f"Error generating S3 key: {str(e)} | Original filename: {repr(file.filename)}")
            return {'success': False, 'error': f'Error generating S3 key: {str(e)}'}
        
        # Upload based on file type
        custom_logger.info(f"Starting {file_type} upload with key: {s3_key}")
        try:
            if file_type == 'audio':
                # For audio: read content into memory (smaller files)
                await file.seek(0)
                file_content = await file.read()
                custom_logger.debug(f"Read audio content: {len(file_content):,} bytes")
                
                result = await AudioUploadService().upload(file, s3_key, file_content)
                
            else:  # video
                # For video: stream upload without loading full content into memory
                await file.seek(0)
                custom_logger.debug(f"Starting video stream upload: {file_size:,} bytes")
                
                result = await VideoUploadService().upload(file, s3_key, file_size)
            
            custom_logger.info(f"Upload service result: {result}")
            
        except Exception as e:
            custom_logger.error(f"Upload service error: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'Upload service error: {str(e)}'}
        
        # Process result
        if not result.get('success', False):
            custom_logger.error(f"Upload failed: {result.get('error', 'Unknown error')}")
            return {'success': False, 'error': result.get('error', 'Upload failed with unknown error')}
        
        custom_logger.info("Upload successful")
        
        # Determine upload method
        upload_method = "direct"
        if file_type == 'video' and result.get('parts_count', 1) > 1:
            upload_method = "multipart"
        
        # Prepare base response data
        response_data = {
            'file_name': file.filename,
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'file_type': file_type,
            's3_key': result['s3_key'],
            's3_url': result['s3_url'],
            'bucket_name': result['bucket_name'],
            'upload_method': upload_method,
            'parts_count': result.get('parts_count', 1),
            'validation': validation_message
        }
        
        audio_s3_key = result['s3_key']  
        audio_metadata = {
            'file_name': file.filename,
            'storage_uri': result['s3_url'],
            'file_size': file_size,
            'file_type': file_type,
            'is_video': file_type == 'video'
        }
        
        try:
            # Save to database
            audio_id = await save_audio_metadata_to_db(audio_metadata, user_id)
            response_data['audio_id'] = str(audio_id)
            custom_logger.info(f"Saved audio metadata to DB with ID: {audio_id}")
        except Exception as e:
            custom_logger.error(f"Failed to save to database: {str(e)}", exc_info=True)
            response_data['audio_id'] = None
            response_data['db_error'] = f"Database save failed: {str(e)}"
        
        if file_type == 'video':
            custom_logger.info("Starting audio extraction from uploaded video...")
            try:
                audio_filename = f"{timestamp}_{unique_id}_{raw_filename.rsplit('.', 1)[0]}"
                
                extraction_result = await extract_audio_from_video_s3(
                    video_s3_key=result['s3_key'],
                    audio_filename=audio_filename
                )
                
                if extraction_result.get('success', False):
                    custom_logger.info(f"Audio extraction successful: {extraction_result['audio_s3_key']}")
                    
                    audio_s3_key = extraction_result['audio_s3_key']
                    
                    response_data.update({
                        'audio_extraction': 'success',
                        'audio_s3_key': extraction_result['audio_s3_key'],
                        'audio_s3_url': extraction_result['audio_s3_url'],
                        'audio_size': extraction_result.get('audio_size', 0),
                        'audio_size_mb': extraction_result.get('audio_size_mb', 0.0)
                    })
                    
                else:
                    custom_logger.error(f"Audio extraction failed: {extraction_result.get('error', 'Unknown extraction error')}")
                    
                    response_data.update({
                        'audio_extraction': 'failed',
                        'audio_extraction_error': extraction_result.get('error', 'Unknown extraction error'),
                        'audio_s3_key': None,
                        'audio_s3_url': None
                    })
                    
                    return {
                        'success': True,
                        'data': response_data
                    }
                    
            except Exception as e:
                custom_logger.error(f"Audio extraction exception: {str(e)}", exc_info=True)
                response_data.update({
                    'audio_extraction': 'failed',
                    'audio_extraction_error': f"Extraction exception: {str(e)}",
                    'audio_s3_key': None,
                    'audio_s3_url': None
                })
                
                return {
                    'success': True,
                    'data': response_data
                }
        
        if response_data.get('audio_id') and audio_s3_key:
            custom_logger.info(f"Starting noise reduction for audio: {audio_s3_key}")
            try:
                noise_result = await process_audio_noise_reduction(response_data['audio_id'], audio_s3_key)
                
                if noise_result and noise_result.get('success', False):
                    custom_logger.info("Noise reduction completed successfully")
                    
                    # Update database with noise analysis
                    try:
                        db_update_success = await update_audio_with_noise_analysis(uuid.UUID(response_data['audio_id']), noise_result)
                    except Exception as db_e:
                        custom_logger.error(f"Database update failed: {str(db_e)}")
                        db_update_success = False
                    
                    # Add noise reduction info to response
                    response_data.update({
                        'noise_reduction': 'success',
                        'cleaned_audio_key': noise_result.get('cleaned_audio_key'),
                        'noise_analysis': {
                            'noise_ratio': noise_result.get('noise_analysis', {}).get('noise_ratio', 0.0),
                            'noise_duration': noise_result.get('noise_analysis', {}).get('noise_duration', 0.0),
                            'segments_count': noise_result.get('noise_analysis', {}).get('segments_count', 0),
                            'total_duration': noise_result.get('noise_analysis', {}).get('total_duration', 0.0)
                        },
                        'db_updated': db_update_success
                    })
                    
                    cleaned_audio_key = noise_result.get('cleaned_audio_key')
                    if cleaned_audio_key:
                        custom_logger.info(f"Starting VAD analysis for cleaned audio: {cleaned_audio_key}")
                        try:
                            vad_service = VoiceActivityDetectionService()
                            vad_result = await vad_service.detect_voice_activity(cleaned_audio_key)
                            
                            if vad_result and vad_result.get('success', False):
                                custom_logger.info(f"VAD analysis completed: {len(vad_result.get('segments', []))} voice segments")
                                segments_saved = await save_audio_segments(uuid.UUID(response_data['audio_id']), vad_result['segments'])
                                response_data.update({
                                    'vad_analysis': 'completed',
                                    'voice_segments': len(vad_result.get('segments', [])),
                                    'voice_activity_ratio': vad_result.get('voice_activity_ratio', 0.0),
                                    'total_voice_duration': vad_result.get('total_voice_duration', 0.0),
                                    'segments_saved_to_db': segments_saved
                                })
                                
                            else:
                                custom_logger.error(f"VAD analysis failed: {vad_result.get('error', 'Unknown VAD error') if vad_result else 'VAD returned None'}")
                                response_data.update({
                                    'vad_analysis': 'failed',
                                    'vad_error': vad_result.get('error', 'Unknown VAD error') if vad_result else 'VAD service returned None'
                                })
                        except Exception as e:
                            custom_logger.error(f"VAD processing error: {str(e)}", exc_info=True)
                            response_data.update({
                                'vad_analysis': 'failed',
                                'vad_error': str(e)
                            })
                    
                else:
                    custom_logger.error(f"Noise reduction failed: {noise_result.get('error', 'Unknown noise reduction error') if noise_result else 'Noise reduction returned None'}")
                    response_data.update({
                        'noise_reduction': 'failed',
                        'noise_reduction_error': noise_result.get('error', 'Unknown noise reduction error') if noise_result else 'Noise reduction service returned None'
                    })
                    
            except Exception as e:
                custom_logger.error(f"Noise reduction exception: {str(e)}", exc_info=True)
                response_data.update({
                    'noise_reduction': 'failed',
                    'noise_reduction_error': f"Noise reduction exception: {str(e)}"
                })
        else:
            custom_logger.warning(f"Skipping noise reduction - audio_id: {response_data.get('audio_id')}, audio_s3_key: {audio_s3_key}")
            response_data.update({
                'noise_reduction': 'skipped',
                'noise_reduction_reason': 'Missing audio_id or audio_s3_key'
            })
        
        return {
            'success': True,
            'data': response_data
        }
        
    except Exception as e:
        custom_logger.error(f"Upload failed with exception: {str(e)}", exc_info=True)
        return {
            'success': False, 
            'error': f"Upload failed: {str(e)}"
        }
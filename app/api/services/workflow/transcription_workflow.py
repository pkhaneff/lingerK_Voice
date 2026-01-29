import uuid
import tempfile
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from loguru import logger as custom_logger

from app.api.services.processing.transcription_service import TranscriptionService
from app.api.services.metadata.db_saver import DBSaver
from app.api.infra.aws.s3.repository.object import get_object
from app.api.infra.aws.s3 import s3_bucket

class TranscriptionWorkflowService:
    """
    Orchestrates the transcription workflow:
    Fetch Tracks -> Download Clean Audio -> Transcribe -> Save Results
    """
    
    def __init__(
        self,
        transcription_service: TranscriptionService,
        db_saver: DBSaver
    ):
        self.transcription_service = transcription_service
        self.db_saver = db_saver

    async def transcribe_audio(
        self, 
        audio_id: str, 
        audio_id_clean: str,
        cleaned_s3_key: str, 
        speaker_tracks: List[Dict],
        separated_regions: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute transcription for processed audio.
        """
        cleaned_audio_path = None
        if separated_regions is None:
            separated_regions = []
            
        try:
            custom_logger.info(f"Starting transcription workflow for {audio_id_clean}")

            # Download Audio (Async)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                cleaned_audio_path = f.name
            
            try:
                audio_object = await get_object(cleaned_s3_key, s3_bucket)
                
                loop = asyncio.get_event_loop()
                def _write_file():
                    with open(cleaned_audio_path, 'wb') as f:
                        f.write(audio_object.body.read())
                await loop.run_in_executor(None, _write_file)
                
                custom_logger.info(f"Cleaned audio downloaded to: {cleaned_audio_path}")
                
            except Exception as e:
                return {'success': False, 'error': f"Failed to download audio: {str(e)}"}

            # Transcribe
            transcription_result = await self.transcription_service.transcribe_all_tracks(
                cleaned_audio_path,
                speaker_tracks,
                separated_regions
            )
            
            if not transcription_result['success']:
                return {'success': False, 'error': transcription_result['error']}
            
            transcribed_tracks = transcription_result['data']['tracks']
            
            # Update DB
            update_result = await self.db_saver.update_track_transcripts(
                audio_id=audio_id,
                transcribed_tracks=transcribed_tracks
            )
            
            if not update_result['success']:
                custom_logger.warning(f"Failed to update transcripts: {update_result['error']}")

            return {
                'success': True,
                'data': {
                    'audio_id_clean': audio_id_clean,
                    'original_audio_id': audio_id,
                    'status': 'transcribed',
                    'summary': {
                        'total_tracks': len(transcribed_tracks),
                        'total_words': sum(len(t.get('words', [])) for t in transcribed_tracks),
                    },
                    'tracks': [
                        {
                            'speaker_id': t.get('speaker_id', 0),
                            'transcript': t.get('transcript', ''),
                            'words_count': len(t.get('words', [])),
                            # Limit data size for response if needed
                        }
                        for t in transcribed_tracks
                    ]
                }
            }

        except Exception as e:
            custom_logger.error(f"Transcription workflow failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
            
        finally:
            if cleaned_audio_path and Path(cleaned_audio_path).exists():
                try:
                    Path(cleaned_audio_path).unlink()
                except:
                    pass

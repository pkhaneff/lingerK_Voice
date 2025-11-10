from typing import Dict, Any, List
from loguru import logger as custom_logger
import numpy as np

from app.api.services.processing.track_audio_generator import TrackAudioGenerator
from app.data_model.implementations.whisper_transcriber import WhisperTranscriber


class TranscriptionService:
    """Service to transcribe all speaker tracks."""
    
    def __init__(self):
        self.generator = TrackAudioGenerator()
        self.transcriber = WhisperTranscriber()
        self._transcriber_loaded = False
    
    async def transcribe_all_tracks(
        self,
        cleaned_audio_path: str,
        speaker_tracks: List[Dict],
        separated_regions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Transcribe all speaker tracks using hybrid approach.
        
        Args:
            cleaned_audio_path: Path to cleaned audio
            speaker_tracks: List of speaker tracks with segments
            separated_regions: Output from Conv-TasNet separation
        """
        track_paths = None
        
        try:
            custom_logger.info(f"Starting transcription for {len(speaker_tracks)} tracks")
            
            if not self._transcriber_loaded:
                custom_logger.info("Loading Whisper model...")
                if not await self.transcriber.load_model():
                    return {'success': False, 'data': None, 'error': 'Whisper model load failed'}
                self._transcriber_loaded = True
            
            if (len(speaker_tracks) == 1 and 
                len(speaker_tracks[0].get('segments', [])) == 1 and
                not separated_regions):
                
                custom_logger.info("Single speaker detected - using direct transcription (bypass track generation)")
                track = speaker_tracks[0]
                speaker_id = track['speaker_id']
                
                result = await self.transcriber.transcribe_track(cleaned_audio_path, language='vi')
                
                if not result['success']:
                    custom_logger.error(f"Direct transcription failed: {result['error']}")
                    return {'success': False, 'data': None, 'error': result['error']}
                
                track['transcript'] = result['data']['text']
                track['words'] = result['data']['words']
                
                custom_logger.info(
                    f"Speaker {speaker_id}: {len(result['data']['words'])} words, "
                    f"text length: {len(result['data']['text'])} chars (direct transcription)"
                )
                
                return {
                    'success': True,
                    'data': {'tracks': [track]},
                    'error': None
                }
            
            custom_logger.info("🔄 Multi-speaker/complex case - using track audio generation")
            track_paths = self.generator.generate_tracks(
                cleaned_audio_path,
                speaker_tracks,
                separated_regions
            )
            
            custom_logger.info(f"Generated {len(track_paths)} track files")
            
            transcribed_tracks = []
            
            for track in speaker_tracks:
                speaker_id = track['speaker_id']
                audio_path = track_paths.get(speaker_id)
                
                if not audio_path:
                    custom_logger.warning(f"No audio file for speaker {speaker_id}")
                    continue
                
                import librosa
                try:
                    audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
                    duration = len(audio_data) / sr
                    non_zero = np.count_nonzero(np.abs(audio_data) > 0.001)
                    voice_duration = non_zero / sr
                    silence_duration = duration - voice_duration
                    
                    custom_logger.info(f"\nðŸŽ¤ Transcribing Speaker {speaker_id}:")
                    custom_logger.info(f"   Audio file: {audio_path}")
                    custom_logger.info(f"   Total duration: {duration:.2f}s")
                    custom_logger.info(f"   Voice: {voice_duration:.2f}s ({voice_duration/duration*100:.1f}%)")
                    custom_logger.info(f"   Silence: {silence_duration:.2f}s ({silence_duration/duration*100:.1f}%)")
                except Exception as e:
                    custom_logger.warning(f"Could not analyze audio: {e}")
                
                result = await self.transcriber.transcribe_track(audio_path, language='vi')
                
                if not result['success']:
                    custom_logger.error(f"Transcription failed for speaker {speaker_id}: {result['error']}")
                    continue
                
                track['transcript'] = result['data']['text']
                track['words'] = result['data']['words']
                
                custom_logger.info(
                    f"Ã¢Å“â€¦ Speaker {speaker_id}: {len(result['data']['words'])} words, "
                    f"text length: {len(result['data']['text'])} chars"
                )
                
                transcribed_tracks.append(track)
            
            custom_logger.info("Cleaning up temp files...")
            self.generator.cleanup(track_paths)
            
            if len(transcribed_tracks) == 0:
                return {
                    'success': False,
                    'data': None,
                    'error': 'No tracks were successfully transcribed'
                }
            
            custom_logger.info(f"Ã¢Å“â€¦ Transcription completed for {len(transcribed_tracks)} tracks")
            
            return {
                'success': True,
                'data': {'tracks': transcribed_tracks},
                'error': None
            }
            
        except Exception as e:
            custom_logger.error(f"Transcription service failed: {e}", exc_info=True)
            
            if track_paths:
                try:
                    self.generator.cleanup(track_paths)
                except:
                    pass
            
            return {'success': False, 'data': None, 'error': str(e)}
    
    def cleanup(self):
        """Cleanup all resources."""
        try:
            self.generator.cleanup()
            self.transcriber.cleanup()
            self._transcriber_loaded = False
            custom_logger.info("Transcription service cleaned up")
        except Exception as e:
            custom_logger.warning(f"Service cleanup error: {e}")
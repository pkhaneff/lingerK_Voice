from typing import List, Dict
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.model.speaker_track_model import SpeakerTrack
from app.api.model.audio_clean import AudioClean
from app.api.services.semantic.sentence_tokenizer import SentenceTokenizer
from app.api.services.semantic.text_normalizer import TextNormalizer
from app.api.services.semantic.document_type_detector import DocumentTypeDetector


class TextPreparationService:
    def __init__(
        self,
        session: AsyncSession,
        tokenizer: SentenceTokenizer,
        normalizer: TextNormalizer,
        type_detector: DocumentTypeDetector
    ):
        self.session = session
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.type_detector = type_detector

    async def prepare(self, audio_clean_id: UUID) -> dict:
        tracks = await self._load_tracks(audio_clean_id)
        
        if not tracks:
            raise ValueError(f"No tracks found for audio_clean_id: {audio_clean_id}")
        
        processed_tracks = await self._process_tracks(tracks)
        sentences = self._build_sentences(processed_tracks)
        speaker_info = self._build_speaker_info(tracks)
        full_text = ' '.join(s['text'] for s in sentences)
        
        return {
            'audio_clean_id': audio_clean_id,
            'full_text': full_text,
            'normalized_text': self.normalizer.normalize(full_text),
            'speaker_count': len(speaker_info),
            'total_duration': max((s['end'] for s in sentences), default=0),
            'word_count': sum(len(t['words']) for t in tracks if t['words']),
            'sentence_count': len(sentences),
            'speaker_info': speaker_info,
            'sentences': sentences,
            'document_type': self.type_detector.detect(speaker_info, sentences),
            'processing_stage': 'prepared'
        }
    
    async def _process_tracks(self, tracks: List[dict]) -> List[dict]:
        from app.api.services.semantic.filler_remover import FillerWordRemover
        from app.api.services.semantic.punctuation_restorer import PunctuationRestorer
        
        filler_remover = FillerWordRemover()
        punct_restorer = PunctuationRestorer()
        
        processed = []
        for track in tracks:
            if not track['transcript']:
                processed.append(track)
                continue
            
            cleaned = filler_remover.remove(track['transcript'], aggressive=False)
            restored = await punct_restorer.restore(cleaned)
            
            processed.append({
                **track,
                'processed_transcript': restored
            })
        
        return processed

    async def _load_tracks(self, audio_clean_id: UUID) -> List[dict]:
        result = await self.session.execute(
            select(AudioClean.original_audio_id)
            .where(AudioClean.cleaned_audio_id == audio_clean_id)
        )
        audio_record = result.first()
        
        if not audio_record:
            return []
        
        original_audio_id = audio_record[0]
        
        result = await self.session.execute(
            select(SpeakerTrack)
            .where(SpeakerTrack.audio_id == original_audio_id)
            .order_by(SpeakerTrack.speaker_id)
        )
        
        tracks = result.scalars().all()
        
        return [
            {
                'speaker_id': t.speaker_id,
                'transcript': t.transcript,
                'words': t.words,
                'total_duration': t.total_duration,
                'coverage': t.coverage
            }
            for t in tracks
        ]

    def _build_sentences(self, tracks: List[dict]) -> List[dict]:
        all_sentences = []
        
        for track in tracks:
            processed_text = track.get('processed_transcript')
            if not processed_text or not track['words']:
                continue
            
            sentences = self.tokenizer.tokenize(processed_text)
            aligned = self.tokenizer.align_with_words(sentences, track['words'])
            
            for sent in aligned:
                sent['speaker_id'] = track['speaker_id']
                all_sentences.append(sent)
        
        all_sentences.sort(key=lambda x: x['start'])
        return all_sentences

    def _build_speaker_info(self, tracks: List[dict]) -> List[dict]:
        return [
            {
                'speaker_id': t['speaker_id'],
                'word_count': len(t['words']) if t['words'] else 0,
                'duration': t['total_duration'],
                'coverage': t['coverage']
            }
            for t in tracks
        ]

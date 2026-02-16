import re
from typing import Optional
from loguru import logger

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class PunctuationRestorer:
    def __init__(self):
        self.model = None
        if HAS_TRANSFORMERS:
            try:
                self.model = pipeline(
                    "token-classification",
                    model="oliverguhr/fullstop-punctuation-multilang-large",
                    aggregation_strategy="simple"
                )
            except Exception as e:
                logger.warning(f"Failed to load punctuation model: {e}")
    
    async def restore(self, text: str) -> str:
        if not text:
            return ""
        
        if self.model:
            return self._restore_with_model(text)
        
        return self._restore_rule_based(text)
    
    def _restore_with_model(self, text: str) -> str:
        try:
            chunks = self._split_text(text, max_length=200)
            restored_chunks = []
            
            for chunk in chunks:
                result = self.model(chunk)
                restored = self._apply_predictions(chunk, result)
                restored_chunks.append(restored)
            
            return ' '.join(restored_chunks)
        except Exception as e:
            logger.error(f"Model punctuation restoration failed: {e}")
            return self._restore_rule_based(text)
    
    def _split_text(self, text: str, max_length: int = 200):
        words = text.split()
        chunks = []
        current = []
        
        for word in words:
            current.append(word)
            if len(current) >= max_length:
                chunks.append(' '.join(current))
                current = []
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def _apply_predictions(self, text: str, predictions):
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            
            for pred in predictions:
                if pred['word'].strip() == word and pred['entity_group'] in ['0', 'PERIOD']:
                    if not word.endswith('.'):
                        result[-1] = word + '.'
                    break
        
        return ' '.join(result)
    
    def _restore_rule_based(self, text: str) -> str:
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            
            if (i + 1) % 15 == 0 and i < len(words) - 1:
                if not word.endswith(('.', '!', '?', ',')):
                    result[-1] = word + '.'
        
        if result and not result[-1].endswith(('.', '!', '?')):
            result[-1] = result[-1] + '.'
        
        return ' '.join(result)

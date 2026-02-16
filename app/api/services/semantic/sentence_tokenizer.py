from typing import List, Dict

try:
    from underthesea import sent_tokenize as underthesea_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


class SentenceTokenizer:
    def __init__(self):
        self.end_markers = ['.', '!', '?', '…']
        self.pause_threshold = 0.8
        
    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        if HAS_UNDERTHESEA:
            return underthesea_tokenize(text)
        
        return self._fallback_tokenize(text)
    
    def _fallback_tokenize(self, text: str) -> List[str]:
        sentences = []
        current = []
        words = text.split()
        
        for word in words:
            current.append(word)
            
            if any(word.endswith(marker) for marker in self.end_markers):
                sentence = ' '.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        
        if current:
            sentence = ' '.join(current).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences if sentences else [text]
    
    def align_with_words(
        self,
        sentences: List[str],
        words: List[Dict]
    ) -> List[Dict]:
        if not sentences or not words:
            return []
        
        aligned = []
        word_idx = 0
        
        for sent_text in sentences:
            sent_word_count = len(sent_text.split())
            sent_words = []
            
            while word_idx < len(words) and len(sent_words) < sent_word_count:
                sent_words.append(words[word_idx])
                word_idx += 1
            
            if sent_words:
                aligned.append({
                    'text': sent_text,
                    'start': sent_words[0]['start'],
                    'end': sent_words[-1]['end'],
                    'words': sent_words
                })
        
        return aligned

import re
from typing import Set


class FillerWordRemover:
    def __init__(self):
        self.filler_words: Set[str] = {
            'á', 'à', 'ừ', 'ờ', 'ơ', 'ư', 'ư', 'hơi',
            'thì', 'là', 'mà', 'nhỉ', 'nhé', 'nha',
            'ấy', 'đấy', 'vậy', 'thế', 'kìa'
        }
        
    def remove(self, text: str, aggressive: bool = False) -> str:
        if not text:
            return ""
        
        words = text.split()
        cleaned = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            
            if aggressive and word_lower in self.filler_words:
                continue
            elif not aggressive and word_lower in {'á', 'à', 'ừ', 'ờ', 'ơ', 'ư'}:
                continue
            
            cleaned.append(word)
        
        return ' '.join(cleaned)

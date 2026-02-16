import re


class TextNormalizer:
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text.strip()

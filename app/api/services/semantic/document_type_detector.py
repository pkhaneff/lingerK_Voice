from typing import List, Dict


class DocumentTypeDetector:
    def detect(self, speaker_info: List[Dict], sentences: List[Dict]) -> str:
        speaker_count = len(speaker_info)
        
        if speaker_count == 1:
            return "other"
        
        if speaker_count == 2:
            durations = [s['duration'] for s in speaker_info]
            ratio = min(durations) / max(durations) if max(durations) > 0 else 0
            
            return "phone_call" if ratio > 0.3 else "online_class"
        
        return "meeting"

class TextNormalizationPrompts:
    """Vietnamese text normalization prompts for LLM"""
    
    @staticmethod
    def gemini_normalization_prompt():
        return """
            You are an expert in Vietnamese text normalization. Please normalize the following speech-to-text output into standard written Vietnamese.

            REQUIREMENTS (do ALL of these in a single pass):

            1. ADD PUNCTUATION:
            - Add a period (.) at the end of each complete sentence
            - Add commas (,) for subordinate clauses, conjunctions, and natural pauses
            - Use colons (:), semicolons (;) when appropriate

            2. SPLIT SENTENCES:
            - Break long sentences into shorter, clearer ones
            - Each sentence should convey one main idea
            - No sentence should exceed 30 words

            3. CAPITALIZATION:
            - Capitalize the first letter of each sentence
            - Capitalize after periods
            - Capitalize proper nouns (if any)

            4. REMOVE FILLER / REDUNDANT WORDS:
            - Repeated words: "cái cái", "em em", "nó nó", "là là"
            - Interjections/particles: "nha", "dạ", "ạ", "luôn", "thôi", "à"
            - Hesitation sounds: "ừm", "ờ", "à", "uhm"
            - Unnecessary connectors: "mà", "đó" (when not needed)

            5. FIX STRUCTURE:
            - Make connectors more natural
            - Ensure coherent, easy-to-read sentences

            MANDATORY RULES:
            - DO NOT add new information
            - DO NOT change the original meaning
            - KEEP the original storytelling style
            - ONLY return the normalized text, with NO explanation

            Original text:
            {text}

            Normalized text:
        """

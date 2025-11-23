import asyncio
from typing import Dict, Any
from loguru import logger as custom_logger
import google.generativeai as genai

from app.api.prompt.norrmalize_analysis import TextNormalizationPrompts


class GeminiTextNormalizer:
    """Gemini 2.0 Flash Vietnamese text normalizer - single pass"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.prompts = TextNormalizationPrompts()
    
    async def normalize_text(self, raw_text: str) -> Dict[str, Any]:
        try:
            start_time = asyncio.get_event_loop().time()
            
            prompt = self.prompts.gemini_normalization_prompt().format(
                text=raw_text
            )
            
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=8192,
                    )
                )
            )
            
            if not hasattr(response, 'text'):
                custom_logger.error("Response missing text attribute")
                return {
                    'success': False,
                    'data': None,
                    'error': 'Response missing text attribute'
                }
            
            try:
                normalized_text = response.text.strip()
                processing_time = asyncio.get_event_loop().time() - start_time
                
                input_tokens = 0
                output_tokens = 0
                
                if hasattr(response, 'usage_metadata'):
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                
                cost = (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000
                
                statistics = {
                    'original_length': len(raw_text),
                    'normalized_length': len(normalized_text),
                    'reduction_ratio': (len(raw_text) - len(normalized_text)) / len(raw_text) if len(raw_text) > 0 else 0,
                    'processing_time': processing_time,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost_estimate': cost,
                    'model': 'gemini-2.0-flash-exp'
                }
                
                custom_logger.info(
                    f"Normalized: {len(raw_text)} -> {len(normalized_text)} chars "
                    f"({statistics['reduction_ratio']:.1%}) in {processing_time:.2f}s, "
                    f"cost: ${cost:.5f}"
                )
                
                return {
                    'success': True,
                    'data': {
                        'normalized_text': normalized_text,
                        'statistics': statistics
                    },
                    'error': None
                }
                
            except ValueError as ve:
                error_msg = f'Cannot extract text: {str(ve)}'
                
                if hasattr(response, 'prompt_feedback'):
                    error_msg += f'. Block reason: {response.prompt_feedback.block_reason}'
                
                custom_logger.error(error_msg)
                
                return {
                    'success': False,
                    'data': None,
                    'error': error_msg
                }
                
            except Exception as e:
                custom_logger.error(f"Text access failed: {type(e).__name__}: {e}")
                return {
                    'success': False,
                    'data': None,
                    'error': f'Text access failed: {str(e)}'
                }
                
        except Exception as e:
            custom_logger.error(f"Gemini normalization failed: {e}", exc_info=True)
            return {
                'success': False,
                'data': None,
                'error': str(e)
            }
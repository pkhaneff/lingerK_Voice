import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key: {api_key[:20]}...")

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.0-flash')

prompt = "Chuẩn hóa câu này: xin chào em em là sinh viên"

try:
    print("Calling Gemini...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=100,
        )
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
    
    if hasattr(response, 'text'):
        print(f"Text: {response.text}")
    else:
        print("No text attribute!")
        print(f"Dir: {dir(response)}")
    
    if hasattr(response, 'prompt_feedback'):
        print(f"Feedback: {response.prompt_feedback}")
    
    if hasattr(response, 'usage_metadata'):
        print(f"Tokens: {response.usage_metadata}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
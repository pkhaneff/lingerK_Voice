from datetime import datetime
import uuid

def generate_s3_key(filename: str) -> str:
    """Generate unique S3 key."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}_{filename}"

def generate_cleaned_s3_key(original_s3_key: str) -> str:
    """
    Generate S3 key for cleaned audio.
    """
    filename = original_s3_key.split('/')[-1]
    name_without_ext = filename.rsplit('.', 1)[0]
    return original_s3_key.replace(filename, f"{name_without_ext}_cleaned.wav")

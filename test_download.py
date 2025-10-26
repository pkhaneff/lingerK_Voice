from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_id = "openai/whisper-medium"
cache_dir = "app/data_model/storage/whisper"

print(f"Downloading {model_id} to {cache_dir}...")
print("This will only download PyTorch files (1.5GB)")

# Download chỉ PyTorch model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    # Không dùng low_cpu_mem_usage để tránh cần accelerate ngay
)

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir
)

print("✅ Download completed!")
print(f"Model saved to: {cache_dir}")
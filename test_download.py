# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
# import torch

# model_id = "vinai/PhoWhisper-medium"
# cache_dir = "app/data_model/storage/phowhisper"

# print(f"Downloading {model_id} to {cache_dir}...")
# print("This will only download PyTorch files (1.5GB)")

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     cache_dir=cache_dir,
# )

# processor = AutoProcessor.from_pretrained(
#     model_id,
#     cache_dir=cache_dir
# )

# print(" Download completed!")
# print(f"Model saved to: {cache_dir}")

from huggingface_hub import snapshot_download
snapshot_download(
    "vinai/PhoWhisper-medium",
    local_dir="app/data_model/storage/phowhisper",
    allow_patterns=["pytorch_model.bin", "*.json", "*.txt", "*.json", "*.md"]
)

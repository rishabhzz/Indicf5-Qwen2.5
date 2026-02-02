from transformers import AutoModel
import torch

print("Loading model...")
model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True
)

print("Model loaded")
print("Using GPU:", torch.cuda.is_available())

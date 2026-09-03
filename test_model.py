import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "/root/quixbugs_lora/models/Qwen2.5-Coder-1.5B-Instruct"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

print("Model loaded successfully!")

print("Device:")
print(model.device)

print("Dtype:")
print(model.dtype)

print("Parameter count:")

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(total_params)

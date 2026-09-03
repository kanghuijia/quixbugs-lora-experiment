from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/root/quixbugs_lora/models/Qwen2.5-Coder-1.5B-Instruct"

ADAPTER_PATH = (
    "/root/quixbugs_lora/outputs/"
    "qwen2.5-coder-1.5b-lora/checkpoint-30"
)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="bfloat16",
    local_files_only=True
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    is_trainable=True
)
model.print_trainable_parameters()

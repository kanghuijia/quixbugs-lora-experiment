import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel




BASE_MODEL = (
    "/root/quixbugs_lora/models/Qwen2.5-Coder-1.5B-Instruct"
)

ADAPTER_PATH = (
    "/root/quixbugs_lora/outputs/"
    "qwen2.5-coder-1.5b-lora/checkpoint-30"
)

VAL_FILE = (
    "/root/quixbugs_lora/data/processed/val.jsonl"
)




tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True
)




print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)




print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()



def generate_fix(
    instruction,
    input_text,
    max_new_tokens=512
):

    prompt = (
        f"{instruction}\n\n"
        f"{input_text}\n\n"
        f"修复后的代码：\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=False,

            temperature=0.0,

            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return result.strip()




with open(
    VAL_FILE,
    "r",
    encoding="utf-8"
) as f:

    samples = [
        json.loads(line)
        for line in f
    ]




for i, sample in enumerate(
    samples,
    start=1
):

    print("\n")
    print("=" * 80)
    print(f"Validation Sample {i}")
    print("=" * 80)

    print("\n[MODEL OUTPUT]\n")

    result = generate_fix(
        sample["instruction"],
        sample["input"]
    )

    print(result)

    print("\n[GROUND TRUTH]\n")
    print(sample["output"])

import os
import re
import json
import csv
import subprocess
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel




BASE_MODEL = "/root/quixbugs_lora/models/Qwen2.5-Coder-1.5B-Instruct"

LORA_ADAPTER = (
    "/root/quixbugs_lora/outputs/"
    "qwen2.5-coder-1.5b-lora/checkpoint-30"
)

VAL_FILE = "/root/quixbugs_lora/data/processed/val.jsonl"

RESULT_DIR = "/root/quixbugs_lora/results"

CSV_FILE = os.path.join(
    RESULT_DIR,
    "evaluation.csv"
)

PREDICTION_FILE = os.path.join(
    RESULT_DIR,
    "predictions.jsonl"
)




MAX_NEW_TOKENS = 512
TEST_TIMEOUT = 10




os.makedirs(RESULT_DIR, exist_ok=True)




print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token




print("Loading Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

base_model.eval()




print("Loading Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

base_model.eval()


print("Loading Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

base_model.eval()




print("Loading Base Model for LoRA...")

lora_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

lora_base_model.eval()




print("Loading LoRA Adapter...")

lora_model = PeftModel.from_pretrained(
    lora_base_model,
    LORA_ADAPTER
)

lora_model.eval()




with open(VAL_FILE, "r", encoding="utf-8") as f:
    samples = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(samples)} validation samples.")




def extract_test(input_text):
    """
    Extract assert statement from the input.

    Current QuixBugs samples contain one-line assert tests.
    """

    lines = input_text.splitlines()

    for line in lines:
        if line.strip().startswith("assert "):
            return line.strip()

    return None




def generate_fix(model, instruction, input_text):

    prompt = (
        f"{instruction}\n\n"
        f"{input_text}\n\n"
        f"修复后的代码：\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id
        )


    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return result.strip()




def extract_code(text):

  

    match = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        text = match.group(1).strip()

  
    match = re.search(
        r"\bdef\s+\w+\s*\(",
        text
    )

    if match:
        text = text[match.start():]



    return text.strip()




def run_test(generated_code, test_code):

    if not generated_code:
        return False, "EMPTY_OUTPUT"

    if not test_code:
        return False, "NO_TEST_FOUND"

   
    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as f:

            temp_path = f.name

            f.write(generated_code)
            f.write("\n\n")
            f.write(test_code)
            f.write("\n")

        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT
        )

        if result.returncode == 0:
            return True, "PASS"

       
        if "AssertionError" in result.stderr:
            return False, "ASSERTION_FAILED"

      
        error = result.stderr.strip()

        if error:
            first_line = error.splitlines()[-1]
            return False, first_line[:200]

        return False, f"RETURN_CODE_{result.returncode}"

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"

    except Exception as e:
        return False, f"ERROR: {str(e)[:200]}"

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)




results = []

print()
print("=" * 70)
print("Starting evaluation")
print("=" * 70)


for i, sample in enumerate(samples, start=1):

    instruction = sample["instruction"]
    input_text = sample["input"]
    ground_truth = sample["output"]

    test_code = extract_test(input_text)

    print()
    print("=" * 70)
    print(f"Sample {i}")
    print("=" * 70)

    print("\n[Test]")
    print(test_code)



    print("\n[Base Model] Generating...")

    base_output = generate_fix(
        base_model,
        instruction,
        input_text
    )

    base_code = extract_code(base_output)

    base_pass, base_detail = run_test(
        base_code,
        test_code
    )

    print("\n[Base Model Result]")
    print("PASS" if base_pass else "FAIL")
    print("Detail:", base_detail)



    print("\n[LoRA Model] Generating...")

    lora_output = generate_fix(
        lora_model,
        instruction,
        input_text
    )

    lora_code = extract_code(lora_output)

    lora_pass, lora_detail = run_test(
        lora_code,
        test_code
    )

    print("\n[LoRA Model Result]")
    print("PASS" if lora_pass else "FAIL")
    print("Detail:", lora_detail)



    results.append({
        "sample": i,

        "test": test_code or "",

        "base_pass": "PASS" if base_pass else "FAIL",

        "base_detail": base_detail,

        "lora_pass": "PASS" if lora_pass else "FAIL",

        "lora_detail": lora_detail,


        "semantic_correct": "",

        "base_output": base_code,

        "lora_output": lora_code,

        "ground_truth": ground_truth
    })




with open(
    PREDICTION_FILE,
    "w",
    encoding="utf-8"
) as f:

    for result in results:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            + "\n"
        )




with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "样本",
        "Base模型",
        "Base测试详情",
        "LoRA模型",
        "LoRA测试详情",
        "测试用例",
        "语义是否正确"
    ])

    for result in results:

        writer.writerow([
            result["sample"],
            result["base_pass"],
            result["base_detail"],
            result["lora_pass"],
            result["lora_detail"],
            result["test"],
            result["semantic_correct"]
        ])




base_pass_count = sum(
    r["base_pass"] == "PASS"
    for r in results
)

lora_pass_count = sum(
    r["lora_pass"] == "PASS"
    for r in results
)

total = len(results)


print()
print("=" * 70)
print("Evaluation Summary")
print("=" * 70)

print(
    f"\nBase Model:"
    f"\nPASS: {base_pass_count} / {total}"
    f"\nRate: {base_pass_count / total * 100:.2f}%"
)

print(
    f"\nLoRA Model:"
    f"\nPASS: {lora_pass_count} / {total}"
    f"\nRate: {lora_pass_count / total * 100:.2f}%"
)

print()
print("=" * 70)

print("CSV saved to:")
print(CSV_FILE)

print()
print("Predictions saved to:")
print(PREDICTION_FILE)

print()
print("Evaluation finished.")

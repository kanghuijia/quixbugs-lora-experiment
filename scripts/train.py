import os
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
)




MODEL_NAME = "/root/quixbugs_lora/models/Qwen2.5-Coder-1.5B-Instruct"

TRAIN_FILE = (
    "/root/quixbugs_lora/data/processed/train.jsonl"
)

VAL_FILE = (
    "/root/quixbugs_lora/data/processed/val.jsonl"
)

OUTPUT_DIR = (
    "/root/quixbugs_lora/outputs/"
    "qwen2.5-coder-1.5b-lora"
)




print("=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token




print("=" * 60)
print("Loading dataset...")
print("=" * 60)

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE
    }
)

print(dataset)




def format_example(example):

    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    text = (
        f"{instruction}\n\n"
        f"{input_text}\n\n"
        f"修复后的代码：\n"
        f"{output_text}"
    )

    return {
        "text": text
    }


dataset = dataset.map(
    format_example
)



MAX_LENGTH = 2048


def tokenize_function(example):

    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    result["labels"] = result["input_ids"].copy()

    return result


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=dataset["train"].column_names
)


print("=" * 60)
print("Tokenized dataset:")
print(tokenized_dataset)
print("=" * 60)




print("=" * 60)
print("Loading model...")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)




lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)




model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


print("=" * 60)
print("LoRA trainable parameters:")
print("=" * 60)

model.print_trainable_parameters()




data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)




training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,



    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=4,



    num_train_epochs=5,



    learning_rate=2e-4,



    optim="adamw_torch",



    logging_strategy="steps",

    logging_steps=1,



    eval_strategy="epoch",



    save_strategy="epoch",

    save_total_limit=2,



    bf16=True,



    report_to="none",

 

    seed=42,

 

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,
)




trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,

    data_collator=data_collator,
)




print("=" * 60)
print("Start training...")
print("=" * 60)

trainer.train()




print("=" * 60)
print("Saving LoRA adapter...")
print("=" * 60)

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)




print("=" * 60)
print("Final evaluation...")
print("=" * 60)

metrics = trainer.evaluate()

print(metrics)

print("=" * 60)
print("Training finished!")
print("=" * 60)

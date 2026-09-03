# Qwen2.5-Coder-1.5B LoRA QuixBugs Experiment

## 1. Experimental Goal

使用 Qwen2.5-Coder-1.5B-Instruct
通过 LoRA 对 QuixBugs 程序修复数据进行参数高效微调。

目标是验证微调后模型在程序修复任务上的表现。

---

## 2. Dataset

Dataset:
QuixBugs

Language:
Python

Total samples:
50

Training samples:
45

Validation samples:
5

Data format:

{
    "instruction": "修复下面代码使测试通过",
    "input": "<buggy fn>\n<failing test>",
    "output": "<fixed fn>"
}

Rationale:
Not used.

---

## 3. Model

Base model:

Qwen/Qwen2.5-Coder-1.5B-Instruct

Fine-tuning method:

LoRA

---

## 4. LoRA Configuration

r = 16

lora_alpha = 32

lora_dropout = 0.05

target_modules:

q_proj
k_proj
v_proj
o_proj

bias = none

---

## 5. Training Configuration

GPU:
RTX 3090

Precision:
BF16

per_device_train_batch_size:
2

gradient_accumulation_steps:
4

effective batch size:
8

learning_rate:
2e-4

num_train_epochs:
5

max_length:
2048

seed:
42

---


## 6. Training Results

Loss curve:
outputs/qwen2.5-coder-1.5b-lora/loss_curve.png

---

## 7. Evaluation

Validation samples:
5

Baseline:
Qwen2.5-Coder-1.5B-Instruct

Fine-tuned:
Qwen2.5-Coder-1.5B-Instruct + LoRA

---

## 9. Evaluation Results

/root/quixbugs_lora/results/evaluation.csv

---

## 10. Conclusion

LoRA 微调后模型在 5 个验证样本中的测试通过数量
相比未微调基线有所提升。

同时通过人工检查模型生成补丁的语义正确性，
进一步判断自动测试结果是否可靠。

可能由于数据样本太少的缘故，曲线不是很好看，尤其是训练集的loss曲线
# QuixBugs-LoRA-Experiment

> A personal experiment on fine-tuning LLMs for Automated Program Repair (APR) using LoRA.

## 📌 Project Description

This repository contains my **personal experimental project** for exploring the application of parameter-efficient fine-tuning (LoRA) on Large Language Models (LLMs) for Automated Program Repair (APR).

The experiment uses **Qwen2.5-Coder-1.5B-Instruct** as the base model and performs LoRA fine-tuning on a small subset of the **QuixBugs** benchmark.

The goal of this project is mainly for:

- Learning the workflow of LLM fine-tuning
- Exploring LLM-based program repair
- Understanding LoRA adaptation in code generation models
- Conducting small-scale APR experiments

> ⚠️ This repository is for **personal research/learning use only**.  
> It is not intended to provide a complete or production-level APR system.

---

## 🧩 Experiment Overview

The overall pipeline is:

```
QuixBugs Dataset
        |
        v
Data Processing
        |
        v
JSONL Repair Samples
        |
        v
LoRA Fine-tuning
        |
        v
Qwen2.5-Coder-1.5B-LoRA
        |
        v
Patch Generation & Evaluation
```

---

## 🛠 Environment

### Hardware

Example environment:

- GPU: NVIDIA RTX 3090
- CUDA supported GPU environment
- AutoDL cloud platform

### Software

- Python >= 3.10
- PyTorch
- Transformers
- PEFT
- Accelerate
- Datasets

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
quixbugs-lora-experiment/
│
├── .gitignore
├── experiment.md              # Experiment notes
├── requirements.txt           # Python dependencies
├── test_model.py              # Base model testing script
│
├── adapter/                   # LoRA adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
├── data/                      
│   ├── metadata/
│   │   └── metadata.json
│   │
│   └── processed/
│       ├── train.jsonl        # Training samples
│       └── val.jsonl          # Validation samples
│
├── QuixBugs/                  # Original QuixBugs benchmark
│
├── results/                   # Experiment results
│   ├── evaluation.csv
│   ├── loss_curve.png
│   └── predictions.jsonl
│
└── scripts/                   # Experiment scripts
    ├── check_params.py
    ├── evaluate.py
    ├── evaluate_all.py
    ├── evaluate_base.py
    ├── plot_loss.py
    ├── plot_loss.py.save
    ├── prepare_data.py
    └── train.py
```

---

## 📊 Dataset Format

The training data follows an instruction tuning format:

```json
{
  "instruction": "修复下面代码使测试通过",
  "input": "<buggy function>\n<failing test>",
  "output": "<fixed function>",
}
```

Each sample contains:

- `instruction`: Repair task description
- `input`: Buggy code and failing test information
- `output`: Corrected code
---

## 🚀 Training

Example:

```bash
python scripts/train.py
```

The training process uses:

- Base Model:
  ```
  Qwen/Qwen2.5-Coder-1.5B-Instruct
  ```

- Fine-tuning method:
  ```
  LoRA (Low-Rank Adaptation)
  ```

Example LoRA configuration:

```python
r = 16
lora_alpha = 32
learning_rate = 2e-4
```

---

## 🔍 Evaluation

After training:

If you want to conduct LoRA model only,execute the following instruction:
```bash
python scripts/evaluate.py
```
else if you want to conduct the base model,execute the following instruction:
```bash
python scripts/evaluate_base.py
```
To obtain the final comparison result of the two with one click,execute the folloing instruction:
```bash
python scripts/evaluate_all.py
```
The evaluation process generates repaired patches and compares them with the ground truth solutions.

Evaluation metrics:

- Successful repair cases
- Test passing rate
- Generated patch quality

---

## 📈 Results

Experiment results will be stored in:

```
results/
```

Including:

- Training loss curve
- Evaluation summary
- Generated repair examples

---

## 📝 Notes

- This experiment only uses a small subset of QuixBugs samples.
- The dataset size is limited and mainly for validating the fine-tuning pipeline.
- Results should not be compared directly with large-scale APR systems.
- The project is maintained as a personal learning/research record.

---

## 📚 References

- QuixBugs: A Multi-Lingual Program Repair Benchmark
- Qwen2.5-Coder
- LoRA: Low-Rank Adaptation of Large Language Models

---

## 👤 Author

Personal experiment repository.

Maintained for:
- LLM-based Automated Program Repair exploration
- LoRA fine-tuning practice
- Conducting small-scale APR experiment

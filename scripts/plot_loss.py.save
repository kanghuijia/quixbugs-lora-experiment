import json
import os

import matplotlib.pyplot as plt


OUTPUT_DIR = (
    "/root/quixbugs_lora/outputs/"
    "qwen2.5-coder-1.5b-lora/checkpoint-30"
)

STATE_FILE = os.path.join(
    OUTPUT_DIR,
    "trainer_state.json"
)

PLOT_FILE = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)




with open(
    STATE_FILE,
    "r",
    encoding="utf-8"
) as f:

    state = json.load(f)


log_history = state["log_history"]


train_steps = []
train_losses = []

eval_steps = []
eval_losses = []


for item in log_history:

    if "loss" in item:

        train_steps.append(
            item["step"]
        )

        train_losses.append(
            item["loss"]
        )

    if "eval_loss" in item:

        eval_steps.append(
            item["step"]
        )

        eval_losses.append(
            item["eval_loss"]
        )


     

plt.figure(figsize=(8, 5))

if train_steps:

    plt.plot(
        train_steps,
        train_losses,
        label="Train Loss"
    )

if eval_steps:

    plt.plot(
        eval_steps,
        eval_losses,
        label="Validation Loss"
    )


plt.xlabel("Step")
plt.ylabel("Loss")

plt.title(
    "Qwen2.5-Coder-1.5B LoRA Training Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=300
)

print(
    f"Loss curve saved to: {PLOT_FILE}"
)

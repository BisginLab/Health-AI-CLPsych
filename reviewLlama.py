import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from huggingface_hub import login

# Paste your token here or read from environment
hf_token = "<token here>"

# This stores the token securely (usually in ~/.cache/huggingface/)
login(token=hf_token)

# 1. Pick a LLaMA model (requires HF token & approval for LLaMA weights)
model_id = "meta-llama/Llama-3.2-3B-Instruct"   # choose 7B/13B depending on your hardware

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. Example dataset (reviews + labels)
train_data = {
    "text": [
        "Review: The room was dirty and staff were rude.\nAnswer: 1",
        "Review: The hotel was okay, nothing special.\nAnswer: 2",
        "Review: Nice service and comfortable beds.\nAnswer: 3",
        "Review: Amazing stay! Excellent staff and very clean rooms.\nAnswer: 4",
    ]
}

test_data = {
    "text": [
        "Review: Rooms were noisy but staff were kind.\nAnswer: 2",
        "Review: Fantastic hotel, I loved everything.\nAnswer: 4",
    ]
}

dataset = Dataset.from_dict(train_data).train_test_split(test_size=0.2)

def tokenize(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 3. Load model in 4-bit (saves GPU memory)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    use_auth_token=True
)

# 4. Apply LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  # common choice for LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Training setup
training_args = TrainingArguments(
    output_dir="./llama-lora-hotel-rating",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=50,
    learning_rate=2e-4,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# 6. Train
trainer.train()

# 7. Save
trainer.save_model("./llama-lora-hotel-rating")


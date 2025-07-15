import datasets as ds
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import os

# === LOAD DATA FROM CSV ===
#NOTE: Dataset must be a dataset object
df_X = ds.load_dataset("csv", data_files="/shared/DATA/reddit/expert/expert_posts.csv")["train"]
df_y = ds.load_dataset("csv", data_files="/shared/DATA/reddit/expert/expert.csv")["train"]
model_id = "meta-llama/Llama-3.2-3B-Instruct"

import pandas as pd

# Convert datasets to pandas
df_X_pd = df_X.to_pandas()
df_y_pd = df_y.to_pandas()

# Group posts by user_id and join into one text block
user_posts = df_X_pd.groupby("user_id")["post_body"].apply(
    lambda posts: "\n\n".join([p for p in posts if pd.notnull(p)])
).reset_index().rename(columns={"post_body": "text"})

# Merge with df_y to get labels + texts
df_merged = pd.merge(df_y_pd, user_posts, on="user_id", how="inner")

# Convert back to HuggingFace Dataset
df = ds.Dataset.from_pandas(df_merged)
print(f"Dataset size: {len(df)}")
print(f"Dataset columns: {df.column_names}")

#Remove non-annotated rows
df = df.filter(lambda x: x['label'] != None, batched=False)

# === LOAD MODEL AND TOKENIZER ===
#Log into huggingface to access gated model
token = os.getenv('token')
login(token)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

def tokenize(example):
    #Use the same prompt structure as zero-shot
    prefix = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a health specialist. Classify the following post content from a social media user as \"a\" for non-suicidal, \"b\" for low-risk of suicide, \"c\" for moderate-risk of suicide, or \"d\" for severe-risk of suicide. Provide your response in the format: \"a\", \"b\", \"c\", or \"d\".\n"
    )
    user_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n{example['text']}\n<|start_header_id|>assistant<|end_header_id|>\n"
    )
    label_suffix = example['label']
    prompt = prefix + user_prompt + label_suffix
    tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)

    #These lines ensure that only the label token is counted for loss calc
    input_len = len(tokenizer(prefix + user_prompt, truncation=True, max_length=1024)["input_ids"])
    labels = tokenized_prompt["input_ids"].copy()
    labels[:input_len] = [-100] * input_len

    #This returns the processed prompt, while making sure that the labels are under 'labels' which is what the trainer expects.
    tokenized_prompt["labels"] = labels
    return tokenized_prompt

tokenized_df = df.map(tokenize, batched=False)
split_df = tokenized_df.train_test_split(test_size=0.1, seed=35)

# === LoRA CONFIGURATION ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir='/home/umflint.edu/brayclou/Health-AI-CLPsych/finetuned',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='/home/umflint.edu/brayclou/Health-AI-CLPsych/finetuned/logs',
    learning_rate=2e-5,
    num_train_epochs=5,
    fp16=True,
    report_to="none"
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_df['train'],
    eval_dataset=split_df['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# === TRAIN ===
trainer.train()
# === SAVE MODEL AND TOKENIZER ===
trainer.save_model("/home/umflint.edu/brayclou/Health-AI-CLPsych/finetuned")
tokenizer.save_pretrained("/home/umflint.edu/brayclou/Health-AI-CLPsych/finetuned")
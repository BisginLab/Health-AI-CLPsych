#Imports 
import peft
from peft import prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed, BitsAndBytesConfig
import datasets as ds
from huggingface_hub import login
import os
from collections import defaultdict
from dotenv import load_dotenv
import argparse

# parser = argparse.ArgumentParser(description="Process model and output csv.")
# parser.add_argument("--model", type=str, help="Model name")
# parser.add_argument("--epochs", type=int, default=25, help="How many epochs the model will be fine-tuned for")
# parser.add_argument("--output", type=str, default="", help="Name of output folder")
# args = parser.parse_args()

#Log in to huggingface with api token
load_dotenv()
token = os.getenv("HF_TOKEN")
assert token, "HF_TOKEN not set correctly!"
login(token)
set_seed(35)

#Set up bitsandbytes
bitsandbytes_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

#Defined changable variables
model_name = "HuggingFaceTB/SmolLM3-3B"#"HuggingFaceTB/SmolLM3-3B"
crowd_test_feature_df_name = "/shared/DATA/reddit/crowd/test/shared_task_posts_test.csv"
crowd_test_label_df_name = "/shared/DATA/reddit/crowd/test/crowd_test.csv"
crowd_train_feature_df_name = "/shared/DATA/reddit/crowd/train/shared_task_posts.csv"
crowd_train_label_df_name = "/shared/DATA/reddit/crowd/train/crowd_train.csv"
max_posts_per_user = 10
max_token_lenth_cap = 2048
unsupervised_epochs = 2
supervised_epochs = 2
output_dir = f"/home/umflint.edu/brayclou/Health-AI-CLPsych/finetuned/SmolLM3-2step-10epochs"

#Load model
print("Setting tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Loading model in kbit...")
untuned_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bitsandbytes_config)
untuned_model = prepare_model_for_kbit_training(untuned_model)
untuned_model.config.pad_token_id = tokenizer.pad_token_id

#If model's max length is an effective infinite, cap at a given number
model_max_len = tokenizer.model_max_length
if model_max_len > 100000:
    print("model's max token length set extremely high; capping to default")
    model_max_len = max_token_lenth_cap
    

#Load dataset
print("Loading dataset...")
df_X_1 = ds.load_dataset("csv", data_files=crowd_test_feature_df_name)["train"]
df_y_1 = ds.load_dataset("csv", data_files=crowd_test_label_df_name)["train"]
df_y_1 = df_y_1.rename_column("raw_label", "label")
df_X_2 = ds.load_dataset("csv", data_files=crowd_train_feature_df_name)["train"]
df_y_2 = ds.load_dataset("csv", data_files=crowd_train_label_df_name)["train"]

df_X = ds.concatenate_datasets([df_X_1, df_X_2])
df_y = ds.concatenate_datasets([df_y_1, df_y_2])

#Set up lora
lora_config = peft.LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    #lora_dropout=0.1, #Learn more about this
    bias="none",
    task_type="CAUSAL_LM"
)

#Set up peft
peft_model = peft.get_peft_model(untuned_model, lora_config)

#Preprocess dataset
print("Sorting dataset into defaultdict...")
user_posts = defaultdict(list)
for row in df_X:
    if len(user_posts[row["user_id"]]) <max_posts_per_user:
        if not row["post_body"]:
            row["post_body"] = ""
        user_posts[row["user_id"]].append(row["post_body"])

##STAGE 1: UNSUPERVISED FINETUNE STEP
def preprocess_unsupervised(row):
    text = row["post_body"] or ""
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=model_max_len,
        return_attention_mask=True,
    )
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

#apply preprocessing function
df_X = df_X.shuffle(seed=35).select(range(min(len(df_X), 5000)))
unsupervised_df = df_X.map(preprocess_unsupervised, desc="preprocessing", remove_columns=df_X.column_names)

#define the unsupervised training setup
unsupervised_args = TrainingArguments(
    output_dir = f"{output_dir}/unsupervised-finetuned",
    learning_rate = 5e-5,
    num_train_epochs = unsupervised_epochs,
    per_device_train_batch_size = 2,
    fp16 = True,
    save_safetensors = True,
)
unsupervised_trainer = Trainer(
    model = peft_model,
    args=unsupervised_args,
    train_dataset = unsupervised_df,
    tokenizer=tokenizer
)
unsupervised_trainer.train()

print("Saving model...")
peft_model.save_pretrained(f"{output_dir}/unsupervised-finetuned")
print(f"Model saved to {output_dir}/unsupervised-finetuned")

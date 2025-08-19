#Imports
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
import datasets as ds
from huggingface_hub import login
import os
from collections import defaultdict

#Log in to huggingface with api token
token = os.getenv("HF_TOKEN")
assert token, "HF_TOKEN not set correctly!"
login(token)
set_seed(35)

#Defined changable variables
model_name = "google/gemma-2-2b"
feature_df_name = "expert.csv"
label_df_name = "expert_posts.csv"
max_posts_per_user = 10
output_dir = "temp_dir" #NOTE: Needs filling upon use

#Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
untuned_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True) #NOTE: need to remvoe if bitsandbytes fails
untuned_model.config.pad_token_id = tokenizer.pad_token_id

#Load dataset
df_X = ds.load_dataset("csv", data_files=feature_df_name)["train"]
df_y = ds.load_dataset("csv", data_files=label_df_name)["train"]

#Preprocess dataset
user_posts = defaultdict(list)
for row in df_X:
    #only append if user post count hasn't capped out
    if len(user_posts[row["user_id"]]) < max_posts_per_user:     #and if row["subreddit"] == "SuicideWatch":
        user_posts[row["user_id"]].append(row["post_body"])
#Cap token count

def preprocess(label_df):
    #convert labels into integer counterparts, 1-4
    label = label_df["raw_label"]
    label_map = {"a": "1", "b": "2", "c": "3", "d": "4"}
    label = label_map[label] if label in label_map else None
    if label is None:
        raise ValueError(f"Label {label} not found in label map for user {label_df['user_id']}!")

    #Retrieve current_labels matching posts, then join them together
    posts = "\n".join(user_posts[label_df["user_id"]])
    prompt = f"{posts}\nIn rating how suicidal the author of the above message is from 1(not suicidal) to 4(extremely suicidal), I choose "
    
    #Tokenize
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    if label_ids and label_ids[-1] != tokenizer.eos_token_id:
        label_ids.append(tokenizer.eos_token_id)

    #If the lenth of the prompt and label is too long, truncate the promp from the left
    full_len = prompt_ids + label_ids
    if len(full_len) > tokenizer.model_max_length:
        trimmed_prompt_ids = prompt_ids[-(tokenizer.model_max_length - len(label_ids)):]
    else:
        trimmed_prompt_ids = prompt_ids
    input_ids = trimmed_prompt_ids + label_ids

    #Zero out post weights, so only labels are changed within the model
    ignore_id = -100
    label_masked = [ignore_id] * len(trimmed_prompt_ids) + label_ids

    #Add right padding
    pad_id = tokenizer.pad_token_id
    pad_to = tokenizer.model_max_length - len(input_ids)
    input_ids += [pad_id] * pad_to
    label_masked += [ignore_id] * pad_to
    
    attention_mask = [1] * (len(input_ids) - pad_to) + [0] * pad_to

    #Return result
    return {"input_ids": input_ids, "labels": label_masked, "attention_mask": attention_mask}

#Trigger preprocessing
df = dict()
preprocessed_df = df_y.map(preprocess, desc="preprocessing", remove_columns=df_y.column_names)
df = preprocessed_df.train_test_split(test_size=0.1, seed=35)

#Set up lora
lora_config = peft.LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    #lora_dropout=0.1, #Learn more about this
    bias="none",
    task_type="CAUSAL_LM"
)
#MAYBE do bitsandbytes quantization(broke something last time)

#Set up peft
peft_model = peft.get_peft_model(untuned_model, lora_config)

#Set up trainer
trainer_config = TrainingArguments(
    output_dir=output_dir,
    logging_dir=f"{output_dir}/.logs",
    save_steps=500,
    eval_strategy="steps",
    gradient_accumulation_steps=3,
    learning_rate=2e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    fp16=True,
    save_safetensors=True,
    optim="paged_adamw_8bit"
)
trainer = Trainer(
    model = peft_model,
    args=trainer_config,
    train_dataset=df["train"],
    eval_dataset=df["test"],
    tokenizer=tokenizer,
    data_collator=None,
)

#Trigger supervised learning
peft_model.print_trainable_parameters()
trainer.train()

#Save model
peft_model.save_pretrained(f"{output_dir}/lora-finetuned")
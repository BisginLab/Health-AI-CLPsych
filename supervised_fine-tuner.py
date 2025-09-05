#Imports
import peft
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed, BitsAndBytesConfig
import datasets as ds
from huggingface_hub import login
import os
from collections import defaultdict
from dotenv import load_dotenv

"""
- This is the current iteration of the fine-tuning script I created for supervised llm generation.  It trains on
the expert labeled data, in order to hopefully get the most skilled classification competence possible.

- Why does this new script replace the older supervised summarizer?
The old script had a couple main issues that made it unworkable.  First, it trained the model at a post level, dispite having only 
user level labels available.  This caused the output to become garbled.  Second, the loss was bouncing all over the place in the old
script and I couldn't figure out why(though it was probably because of the first issue.)  Finally, the code was just bad in general, and I had learned
more since that attempt.  While I could have modified the old script instead of starting from scratch, the effort involved would likely have been greater
what with the old code causing errors in the new code.

Changelog:
- Reduced max token count by 2000
- added config adjustment to turn expandable_segments to True so that there is less data fragmentation(moved to slurm script)
- Explicitely set trainer's batch size to 2

"""

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
model_name = "meta-llama/Llama-3.2-3B-Instruct"
feature_df_name = "../expert/expert_posts.csv"
label_df_name = "../expert/expert.csv"
max_posts_per_user = 10
max_token_lenth_cap = 2048
output_dir = "../finetuned/llama-3.2-base-finetuned"

#Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
untuned_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bitsandbytes_config)
untuned_model = prepare_model_for_kbit_training(untuned_model)
untuned_model.config.pad_token_id = tokenizer.pad_token_id

#If model's max length is an effective infinite, cap at a given number
model_max_len = tokenizer.model_max_length
if model_max_len > 100000:
    print("model's max token length set extremely high; capping to default")
    model_max_len = max_token_lenth_cap
    

#Load dataset
df_X = ds.load_dataset("csv", data_files=feature_df_name)["train"]
df_y = ds.load_dataset("csv", data_files=label_df_name)["train"]

#Preprocess dataset
user_posts = defaultdict(list)
for row in df_X:
    if len(user_posts[row["user_id"]]) <max_posts_per_user:
        if not row["post_body"]:
            row["post_body"] = ""
        user_posts[row["user_id"]].append(row["post_body"])

def preprocess(label_df):
    #convert labels into integer counterparts, 1-4
    label = label_df["label"]
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
    if len(full_len) > model_max_len:
        trimmed_prompt_ids = prompt_ids[-(model_max_len - len(label_ids)):]
    else:
        trimmed_prompt_ids = prompt_ids
    input_ids = trimmed_prompt_ids + label_ids

    #Zero out post weights, so only labels are changed within the model
    ignore_id = -100
    label_masked = [ignore_id] * len(trimmed_prompt_ids) + label_ids

    #Add right padding
    pad_id = tokenizer.pad_token_id
    pad_to = model_max_len - len(input_ids)
    input_ids += [pad_id] * pad_to
    label_masked += [ignore_id] * pad_to
    
    attention_mask = [1] * (len(input_ids) - pad_to) + [0] * pad_to

    #Return result
    return {"input_ids": input_ids, "labels": label_masked, "attention_mask": attention_mask}

#Trigger preprocessing
df = dict()

#Remove null labels
df_y = df_y.filter(lambda row: row["label"] != None)

preprocessed_df = df_y.map(preprocess, desc="preprocessing", remove_columns=df_y.column_names)
df = preprocessed_df.train_test_split(test_size=0.1, seed=35)
print(f"The columns of preprocessed are {preprocessed_df.column_names}")#NOTE: test code
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
    gradient_accumulation_steps=6,
    learning_rate=2e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    fp16=True,
    save_safetensors=True,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    label_names=["labels"],
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
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

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
from tqdm import tqdm

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
model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_dir = "/home/umflint.edu/brayclou/Health-AI-CLPsych/finetuned/llama-2step-10epochs/unsupervised-finetuned"
crowd_test_feature_df_name = "/shared/DATA/reddit/crowd/test/shared_task_posts_test.csv"
crowd_test_label_df_name = "/shared/DATA/reddit/crowd/test/crowd_test.csv"
crowd_train_feature_df_name = "/shared/DATA/reddit/crowd/train/shared_task_posts.csv"
crowd_train_label_df_name = "/shared/DATA/reddit/crowd/train/crowd_train.csv"
max_posts_per_user = 10
max_token_lenth_cap = 2048
epochs = 18
output_dir = f"/home/umflint.edu/brayclou/Health-AI-CLPsych/finetuned/llama-2step-2&18epochs/supervised-finetuned"

#Load model
print("Setting tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Loading model in kbit...")

def load_model(model_name: str, adapter_dir: str, tokenizer_for_eos):
    """
    This function handles the instantiation and loading of the huggingface model.  It is used instead of a simple AutoModelForCausalLM call
    so that if a fine-tuned model adapter is present and passed in as an argument, it is merged into the model for use in inference.

    args:
        model_name (str): the huggingface repo name for the model
        adapter_dir (str): the directory path to the saved model weights.  ENSURE that these model weights are only applied to their
                           appropriate model.
        tokenizer_for_eos: Passes in the tokenizer so AutoModelForCausalLM can set eos_token_id on the model.
    returns: 
        output_model: The final model, fine-tuned or base depending on whether adapter_dir is passed in as an empty string.
    """
    if adapter_dir == "":
        #Load the base model.
        output_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            quantization_config=bitsandbytes_config
        )
        output_model = prepare_model_for_kbit_training(output_model)
        output_model.config.pad_token_id = tokenizer.pad_token_id
        return output_model
    else:
        #Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bitsandbytes_config
        )

        #Combine base model with adapter
        output_model = PeftModel.from_pretrained(base_model, adapter_dir)

        #Prepare for training
        output_model = prepare_model_for_kbit_training(output_model)
        output_model.gradient_checkpointing_enable()
        output_model.config.pad_token_id = tokenizer.pad_token_id
        output_model.enable_input_require_grads()
        return output_model

untuned_model = load_model(model_name, adapter_dir, tokenizer)

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

#Preprocess dataset
print("Sorting dataset into defaultdict...")
user_posts = defaultdict(list)
for row in tqdm(df_X):
    if len(user_posts[row["user_id"]]) <max_posts_per_user:
        if not row["post_body"]:
            row["post_body"] = ""
        user_posts[row["user_id"]].append(row["post_body"])

def preprocess(label_df):
    #convert labels into integer counterparts, 1-4
    label = label_df["label"]
    label_map = {None: "no", "b": "yes", "c": "yes", "d": "yes"}
    label = label_map[label] if label in label_map else None
    if label is None:
        raise ValueError(f"Label {label} not found in label map for user {label_df['user_id']}!")

    #Retrieve current_labels matching posts, then join them together
    posts = "\n".join(user_posts[label_df["user_id"]])
    prompt = f"{posts}\nIs the author of the post suicidal? yes or no.  Answer: "
    
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

#Remove "a" labels
df_y = df_y.filter(lambda row: row["label"] != "a")

#Trigger preprocessing
print("Preprocessing sorted dataset...")
preprocessed_df = df_y.map(preprocess, desc="preprocessing", remove_columns=df_y.column_names)
df = dict()
df = preprocessed_df.train_test_split(test_size=0.1, seed=35)
print(f"length of dataset is {len(df['train'])}!")

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
do_fp16 = False
if adapter_dir == "":
    peft_model = peft.get_peft_model(untuned_model, lora_config)
    do_fp16 = True
else: 
    peft_model = untuned_model

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
    fp16=do_fp16,
    num_train_epochs=epochs, #args.epochs,
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
print("Printing trainable params...")
peft_model.print_trainable_parameters()

print("Beginning training...")
trainer.train()

#Save model
print("Saving model...")
peft_model.save_pretrained(f"{output_dir}/lora-finetuned")
print(f"Model saved to {output_dir}/lora-finetuned")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import datasets as ds
from huggingface_hub import login
import os
import re
import argparse
import torch

"""
This script handles the crowd test predictions for a model that is passed in via args.  
This script works agnostic of model inputted.
"""

parser = argparse.ArgumentParser(description="Process model and output csv.")
parser.add_argument("--model", type=str, help="Model name")
parser.add_argument("--adapter_dir", type=str, default="", help="Path to adapter save directory.  Leave blank for base model classification.")
parser.add_argument("--output", type=str, default="", help="Name of output csv")
args = parser.parse_args()

token = os.getenv('token')
assert token, "Environment variable 'token' is not set"
login(token)

######
print("Note for Logs: This is the crowd test classifier")
model_name = args.model
adapter = args.adapter_dir
df_X_path = "../expert/expert_posts.csv"
df_y_path = "../expert/expert.csv"
output_path = f"../results/crowd-test-{args.output}"

######

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
        output_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", pad_token_id=tokenizer_for_eos.eos_token_id)
        return output_model
    else:
        #Use the same conf as when finetuning?
        bitsandbytes_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        #Load base model, but with config for quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            pad_token_id=tokenizer_for_eos.eos_token_id, 
            quantization_config=bitsandbytes_config
        )

        #base_model.config.use_cache=False #NOTE: Suggested by GPT, need to figure out if it is helpful or not before implementing
        #Take base model, and merge adapters to it.
        output_model = PeftModel.from_pretrained(base_model, adapter_dir)
        return output_model

#load model and tokenizer
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for Llama models
model = load_model(model_name, adapter, tokenizer)
model.eval()

#Load in /shared/DATA/reddit/crowd/test/shared_task_posts_test.csv for post level features
df_X = ds.load_dataset("csv", data_files=df_X_path)['train']#/shared/DATA/reddit/crowd/test
#Load in /shared/DATA/reddit/crowd/test/crowd_test.csv for user level labels
df_y = ds.load_dataset("csv", data_files=df_y_path)['train']#/shared/DATA/reddit/crowd/test
separator = "\n\n"

#Because the different datasets have different column naming conventions, rename label to raw_label
df_y = df_y.rename_column("label", "raw_label")

def get_matching_posts(user):
    """
    This function takes a dataset row from a y dataset, and stiches together data from both X and y into a user-level, matched dataset.

    Args:
        user (dict): A dictionary representing a row from the y dataset, containing 'user_id' and 'label'.
    
    Returns:
        dict: All posts from a given user, concatenated into a single text string.
    """
    user_id = user['user_id']
    matching_posts = df_X.filter(lambda row: row['user_id'] == user_id)
    return {"text": separator.join([row['post_body'] for row in matching_posts if row['post_body'] is not None][:10])}

def get_predictions(batch):
    """
    High level controller of the llm pipeline.

    Args:
    batch() - A sub-dataframe passed in by a mapping loop.

    Returns:
    NOTE: Need to add
    """
    prompts = [ 
        f'"{text}"\nIs the author of the post suicidal? yes or no.  Answer: ' for text in batch['text']
    ]
    
    predictions = []
    for prompt in prompts:
        full_probabilities = generate(prompt)
        predictions.append(reduce_to_label(full_probabilities))

    return {"predictions": predictions}


def generate(prepared_prompt):
    """
    A function that handles the huggingface api calls for triggering the model directly.

    Args:
    prepared_prompts() - the tokenized full prompts from get_predictions

    Returns:
    output_probabilities() - the probabilities for which token would be generated by the model(full dictionary)
    """

    #Tokenize the prompts
    tokenized_inputs = tokenizer(
        prepared_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    #Do a forward pass on the model with tokenized inputs
    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    #Get logits for next token
    next_logits = outputs.logits[0, -1, :]

    #Turn logits into probabilities
    output_probabilities = torch.softmax(next_logits, dim=-1)
    return output_probabilities

def reduce_to_label(output_logits):
    """
    Takes still-tokenized output from a model and returns 

    Args:
    output_logits() - The logits from the model generation output

    Returns:
    highest_probability_label(string) - the label with the highest probability
    """
    #From gpt
    labels = ["yes", "no"]
    label_prob_dict = dict()
    for label in labels:
        tid = tokenizer.convert_tokens_to_ids(label)
        label_prob_dict[label] = output_logits[tid].item()
    
    return max(label_prob_dict, key=label_prob_dict.get)

#Create a single dataset out of the key-value pairs of the original dataset
print("Mapping user posts to their corresponding texts...")
df = df_y.map(get_matching_posts, batched=False, desc="Mapping user posts")

#Feedback on the mapping output
print(f"Mapped dataset columns: {df.column_names}")
print(f"Mapped dataset size: {len(df)}")

#remove rows where df['raw_label'] is equal to the string 'a'
df = df.filter(lambda x: x['raw_label'] != "a", batched=False)
print("Nones filtered out")

df = df.map(get_predictions, batched=True, batch_size=2, desc="Generating predictions")
print("Predictions generated")

#Dropping text column for easier readability
df = df.remove_columns("text")

df.to_pandas().to_csv(output_path, index=True)#/home/umflint.edu/brayclou/Health-AI-CLPsych/results
print(f"Predictions saved to {output_path}")

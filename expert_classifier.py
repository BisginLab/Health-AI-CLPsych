from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as ds
from huggingface_hub import login
import os
import re
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Process model and output csv.")
parser.add_argument("--model", type=str, help="Model name")
parser.add_argument("--output", type=str, default="", help="Name of output csv")
args = parser.parse_args()

token = os.getenv('token')
assert token, "Environment variable 'token' is not set"
login(token)

######
print("Note for Logs: This is the expert classifier")
model_name = args.model
df_X_path = "../expert/expert_posts.csv"
df_y_path = "../expert/expert.csv"
output_path = f"../results/expert-{args.output}"
separator = "\n\n"

######

#load model and tokenizer
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for Llama models
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", pad_token_id=tokenizer.eos_token_id)

#Load in /shared/DATA/reddit/crowd/test/shared_task_posts_test.csv for post level features
df_X = ds.load_dataset("csv", data_files=df_X_path)['train']
#Load in /shared/DATA/reddit/crowd/test/crowd_test.csv for user level labels
df_y = ds.load_dataset("csv", data_files=df_y_path)['train']
#Because the different datasets have different column naming conventions, rename label to raw_label
df_y = df_y.rename_column("label", "raw_label")

#Add content of df_X into a defaultdict keyed by user id
user_posts = defaultdict(list)
for row in df_X:
    #If statement filters by SuicideWatch subreddit
    if row["subreddit"] == "SuicideWatch":
        user_posts[row["user_id"]].append(row["post_body"])

def get_matching_posts(user):
    """
    This function takes a dataset row from a y dataset, and stiches together data from both X and y into a user-level, matched dataset.

    Args:
        user (dict): A dictionary representing a row from the y dataset, containing 'user_id' and 'label'.
    
    Returns:
        dict: All posts from a given user, concatenated into a single text string.
    """
    posts = user_posts.get(user["user_id"], [])
    # if len(posts) == 0:
    #     raise ValueError(f"User \"{user['user_id']}\" found to be without any posts!")
    try:
        return {
            "text": separator.join(posts[:10]) if posts else None
        }
    except TypeError:
        return {"text": None}

def clean_label(pred):
    """This function extracts the first occurrence of a match from the prediction string."""
    match = re.search(r'(1|2|3|4)', pred.lower())
    try:
        if match[0] == "1":
            return "a"
        elif match[0] == "2":
            return "b"
        elif match[0] == "3":
            return "c"
        elif match[0] == "4":
            return "d"
        else:
            return "?"
    except TypeError:
        return "?"

def get_predictions(batch):
    prompts = [ 
        f'"{text}"\nIn rating how suicidal the author of the above message is from 1(not suicidal) to 4(extremely suicidal), I choose ' for text in batch['text']
    ]

    #Tokenize prompts
    tokenized_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    #record length of prompt, so that the script doesn't need to decode that part.
    input_ids = tokenized_inputs['input_ids']
    input_lengths = [len(seq) for seq in input_ids]

    responses = model.generate(
        **tokenized_inputs,
        max_new_tokens=15,
        do_sample=False,
        num_beams=1,
        use_cache=True
    )

    decoded_responses = [
        tokenizer.decode(responses[i][input_lengths[i]:], skip_special_tokens=True)
        for i in range(len(responses))
    ]

    # Return exactly one prediction per input text
    predictions = [clean_label(response) for response in decoded_responses]

    return {
        "predictions": predictions,
        "raw_predictions": decoded_responses
    }


#Create a single dataset out of the key-value pairs of the original dataset
print("Mapping user posts to their corresponding texts...")
df = df_y.map(get_matching_posts, batched=False, desc="Mapping user posts")

#Feedback on the mapping output
print(f"Mapped dataset columns: {df.column_names}")
print(f"Mapped dataset size: {len(df)}")

#remove rows where df['raw_label'] is equal to the string 'nan'
df = df.filter(lambda x: x['raw_label'] != None, batched=False)
df = df.filter(lambda x: x['text'] != None, batched=False)
print("Nones filtered out")

df = df.map(get_predictions, batched=True, batch_size=2, desc="Generating predictions")
print("Predictions generated")

df.to_pandas().to_csv(output_path, index=True)#/home/umflint.edu/brayclou/Health-AI-CLPsych/results
print(f"Predictions saved to {output_path}")

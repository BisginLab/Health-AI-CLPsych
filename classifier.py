from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as ds
from huggingface_hub import login
import os
import re

token = os.getenv('token')
assert token, "Environment variable 'token' is not set"
login(token)

#load model and tokenizer
print("Loading model and tokenizer...")
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for Llama models
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", pad_token_id=tokenizer.eos_token_id)

#Load in /shared/DATA/reddit/crowd/test/shared_task_posts_test.csv for post level features
df_X = ds.load_dataset("csv", data_files="../shared_task_posts_test.csv")['train']#/shared/DATA/reddit/crowd/test
#Load in /shared/DATA/reddit/crowd/test/crowd_test.csv for user level labels
df_y = ds.load_dataset("csv", data_files="../crowd_test.csv")['train']#/shared/DATA/reddit/crowd/test
separator = "\n\n"

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
    return {"text": separator.join([row['post_body'] for row in matching_posts if row['post_body'] is not None and row["subreddit"] == "SuicideWatch"][:10])}

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

df = df.map(get_predictions, batched=True, batch_size=2, desc="Generating predictions")

df.to_pandas().to_csv("../llama_8b.csv", index=True)#/home/umflint.edu/brayclou/Health-AI-CLPsych/results

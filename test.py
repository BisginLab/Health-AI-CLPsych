from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as ds
from huggingface_hub import login
import os
import re

token = os.getenv('token')
login(token)

#load model and tokenizer
print("Loading model and tokenizer...")
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#Load in /shared/DATA/reddit/crowd/test/shared_task_posts_test.csv for post level features
df_X = ds.load_dataset("csv", data_files="/shared/DATA/reddit/crowd/test/shared_task_posts_test.csv")['train']
#Load in /shared/DATA/reddit/crowd/test/crowd_test.csv for user level labels
df_y = ds.load_dataset("csv", data_files="/shared/DATA/reddit/crowd/test/crowd_test.csv")['train']
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
    return {"text": separator.join([text for text in matching_posts['post_body'] if text is not None])}

def clean_label(pred):
    match = re.search(r'[abcd]', pred.lower())
    return match.group(0) if match else "?"

def get_predictions(input_batch):
    # get the prompt for text generation, formatted in role/content style
    formatted_prompts = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a health specialist. Classify the following post content from a social media user as \"a\" for non-suicidal, \"b\" for low-risk of suicide, \"c\" for moderate-risk of suicide, or \"d\" for severe-risk of suicide. Provide your response in the format: \"a\", \"b\", \"c\", or \"d\".\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{text}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        for text in input_batch['text']
    ]

    # Generate a response from the model
    tokenized_inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    responses = model.generate(**tokenized_inputs, max_new_tokens=4)

    # Decode the response and extract the relevant information
    decoded_response = tokenizer.batch_decode(responses[0], skip_special_tokens=True)
    cleaned_response = [clean_label(response) for response in decoded_response]
    return {"predictions": cleaned_response}

#Create a single dataset out of the key-value pairs of the original dataset
print("Mapping user posts to their corresponding texts...")
df = df_y.map(get_matching_posts, batched=False, desc="Mapping user posts")

#Feedback on the mapping output
print(f"Mapped dataset columns: {df.column_names}")
print(f"Mapped dataset size: {len(df)}")

df = df.map(get_predictions, batched=True, batch_size=4, desc="Generating predictions")

df.to_pandas().to_csv("/shared/DATA/reddit/crowd/test/baseline_llama_predictions.csv", index=True)
#Imports
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as ds
from huggingface_hub import login
import os
import re
from collections import defaultdict

#Log in to huggingface with api token
token = os.get_env("token")
login("token")

#Defined changable variables
model_name = "meta-llama/Llama-3.2-3B"
feature_df_name = "expert.csv"
label_df_name = "expert_posts.csv"

#Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
untuned_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", pad_token_id=tokenizer.eos_token_id)

#Load dataset
df_X = ds.load_dataset("csv", feature_df_name)
df_y = ds.load_dataset("csv", label_df_name)

#Preprocess dataset
user_posts = defaultdict(list)
for row in df_X:
    #If statement filters by SuicideWatch subreddit
    #if row["subreddit"] == "SuicideWatch":
    user_posts[row["user_id"]].append(row["post_body"])
def preprocess(label_df):
    #convert labels into integer counterparts, 1-4
    label = label_df["raw_label"]
    if label == "a":
        label = "1"
    elif label == "b":
        label = "2"
    elif label == "c":
        label = "3"
    elif label == "d":
        label = "4"

    #Retrieve current_labels matching posts, then join them together
    posts = "\n".join(user_posts[label_df["user_id"]])

    #TODO: tokenize posts
    #TODO: tokenize labels
    #TODO: zero out post weights, so only labels are changed within the model
    #TODO: concatenate posts and labels together
    #return the result


#TODO: Set up lora

#TODO: Set up peft?

#TODO: Set up trainer

#Trigger supervised learning
tuned_model = untuned_model.train()

#TODO: Save model(to hub or locally?)
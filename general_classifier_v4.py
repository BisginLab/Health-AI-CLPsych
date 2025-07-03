from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

#load dataset
print("Loading dataset...")
dataset = load_dataset("csv", data_files="/home/umflint.edu/brayclou/Github repo/datasets/Test_Suicide_Detection.csv")
dataset = dataset.map(lambda row: {"class": 0 if row["class"] == "non-suicide" else 1})

#load model and tokenizer
print("Loading model and tokenizer...")
model_name = "/home/umflint.edu/brayclou/real-llama-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#Define prompt constructor function
def construct_prompt(datapoint):
    return {"prompt":(
        'You are a health specialist. Answer the following text by stating whether or not the patient is suicidal '
        'or not by giving rating of 1 to 10 formatted as number/10.\n\nPatient text: "'
        + datapoint['text'] + '"\n'
    )}
def get_prediction(input_text):
    prompt = construct_prompt({"text": input_text})
    inputs = tokenizer(prompt["prompt"], return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=15)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r'\b(\d{1,2})/10\b', decoded_output)
    prediction = int(match.group(1)) / 10 if match else None
    return {
        "ID": input_text["Unnamed: 0"],
        "True": input_text["class"],
        "Predicted": prediction
    }

#apply prompt structure and tokenizer to dataset
pred_dataset = dataset.map(get_prediction, batched=False, desc="Generating predictions")

#Save to csv
print("Saving predictions to CSV...")
pred_dataset.to_pandas().to_csv("/home/umflint.edu/brayclou/Github repo/results/llama-finetuned/predicted_scores.csv", index=False)
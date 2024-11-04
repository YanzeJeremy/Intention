import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split
import json

with open('../noprompt_merged_casino_dataset.json', 'r') as file:
    data = json.load(file)
train_df = pd.DataFrame(data["train"])
eval_df = pd.DataFrame(data["valid"])
test_df = pd.DataFrame(data["test"])

# Function to split 'text' into 'utterance' and 'label'
def split_text(row):
    utterance, label = row['text'].split(", Label: ")
    return pd.Series([utterance.replace("Utterance: ", "").strip(), label.strip()])

# Apply the split_text function and create separate columns for each DataFrame
train_df[['utterance', 'label']] = train_df.apply(split_text, axis=1)
eval_df[['utterance', 'label']] = eval_df.apply(split_text, axis=1)
test_df[['utterance', 'label']] = test_df.apply(split_text, axis=1)

# Drop the original 'text' column
train_df = train_df.drop(columns=['text'])
eval_df = eval_df.drop(columns=['text'])
test_df = test_df.drop(columns=['text'])

# Filter rows to keep only single-label entries
train_df = train_df[~train_df['label'].str.contains(",")]
eval_df = eval_df[~eval_df['label'].str.contains(",")]
test_df = test_df[~test_df['label'].str.contains(",")]

print(train_df.label.value_counts())
print(eval_df.label.value_counts())
print(test_df.label.value_counts())
# def generate_prompt(data_point):
#     return f"""
#             Classify the text into small-talk, showing-empathy, promote-coordination, no-need, non-strategic, 
#         other-need, uv-part, vouch-fair, elicit-pref, self-need, and return the answer as the corresponding negotiation strategy label. 
# text: {data_point["utterance"]}
# label: {data_point["label"]}""".strip()

def generate_test_prompt(data_point):
    return f"""
You are an excellent assistant able to classify given utterances inside a conversation for their intent.
They are negotiating over a common resource allocation. Here are the definitions for the intents:
+ small-talk: When they engage in small talk while discussing topics apart from the negotiation, in an attempt to build a rapport with the partner.
+ showing-empathy: when there is evidence of positive acknowledgments or empathetic behavior towards a personal context of the partner.
+ promote-coordination: when a participant promotes coordination among the two partners. This can be, for instance, through an explicit offer of a trade or mutual concession, or via an implicit remark suggesting to work together towards a deal.
+ no-need: when a participant points out that they do not need an item based on personal context such as suggesting that they have ample water to spare.
+ elicit-pref: an attempt to discover the preference order of the opponent.
+ uv-part: where a participant undermines the requirements of their opponent.
+ vouch-fair: a callout to fairness for personal benefit, either when acknowledging a fair deal or when the opponent offers a deal that benefits them.
+ self-need: arguments for creating a personal need for an item in the negotiation.
+ other-need: similar to self-Need but is used when the participants discuss a need for someone else rather than themselves
+ non-strategic: If no strategy is evident, the utterance is labelled as non-strategic.
REMEMBER TO CHOOSE ONLY THE CORRECT INTENT. YOU SHOULD USE EXACT SAME LABELS AS GIVEN.
THE FORMAT IS:
INPUT:
UTTERANCE: [utterance]

OUTPUT:
INTENT: [intent]
*rationale: [rationale]

Let's see couple of examples.
UTTERANCE: Hello I need a lot of water to survive.
INTENT: self-need
UTTERANCE: I have a lot of water around me I can use. So I can give you a lot of extra.
INTENT: no-need
UTTERANCE: Ok nice I will give you 3 food if you give me 3 water.
INTENT: promote-coordination
UTTERANCE: OK, that works for me. But I would like a lot of firewood. I like to hunt and cook what I kill.
INTENT: self-need
UTTERANCE: Ok I will give you 2 firewood.
INTENT: non-strategic
UTTERANCE: So I get 3 food and 2 firewood? And you get all of the water?
INTENT: non-strategic
UTTERANCE: Am fine what about you?
INTENT: small-talk
UTTERANCE: Good thanks! So I'm hoping I can get all of the Food, but am willing to negotiate on the other items. 
INTENT: promote-coordination
UTTERANCE: Ohh food?.Thats a very essential property over here.I mean very important.Do you really need it that bad?
INTENT: uv-part
UTTERANCE: It aslo my top priority but i would not if u give me 3 packages of water and  2 firewood in exchange for all food.Agree?
INTENT: promote-coordination

Now let's start.
UTTERANCE: {data_point["utterance"]}
INTENT: 
""".strip()

train_df.loc[:,'text'] = train_df.apply(generate_prompt, axis=1)
eval_df.loc[:,'text'] = eval_df.apply(generate_prompt, axis=1)
print(train_df.head())

y_true = test_df.loc[:,'label']
print(y_true)
X_test = pd.DataFrame(test_df.apply(generate_test_prompt, axis=1), columns=["text"])


train_data = Dataset.from_pandas(train_df[["text"]])
eval_data = Dataset.from_pandas(train_df[["text"]])


base_model_name = "meta-llama/Meta-Llama-3.1-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config, 
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

tokenizer.pad_token_id = tokenizer.eos_token_id


def predict(test, model, tokenizer):
    y_pred = []
    categories = ["small-talk", "showing-empathy", "promote-coordination", "no-need", "non-strategic", 
        "other-need", "uv-part", "vouch-fair", "elicit-pref", "self-need"]
    
    for i in tqdm(range(len(test))):
        string = ''
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=100, 
                        temperature=0.1)
        print(prompt)
        print('--------')
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("INTENT:")[-1].strip()
        print(answer)
        
        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")

    
    return y_pred


y_pred = predict(X_test, model, tokenizer)
print(y_pred)

def evaluate(y_true, y_pred):
    labels = ["small-talk", "showing-empathy", "promote-coordination", "no-need", "non-strategic", 
        "other-need", "uv-part", "vouch-fair", "elicit-pref", "self-need"]
    mapping = {label: idx for idx, label in enumerate(labels)}
    
    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data
    
    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true_mapped)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels, labels=list(range(len(labels))))
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
    print('\nConfusion Matrix:')
    print(conf_matrix)

evaluate(y_true, y_pred)